// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::indexer::TieredMatchDetails;
use crate::protocols::{ExternalSequenceBlockHash, LocalBlockHash, StorageTier, WorkerWithDpRank};

pub const REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY: &str = "remote_kv_reuse_plan";
pub const REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY: &str = "remote_kv_reuse_no_plan_reason";
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteKvReusePlan {
    pub request_id: String,
    pub source_control_endpoint: String,
    /// Source-side sequence/block hashes in the engine's external hash space.
    /// These are parallel to the request blocks starting at `start_block_index`.
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
    /// Position in the request's prefix where `block_hashes[0]` lives.
    pub start_block_index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteKvReuseCandidate {
    pub request_id: String,
    pub source: WorkerWithDpRank,
    pub routing_block_hashes: Vec<LocalBlockHash>,
    pub start_block_index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RemoteKvReuseNoPlanReason {
    Disabled,
    NoRemoteG2Candidate,
    NoContiguousPrefix,
    SourceIsTarget,
    NoSourceControlEndpoint,
    IncompatibleBlockSize,
    SerializationFailed,
}

impl RemoteKvReuseNoPlanReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::NoRemoteG2Candidate => "no_remote_g2_candidate",
            Self::NoContiguousPrefix => "no_contiguous_prefix",
            Self::SourceIsTarget => "source_is_target",
            Self::NoSourceControlEndpoint => "no_source_control_endpoint",
            Self::IncompatibleBlockSize => "incompatible_block_size",
            Self::SerializationFailed => "serialization_failed",
        }
    }
}

pub struct RemoteKvReuseSelectionInput<'a> {
    pub request_id: &'a str,
    pub target: WorkerWithDpRank,
    pub block_hashes: &'a [LocalBlockHash],
    pub tiered_matches: &'a TieredMatchDetails,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RemoteKvReuseSelectionStats {
    pub rejected_g1_candidates: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteKvReuseDecision {
    Candidate {
        candidate: RemoteKvReuseCandidate,
        stats: RemoteKvReuseSelectionStats,
    },
    Plan {
        plan: RemoteKvReusePlan,
        stats: RemoteKvReuseSelectionStats,
    },
    NoPlan {
        reason: RemoteKvReuseNoPlanReason,
        stats: RemoteKvReuseSelectionStats,
    },
}

pub fn select_remote_g2_reuse_plan(
    input: RemoteKvReuseSelectionInput<'_>,
) -> RemoteKvReuseDecision {
    let stats = RemoteKvReuseSelectionStats {
        rejected_g1_candidates: input
            .tiered_matches
            .device
            .overlap_scores
            .scores
            .values()
            .filter(|&&overlap| overlap > 0)
            .count() as u32,
    };

    let Some(host_pinned_matches) = input
        .tiered_matches
        .lower_tier
        .get(&StorageTier::HostPinned)
    else {
        return RemoteKvReuseDecision::NoPlan {
            reason: RemoteKvReuseNoPlanReason::NoRemoteG2Candidate,
            stats,
        };
    };

    let mut saw_remote_candidate = false;
    let mut best: Option<(WorkerWithDpRank, usize)> = None;
    for (&worker, &hits) in &host_pinned_matches.cross_worker_hits {
        if worker == input.target {
            continue;
        }
        saw_remote_candidate = true;
        if hits == 0 {
            continue;
        }
        match best {
            None => best = Some((worker, hits)),
            Some((best_worker, best_hits))
                if hits > best_hits || (hits == best_hits && worker < best_worker) =>
            {
                best = Some((worker, hits));
            }
            Some(_) => {}
        }
    }

    let Some((source, hits)) = best else {
        return RemoteKvReuseDecision::NoPlan {
            reason: if saw_remote_candidate {
                RemoteKvReuseNoPlanReason::NoContiguousPrefix
            } else {
                RemoteKvReuseNoPlanReason::NoRemoteG2Candidate
            },
            stats,
        };
    };

    // Plan window math.
    //
    // The plan covers a contiguous range [start, end) of the request's block
    // hashes that the SOURCE has on HostPinned and that the TARGET needs to
    // load. The right anchor for `start` is the TARGET's device-tier coverage:
    // the target will satisfy positions [0, target_device_match) from its own
    // GPU prefix cache (the v1 scheduler short-circuits OffloadingConnector
    // lookup on a prefix-cache hit), so the plan only needs to cover from
    // there onward.
    //
    // Using the SOURCE's device-tier coverage instead is only correct when
    // target == source (the within-worker case the predicate already skips
    // above), and is actively wrong for cross-worker pulls: in a co-located
    // setup the source typically still holds the prefix on its GPU prefix
    // cache after also offloading it, so source_device_match == request_blocks
    // and the plan window collapses to zero — silently dropping pulls that
    // are correct and available.
    //
    // The MAX with the source's host-pinned-start handles the case where the
    // source's HostPinned chain itself begins past position 0 (an extension
    // of its own device chain). In our patched `query_lower_tiers` we always
    // also consider a fresh-start path for each tier, so when the source's
    // host-pinned blocks are accessible from position 0 the chain start is 0
    // and `target_device_match` dominates.
    let target_device_match = input
        .tiered_matches
        .device
        .overlap_scores
        .scores
        .get(&input.target)
        .copied()
        .unwrap_or(0) as usize;
    let source_hp_start = input
        .tiered_matches
        .lower_tier
        .get(&StorageTier::HostPinned)
        .and_then(|m| m.next_continuations.get(&source))
        .map(|c| c.start_pos.saturating_sub(hits as usize))
        .unwrap_or(0);

    let request_blocks = input.block_hashes.len();
    let start = target_device_match.max(source_hp_start).min(request_blocks);
    let available_after_target_device = request_blocks.saturating_sub(start);

    // `hits` is the number of source's host-pinned blocks measured from
    // `source_hp_start`. The portion of those that is useful to the target
    // is the overlap with [start, request_blocks).
    let source_hp_end = source_hp_start.saturating_add(hits as usize);
    let useful_source_hp = source_hp_end.saturating_sub(start);
    let planned_prefix_blocks = useful_source_hp.min(available_after_target_device) as u32;
    if planned_prefix_blocks == 0 {
        return RemoteKvReuseDecision::NoPlan {
            reason: RemoteKvReuseNoPlanReason::NoContiguousPrefix,
            stats,
        };
    }
    let end = start + planned_prefix_blocks as usize;

    RemoteKvReuseDecision::Candidate {
        candidate: RemoteKvReuseCandidate {
            request_id: input.request_id.to_string(),
            source,
            routing_block_hashes: input.block_hashes[start..end].to_vec(),
            start_block_index: start as u32,
        },
        stats,
    }
}

#[cfg(test)]
mod tests {
    use crate::indexer::{
        LowerTierContinuation, LowerTierMatchDetails, MatchDetails, TieredMatchDetails,
    };
    use crate::protocols::{
        ExternalSequenceBlockHash, LocalBlockHash, OverlapScores, StorageTier, WorkerWithDpRank,
    };
    use crate::remote_g2_plan::{
        RemoteKvReuseDecision, RemoteKvReuseNoPlanReason, RemoteKvReusePlan,
        RemoteKvReuseSelectionInput, select_remote_g2_reuse_plan,
    };

    fn test_plan() -> RemoteKvReusePlan {
        RemoteKvReusePlan {
            request_id: "request-1".to_string(),
            source_control_endpoint: "tcp://source:1234".to_string(),
            block_hashes: vec![ExternalSequenceBlockHash(11), ExternalSequenceBlockHash(22)],
            start_block_index: 0,
        }
    }

    fn block_hashes(count: u64) -> Vec<LocalBlockHash> {
        (0..count).map(LocalBlockHash).collect()
    }

    fn tiered_matches(
        device_hits: &[(WorkerWithDpRank, u32)],
        host_pinned_hits: &[(WorkerWithDpRank, usize)],
    ) -> TieredMatchDetails {
        let mut device = MatchDetails {
            overlap_scores: OverlapScores::new(),
            ..Default::default()
        };
        device
            .overlap_scores
            .scores
            .extend(device_hits.iter().copied());

        let mut lower_tier = std::collections::HashMap::new();
        let mut host_pinned = LowerTierMatchDetails::default();
        host_pinned.hits.extend(host_pinned_hits.iter().copied());
        host_pinned
            .cross_worker_hits
            .extend(host_pinned_hits.iter().copied());
        for &(worker, hits) in host_pinned_hits {
            let device_hit = device_hits
                .iter()
                .find_map(|&(device_worker, device_hit)| {
                    (device_worker == worker).then_some(device_hit as usize)
                })
                .unwrap_or(0);
            host_pinned.next_continuations.insert(
                worker,
                LowerTierContinuation::new(
                    device_hit + hits,
                    ExternalSequenceBlockHash(device_hit.saturating_add(hits) as u64),
                ),
            );
        }
        lower_tier.insert(StorageTier::HostPinned, host_pinned);

        TieredMatchDetails { device, lower_tier }
    }

    fn selection_input<'a>(
        target: WorkerWithDpRank,
        block_hashes: &'a [LocalBlockHash],
        tiered_matches: &'a TieredMatchDetails,
    ) -> RemoteKvReuseSelectionInput<'a> {
        RemoteKvReuseSelectionInput {
            request_id: "request-1",
            target,
            block_hashes,
            tiered_matches,
        }
    }

    #[test]
    fn serde_plan_round_trips_basic() {
        let plan = test_plan();
        let json = serde_json::to_string(&plan).unwrap();
        let decoded: RemoteKvReusePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, plan);
    }

    #[test]
    fn serde_plan_has_only_target_contract_fields() {
        let json = serde_json::to_string(&test_plan()).unwrap();
        assert!(json.contains("source_control_endpoint"));
        assert!(json.contains("block_hashes"));
        for forbidden in [
            "plan_id",
            "target_worker_id",
            "target_dp_rank",
            "source_worker_id",
            "source_dp_rank",
            "source_control_location",
            "source_tier",
            "planned_prefix_blocks",
            "block_size_tokens",
            "created_at_ms",
            "expires_at_ms",
            "plan_version",
            "kv_block_hashes",
            "virtual_address",
            "physical_address",
            "nixl_descriptor",
            "descriptor",
            "target_g1_block_id",
            "source_block_id",
            "block_ptr",
            "handle",
        ] {
            assert!(
                !json.contains(forbidden),
                "serialized plan contains forbidden field: {forbidden}"
            );
        }
    }

    #[test]
    fn serde_no_plan_reason_uses_snake_case() {
        let json = serde_json::to_string(&RemoteKvReuseNoPlanReason::NoRemoteG2Candidate).unwrap();
        assert_eq!(json, "\"no_remote_g2_candidate\"");
    }

    #[test]
    fn select_longest_remote_g2_prefix() {
        let hashes = block_hashes(5);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(
            &[],
            &[
                (WorkerWithDpRank::new(7, 0), 2),
                (WorkerWithDpRank::new(8, 0), 4),
            ],
        );

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Candidate { candidate, .. } => {
                assert_eq!(candidate.source, WorkerWithDpRank::new(8, 0));
                assert_eq!(candidate.routing_block_hashes, hashes[..4].to_vec());
            }
            other => panic!("expected candidate, got {other:?}"),
        }
    }

    #[test]
    fn select_tie_break_by_worker_then_rank() {
        let hashes = block_hashes(4);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(
            &[],
            &[
                (WorkerWithDpRank::new(8, 1), 3),
                (WorkerWithDpRank::new(7, 3), 3),
                (WorkerWithDpRank::new(7, 1), 3),
            ],
        );

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Candidate { candidate, .. } => {
                assert_eq!(candidate.source, WorkerWithDpRank::new(7, 1));
                assert_eq!(candidate.routing_block_hashes.len(), 3);
            }
            other => panic!("expected candidate, got {other:?}"),
        }
    }

    #[test]
    fn select_rejects_g1_only_device_hits() {
        let hashes = block_hashes(2);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = TieredMatchDetails {
            device: {
                let mut device = MatchDetails {
                    overlap_scores: OverlapScores::new(),
                    ..Default::default()
                };
                device
                    .overlap_scores
                    .scores
                    .insert(WorkerWithDpRank::new(7, 0), 2);
                device
            },
            lower_tier: Default::default(),
        };

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, stats } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoRemoteG2Candidate);
                assert!(stats.rejected_g1_candidates > 0);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    // Target identity is the full (worker_id, dp_rank) pair, not just
    // worker_id. A peer with the same worker_id but a different dp_rank is a
    // distinct KV-cache owner and must be eligible as a remote source.
    #[test]
    fn select_distinguishes_target_by_dp_rank() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(9, 1);
        let matches = tiered_matches(&[], &[(source, 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Candidate { candidate, .. } => {
                assert_eq!(candidate.source, source);
                assert_eq!(candidate.routing_block_hashes.len(), 3);
            }
            other => panic!("expected candidate, got {other:?}"),
        }
    }

    #[test]
    fn select_candidate_metadata_propagates_from_input() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(42, 2);
        let source = WorkerWithDpRank::new(7, 1);
        let matches = tiered_matches(&[], &[(source, 3)]);
        let input = RemoteKvReuseSelectionInput {
            request_id: "req-meta",
            target,
            block_hashes: &hashes,
            tiered_matches: &matches,
        };

        let decision = select_remote_g2_reuse_plan(input);

        match decision {
            RemoteKvReuseDecision::Candidate { candidate, .. } => {
                assert_eq!(candidate.request_id, "req-meta");
                assert_eq!(candidate.source, source);
                assert_eq!(candidate.start_block_index, 0);
                assert_eq!(candidate.routing_block_hashes, hashes);
            }
            other => panic!("expected candidate, got {other:?}"),
        }
    }

    #[test]
    fn scenario_zero_host_pinned_hits_no_contiguous_prefix() {
        let hashes = block_hashes(2);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(&[], &[(WorkerWithDpRank::new(7, 0), 0)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, .. } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoContiguousPrefix);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    #[test]
    fn scenario_empty_request_no_contiguous_prefix() {
        let hashes: Vec<LocalBlockHash> = Vec::new();
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(&[], &[(WorkerWithDpRank::new(7, 0), 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, .. } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoContiguousPrefix);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    #[test]
    fn scenario_g1_partial_g2_tail_start_at_device_match() {
        let hashes = block_hashes(6);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(source, 2)], &[(source, 2)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Candidate { candidate, .. } => {
                assert_eq!(candidate.start_block_index, 2);
                assert_eq!(candidate.routing_block_hashes, hashes[2..4].to_vec());
            }
            other => panic!("expected candidate, got {other:?}"),
        }
    }

    #[test]
    fn scenario_zero_overlap_no_remote_g2_candidate() {
        let hashes = block_hashes(4);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(&[], &[]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, stats } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoRemoteG2Candidate);
                assert_eq!(stats.rejected_g1_candidates, 0);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    #[test]
    fn scenario_target_only_host_pinned_no_remote_g2_candidate() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(&[], &[(target, 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, stats } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoRemoteG2Candidate);
                assert_eq!(stats.rejected_g1_candidates, 0);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    #[test]
    fn scenario_no_host_pinned_tier_no_remote_g2_candidate() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = TieredMatchDetails {
            device: MatchDetails {
                overlap_scores: OverlapScores::new(),
                ..Default::default()
            },
            lower_tier: Default::default(),
        };

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, stats } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoRemoteG2Candidate);
                assert_eq!(stats.rejected_g1_candidates, 0);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    #[test]
    fn scenario_full_g2_no_g1_full_coverage_candidate() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Candidate { candidate, stats } => {
                assert_eq!(candidate.source, source);
                assert_eq!(candidate.start_block_index, 0);
                assert_eq!(candidate.routing_block_hashes, hashes);
                assert_eq!(stats.rejected_g1_candidates, 0);
            }
            other => panic!("expected candidate, got {other:?}"),
        }
    }

    #[test]
    fn scenario_partial_g2_no_g1_matched_prefix_only() {
        let hashes = block_hashes(5);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Candidate { candidate, stats } => {
                assert_eq!(candidate.source, source);
                assert_eq!(candidate.start_block_index, 0);
                assert_eq!(candidate.routing_block_hashes, hashes[..3].to_vec());
                assert!(candidate.routing_block_hashes.len() < hashes.len());
                assert_eq!(stats.rejected_g1_candidates, 0);
            }
            other => panic!("expected candidate, got {other:?}"),
        }
    }

    #[test]
    fn scenario_full_g1_extra_g2_no_contiguous_prefix() {
        let hashes = block_hashes(4);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(source, 4)], &[(source, 2)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, stats } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoContiguousPrefix);
                assert!(stats.rejected_g1_candidates > 0);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }
}
