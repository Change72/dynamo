// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::indexer::{CrossWorkerSpan, TieredMatchDetails};
use crate::protocols::{
    DpRank, ExternalSequenceBlockHash, LocalBlockHash, StorageTier, WorkerId, WorkerWithDpRank,
};

pub const REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY: &str = "remote_kv_reuse_plan";
pub const REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY: &str = "remote_kv_reuse_no_plan_reason";
pub const REMOTE_KV_REUSE_PLAN_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteKvReusePlan {
    pub plan_id: String,
    pub request_id: String,
    pub target_worker_id: WorkerId,
    pub target_dp_rank: DpRank,
    pub source_worker_id: WorkerId,
    pub source_dp_rank: DpRank,
    pub source_tier: StorageTier,
    pub block_hashes: Vec<LocalBlockHash>,
    /// Position in the request's prefix where `block_hashes[0]` lives.
    /// Equals the target worker's device-tier prefix length at plan time.
    /// The target's connector uses this to verify alignment with its own
    /// `num_computed_tokens` before attaching descriptors.
    pub start_block_index: u32,
    pub planned_prefix_blocks: u32,
    pub block_size_tokens: u32,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
    pub plan_version: u32,
    /// Parallel to `block_hashes`, carrying each block's source-side
    /// KV-cache-manager hash (TRT-LLM splitmix). The TRT-LLM source side
    /// uses these values to look up blocks via `find_block_by_hash`;
    /// `block_hashes` (XXH3 tokens hash) remains the plan's identity.
    /// Empty when the producer has not been updated to populate the new
    /// field — TRT-LLM's source falls back to using `block_hashes` for
    /// the lookup (legacy behavior).
    #[serde(default)]
    pub kv_block_hashes: Vec<u64>,
}

// Compatibility identity is intentionally deferred in v1; source resolve remains authoritative.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RemoteKvReuseNoPlanReason {
    Disabled,
    NoRemoteG2Candidate,
    NoContiguousPrefix,
    SourceIsTarget,
    IncompatibleBlockSize,
    PlanExpired,
    SerializationFailed,
}

impl RemoteKvReuseNoPlanReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::NoRemoteG2Candidate => "no_remote_g2_candidate",
            Self::NoContiguousPrefix => "no_contiguous_prefix",
            Self::SourceIsTarget => "source_is_target",
            Self::IncompatibleBlockSize => "incompatible_block_size",
            Self::PlanExpired => "plan_expired",
            Self::SerializationFailed => "serialization_failed",
        }
    }
}

pub struct RemoteKvReuseSelectionInput<'a> {
    pub request_id: &'a str,
    pub target: WorkerWithDpRank,
    pub block_hashes: &'a [LocalBlockHash],
    pub block_size_tokens: u32,
    pub tiered_matches: &'a TieredMatchDetails,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RemoteKvReuseSelectionStats {
    pub rejected_g1_candidates: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteKvReuseDecision {
    Plan {
        plan: RemoteKvReusePlan,
        stats: RemoteKvReuseSelectionStats,
        /// Exact local lower-tier span chosen by the target-aware selector.
        /// This is router-internal materialization state and is intentionally
        /// absent from the serialized [`RemoteKvReusePlan`].
        selected_span: CrossWorkerSpan,
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

    let request_blocks = input.block_hashes.len();
    let target_device_match = (input
        .tiered_matches
        .device
        .overlap_scores
        .scores
        .get(&input.target)
        .copied()
        .unwrap_or(0) as usize)
        .min(request_blocks);

    // Plan selection is target-relative. Raw lower-tier hit counts cannot be
    // ranked before the target is known: a long source span may end before the
    // target's local prefix, while a shorter span beginning at that boundary is
    // fully useful. A span is consumable only when it covers the very next
    // target block; vLLM's maximal-prefix lookup stops on the first gap.
    let mut saw_remote_candidate = false;
    let mut best: Option<(WorkerWithDpRank, CrossWorkerSpan, usize)> = None;
    for (&worker, spans) in &host_pinned_matches.cross_worker_spans {
        if worker == input.target {
            continue;
        }
        if spans.iter().any(|span| span.end_pos > span.start_pos) {
            saw_remote_candidate = true;
        }
        for &span in spans {
            let end = span.end_pos.min(request_blocks);
            if span.start_pos > target_device_match || end <= target_device_match {
                continue;
            }
            let useful = end - target_device_match;
            let better = match best {
                None => true,
                Some((best_worker, best_span, best_useful)) => {
                    useful > best_useful
                        || (useful == best_useful
                            && (worker < best_worker
                                || (worker == best_worker
                                    && (span.start_pos > best_span.start_pos
                                        || (span.start_pos == best_span.start_pos
                                            && span.parent_hash < best_span.parent_hash)))))
                }
            };
            if better {
                best = Some((worker, span, useful));
            }
        }
    }

    // Wire-inbound results currently carry only the raw summary and no local
    // materialization spans. Count them as candidates, but never guess their
    // anchors: they must fail closed until remote materialization is supported.
    saw_remote_candidate |= host_pinned_matches
        .cross_worker_hits
        .iter()
        .any(|(&worker, &hits)| worker != input.target && hits > 0);

    let Some((source, selected_span, useful)) = best else {
        return RemoteKvReuseDecision::NoPlan {
            reason: if saw_remote_candidate {
                RemoteKvReuseNoPlanReason::NoContiguousPrefix
            } else {
                RemoteKvReuseNoPlanReason::NoRemoteG2Candidate
            },
            stats,
        };
    };

    let start = target_device_match;
    let end = start + useful;
    let planned_prefix_blocks = useful as u32;

    RemoteKvReuseDecision::Plan {
        plan: RemoteKvReusePlan {
            plan_id: format!(
                "remote-g2:{}:{}:{}:{}",
                input.request_id, source.worker_id, source.dp_rank, input.created_at_ms
            ),
            request_id: input.request_id.to_string(),
            target_worker_id: input.target.worker_id,
            target_dp_rank: input.target.dp_rank,
            source_worker_id: source.worker_id,
            source_dp_rank: source.dp_rank,
            source_tier: StorageTier::HostPinned,
            block_hashes: input.block_hashes[start..end].to_vec(),
            start_block_index: start as u32,
            planned_prefix_blocks,
            block_size_tokens: input.block_size_tokens,
            created_at_ms: input.created_at_ms,
            expires_at_ms: input.expires_at_ms,
            plan_version: REMOTE_KV_REUSE_PLAN_VERSION,
            // Caller fills this in post-selection by walking the indexer for
            // the chosen source. Left empty here so the planner stays a pure
            // function of `tiered_matches` and does not depend on the indexer.
            kv_block_hashes: Vec::new(),
        },
        stats,
        selected_span,
    }
}

/// Populate a selected plan with source-side external block hashes.
///
/// The selector carries its exact local span outside the serialized plan. This
/// function validates that the plan window remains inside that span, walks from
/// the recorded parent anchor, and drops only the prefix already present on the
/// target. Any inconsistent or stale state fails closed. If an eviction races
/// the walk, a non-empty shorter chain shrinks the plan to the materialized
/// prefix; an empty chain becomes `NoContiguousPrefix`.
pub fn materialize_remote_g2_reuse_plan<F>(
    decision: RemoteKvReuseDecision,
    request_hashes: &[LocalBlockHash],
    mut resolve_chain: F,
) -> RemoteKvReuseDecision
where
    F: FnMut(
        WorkerWithDpRank,
        Option<ExternalSequenceBlockHash>,
        &[LocalBlockHash],
    ) -> Vec<ExternalSequenceBlockHash>,
{
    let (mut plan, stats, selected_span) = match decision {
        RemoteKvReuseDecision::Plan {
            plan,
            stats,
            selected_span,
        } => (plan, stats, selected_span),
        no_plan @ RemoteKvReuseDecision::NoPlan { .. } => return no_plan,
    };

    let fail = || RemoteKvReuseDecision::NoPlan {
        reason: RemoteKvReuseNoPlanReason::NoContiguousPrefix,
        stats,
    };
    let start = plan.start_block_index as usize;
    let Some(end) = start.checked_add(plan.planned_prefix_blocks as usize) else {
        return fail();
    };
    if selected_span.start_pos > start
        || start >= end
        || end > selected_span.end_pos
        || end > request_hashes.len()
        || plan.block_hashes.len() != end - start
        || plan.block_hashes.as_slice() != &request_hashes[start..end]
    {
        return fail();
    }

    let source = WorkerWithDpRank::new(plan.source_worker_id, plan.source_dp_rank);
    let full_chain = resolve_chain(
        source,
        selected_span.parent_hash,
        &request_hashes[selected_span.start_pos..end],
    );
    let skip = start - selected_span.start_pos;
    if full_chain.len() <= skip {
        return fail();
    }

    let materialized = (full_chain.len() - skip).min(end - start);
    if materialized == 0 {
        return fail();
    }
    plan.planned_prefix_blocks = materialized as u32;
    plan.block_hashes.truncate(materialized);
    plan.kv_block_hashes = full_chain[skip..skip + materialized]
        .iter()
        .map(|hash| hash.0)
        .collect();

    RemoteKvReuseDecision::Plan {
        plan,
        stats,
        selected_span,
    }
}

#[cfg(test)]
mod tests {
    // Test naming convention:
    // - `serde_*`    — wire-format contract (round-trip, back-compat,
    //                  forbidden fields, enum casing).
    // - `select_*`   — selection algorithm (which source wins, what
    //                  the constructor populates on the chosen plan).
    // - `scenario_*` — full plan-shape behavior under a given input
    //                  (request × G1 × G2 combinations, plan-or-no-plan
    //                  outcome and reason).

    use crate::indexer::{
        CrossWorkerSpan, LowerTierContinuation, LowerTierMatchDetails, MatchDetails,
        TieredMatchDetails, WireTieredMatchDetails,
    };
    use crate::protocols::{
        ExternalSequenceBlockHash, LocalBlockHash, OverlapScores, StorageTier, WorkerWithDpRank,
    };
    use crate::remote_g2_plan::{
        REMOTE_KV_REUSE_PLAN_VERSION, RemoteKvReuseDecision, RemoteKvReuseNoPlanReason,
        RemoteKvReusePlan, RemoteKvReuseSelectionInput, materialize_remote_g2_reuse_plan,
        select_remote_g2_reuse_plan,
    };

    fn test_plan() -> RemoteKvReusePlan {
        RemoteKvReusePlan {
            plan_id: "plan-1".to_string(),
            request_id: "request-1".to_string(),
            target_worker_id: 9,
            target_dp_rank: 0,
            source_worker_id: 7,
            source_dp_rank: 1,
            source_tier: StorageTier::HostPinned,
            block_hashes: vec![LocalBlockHash(11), LocalBlockHash(22)],
            start_block_index: 0,
            planned_prefix_blocks: 2,
            block_size_tokens: 16,
            created_at_ms: 1000,
            expires_at_ms: 2000,
            plan_version: REMOTE_KV_REUSE_PLAN_VERSION,
            kv_block_hashes: vec![],
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
        // Keep the legacy summary and local materialization spans consistent
        // in ordinary fixtures; dedicated malformed/wire tests break that
        // relationship explicitly to verify fail-closed behavior.
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
            // These fixtures model a source whose host-pinned chain extends its
            // own device coverage, so its span begins at `device_hit` and is
            // anchored by the preceding external hash.
            host_pinned.cross_worker_spans.insert(
                worker,
                vec![CrossWorkerSpan {
                    start_pos: device_hit,
                    end_pos: device_hit.saturating_add(hits),
                    parent_hash: (device_hit > 0)
                        .then_some(ExternalSequenceBlockHash(device_hit as u64)),
                }],
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
            block_size_tokens: 16,
            tiered_matches,
            created_at_ms: 1000,
            expires_at_ms: 2000,
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
    fn serde_plan_round_trips_kv_block_hashes() {
        // Populated kv_block_hashes must appear in the JSON and survive
        // a serialize → deserialize round trip with the exact same values.
        let mut plan = test_plan();
        plan.kv_block_hashes = vec![
            0xAAAA_AAAA_AAAA_AAAA,
            0xBBBB_BBBB_BBBB_BBBB,
            0xCCCC_CCCC_CCCC_CCCC,
        ];
        let json = serde_json::to_string(&plan).unwrap();
        assert!(
            json.contains("\"kv_block_hashes\""),
            "serialized plan missing kv_block_hashes field: {json}"
        );
        // Big values must serialize as integers, not stringified
        assert!(json.contains("12297829382473034410"));
        let decoded: RemoteKvReusePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.kv_block_hashes, plan.kv_block_hashes);
    }

    #[test]
    fn serde_plan_accepts_legacy_payload_without_kv_block_hashes() {
        // A producer that has not been updated to populate kv_block_hashes
        // emits the field-less JSON; it must still deserialize, with the
        // new field defaulting to empty.
        let plan = test_plan();
        let json = serde_json::to_string(&plan).unwrap();
        // Strip the kv_block_hashes field from the JSON to simulate a
        // legacy producer.
        let legacy = json.replace(",\"kv_block_hashes\":[]", "");
        assert!(
            !legacy.contains("kv_block_hashes"),
            "legacy payload should not contain kv_block_hashes"
        );
        let decoded: RemoteKvReusePlan = serde_json::from_str(&legacy).unwrap();
        assert!(decoded.kv_block_hashes.is_empty());
    }

    #[test]
    fn serde_plan_has_no_router_truth_fields() {
        let json = serde_json::to_string(&test_plan()).unwrap();
        for forbidden in [
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
                "serialized plan contains forbidden router truth: {forbidden}"
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
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.source_worker_id, 8);
                assert_eq!(plan.source_dp_rank, 0);
                assert_eq!(plan.planned_prefix_blocks, 4);
                assert_eq!(plan.block_hashes, hashes[..4].to_vec());
            }
            other => panic!("expected plan, got {other:?}"),
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
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.source_worker_id, 7);
                assert_eq!(plan.source_dp_rank, 1);
                assert_eq!(plan.planned_prefix_blocks, 3);
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn select_uses_target_relative_span_not_raw_path_winner() {
        let hashes = block_hashes(5);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let mut matches = tiered_matches(&[(target, 3)], &[(source, 5)]);
        matches
            .lower_tier
            .get_mut(&StorageTier::HostPinned)
            .unwrap()
            .cross_worker_spans
            .insert(
                source,
                vec![
                    CrossWorkerSpan {
                        start_pos: 0,
                        end_pos: 2,
                        parent_hash: None,
                    },
                    CrossWorkerSpan {
                        start_pos: 3,
                        end_pos: 5,
                        parent_hash: Some(ExternalSequenceBlockHash(30)),
                    },
                ],
            );

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan {
                plan,
                selected_span,
                ..
            } => {
                assert_eq!(plan.start_block_index, 3);
                assert_eq!(plan.planned_prefix_blocks, 2);
                assert_eq!(plan.block_hashes, hashes[3..5].to_vec());
                assert_eq!(selected_span.start_pos, 3);
            }
            other => panic!("expected target-useful extension plan, got {other:?}"),
        }
    }

    #[test]
    fn select_useful_source_beats_larger_unusable_raw_hit_count() {
        let hashes = block_hashes(6);
        let target = WorkerWithDpRank::new(9, 0);
        let unusable = WorkerWithDpRank::new(7, 0);
        let usable = WorkerWithDpRank::new(8, 0);
        let mut matches = tiered_matches(&[(target, 3)], &[(unusable, 8), (usable, 3)]);
        let host = matches
            .lower_tier
            .get_mut(&StorageTier::HostPinned)
            .unwrap();
        host.cross_worker_spans.insert(
            unusable,
            vec![CrossWorkerSpan {
                start_pos: 0,
                end_pos: 3,
                parent_hash: None,
            }],
        );
        host.cross_worker_spans.insert(
            usable,
            vec![CrossWorkerSpan {
                start_pos: 2,
                end_pos: 5,
                parent_hash: Some(ExternalSequenceBlockHash(20)),
            }],
        );

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.source_worker_id, usable.worker_id);
                assert_eq!(plan.start_block_index, 3);
                assert_eq!(plan.planned_prefix_blocks, 2);
            }
            other => panic!("expected usable source plan, got {other:?}"),
        }
    }

    #[test]
    fn select_rejects_span_separated_from_target_prefix_by_gap() {
        let hashes = block_hashes(5);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(source, 2)], &[(source, 2)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        assert!(matches!(
            decision,
            RemoteKvReuseDecision::NoPlan {
                reason: RemoteKvReuseNoPlanReason::NoContiguousPrefix,
                ..
            }
        ));
    }

    #[test]
    fn select_wire_hits_without_local_span_fail_closed() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let local_matches = tiered_matches(&[], &[(source, 3)]);
        let wire = WireTieredMatchDetails::from(&local_matches);
        let matches = TieredMatchDetails::from(wire);
        assert!(
            matches.lower_tier[&StorageTier::HostPinned]
                .cross_worker_spans
                .is_empty(),
            "materialization spans are deliberately local-only"
        );

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        assert!(matches!(
            decision,
            RemoteKvReuseDecision::NoPlan {
                reason: RemoteKvReuseNoPlanReason::NoContiguousPrefix,
                ..
            }
        ));
    }

    #[test]
    fn materialize_empty_chain_fails_closed() {
        let hashes = block_hashes(5);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(target, 2)], &[(source, 5)]);
        let selected = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        let decision = materialize_remote_g2_reuse_plan(selected, &hashes, |_, _, _| Vec::new());

        assert!(matches!(
            decision,
            RemoteKvReuseDecision::NoPlan {
                reason: RemoteKvReuseNoPlanReason::NoContiguousPrefix,
                ..
            }
        ));
    }

    #[test]
    fn materialize_racing_short_chain_shrinks_plan_consistently() {
        let hashes = block_hashes(5);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(target, 2)], &[(source, 5)]);
        let selected = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        let decision = materialize_remote_g2_reuse_plan(selected, &hashes, |_, _, requested| {
            assert_eq!(requested, hashes.as_slice());
            vec![
                ExternalSequenceBlockHash(100),
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
                ExternalSequenceBlockHash(103),
            ]
        });

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.planned_prefix_blocks, 2);
                assert_eq!(plan.block_hashes, hashes[2..4]);
                assert_eq!(plan.kv_block_hashes, vec![102, 103]);
            }
            other => panic!("expected a shortened materialized plan, got {other:?}"),
        }
    }

    #[test]
    fn materialize_inconsistent_plan_window_fails_before_resolver() {
        let hashes = block_hashes(5);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(target, 2)], &[(source, 5)]);
        let mut selected = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));
        let RemoteKvReuseDecision::Plan { plan, .. } = &mut selected else {
            panic!("fixture must select a plan");
        };
        plan.block_hashes[0] = LocalBlockHash(u64::MAX);

        let decision = materialize_remote_g2_reuse_plan(selected, &hashes, |_, _, _| {
            panic!("resolver must not run for an inconsistent plan")
        });

        assert!(matches!(
            decision,
            RemoteKvReuseDecision::NoPlan {
                reason: RemoteKvReuseNoPlanReason::NoContiguousPrefix,
                ..
            }
        ));
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

    #[test]
    fn select_preserves_target_identity() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(42, 2);
        let matches = tiered_matches(&[], &[(WorkerWithDpRank::new(7, 0), 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.target_worker_id, 42);
                assert_eq!(plan.target_dp_rank, 2);
                assert_eq!(plan.source_worker_id, 7);
                assert_eq!(plan.source_dp_rank, 0);
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    // Target identity is the full (worker_id, dp_rank) pair, not just
    // worker_id. A peer with the same worker_id but a different dp_rank
    // (e.g. another DP rank of the same physical worker) is a distinct
    // KV-cache owner and must be eligible as a remote source.
    #[test]
    fn select_distinguishes_target_by_dp_rank() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(9, 1);
        let matches = tiered_matches(&[], &[(source, 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.source_worker_id, 9);
                assert_eq!(plan.source_dp_rank, 1);
                assert_eq!(plan.planned_prefix_blocks, 3);
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    // Wire-contract pinning: every metadata field on the constructed plan
    // must flow through from the input (or be a known constant). The
    // other select_* tests assert *which* source the planner picks; this
    // one asserts the constructor populates the plan correctly once the
    // pick is made. A regression here breaks the target worker's view of
    // the plan without breaking any existing test.
    //
    // The asserted invariants:
    // - request_id, block_size_tokens, created_at_ms, expires_at_ms come
    //   straight from input
    // - plan_version equals the REMOTE_KV_REUSE_PLAN_VERSION constant
    //   (not a literal — bumping the constant must propagate)
    // - source_tier is always HostPinned (the only tier the planner
    //   considers)
    // - kv_block_hashes is empty here; the caller in kv_router.rs
    //   populates it post-selection by walking the indexer
    // - plan_id format is "remote-g2:<request_id>:<source_worker>:<source_dp_rank>:<created_at_ms>"
    //   — this string is the lookup key the target uses to retrieve the
    //   plan, so any format change is a coordinated breaking change with
    //   targets running an older version
    #[test]
    fn select_plan_metadata_propagates_from_input() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(42, 2);
        let source = WorkerWithDpRank::new(7, 1);
        let matches = tiered_matches(&[], &[(source, 3)]);
        let input = RemoteKvReuseSelectionInput {
            request_id: "req-meta",
            target,
            block_hashes: &hashes,
            block_size_tokens: 32,
            tiered_matches: &matches,
            created_at_ms: 1234,
            expires_at_ms: 5678,
        };

        let decision = select_remote_g2_reuse_plan(input);

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.request_id, "req-meta");
                assert_eq!(plan.block_size_tokens, 32);
                assert_eq!(plan.created_at_ms, 1234);
                assert_eq!(plan.expires_at_ms, 5678);
                assert_eq!(plan.plan_version, REMOTE_KV_REUSE_PLAN_VERSION);
                assert_eq!(plan.source_tier, StorageTier::HostPinned);
                assert!(plan.kv_block_hashes.is_empty());
                assert_eq!(plan.plan_id, "remote-g2:req-meta:7:1:1234");
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn scenario_zero_host_pinned_hits_no_remote_candidate() {
        let hashes = block_hashes(2);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(&[], &[(WorkerWithDpRank::new(7, 0), 0)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, .. } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoRemoteG2Candidate);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    // Empty request with a viable HostPinned candidate. The arithmetic
    // falls through to `planned_prefix_blocks == 0` and returns
    // NoContiguousPrefix. Defense-in-depth: pins this output so a future
    // refactor that drops `saturating_sub` or reorders the zero-check
    // can't turn this path into a panic on `block_hashes[start..end]`.
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
        // Source A has 2 device-tier matches and 2 HostPinned hits chained
        // past them. The target also has the first 2 blocks locally, so the
        // source span is contiguous with the target's next missing block and
        // the plan covers request positions [2, 4).
        let hashes = block_hashes(6);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(source, 2), (target, 2)], &[(source, 2)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.start_block_index, 2);
                assert_eq!(plan.planned_prefix_blocks, 2);
                assert_eq!(plan.block_hashes, hashes[2..4].to_vec());
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    // Cold prefix: no remote candidate at any tier. Planner emits no plan
    // and reports zero rejected G1 candidates.
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

    // Target itself has HostPinned hits but no other worker does. The
    // selector skips the target before marking `saw_remote_candidate`, so
    // the loop ends with no candidate seen and the reason is
    // NoRemoteG2Candidate (not NoContiguousPrefix). Pins the ordering of
    // the `continue` vs `saw_remote_candidate = true` lines.
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

    // The lower_tier map has no HostPinned entry at all (distinct from
    // the zero-overlap scenario where the entry exists but is empty).
    // Planner short-circuits at the `get(&HostPinned)` step before
    // scanning any worker.
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

    // Full G2 coverage: no G1 anywhere, source has HostPinned hits equal
    // to the request length. Plan covers the entire request.
    #[test]
    fn scenario_full_g2_no_g1_full_coverage_plan() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, stats, .. } => {
                assert_eq!(plan.source_worker_id, 7);
                assert_eq!(plan.start_block_index, 0);
                assert_eq!(plan.planned_prefix_blocks, hashes.len() as u32);
                assert_eq!(plan.block_hashes, hashes);
                assert_eq!(stats.rejected_g1_candidates, 0);
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    // Partial G2 only: no G1 anywhere, source has fewer HostPinned hits
    // than the request length. Plan covers the matched prefix [0, hits)
    // and leaves the tail to be computed freshly by the target.
    #[test]
    fn scenario_partial_g2_no_g1_matched_prefix_only() {
        let hashes = block_hashes(5);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, stats, .. } => {
                assert_eq!(plan.source_worker_id, 7);
                assert_eq!(plan.start_block_index, 0);
                assert_eq!(plan.planned_prefix_blocks, 3);
                assert_eq!(plan.block_hashes, hashes[..3].to_vec());
                assert!(plan.planned_prefix_blocks < hashes.len() as u32);
                assert_eq!(stats.rejected_g1_candidates, 0);
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    // 100% G1 + extra G2: a source's device chain already covers the full
    // request. There is nothing left for remote-G2 to fill, so the planner
    // emits NoContiguousPrefix even though HostPinned hits exist chained
    // past the device match. G1 wins the local-reuse path.
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
