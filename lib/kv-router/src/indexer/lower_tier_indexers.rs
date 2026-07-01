// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-tier registry of [`LowerTierIndexer`] instances and helpers for walking
//! the device → host → disk continuation chain.
//!
//! The primary KV indexer (radix tree) handles device-tier overlap scoring.
//! When a request arrives, we want to extend the per-worker match by walking
//! whichever lower tiers a worker has registered. [`LowerTierIndexers`] holds
//! one [`ThreadPoolIndexer<LowerTierIndexer>`] per non-device [`StorageTier`]
//! and lazily allocates each tier on first event arrival.
//!
//! Both the request-plane indexer (`dynamo-llm`) and the standalone HTTP
//! indexer (this crate's `services::indexer` module) share this implementation
//! so tier semantics stay aligned across the two surfaces.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rustc_hash::FxHashMap;

use crate::indexer::{
    CrossWorkerSpan, KvIndexerMetrics, LowerTierContinuation, LowerTierIndexer,
    LowerTierMatchDetails, MatchDetails, ThreadPoolIndexer, WireTieredMatchDetails,
};
use crate::protocols::{LocalBlockHash, StorageTier, WorkerWithDpRank};

fn insert_cross_worker_span(
    spans: &mut FxHashMap<WorkerWithDpRank, Vec<CrossWorkerSpan>>,
    worker: WorkerWithDpRank,
    candidate: CrossWorkerSpan,
) {
    if candidate.end_pos <= candidate.start_pos {
        return;
    }

    let worker_spans = spans.entry(worker).or_default();
    if let Some(existing) = worker_spans.iter_mut().find(|span| {
        span.start_pos == candidate.start_pos && span.parent_hash == candidate.parent_hash
    }) {
        existing.end_pos = existing.end_pos.max(candidate.end_pos);
    } else {
        worker_spans.push(candidate);
    }
}

/// Holds one per-tier [`ThreadPoolIndexer<LowerTierIndexer>`] for every
/// non-device [`StorageTier`] that has received at least one event.
#[derive(Clone)]
pub struct LowerTierIndexers {
    metrics: Option<Arc<KvIndexerMetrics>>,
    num_threads: usize,
    block_size: u32,
    indexers: Arc<RwLock<HashMap<StorageTier, Arc<ThreadPoolIndexer<LowerTierIndexer>>>>>,
}

impl LowerTierIndexers {
    /// Metrics-less constructor for call sites without a `KvIndexerMetrics` handle.
    /// Router production assembly should use [`new_with_metrics`](Self::new_with_metrics)
    /// so lower-tier traffic is included in `kv_cache_events_applied`.
    pub fn new(num_threads: usize, block_size: u32) -> Self {
        Self::new_with_metrics(num_threads, block_size, None)
    }

    /// Same as [`new`](Self::new) but wires `kv_cache_events_applied`
    /// counters into every lazily created per-tier indexer, matching the
    /// observability of the device-tier path.
    pub fn new_with_metrics(
        num_threads: usize,
        block_size: u32,
        metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> Self {
        assert!(
            num_threads > 0,
            "lower-tier indexer threads must be non-zero"
        );
        Self {
            num_threads,
            block_size,
            metrics,
            indexers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Return the per-tier indexer for `storage_tier`, lazily allocating it
    /// the first time a non-device tier is seen.
    pub fn get_or_create(
        &self,
        storage_tier: StorageTier,
    ) -> Arc<ThreadPoolIndexer<LowerTierIndexer>> {
        debug_assert!(!storage_tier.is_gpu());
        if let Some(indexer) = self.indexers.read().unwrap().get(&storage_tier).cloned() {
            return indexer;
        }
        self.indexers
            .write()
            .unwrap()
            .entry(storage_tier)
            .or_insert_with(|| {
                Arc::new(ThreadPoolIndexer::new_with_metrics(
                    LowerTierIndexer::new(),
                    self.num_threads,
                    self.block_size,
                    self.metrics.clone(),
                ))
            })
            .clone()
    }

    /// All currently allocated lower-tier indexers, in unspecified order.
    pub fn all(&self) -> Vec<Arc<ThreadPoolIndexer<LowerTierIndexer>>> {
        self.indexers.read().unwrap().values().cloned().collect()
    }

    /// All currently allocated lower-tier indexers paired with the
    /// [`StorageTier`] each one indexes. Used by callers that need to retag
    /// per-tier dumps (e.g. peer-recovery).
    pub fn entries(&self) -> Vec<(StorageTier, Arc<ThreadPoolIndexer<LowerTierIndexer>>)> {
        self.indexers
            .read()
            .unwrap()
            .iter()
            .map(|(tier, indexer)| (*tier, indexer.clone()))
            .collect()
    }

    /// Lookup without allocation; returns `None` if the tier is unseen.
    pub fn get(
        &self,
        storage_tier: StorageTier,
    ) -> Option<Arc<ThreadPoolIndexer<LowerTierIndexer>>> {
        self.indexers.read().unwrap().get(&storage_tier).cloned()
    }
}

/// Native tiered-match container: the device-tier match plus a per-tier map
/// of lower-tier hits. Wire-friendly representations live in
/// [`WireTieredMatchDetails`]; conversions in both directions are provided.
#[derive(Debug, Clone, Default)]
pub struct TieredMatchDetails {
    pub device: MatchDetails,
    pub lower_tier: HashMap<StorageTier, LowerTierMatchDetails>,
}

impl From<&TieredMatchDetails> for WireTieredMatchDetails {
    fn from(d: &TieredMatchDetails) -> Self {
        Self {
            device: d.device.overlap_scores.clone().into(),
            lower_tier: d
                .lower_tier
                .iter()
                .map(|(tier, details)| (*tier, details.into()))
                .collect(),
        }
    }
}

impl From<WireTieredMatchDetails> for TieredMatchDetails {
    fn from(w: WireTieredMatchDetails) -> Self {
        // `last_matched_hashes` is only needed server-side to seed the tier walk,
        // so we leave it empty on the inbound side.
        let mut lower_tier = HashMap::with_capacity(w.lower_tier.len());
        for (tier, details) in w.lower_tier {
            if lower_tier.insert(tier, details.into()).is_some() {
                tracing::warn!(
                    ?tier,
                    "Duplicate StorageTier in WireTieredMatchDetails; keeping last entry"
                );
            }
        }
        Self {
            device: MatchDetails {
                overlap_scores: w.device.into(),
                ..Default::default()
            },
            lower_tier,
        }
    }
}

/// The order in which lower tiers are walked when extending a match. Device
/// → HostPinned → Disk → External.
pub fn lower_tier_query_order() -> [StorageTier; 3] {
    [
        StorageTier::HostPinned,
        StorageTier::Disk,
        StorageTier::External,
    ]
}

/// Walk every allocated lower tier in [`lower_tier_query_order`] and build a
/// per-tier match map seeded from `device_matches`. Per-worker continuations
/// flow forward: a worker that matched N device blocks starts the host walk
/// at block N (anchored on its last device hash), and so on.
pub fn query_lower_tiers(
    indexers: &LowerTierIndexers,
    sequence: &[LocalBlockHash],
    device_matches: &MatchDetails,
) -> HashMap<StorageTier, LowerTierMatchDetails> {
    // No lower-tier indexers are allocated, so there is no continuation
    // work to perform. Return before validating device score/hash lockstep;
    // that invariant only matters when a lower tier will consume the
    // continuations.
    if indexers.indexers.read().unwrap().is_empty() {
        return HashMap::new();
    }

    let mut continuations = LowerTierMatchDetails::default().next_continuations;
    for (worker, matched_blocks) in &device_matches.overlap_scores.scores {
        let Some(last_hash) = device_matches.last_matched_hashes.get(worker).copied() else {
            debug_assert!(
                false,
                "device match result missing last matched hash for worker {worker:?}"
            );
            continue;
        };

        continuations.insert(
            *worker,
            LowerTierContinuation::new(*matched_blocks as usize, last_hash),
        );
    }

    let mut lower_tier_matches = HashMap::new();

    for storage_tier in lower_tier_query_order() {
        let Some(indexer) = indexers.get(storage_tier) else {
            continue;
        };

        // Path A: extension from device coverage (and from-root for workers
        // that the tier knows about but device didn't see). This is the
        // existing logic and answers "how much can this worker contribute
        // BEYOND its device chain?" — the right question when the worker is
        // both source and target of the load.
        let mut extension_continuations = continuations.clone();
        if let Some(&first_hash) = sequence.first() {
            let root_workers: Vec<_> = indexer.backend().root_workers(first_hash);
            for worker in root_workers.iter() {
                extension_continuations
                    .entry(*worker)
                    .or_insert_with(|| LowerTierContinuation::from_root(0));
            }
        }
        let extension_matches = indexer
            .backend()
            .query_match_details(sequence, &extension_continuations);

        // Path B: fresh start (from position 0) for every worker the tier
        // knows about. This is the right question when target ≠ source: a
        // remote target asking "what does worker W have on host-pinned" must
        // not have its view of W's host-pinned suppressed by W's own device
        // chain coverage. Without this, when target=W2 wants to pull blocks
        // that W1 has on host-pinned, the indexer would report 0 hits for W1
        // whenever W1 ALSO happens to still hold the prefix on its GPU prefix
        // cache (which is the common case — vLLM emits BlockRemoved only on
        // actual GPU prefix-cache eviction, which doesn't fire under light
        // load).
        let fresh_continuations: std::collections::HashMap<_, _> =
            if let Some(&first_hash) = sequence.first() {
                indexer
                    .backend()
                    .root_workers(first_hash)
                    .into_iter()
                    .map(|w| (w, LowerTierContinuation::from_root(0)))
                    .collect()
            } else {
                std::collections::HashMap::new()
            };
        let fresh_matches = indexer
            .backend()
            .query_match_details(sequence, &fresh_continuations);

        // Two views of the SAME tier query, kept separate so the two
        // consumers do not pollute each other (see LowerTierMatchDetails):
        //   * `hits` = Path A (extension) ONLY = the dedup'd view. A worker's
        //     host-pinned blocks already covered by its own device chain are
        //     suppressed. Consumed by cache-hit estimation; pins the
        //     `does_not_double_count` invariant.
        //   * `cross_worker_hits` = maximum raw span length across Path A and
        //     Path B, retained as a summary for metrics/wire compatibility.
        //   * `cross_worker_spans` = BOTH materializable candidates, including
        //     their exact start/end and parent anchor. Plan selection cannot
        //     pick a winner here: which span is useful depends on the eventual
        //     target's local prefix. For example, root [0, 2) is useless to a
        //     target that already has 3 blocks, while extension [3, 5) is not.
        let mut cross_worker_hits = extension_matches.hits.clone();
        let mut cross_worker_spans: FxHashMap<_, Vec<CrossWorkerSpan>> = FxHashMap::default();
        for (worker, &hits) in &extension_matches.hits {
            let Some(seed) = extension_continuations.get(worker) else {
                debug_assert!(false, "lower-tier hit missing its query continuation");
                continue;
            };
            let start_pos = seed.start_pos;
            insert_cross_worker_span(
                &mut cross_worker_spans,
                *worker,
                CrossWorkerSpan {
                    start_pos,
                    end_pos: start_pos.saturating_add(hits),
                    parent_hash: seed.last_matched_hash,
                },
            );
        }
        for (worker, &fresh_hits) in &fresh_matches.hits {
            let cw = cross_worker_hits.get(worker).copied().unwrap_or(0);
            if fresh_hits > cw {
                cross_worker_hits.insert(*worker, fresh_hits);
            }
            insert_cross_worker_span(
                &mut cross_worker_spans,
                *worker,
                CrossWorkerSpan {
                    start_pos: 0,
                    end_pos: fresh_hits,
                    parent_hash: None,
                },
            );
        }
        for spans in cross_worker_spans.values_mut() {
            spans.sort_unstable_by_key(|span| {
                (
                    span.start_pos,
                    span.end_pos,
                    span.parent_hash.map(|hash| hash.0),
                )
            });
        }

        let result = LowerTierMatchDetails {
            hits: extension_matches.hits,
            cross_worker_hits,
            cross_worker_spans,
            next_continuations: extension_matches.next_continuations,
        };

        let matched_workers = result
            .cross_worker_hits
            .values()
            .filter(|&&h| h > 0)
            .count();
        tracing::debug!(
            ?storage_tier,
            queried_workers_extension = extension_continuations.len(),
            queried_workers_fresh = fresh_continuations.len(),
            matched_workers,
            "Queried lower-tier indexer (dedup hits + cross-worker view)"
        );
        // Carry Path A (dedup'd) continuations to the next tier so the dedup'd
        // `hits` chain stays consistent across tiers (pre-remote-G2 behaviour).
        continuations = result.next_continuations.clone();
        lower_tier_matches.insert(storage_tier, result);
    }

    lower_tier_matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::{KvIndexerInterface, TieredMatchDetails};
    use crate::protocols::{
        ExternalSequenceBlockHash, KvCacheEventData, KvCacheStoreData, LocalBlockHash,
        OverlapScores, RouterEvent, WorkerWithDpRank,
    };
    use crate::remote_g2_plan::{
        RemoteKvReuseDecision, RemoteKvReuseSelectionInput, materialize_remote_g2_reuse_plan,
        select_remote_g2_reuse_plan,
    };
    use crate::test_utils::{router_event, stored_blocks_with_sequence_hashes};

    fn hp_store(
        worker_id: u64,
        event_id: u64,
        parent: Option<u64>,
        locals: &[u64],
        externals: &[u64],
    ) -> RouterEvent {
        let locals: Vec<LocalBlockHash> = locals.iter().copied().map(LocalBlockHash).collect();
        router_event(
            worker_id,
            event_id,
            0,
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks: stored_blocks_with_sequence_hashes(&locals, externals),
            }),
        )
    }

    fn device_match(worker: WorkerWithDpRank, hits: u32, last: Option<u64>) -> MatchDetails {
        let mut d = MatchDetails {
            overlap_scores: OverlapScores::new(),
            ..Default::default()
        };
        d.overlap_scores.scores.insert(worker, hits);
        if let Some(h) = last {
            d.last_matched_hashes
                .insert(worker, ExternalSequenceBlockHash(h));
        }
        d
    }

    #[test]
    fn query_lower_tiers_returns_empty_when_no_tiers_allocated() {
        let indexers = LowerTierIndexers::new(1, 4);

        // Mismatched device_matches: a score entry with no paired
        // `last_matched_hashes` entry. Would `debug_assert!`-panic in the
        // old body; the early-return must skip the seeding loop entirely.
        let mut overlap_scores = OverlapScores::new();
        overlap_scores
            .scores
            .insert(WorkerWithDpRank::new(99, 0), 3);
        let device_matches = MatchDetails {
            overlap_scores,
            last_matched_hashes: Default::default(),
        };

        let sequence = vec![LocalBlockHash(1), LocalBlockHash(2)];
        let result = query_lower_tiers(&indexers, &sequence, &device_matches);
        assert!(result.is_empty());
    }

    // Regression 1 (the observed 4×TP1 gated-microbenchmark failure): a source
    // that holds the FULL host-pinned chain from root AND the full device
    // prefix, with a target that already has the first 2 blocks locally. The
    // query must preserve the root span [0, 5), so the later target-aware
    // materializer can walk from root and skip the target's first 2 blocks —
    // not anchor at position 2 on the source's device tail (which misses).
    #[tokio::test]
    async fn cross_worker_chain_from_root_skips_target_local_prefix() {
        let src = WorkerWithDpRank::new(1, 0);
        let target = WorkerWithDpRank::new(2, 0);
        let indexers = LowerTierIndexers::new(1, 1);
        let hp = indexers.get_or_create(StorageTier::HostPinned);
        hp.apply_event(hp_store(
            1,
            0,
            None,
            &[10, 11, 12, 13, 14],
            &[100, 101, 102, 103, 104],
        ))
        .await;
        let _ = hp.dump_events().await; // flush the worker thread

        let sequence: Vec<LocalBlockHash> = [10, 11, 12, 13, 14]
            .into_iter()
            .map(LocalBlockHash)
            .collect();
        // Source matched the whole prefix on device too (tail external 104).
        let mut device = device_match(src, 5, Some(104));
        device.overlap_scores.scores.insert(target, 2);
        device
            .last_matched_hashes
            .insert(target, ExternalSequenceBlockHash(201));
        let result = query_lower_tiers(&indexers, &sequence, &device);
        let details = &result[&StorageTier::HostPinned];

        // The recorded span is the from-root chain, NOT the device tail.
        assert_eq!(
            details.cross_worker_spans[&src],
            vec![CrossWorkerSpan {
                start_pos: 0,
                end_pos: 5,
                parent_hash: None,
            }]
        );
        assert_eq!(details.cross_worker_hits[&src], 5);

        // Guard: the OLD buggy anchor (device tail 104 at position 2) misses,
        // which is exactly why every p2..p8 pull silently recomputed.
        let wrong = hp.backend().chain_block_hashes_for_worker(
            src,
            Some(ExternalSequenceBlockHash(104)),
            &sequence[2..5],
        );
        assert!(wrong.is_empty(), "device-tail anchor at start>0 must miss");

        let tiered = TieredMatchDetails {
            device,
            lower_tier: result,
        };
        let decision = select_remote_g2_reuse_plan(RemoteKvReuseSelectionInput {
            request_id: "root-span",
            target,
            block_hashes: &sequence,
            block_size_tokens: 1,
            tiered_matches: &tiered,
            created_at_ms: 1,
            expires_at_ms: 2,
        });
        let decision = materialize_remote_g2_reuse_plan(
            decision,
            &sequence,
            |source, parent_hash, token_hashes| {
                hp.backend()
                    .chain_block_hashes_for_worker(source, parent_hash, token_hashes)
            },
        );
        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.start_block_index, 2);
                assert_eq!(plan.block_hashes, sequence[2..5]);
                assert_eq!(plan.kv_block_hashes, vec![102, 103, 104]);
            }
            other => panic!("expected materialized root-span plan, got {other:?}"),
        }
    }

    // Regression 2: a source whose host-pinned chain is NOT walkable from root
    // — it extends from the source's device tail (parent 101 at position 2).
    // The query must preserve that exact device-tail span for materialization.
    #[tokio::test]
    async fn cross_worker_chain_non_root_extends_from_device_tail() {
        let src = WorkerWithDpRank::new(1, 0);
        let target = WorkerWithDpRank::new(2, 0);
        let indexers = LowerTierIndexers::new(1, 1);
        let hp = indexers.get_or_create(StorageTier::HostPinned);
        // Host-pinned blocks 2..5 hang off device tail external 101.
        hp.apply_event(hp_store(1, 0, Some(101), &[12, 13, 14], &[102, 103, 104]))
            .await;
        let _ = hp.dump_events().await;

        let sequence: Vec<LocalBlockHash> = [10, 11, 12, 13, 14]
            .into_iter()
            .map(LocalBlockHash)
            .collect();
        let mut device = device_match(src, 2, Some(101));
        device.overlap_scores.scores.insert(target, 2);
        device
            .last_matched_hashes
            .insert(target, ExternalSequenceBlockHash(201));
        let result = query_lower_tiers(&indexers, &sequence, &device);
        let details = &result[&StorageTier::HostPinned];

        assert_eq!(
            details.cross_worker_spans[&src],
            vec![CrossWorkerSpan {
                start_pos: 2,
                end_pos: 5,
                parent_hash: Some(ExternalSequenceBlockHash(101)),
            }],
            "chain extends from device tail at pos 2"
        );
        assert_eq!(details.cross_worker_hits[&src], 3);

        let tiered = TieredMatchDetails {
            device,
            lower_tier: result,
        };
        let decision = select_remote_g2_reuse_plan(RemoteKvReuseSelectionInput {
            request_id: "non-root-span",
            target,
            block_hashes: &sequence,
            block_size_tokens: 1,
            tiered_matches: &tiered,
            created_at_ms: 1,
            expires_at_ms: 2,
        });
        let decision = materialize_remote_g2_reuse_plan(
            decision,
            &sequence,
            |source, parent_hash, token_hashes| {
                hp.backend()
                    .chain_block_hashes_for_worker(source, parent_hash, token_hashes)
            },
        );
        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.start_block_index, 2);
                assert_eq!(plan.block_hashes, sequence[2..5]);
                assert_eq!(plan.kv_block_hashes, vec![102, 103, 104]);
            }
            other => panic!("expected materialized non-root plan, got {other:?}"),
        }
    }

    /// Path A and Path B may expose different chains for the same source. The
    /// query must retain both until the target is known; choosing by raw hits
    /// here would discard the only span contiguous with target prefix 3.
    #[tokio::test]
    async fn query_select_materialize_keeps_both_source_paths() {
        let src = WorkerWithDpRank::new(1, 0);
        let target = WorkerWithDpRank::new(2, 0);
        let indexers = LowerTierIndexers::new(1, 1);
        let hp = indexers.get_or_create(StorageTier::HostPinned);

        // Path B: a root chain covering only [0, 2).
        hp.apply_event(hp_store(1, 0, None, &[10, 11], &[100, 101]))
            .await;
        // Path A: a separate chain extending the source's device tail at 3.
        hp.apply_event(hp_store(1, 1, Some(201), &[13, 14], &[203, 204]))
            .await;
        let _ = hp.dump_events().await;

        let sequence: Vec<LocalBlockHash> = [10, 11, 12, 13, 14]
            .into_iter()
            .map(LocalBlockHash)
            .collect();
        let mut device = device_match(src, 3, Some(201));
        device.overlap_scores.scores.insert(target, 3);
        device
            .last_matched_hashes
            .insert(target, ExternalSequenceBlockHash(301));
        let result = query_lower_tiers(&indexers, &sequence, &device);
        assert_eq!(
            result[&StorageTier::HostPinned].cross_worker_spans[&src],
            vec![
                CrossWorkerSpan {
                    start_pos: 0,
                    end_pos: 2,
                    parent_hash: None,
                },
                CrossWorkerSpan {
                    start_pos: 3,
                    end_pos: 5,
                    parent_hash: Some(ExternalSequenceBlockHash(201)),
                },
            ]
        );

        let tiered = TieredMatchDetails {
            device,
            lower_tier: result,
        };
        let decision = select_remote_g2_reuse_plan(RemoteKvReuseSelectionInput {
            request_id: "both-paths",
            target,
            block_hashes: &sequence,
            block_size_tokens: 1,
            tiered_matches: &tiered,
            created_at_ms: 1,
            expires_at_ms: 2,
        });
        let decision = materialize_remote_g2_reuse_plan(
            decision,
            &sequence,
            |source, parent_hash, token_hashes| {
                hp.backend()
                    .chain_block_hashes_for_worker(source, parent_hash, token_hashes)
            },
        );
        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.start_block_index, 3);
                assert_eq!(plan.block_hashes, sequence[3..5]);
                assert_eq!(plan.kv_block_hashes, vec![203, 204]);
            }
            other => panic!("expected extension-path plan, got {other:?}"),
        }
    }
}
