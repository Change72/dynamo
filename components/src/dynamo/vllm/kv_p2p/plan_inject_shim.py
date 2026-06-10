# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan injector shim for the dynamo + vLLM + KV-P2P end-to-end demo.

Background: the KV Router's native plan emitter
(``lib/kv-router/src/remote_g2_plan.rs`` on
``oandreeva-kv-p2p-v1-followups``) only fires
``select_remote_g2_reuse_plan`` when **target != source** AND the
target has no device-tier prefix AND some other worker has a
``CPU_PINNED`` prefix for the same blocks. In a typical KV-router
deployment with prefix-aware routing, the Router naturally sends any
prefix-bearing request to whichever worker already has the prefix on
device, so target == source and the predicate short-circuits to no
plan. The native path therefore rarely emits in single-host or
co-located 2-worker topologies — the exact topologies you want to
use to verify the source RPC + NIXL data plane.

This shim sidesteps that. When the vLLM worker's handler is about to
build its sampling params, it asks every known peer for the
block-hash list its source pool has published and stuffs a single
``RemoteKvReusePlan`` referencing all of them into
``sampling_params.extra_args["kv_transfer_params"]["remote_g2_plan"]``.

The vLLM-side manager (``RemoteG2OffloadingManager``) iterates
through the plan's hashes when resolving prefix matches — only the
hashes that also appear in the target request's per-block hashes
actually trigger a NIXL READ. An "all-hashes" plan is therefore a
correctness-preserving stand-in for a Router that produces precise
per-request plans: identical load decisions on the matching blocks,
and a no-op on the rest.

Audience: anybody verifying KV-P2P end-to-end without engineering a
cross-worker routing scenario. Also the only practical option when
running against prebuilt Dynamo binaries that do not yet ship
``remote_g2_plan.rs``.

How peers are discovered:

* ``KVP2P_PEER_SOCKETS`` env var, comma-separated
  ``<worker_id>=<socket_path>`` pairs (same shape as the vLLM
  spec's ``peer_endpoints`` config).
* ``REMOTE_G2_SOURCE_WORKER_ID`` env var identifies the local worker
  so the shim skips a useless self-pull.

Cost-of-injection budget: one ZMQ round-trip per peer per request,
fully concurrent-safe; the result is cached for 1 second so a burst
of requests on the same target reuses the same plan.

Patch site: insert a call to ``maybe_inject_plan(sampling_params,
request_id=...)`` at the end of ``build_sampling_params`` in
``components/src/dynamo/vllm/handlers.py`` (the non-OpenAI variant
— ``dynamo.vllm`` chat completions go through that builder, not the
``*_openai`` one). See ``POC_OVERVIEW.md`` §2.7(b) in the vLLM
KV-P2P design folder for the exact patch.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TTL_S = 1.0
_DEFAULT_PEER_TIMEOUT_MS = 1500


class _PeerHashCache:
    """Per-peer cache of (block_hash_int list, fetched_at). TTL-bounded."""

    def __init__(self, ttl_s: float = _DEFAULT_TTL_S) -> None:
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._entries: dict[int, dict[str, Any]] = {}

    def fetch(self, peer_worker_id: int, socket_path: str) -> list[int]:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(peer_worker_id)
            if entry is not None and now - entry["ts"] < self._ttl_s:
                return list(entry["hashes"])
            client = entry["client"] if entry is not None else None

        if client is None:
            from vllm.v1.kv_offload.remote_g2.target_client import TargetG2RpcClient

            client = TargetG2RpcClient(socket_path, timeout_ms=_DEFAULT_PEER_TIMEOUT_MS)

        try:
            stats = client.stats(sample_limit=8192) or {}
        except Exception as exc:
            logger.warning(
                "kvp2p plan_inject: stats RPC to peer %d failed: %s",
                peer_worker_id,
                exc,
            )
            return []

        hashes = [int(h) for h in stats.get("sample_block_hashes", [])]
        with self._lock:
            self._entries[peer_worker_id] = {
                "hashes": hashes,
                "ts": now,
                "client": client,
            }
        return hashes


_cache = _PeerHashCache()


def _parse_peers(spec: str | None) -> list[tuple[int, str]]:
    if not spec:
        return []
    out = []
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        wid, path = pair.split("=", 1)
        try:
            out.append((int(wid.strip()), path.strip()))
        except ValueError:
            continue
    return out


def maybe_inject_plan(
    sampling_params: Any,
    *,
    request_id: str = "req",
) -> bool:
    """Inject a plan into ``sampling_params.extra_args[
    "kv_transfer_params"]["remote_g2_plan"]`` if any peer reports
    descriptors.

    Returns True when a plan was injected, False otherwise.
    """
    peers = _parse_peers(os.environ.get("KVP2P_PEER_SOCKETS"))
    if not peers:
        logger.warning("kvp2p plan_inject: no peers configured (req=%s)", request_id)
        return False

    self_worker_id = int(os.environ.get("REMOTE_G2_SOURCE_WORKER_ID", "-1"))

    plan_hashes: list[int] = []
    chosen_peer_worker_id: int | None = None
    peer_summary: list[str] = []
    for peer_worker_id, socket_path in peers:
        if peer_worker_id == self_worker_id:
            peer_summary.append(f"peer{peer_worker_id}=self")
            continue
        if not os.path.exists(socket_path):
            peer_summary.append(f"peer{peer_worker_id}=no_socket")
            continue
        hashes = _cache.fetch(peer_worker_id, socket_path)
        peer_summary.append(f"peer{peer_worker_id}=n_hashes={len(hashes)}")
        if not hashes:
            continue
        plan_hashes = hashes
        chosen_peer_worker_id = peer_worker_id
        break

    if not plan_hashes or chosen_peer_worker_id is None:
        logger.warning(
            "kvp2p plan_inject: no plan attached for req=%s "
            "self_worker=%d (peers: %s)",
            request_id,
            self_worker_id,
            ", ".join(peer_summary),
        )
        return False

    from vllm.v1.kv_offload.remote_g2.data_model import REMOTE_KV_REUSE_PLAN_VERSION

    plan = {
        "plan_id": f"router-shim-{request_id}",
        "request_id": str(request_id),
        "target_worker_id": int(self_worker_id),
        "target_dp_rank": 0,
        "source_worker_id": int(chosen_peer_worker_id),
        "source_dp_rank": 0,
        "source_tier": "host_pinned",
        "block_hashes": plan_hashes,
        "kv_block_hashes": [],
        "start_block_index": 0,
        "planned_prefix_blocks": len(plan_hashes),
        "block_size_tokens": 16,
        "created_at_ms": 0,
        "expires_at_ms": 10**15,
        "plan_version": REMOTE_KV_REUSE_PLAN_VERSION,
    }

    if sampling_params.extra_args is None:
        sampling_params.extra_args = {}
    kv_params = sampling_params.extra_args.setdefault("kv_transfer_params", {})
    kv_params["remote_g2_plan"] = plan
    logger.info(
        "kvp2p plan_inject: attached plan for req=%s target=%d source=%d, "
        "n_hashes=%d",
        request_id,
        self_worker_id,
        chosen_peer_worker_id,
        len(plan_hashes),
    )
    return True
