# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo vLLM Remote G2 (KV-P2P) parent-process bridges.

The actual ZMQ REP loop lives inside the vLLM EngineCore subprocess
(``vllm.v1.kv_offload.remote_g2.source_rpc.SourceG2RpcServer``). It
binds to the same ``/tmp/dynamo_remote_g2_ipc_<dynamo_pid>.sock`` path
the TRT-LLM POC uses, and speaks the same pickle wire format.

That makes the dynamo parent-side bridges fully engine-agnostic, so we
re-export them from ``dynamo.trtllm.kv_p2p`` instead of duplicating the
code. When the proposal in
``vllm_kvp2p_design_for_review.md`` §7.1 (move to ``dynamo.common.kv_p2p``)
lands, only these re-exports need to update; the call sites stay put.
"""

from dynamo.trtllm.kv_p2p.source_rpc_server import (  # noqa: F401
    setup_source_rpc_endpoints,
)
from dynamo.trtllm.kv_p2p.target_rpc_client import (  # noqa: F401
    _TargetRpcClient as TargetRpcClient,
)
from dynamo.trtllm.kv_p2p.target_rpc_local import setup_target_rpc_local  # noqa: F401

__all__ = [
    "setup_source_rpc_endpoints",
    "setup_target_rpc_local",
    "TargetRpcClient",
]
