# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic Remote-G2 connector-config canonicalization tests.

Guards the foot-guns of an env-only approach: a static ``source_worker_id``
shadowing the real WorkerId, a ``tcp://`` source socket the parent bridge can't
find, and split engine-vs-dynamo mode ownership (env vs config). Pure Python
(fake config objects) — no engine, no ZMQ."""
import os
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

pytest.importorskip("vllm")

from dynamo.vllm.args import remote_g2_extra_configs  # noqa: E402
from dynamo.vllm.worker_factory import (  # noqa: E402
    _canonicalize_remote_g2_dynamic_config,
)

WID = "7587895884685104905"  # a realistic 64-bit Dynamo WorkerId
ENV_BRIDGE = "REMOTE_G2_USE_DYNAMO_BRIDGE"
ENV_WID = "REMOTE_G2_SOURCE_WORKER_ID"


def _direct_config(extra: dict) -> SimpleNamespace:
    kv = SimpleNamespace(
        kv_connector="OffloadingConnector", kv_connector_extra_config=extra
    )
    return SimpleNamespace(engine_args=SimpleNamespace(kv_transfer_config=kv))


def _pd_config(nested_extra: dict) -> SimpleNamespace:
    kv = SimpleNamespace(
        kv_connector="PdConnector",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "NixlConnector", "kv_connector_extra_config": {}},
                {
                    "kv_connector": "OffloadingConnector",
                    "kv_connector_extra_config": nested_extra,
                },
            ]
        },
    )
    return SimpleNamespace(engine_args=SimpleNamespace(kv_transfer_config=kv))


def _rg2_extra(**over) -> dict:
    e = {
        "spec_name": "RemoteG2OffloadingSpec",
        "source_worker_id": 1,
        "source_rpc_socket_path": "tcp://0.0.0.0:19090",
    }
    e.update(over)
    return e


@pytest.fixture(autouse=True)
def _clean_env():
    # The function under test writes os.environ directly, and monkeypatch.delenv
    # on an already-absent key registers no undo, so save+restore explicitly to
    # avoid leaking either var across tests.
    saved = {k: os.environ.get(k) for k in (ENV_BRIDGE, ENV_WID)}
    for k in (ENV_BRIDGE, ENV_WID):
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_extra_configs_direct():
    extra = _rg2_extra()
    got = remote_g2_extra_configs(_direct_config(extra).engine_args)
    assert got == [extra]
    assert got[0] is extra  # same object -> mutations propagate


def test_extra_configs_pd_nested():
    nested = _rg2_extra()
    got = remote_g2_extra_configs(_pd_config(nested).engine_args)
    assert got == [nested]
    assert got[0] is nested


def test_canonicalize_rewrites_identity_and_sockets():
    os.environ[ENV_BRIDGE] = "1"
    extra = _rg2_extra(source_worker_id=1)
    _canonicalize_remote_g2_dynamic_config(
        _direct_config(extra), WID, snapshot_engine=None
    )
    pid = os.getpid()
    assert extra["source_worker_id"] == int(WID)  # stale 1 overridden
    assert extra["source_rpc_socket_path"] == f"/tmp/dynamo_remote_g2_ipc_{pid}.sock"
    assert extra["target_bridge_socket_path"] == (
        f"/tmp/dynamo_remote_g2_target_{pid}.sock"
    )
    assert extra["use_dynamo_bridge"] is True
    assert os.environ[ENV_WID] == WID


def test_canonicalize_pd_nested_rewrites():
    os.environ[ENV_BRIDGE] = "1"
    nested = _rg2_extra()
    _canonicalize_remote_g2_dynamic_config(
        _pd_config(nested), WID, snapshot_engine=None
    )
    assert nested["source_worker_id"] == int(WID)
    assert nested["use_dynamo_bridge"] is True


def test_extra_configs_pd_multiple_rg2():
    # A PdConnector may nest more than one RemoteG2 connector (e.g. prefill +
    # decode roles); every one must be surfaced for rewrite.
    a = _rg2_extra(source_worker_id=1)
    b = _rg2_extra(source_worker_id=2)
    kv = SimpleNamespace(
        kv_connector="PdConnector",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "OffloadingConnector", "kv_connector_extra_config": a},
                {"kv_connector": "OffloadingConnector", "kv_connector_extra_config": b},
            ]
        },
    )
    cfg = SimpleNamespace(engine_args=SimpleNamespace(kv_transfer_config=kv))
    got = remote_g2_extra_configs(cfg.engine_args)
    assert got == [a, b]
    os.environ[ENV_BRIDGE] = "1"
    _canonicalize_remote_g2_dynamic_config(cfg, WID, snapshot_engine=None)
    assert a["source_worker_id"] == int(WID)
    assert b["source_worker_id"] == int(WID)
    assert a["use_dynamo_bridge"] is True and b["use_dynamo_bridge"] is True


def test_config_only_enables_bridge():
    # No env flag, but the connector config opts in -> must canonicalize (else
    # vLLM would bridge while dynamo stayed static).
    os.environ.pop(ENV_BRIDGE, None)
    extra = _rg2_extra(use_dynamo_bridge=True)
    _canonicalize_remote_g2_dynamic_config(
        _direct_config(extra), WID, snapshot_engine=None
    )
    assert extra["source_worker_id"] == int(WID)
    assert os.environ[ENV_BRIDGE] == "1"  # env normalized for dynamo readers


def test_canonicalize_noop_when_disabled():
    os.environ.pop(ENV_BRIDGE, None)
    extra = _rg2_extra(source_worker_id=1)
    _canonicalize_remote_g2_dynamic_config(
        _direct_config(extra), WID, snapshot_engine=None
    )
    assert extra["source_worker_id"] == 1  # untouched
    assert "use_dynamo_bridge" not in extra


def test_config_explicit_off_is_noop():
    os.environ.pop(ENV_BRIDGE, None)
    extra = _rg2_extra(use_dynamo_bridge=False)
    _canonicalize_remote_g2_dynamic_config(
        _direct_config(extra), WID, snapshot_engine=None
    )
    assert extra["source_worker_id"] == 1  # untouched (explicit off)


def test_conflict_env_on_config_off():
    os.environ[ENV_BRIDGE] = "1"
    extra = _rg2_extra(use_dynamo_bridge=False)
    with pytest.raises(ValueError, match="conflicting"):
        _canonicalize_remote_g2_dynamic_config(
            _direct_config(extra), WID, snapshot_engine=None
        )


def test_canonicalize_rejects_snapshot_engine():
    os.environ[ENV_BRIDGE] = "1"
    with pytest.raises(ValueError, match="snapshot"):
        _canonicalize_remote_g2_dynamic_config(
            _direct_config(_rg2_extra()), WID, snapshot_engine=object()
        )


def test_canonicalize_rejects_disabled_source_rpc():
    os.environ[ENV_BRIDGE] = "1"
    extra = _rg2_extra(enable_source_rpc=False)
    with pytest.raises(ValueError, match="source RPC"):
        _canonicalize_remote_g2_dynamic_config(
            _direct_config(extra), WID, snapshot_engine=None
        )


def test_canonicalize_noop_for_non_remote_g2():
    os.environ[ENV_BRIDGE] = "1"
    kv = SimpleNamespace(kv_connector="NixlConnector", kv_connector_extra_config={})
    cfg = SimpleNamespace(engine_args=SimpleNamespace(kv_transfer_config=kv))
    _canonicalize_remote_g2_dynamic_config(cfg, WID, snapshot_engine=None)
    assert kv.kv_connector_extra_config == {}
