"""trajectory_v1 record construction, validation, and JSONL I/O (SPEC-001 §7)."""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from rlbot.state.enums import REWARD_VERSION, SCHEMA_VERSION

_STATE_DIR = Path(__file__).parent.parent / "state"
SCHEMA_PATH = _STATE_DIR / "trajectory_schema.json"
_SCHEMAS = {
    "trajectory_v1": json.loads(SCHEMA_PATH.read_text()),
    "trajectory_v2": json.loads((_STATE_DIR / "trajectory_schema_v2.json").read_text()),
}
_SCHEMA = _SCHEMAS["trajectory_v1"]


def decision_to_record(d, ticker: str, run_id: str, episode_id: str, seq: int,
                       policy_meta: dict | None = None,
                       counterfactuals: dict | None = None,
                       schema_version: str = SCHEMA_VERSION) -> dict:
    """v1 for opening decisions; pass schema_version="trajectory_v2" for runs
    that include management decisions (SPEC-001A §6)."""
    mgmt = getattr(d, "mgmt_state", None)
    if mgmt is not None and schema_version == "trajectory_v1":
        raise ValueError("management decisions require schema_version=trajectory_v2")
    extra = {"mgmt_state": list(mgmt) if mgmt is not None else None} \
        if schema_version == "trajectory_v2" else {}
    reward_version = "diff_v2" if mgmt is not None else REWARD_VERSION
    return {
        **extra,
        "schema_version": schema_version,
        "reward_version": reward_version,
        "run_id": run_id,
        "episode_id": episode_id,
        "decision_id": f"{episode_id}:{seq}",
        "date": str(d.date.date()),
        "ticker": ticker,
        "position_state": d.position_state,
        "q_state": list(d.q_state),
        "features": {
            "market_regime": d.q_state[0],
            "valuation_state": d.q_state[1],
            "vol_compensation": d.q_state[2],
            "raw": d.features_raw,
        },
        "available_actions": d.available_actions,
        "chosen_action": d.chosen_action,
        "action_source": d.action_source,
        "policy_meta": policy_meta or {},
        "contract": d.contract,
        "risk_checks": {"passed": d.risk.passed, "flags": d.risk.flags},
        "portfolio_before": d.portfolio_before,
        "next_epoch_date": str(d.next_epoch_date.date()) if d.next_epoch_date is not None else None,
        "portfolio_after": d.portfolio_after,
        "reward": d.reward,
        "reference_return": d.reference_return,
        "next_q_state": list(d.next_q_state) if d.next_q_state is not None else None,
        "next_position_state": d.next_position_state,
        "terminal": d.terminal,
        "counterfactuals": counterfactuals,
    }


def validate_record(record: dict) -> None:
    schema = _SCHEMAS.get(record.get("schema_version"))
    if schema is None:
        raise jsonschema.ValidationError(
            f"unsupported schema_version: {record.get('schema_version')!r}")
    jsonschema.validate(record, schema)


def write_jsonl(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            validate_record(r)
            f.write(json.dumps(r) + "\n")
