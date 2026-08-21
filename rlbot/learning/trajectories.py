"""trajectory_v1 record construction, validation, and JSONL I/O (SPEC-001 §7)."""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from rlbot.state.enums import REWARD_VERSION, SCHEMA_VERSION

SCHEMA_PATH = Path(__file__).parent.parent / "state" / "trajectory_schema.json"
_SCHEMA = json.loads(SCHEMA_PATH.read_text())


def decision_to_record(d, ticker: str, run_id: str, episode_id: str, seq: int,
                       policy_meta: dict | None = None,
                       counterfactuals: dict | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "reward_version": REWARD_VERSION,
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
    jsonschema.validate(record, _SCHEMA)


def write_jsonl(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            validate_record(r)
            f.write(json.dumps(r) + "\n")
