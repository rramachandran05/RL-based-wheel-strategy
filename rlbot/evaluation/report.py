"""Render g2_verdict.json into a markdown evaluation report (SPEC-006 §6).

Run:  python -m rlbot.evaluation.report
"""
from __future__ import annotations

import json

from rlbot.config import RlbotConfig
from rlbot.state.enums import CashAction, MarketRegime

REGIME_NAMES = {str(int(r)): r.name for r in MarketRegime}


def _pct(x, digits=2):
    return f"{x * 100:+.{digits}f}%" if x is not None else "—"


def render(out: dict) -> str:
    g2 = out["g2"]
    lines = [
        "# Wheel-RLBot — Walk-Forward Evaluation Report",
        "",
        f"**GATE G1 (simulator calibration):** {'PASS' if out['g1']['pass'] else 'FAIL'}"
        f" — iv_uplift = {out['g1']['iv_uplift']}",
        f"**GATE G2 (learned policy vs Baseline 3):** {'PASS' if g2['pass'] else 'FAIL'}",
        "",
        f"Pooled test differential return: **{_pct(g2['pooled_test_diff_ann'])}/yr** "
        f"(per fold: {', '.join(_pct(t) for t in g2['per_fold_test_diff_ann'])})",
        "",
        "## G2 criteria",
        "",
        "| Criterion | Result |",
        "|---|---|",
    ]
    for k, v in g2["criteria"].items():
        lines.append(f"| {k} | {'✅' if v else '❌'} |")

    for fold in out["folds"]:
        lines += ["", f"## Fold {fold['fold']}", ""]
        for split in ("val", "test"):
            s = fold[split]
            lines += [
                f"### {split} window",
                "",
                f"- Episodes: {s['n_episodes']}; mean diff {_pct(s['mean_diff_ann'])}/yr; "
                f"median {_pct(s['median_diff_ann'])}/yr; "
                f"{s['pct_episodes_positive'] * 100:.0f}% of episodes positive",
                f"- Drawdown ratio vs B3 (mean): {s['dd_ratio_mean']:.2f}"
                if s["dd_ratio_mean"] is not None else "- Drawdown ratio: —",
                "- Regime segments (annualized diff vs B3): "
                + (", ".join(f"{REGIME_NAMES.get(k, k)} {_pct(v)}"
                             for k, v in s["regime_segments_ann_diff"].items()) or "—"),
                "",
            ]
        ab = fold["ab_check"]
        lines.append(f"A/B halves (deployed-vs-rule advantage): "
                     f"A={ab['A']}, B={ab['B']}")
        lines += ["", "### Q-state coverage (n_eff per action)", "",
                  "| q_state | " + " | ".join(a.name for a in CashAction) + " |",
                  "|" + "---|" * 7]
        for q, row in fold["coverage"].items():
            cells = " | ".join(str(row[str(int(a))]) if str(int(a)) in row
                               else str(row.get(int(a), 0)) for a in CashAction)
            lines.append(f"| {q} | {cells} |")

    lines += ["", "---", "", f"_{out['disclaimer']}_", ""]
    return "\n".join(lines)


def main():
    cfg = RlbotConfig()
    verdict_path = cfg.data.base_path / "reports" / "g2_verdict.json"
    out = json.loads(verdict_path.read_text())
    md = render(out)
    report_path = cfg.data.base_path / "reports" / "g2_report.md"
    report_path.write_text(md)
    print(f"wrote {report_path}")
    return report_path


if __name__ == "__main__":
    main()
