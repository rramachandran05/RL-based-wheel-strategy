"""Merge predecessor-symbol chains into the successor's directory
(FB -> META, GOOG -> GOOGL): same equity, renamed ticker; AV serves each era
only under the symbol of the day. Collision years are concatenated and
deduplicated on (snapshot_date, cp, expiration, strike).

Run:  python -m rlbot.data.merge_predecessor_chains
"""
from __future__ import annotations

import pandas as pd

from rlbot.config import RlbotConfig

MERGES = {"FB": "META", "GOOG": "GOOGL"}


def merge_all(cfg: RlbotConfig | None = None) -> None:
    cfg = cfg or RlbotConfig()
    chains = cfg.data.base_path / "chains"
    for pred, succ in MERGES.items():
        pdir, sdir = chains / pred, chains / succ
        if not pdir.exists():
            print(f"{pred}: nothing to merge")
            continue
        sdir.mkdir(exist_ok=True)
        for pfile in sorted(pdir.glob("*.parquet")):
            sfile = sdir / pfile.name
            pred_df = pd.read_parquet(pfile)
            if pred_df.empty:
                continue
            if sfile.exists():
                combined = pd.concat([pred_df, pd.read_parquet(sfile)])
                combined = combined.drop_duplicates(
                    subset=["snapshot_date", "cp", "expiration", "strike"])
                combined = combined.sort_values(
                    ["snapshot_date", "cp", "expiration", "strike"])
                combined.to_parquet(sfile)
                print(f"  {pred}/{pfile.name} merged into {succ} "
                      f"({len(pred_df)} + existing -> {len(combined)})")
            else:
                pred_df.to_parquet(sfile)
                print(f"  {pred}/{pfile.name} -> {succ}/{pfile.name} "
                      f"({len(pred_df)} rows)")
        # keep the predecessor dir as provenance; consumers read the successor
    print("merge complete")


if __name__ == "__main__":
    merge_all()
