from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_features(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    features = payload.get("features", [])
    if not isinstance(features, list):
        raise ValueError(f"'features' is not a list in {path}")

    return [str(feature) for feature in features]


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    iot23_path = base_dir / "Full Dataset IoT23" / "iot23_full_raw_features.json"
    unsw_path = base_dir / "Full Dataset UNSW_NB15" / "unsw_full_raw_features.json"

    out_dir = base_dir / "comparison_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    iot23_features = sorted(set(load_features(iot23_path)))
    unsw_features = sorted(set(load_features(unsw_path)))

    shared_features = sorted(set(iot23_features) & set(unsw_features))
    only_in_iot23 = sorted(set(iot23_features) - set(unsw_features))
    only_in_unsw = sorted(set(unsw_features) - set(iot23_features))

    pd.DataFrame({"shared_features": shared_features}).to_csv(
        out_dir / "shared_features.csv", index=False
    )
    pd.DataFrame({"only_in_iot23": only_in_iot23}).to_csv(
        out_dir / "only_in_iot23.csv", index=False
    )
    pd.DataFrame({"only_in_unsw": only_in_unsw}).to_csv(
        out_dir / "only_in_unsw.csv", index=False
    )

    summary = {
        "iot23_feature_count": len(iot23_features),
        "unsw_feature_count": len(unsw_features),
        "shared_feature_count": len(shared_features),
        "iot23_only_count": len(only_in_iot23),
        "unsw_only_count": len(only_in_unsw),
    }

    with open(out_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Feature comparison complete.\n")
    print(f"IoT-23 raw features: {len(iot23_features)}")
    print(f"UNSW-NB15 raw features: {len(unsw_features)}")
    print(f"Shared features: {len(shared_features)}")
    print(f"Only in IoT-23: {len(only_in_iot23)}")
    print(f"Only in UNSW-NB15: {len(only_in_unsw)}")
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
