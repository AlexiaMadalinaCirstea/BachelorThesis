from __future__ import annotations

import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
IOT23_PATH = BASE_DIR / "Full Dataset IoT23" / "iot23_full_raw_features.json"
UNSW_PATH = BASE_DIR / "Full Dataset UNSW_NB15" / "unsw_full_raw_features.json"
OUT_CSV = BASE_DIR / "comparison_outputs" / "aligned_features_curated.csv"
OUT_SUMMARY = BASE_DIR / "comparison_outputs" / "aligned_features_curated_summary.json"


CURATED_ALIGNMENT = [
    {
        "aligned_feature": "protocol",
        "iot23_feature": "proto",
        "unsw_feature": "proto",
        "status": "accepted",
        "match_type": "direct_name_match",
        "semantic_group": "categorical_flow_metadata",
        "thesis_use": "cross_domain_alignment",
        "transformation_notes": "Use the same categorical encoding vocabulary across datasets.",
        "comparability_notes": "Protocol identity is directly shared across both schemas.",
    },
    {
        "aligned_feature": "service",
        "iot23_feature": "service",
        "unsw_feature": "service",
        "status": "accepted_with_normalization",
        "match_type": "direct_name_match",
        "semantic_group": "categorical_flow_metadata",
        "thesis_use": "cross_domain_alignment",
        "transformation_notes": "Normalize missing markers and unseen categories before encoding.",
        "comparability_notes": "The field is shared by name, but category coverage may differ between datasets.",
    },
    {
        "aligned_feature": "connection_state",
        "iot23_feature": "conn_state",
        "unsw_feature": "state",
        "status": "review_required",
        "match_type": "semantic_candidate",
        "semantic_group": "connection_state",
        "thesis_use": "candidate_for_ablation",
        "transformation_notes": "Build an explicit category mapping only if state taxonomies can be justified.",
        "comparability_notes": "Both fields describe connection state, but the label sets may not be semantically identical.",
    },
    {
        "aligned_feature": "flow_duration",
        "iot23_feature": "duration",
        "unsw_feature": "dur",
        "status": "accepted",
        "match_type": "semantic_candidate",
        "semantic_group": "time",
        "thesis_use": "cross_domain_alignment",
        "transformation_notes": "Confirm unit consistency and apply the same scaling or log transform in both datasets.",
        "comparability_notes": "Both features describe connection duration and are strong alignment candidates.",
    },
    {
        "aligned_feature": "source_to_destination_packets",
        "iot23_feature": "orig_pkts",
        "unsw_feature": "spkts",
        "status": "accepted",
        "match_type": "semantic_candidate",
        "semantic_group": "traffic_volume",
        "thesis_use": "cross_domain_alignment",
        "transformation_notes": "Treat IoT-23 originator as the UNSW source direction after confirming directional conventions.",
        "comparability_notes": "These features both measure packet count in the forward direction.",
    },
    {
        "aligned_feature": "destination_to_source_packets",
        "iot23_feature": "resp_pkts",
        "unsw_feature": "dpkts",
        "status": "accepted",
        "match_type": "semantic_candidate",
        "semantic_group": "traffic_volume",
        "thesis_use": "cross_domain_alignment",
        "transformation_notes": "Treat IoT-23 responder as the UNSW destination direction after confirming directional conventions.",
        "comparability_notes": "These features both measure packet count in the reverse direction.",
    },
    {
        "aligned_feature": "source_to_destination_bytes",
        "iot23_feature": "orig_bytes",
        "unsw_feature": "sbytes",
        "status": "accepted",
        "match_type": "semantic_candidate",
        "semantic_group": "traffic_volume",
        "thesis_use": "cross_domain_alignment",
        "transformation_notes": "Confirm whether both fields count payload bytes rather than link-layer totals.",
        "comparability_notes": "Both features represent bytes sent in the forward direction.",
    },
    {
        "aligned_feature": "destination_to_source_bytes",
        "iot23_feature": "resp_bytes",
        "unsw_feature": "dbytes",
        "status": "accepted",
        "match_type": "semantic_candidate",
        "semantic_group": "traffic_volume",
        "thesis_use": "cross_domain_alignment",
        "transformation_notes": "Confirm whether both fields count payload bytes rather than link-layer totals.",
        "comparability_notes": "Both features represent bytes sent in the reverse direction.",
    },
    {
        "aligned_feature": "source_mean_bytes_per_packet",
        "iot23_feature": "orig_bytes_per_pkt",
        "unsw_feature": "smean",
        "status": "accepted",
        "match_type": "derived_semantic_candidate",
        "semantic_group": "traffic_intensity",
        "thesis_use": "cross_domain_alignment",
        "transformation_notes": "Verify that both datasets define the statistic as bytes divided by packets for the forward direction.",
        "comparability_notes": "The names differ, but both fields appear to encode forward-direction average packet size.",
    },
    {
        "aligned_feature": "destination_mean_bytes_per_packet",
        "iot23_feature": "resp_bytes_per_pkt",
        "unsw_feature": "dmean",
        "status": "accepted",
        "match_type": "derived_semantic_candidate",
        "semantic_group": "traffic_intensity",
        "thesis_use": "cross_domain_alignment",
        "transformation_notes": "Verify that both datasets define the statistic as bytes divided by packets for the reverse direction.",
        "comparability_notes": "The names differ, but both fields appear to encode reverse-direction average packet size.",
    },
    {
        "aligned_feature": "bidirectional_byte_ratio",
        "iot23_feature": "bytes_ratio",
        "unsw_feature": "",
        "status": "rejected_no_unsw_match",
        "match_type": "iot23_only",
        "semantic_group": "derived_ratio",
        "thesis_use": "exclude_from_cross_domain_subset",
        "transformation_notes": "Would require deriving a custom ratio from UNSW byte features.",
        "comparability_notes": "No raw UNSW feature exposes the same ratio directly.",
    },
    {
        "aligned_feature": "bidirectional_packet_ratio",
        "iot23_feature": "pkts_ratio",
        "unsw_feature": "",
        "status": "rejected_no_unsw_match",
        "match_type": "iot23_only",
        "semantic_group": "derived_ratio",
        "thesis_use": "exclude_from_cross_domain_subset",
        "transformation_notes": "Would require deriving a custom ratio from UNSW packet features.",
        "comparability_notes": "No raw UNSW feature exposes the same ratio directly.",
    },
    {
        "aligned_feature": "source_ip_bytes",
        "iot23_feature": "orig_ip_bytes",
        "unsw_feature": "",
        "status": "rejected_no_unsw_match",
        "match_type": "iot23_only",
        "semantic_group": "traffic_volume",
        "thesis_use": "exclude_from_cross_domain_subset",
        "transformation_notes": "No direct UNSW counterpart was identified in the exported raw schema.",
        "comparability_notes": "UNSW does not expose an obvious raw feature for forward-direction IP byte totals.",
    },
    {
        "aligned_feature": "destination_ip_bytes",
        "iot23_feature": "resp_ip_bytes",
        "unsw_feature": "",
        "status": "rejected_no_unsw_match",
        "match_type": "iot23_only",
        "semantic_group": "traffic_volume",
        "thesis_use": "exclude_from_cross_domain_subset",
        "transformation_notes": "No direct UNSW counterpart was identified in the exported raw schema.",
        "comparability_notes": "UNSW does not expose an obvious raw feature for reverse-direction IP byte totals.",
    },
    {
        "aligned_feature": "missed_bytes",
        "iot23_feature": "missed_bytes",
        "unsw_feature": "",
        "status": "rejected_no_unsw_match",
        "match_type": "iot23_only",
        "semantic_group": "capture_artifact",
        "thesis_use": "exclude_from_cross_domain_subset",
        "transformation_notes": "This would require a dataset-specific reconstruction that is not available in UNSW.",
        "comparability_notes": "This appears to be capture-specific rather than a stable shared traffic descriptor.",
    },
]


def load_feature_set(path: Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    features = payload.get("features", [])
    if not isinstance(features, list):
        raise ValueError(f"'features' must be a list in {path}")

    return {str(feature) for feature in features}


def validate_alignment(rows: list[dict[str, str]], iot23_features: set[str], unsw_features: set[str]) -> None:
    for row in rows:
        iot23_feature = row["iot23_feature"]
        unsw_feature = row["unsw_feature"]

        if iot23_feature and iot23_feature not in iot23_features:
            raise ValueError(f"IoT-23 feature '{iot23_feature}' not found in exported feature list.")
        if unsw_feature and unsw_feature not in unsw_features:
            raise ValueError(f"UNSW feature '{unsw_feature}' not found in exported feature list.")


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    accepted = [
        row for row in rows
        if row["status"] in {"accepted", "accepted_with_normalization"}
    ]
    review = [row for row in rows if row["status"] == "review_required"]
    rejected = [row for row in rows if row["status"].startswith("rejected")]

    return {
        "n_rows": len(rows),
        "n_accepted": len(accepted),
        "n_review_required": len(review),
        "n_rejected": len(rejected),
        "accepted_aligned_features": [row["aligned_feature"] for row in accepted],
        "review_required_features": [row["aligned_feature"] for row in review],
    }


def main() -> None:
    iot23_features = load_feature_set(IOT23_PATH)
    unsw_features = load_feature_set(UNSW_PATH)

    validate_alignment(CURATED_ALIGNMENT, iot23_features, unsw_features)
    write_csv(CURATED_ALIGNMENT, OUT_CSV)

    summary = build_summary(CURATED_ALIGNMENT)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved curated alignment table to: {OUT_CSV}")
    print(f"Saved curated alignment summary to: {OUT_SUMMARY}")
    print(f"Accepted aligned features: {summary['n_accepted']}")
    print(f"Review-required features: {summary['n_review_required']}")
    print(f"Rejected features: {summary['n_rejected']}")


if __name__ == "__main__":
    main()
