"""
Scenario level split for utilities for IoT-23.
"""

from typing import Dict, List

TEST_SCENARIOS = [
    "CTU-IoT-Malware-Capture-34-1",
    "CTU-IoT-Malware-Capture-43-1",
    "CTU-IoT-Malware-Capture-44-1",
    "CTU-Honeypot-Capture-4-1",
    "CTU-Honeypot-Capture-5-1",
    "CTU-Honeypot-Capture-7-1",
]

VAL_SCENARIOS = [
    "CTU-IoT-Malware-Capture-49-1",
    "CTU-IoT-Malware-Capture-52-1",
    "CTU-IoT-Malware-Capture-20-1",
]


def get_fixed_split_map() -> Dict[str, str]:
    split_map = {}
    for s in TEST_SCENARIOS:
        split_map[s] = "test"
    for s in VAL_SCENARIOS:
        if s in split_map:
            raise ValueError(f"Scenario appears in multiple fixed splits: {s}")
        split_map[s] = "val"
    return split_map


def get_fixed_split(scenario_name: str) -> str:
    return get_fixed_split_map().get(scenario_name, "train")


def generate_loso_splits(all_scenarios: List[str]) -> List[dict]:
    """
    Leave-one-scenario-out split generator.
    Each split holds out one scenario for test and uses the rest for train.
    You can later extend this to carve validation out of train.
    """
    unique_scenarios = sorted(set(all_scenarios))
    splits = []

    for held_out in unique_scenarios:
        train_scenarios = [s for s in unique_scenarios if s != held_out]
        splits.append({
            "name": f"loso_{held_out}",
            "train": train_scenarios,
            "test": [held_out],
        })

    return splits