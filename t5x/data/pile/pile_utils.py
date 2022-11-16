# Copyright 2022
import os

def get_pile_files():
    """name of the pile files"""

    return {
        "train": [f"{i:02}.jsonl" for i in range(19)],
        "validation": ["19.jsonl"],
        "test": ["19.jsonl"],
        }

def get_minipile_files(num_files):
    """name of the minipile files"""

    return {
        "train": [f"00_x0{i:02}.jsonl" for i in range(num_files)],
        "validation": ["19.jsonl"],
        "test": ["19.jsonl"],
        }
