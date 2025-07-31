#!/usr/bin/env python3

import os
import json
from pathlib import Path

def merge_summaries_in_folder(folder_path, output_file="merged_summaries.json"):
    """
    This script looks in `folder_path` for all .json files,
    merges their question summaries (excluding 'answers'),
    and writes them to `output_file`. The summaries for each question are then
    ordered by a predefined list of strategies:
        1) greedy
        2) basic
        3) CoT
        4) beam_search
        5) answer_dist
    Any unrecognized strategy is placed at the end, in the order encountered.
    """

    # Desired strategy order
    strategy_order_list = ["greedy", "basic", "CoT", "beam_search", "answer_dist"]
    # Create a lookup dict: strategy -> rank
    strategy_rank = {strategy: i for i, strategy in enumerate(strategy_order_list)}

    merged_data = {}

    # 1. Iterate over JSON files in the given folder
    for file_path in Path(folder_path).glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid or non-JSON file: {file_path}")
                continue

        # 2. Each file's top-level keys are question IDs
        for question_id, question_obj in data.items():
            # Initialize if new question ID
            if question_id not in merged_data:
                merged_data[question_id] = {
                    "question_text": question_obj.get("question_text", ""),
                    "summaries_across_files": []
                }
            else:
                # Update question_text if missing
                if not merged_data[question_id]["question_text"]:
                    merged_data[question_id]["question_text"] = question_obj.get("question_text", "")

            # 3. Extract the summary and summary_strategy
            summary_strategy = question_obj.get("summary_strategy", "unknown_strategy")
            summary_text = ""
            if "summaries" in question_obj:
                summary_text = question_obj["summaries"].get("summary", "")

            # 4. Append an entry with only strategy and summary
            merged_data[question_id]["summaries_across_files"].append({
                "summary_strategy": summary_strategy,
                "summary": summary_text
            })

    # 5. Sort the summaries according to the defined strategy order
    for question_id, qdata in merged_data.items():
        summaries_list = qdata["summaries_across_files"]
        # Sort by the rank of the summary_strategy
        # Unrecognized strategies get a default rank (999) and end up last
        summaries_list.sort(
            key=lambda item: strategy_rank.get(item["summary_strategy"], 999)
        )

    # 6. Write merged data to disk
    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(merged_data, out, indent=2, ensure_ascii=False)

    print(f"Merged summaries written to: {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python merge_summaries.py <folder_path> [output_file]")
        sys.exit(1)

    folder = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else "merged_summaries.json"

    merge_summaries_in_folder(folder, out_file)

