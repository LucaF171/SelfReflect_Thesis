#!/usr/bin/env python3

import sys
import json
import statistics
from collections import defaultdict
from pathlib import Path

def main(json_file_path):
    # 1. Load the data from the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # A dictionary of lists; each key is a path to the metric,
    # and each value is a list of numeric values found for that metric.
    metrics_collection = defaultdict(list)

    def collect_metrics(d, prefix=""):
        """
        Recursively traverse the dictionary `d`. Whenever a numeric
        value is found, store it under a key reflecting its path.
        """
        for k, v in d.items():
            key_path = prefix + k
            if isinstance(v, dict):
                # If value is a dict, recurse
                collect_metrics(v, prefix=key_path + ".")
            elif isinstance(v, (int, float)):
                # If value is numeric, store it
                metrics_collection[key_path].append(v)
            # Ignore other types (str, None, etc.)

    # 2. Loop over each question and collect all metrics
    for question in data:
        question_metrics = question.get("metrics", {})
        for approach_name, approach_data in question_metrics.items():
            summary_dict = approach_data.get("summary", {})
            collect_metrics(summary_dict, prefix=approach_name + ".")

    # 3. Compute means
    results = {}
    for metric_path, values in metrics_collection.items():
        mean_value = statistics.mean(values)
        results[metric_path] = mean_value

    # 4. Derive output file name from input path,
    #    ignoring any directories and slashes.
    in_path = Path(json_file_path)
    out_stem = in_path.stem  # e.g. "my_data" if the file is "my_data.json"
    out_stem = out_stem.replace("/", "")  # Just in case, remove any stray slash
    out_file_name = f"{out_stem}_means.json"

    # 5. Save results to JSON in the current directory
    with open(out_file_name, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=2)

    print(f"Mean values saved to: {out_file_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_means.py path_to_json_file")
        sys.exit(1)

    json_file_path = sys.argv[1]
    main(json_file_path)

