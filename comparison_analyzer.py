import json
import argparse
import os
from itertools import combinations
from collections import defaultdict

def nested_dict():
    """Creates a default dictionary that allows for deep nesting."""
    return defaultdict(nested_dict)

def compare_metrics_recursively(metrics1, metrics2, comparison_results):
    for metric_name, value1 in metrics1.items():
        if metric_name in metrics2:
            value2 = metrics2[metric_name]

            # If the values are dictionaries, recurse deeper
            if isinstance(value1, dict) and isinstance(value2, dict):
                compare_metrics_recursively(value1, value2, comparison_results[metric_name])
            # If the values are numbers, perform the comparison
            elif isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                if 'total' not in comparison_results[metric_name]:
                    comparison_results[metric_name]['total'] = 0
                    comparison_results[metric_name]['smaller'] = 0

                comparison_results[metric_name]['total'] += 1
                if value1 < value2:
                    comparison_results[metric_name]['smaller'] += 1

def print_results_recursively(results, path=""):
    """
    Recursively traverses the results dictionary to print the final percentages.
    """
    for key, value in results.items():
        current_path = f"{path} -> {key}" if path else key
        if 'total' in value and 'smaller' in value:
            total = value['total']
            smaller = value['smaller']
            if total > 0:
                percentage = (smaller / total) * 100
                print(f"{current_path}: {percentage:.2f}% ({smaller}/{total} cases)")
        else:
            # It's a nested dictionary, so recurse
            print_results_recursively(value, current_path)


def analyze_folder_results(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: The provided path '{folder_path}' is not a valid directory.")
        return

    all_data = []
    # Walk through the directory and find all JSON files
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Ensure data is a list and extend the main data list
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        print(f"Warning: Content of '{filename}' is not a list and will be skipped.")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from the file '{file_path}'. Skipping this file.")
            except Exception as e:
                print(f"An unexpected error occurred with file '{file_path}': {e}")


    if not all_data:
        print("No valid data was found in any JSON file in the specified folder.")
        return

    # Dynamically get summary labels from the first entry
    first_item_summaries = all_data[0].get("summaries", {})
    if not first_item_summaries:
        print("Error: Could not find 'summaries' in the first data entry of the aggregated data.")
        return
        
    summary_labels = list(first_item_summaries.keys())
    summary_pairs = list(combinations(summary_labels, 2))

    comparison_counts = nested_dict()

    # The rest of the logic remains the same, but operates on the aggregated `all_data`
    for item in all_data:
        metrics_data = item.get("metrics", {})
        for approach, approach_data in metrics_data.items():
            for summary1, summary2 in summary_pairs:
                if summary1 in approach_data and summary2 in approach_data:
                    # Comparison for (summary1 vs summary2)
                    pair_key = f"{summary1}_vs_{summary2}"
                    compare_metrics_recursively(
                        approach_data[summary1],
                        approach_data[summary2],
                        comparison_counts[approach][pair_key]
                    )
                    
                    # Comparison for (summary2 vs summary1)
                    pair_key_rev = f"{summary2}_vs_{summary1}"
                    compare_metrics_recursively(
                        approach_data[summary2],
                        approach_data[summary1],
                        comparison_counts[approach][pair_key_rev]
                    )

    print("\n--- Comparison Analysis Report ---")
    print(f"Aggregated results from all JSON files in: '{folder_path}'")
    print("Showing percentage of cases where the first summary type's metric is SMALLER than the second's.\n")
    
    for approach, approach_results in comparison_counts.items():
        print(f"--- Approach: {approach} ---")
        for pair_key, pair_results in approach_results.items():
            print(f"\nComparison: {pair_key}")
            print_results_recursively(pair_results)
        print("-" * (len(approach) + 14))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze metric results from a folder of JSON files by comparing pairs of summaries."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder containing JSON files with metric results."
    )
    args = parser.parse_args()
    analyze_folder_results(args.folder_path)
