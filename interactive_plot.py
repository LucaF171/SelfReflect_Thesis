import plotly.express as px
import plotly.graph_objects as go
import plotly
import pandas as pd
import json
import textwrap


# User settings
COMPARE_TO_GOOD_SUMMARY = True   # Set this to True if you want to substract the good summary score from all scores, so
                                 # that "better than good" is left from the 0 line and "worse than good" is on the right
different_experiment_files = [
        "results/qwen_top3/output_evaluation_dataset_detailed.json",
        #"results/qwen_7b/output_edgecase_dirac_verbosity_detailed.json",
        #"results/qwen_7b/output_edgecase_majority_almostdirac_detailed.json",
        #"results/qwen_7b/output_edgecase_number_ranges_detailed.json",
        #"results/qwen_7b/output_edgecase_idk_cases_detailed.json",
        #"results/qwen_7b/output_edgecase_wording_dirac_detailed.json",
        #"results/qwen_7b/output_edgecase_wording_despite_majority_almostdirac_detailed.json"
    ]
y_space_between_files = 20  # If files are different experiments, visually separate them.
                            # If you concat multiple files from, e.g., a split run, set this to 0

# load our data
data = []
y_indices = {}
group_sizes = {}
start_idx = 0
for idx, file in enumerate(different_experiment_files):
    with open(file, "r", encoding="utf-8") as f:
        new_data = json.load(f)
        group_sizes[idx] = len(new_data)
        for i in range(len(new_data)):
            y_indices[start_idx] = idx
            start_idx += 1
        data += new_data

# Convert data into a DataFrame for plotting
plot_data = []

for idx, entry in enumerate(data):
    for key in entry["metrics"]["approach_2"]:
        if ((key not in ["question_as_summary", "no_question_as_summary"]) and
            ("KL" in entry["metrics"]["approach_2"][key]["masked_out_infilling"].keys())):
            group = y_indices[idx]
            plot_data.append({
                "x": entry["metrics"]["approach_2"][key]["masked_out_infilling"]["KL"]["approach_2_2"] -
                     (entry["metrics"]["approach_2"]["good"]["masked_out_infilling"]["KL"]["approach_2_2"] if COMPARE_TO_GOOD_SUMMARY else 0),
                "y": y_space_between_files * group + idx,  # Spread the points a bit along the y axis and group by file/experiment
                "category": key,
                "question": entry["prompt"],
                "summary": entry["summaries"][key][:min(200, len(entry["summaries"][key]))],
                "metric": entry["metrics"]["approach_2"][key]["masked_out_infilling"]["KL"]["approach_2_2"],
                "For comparison: Good summary": entry["summaries"]["good"][:min(200, len(entry["summaries"]["good"]))] if key != "good" else "",
                "For comparison: Good summary metric": entry["metrics"]["approach_2"]["good"]["masked_out_infilling"]["KL"]["approach_2_2"] if key != "good" else "",
                "index": idx,
                "experiment": different_experiment_files[group],
            })

# Create DataFrame
df = pd.DataFrame(plot_data)

# Create scatter plot
color_map = {
                     "good": "forestgreen",
                     "mid": "gold",
                     "bad": "orangered",
                     "question_as_summary": "blue",
                     "no_question_as_summary": "purple"
                 }
fig = px.scatter(df,
                 x="x",
                 y="y",
                 color="category",
                 color_discrete_map=color_map,
                 hover_data={
                     "question": True,
                     "summary": True,
                     "metric": True,
                     "For comparison: Good summary": True,
                     "For comparison: Good summary metric": True,
                     "index": True,
                     "x": False,
                     "y": False,
                     "experiment": True
                 })

"""
# Boxplots
fig = plotly.tools.make_subplots(rows=1, cols=1)
for key in data[0]["metrics"]["approach_2"].keys():
    # Create box plots to show category distributions
    fig.add_trace(go.Box(x=df["x"][df["category"] == key], yaxis='y2', fillcolor=color_map[key]))
    fig.update_layout(yaxis2=dict(
            matches='y',
            layer="below traces",
            overlaying="y",
        ),)
"""

# Update layout for better visualization
fig.update_traces(marker=dict(size=10))
fig.update_layout(title="", xaxis_title="KL Score (lower = better summary)", yaxis_title="Question index")

# Show the plot
fig.show()
