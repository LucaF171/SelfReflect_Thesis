import json
import torch
import pprint

majority_answers_per_question = {
    "0": [0, 2, 4, 6, 7, 8, 9],
    "1": [0, 1, 2, 3, 5, 7, 9],
    "2": [0, 1, 3, 5, 7, 8, 9],
    "3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # interesting case because high dependency on formatting
    "4": [0, 2, 3, 4, 7, 8, 9],
    "5": [0, 1, 2, 3, 4, 5, 6, 8],
    "6": [1, 2, 4, 5, 6, 7, 9],
    "7": [2, 3, 4, 5, 6, 7, 8, 9],
    "8": [0, 1, 2, 5, 6, 7],
    "9": [0, 1, 2, 6],
    "10": [0, 1, 2, 3, 6, 7, 8, 9],
    "11": [1, 2, 7, 9],
    "12": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "13": [0, 4, 5],
    "14": [0, 2, 3, 4, 6, 8, 9],
    "15": [1, 3, 4, 5, 7, 8, 9],
    "16": [2, 3, 5, 7, 8, 9],
    "17": [0, 1, 2, 4, 5, 6, 7, 9],
    "18": [0, 1, 3, 4, 5, 6, 7, 8],
    "19": [1, 2, 6],
    "20": [1, 3, 4, 5, 6, 8],
    "21": [0, 2, 3, 4, 5, 6, 7, 8]
}

# here the softmax rescaling did not work because the "others" class was actually >50%,
# or because the distribution was too close to random (so we don't really have a signal)
skip_questions = {}  # {"15", "13", "1"}

def get_scores(data, question_id, summary_key):
    all_res = {}
    n_tokens = 0
    for answer in data[question_id]["metrics"]["approach_2"][summary_key]["masked_out_infilling"]["detailed_logging"][
        "answers"]:
        answer_idx = answer["masked_words"]["answer_id"]
        if answer_idx not in all_res:
            all_res[answer_idx] = [0., answer["answer_text"]]
        all_res[answer_idx][0] += answer["masked_words"]["KL_div_sum"]
        n_tokens += len(answer["masked_words"]["KL_div_per_token"])

    return all_res, n_tokens

def print_per_answer(data, question_id="0", summary_key = "temp_1"):
    print(f'Distance {(data[question_id]["metrics"]["approach_2"][summary_key]["masked_out_infilling"]["KL"]*1000):.3f} for summary: {data[question_id]["summaries"][summary_key]}')
    print("Here's your score broken down into the distance to each sample of the LLM distribution:")
    all_res, n_tokens = get_scores(data, question_id, summary_key)
    for key, value in all_res.items():
        print(f"Distance {(value[0] / n_tokens * 1000):.3f} for answer: {value[1]}")

    print()

def flip_minority_answers(scores, question_id):
    return [score if (i in majority_answers_per_question[question_id]) else -score for i, score in enumerate(scores)]

def check_monotonicity(data):
    # This checks if increasing the probability leads to a lower distance,
    # and decreasing the probability leads to a higher distance
    res = {}
    for question_id in data:
        if question_id in skip_questions:
            continue
        temp_low, _ = get_scores(data, question_id, "temp_0.4")
        temp_normal, _ = get_scores(data, question_id, "temp_1")
        temp_high, _ = get_scores(data, question_id, "temp_4")
        metric_temp_low = []
        metric_temp_normal = []
        metric_temp_high = []
        for answer_idx in temp_low:
            metric_temp_low.append(temp_low[answer_idx][0])
            metric_temp_normal.append(temp_normal[answer_idx][0])
            metric_temp_high.append(temp_high[answer_idx][0])

        # For majority answers, we want to check that the distance _decreases_ with higher temp,
        # for minority answers vice versa
        metric_temp_low = flip_minority_answers(metric_temp_low, question_id)
        metric_temp_normal = flip_minority_answers(metric_temp_normal, question_id)
        metric_temp_high = flip_minority_answers(metric_temp_high, question_id)

        pct_ordering_correct = ((torch.Tensor(metric_temp_low) > torch.Tensor(metric_temp_normal)) * (torch.Tensor(metric_temp_normal) > torch.Tensor(metric_temp_high))).float().mean().item()
        res[question_id] = pct_ordering_correct

    overall = torch.Tensor(list(res.values())).mean().item()
    print("Monotonicity is fulfilled per question:")
    pprint.pprint(res)
    print(f"Overall: {overall}")

with open("results/pct_analysis/metric_results_edgecase_pct_monotonicity_approach_2_date_6_3_detailed.json", "r", encoding="utf-8") as f:
    data = json.load(f)


for question_id in data:
    print(question_id)
    print()
    print_per_answer(data, question_id, "temp_0.4")
    print_per_answer(data, question_id, "temp_1")
    print_per_answer(data, question_id, "temp_4")

check_monotonicity(data)