import json
import copy
import matplotlib.pyplot as plt
import random
from sklearn.cluster import AgglomerativeClustering  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
#from rouge_score import rouge_scorer  # type: ignore
import numpy as np
from scipy.spatial.distance import pdist, squareform  # type: ignore
from utils import TFIDF

"""
def rouge_l_similarity(s1, s2):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(s1, s2)['rougeL'].fmeasure

def compute_similarity_matrix(sentences):
    n = len(sentences)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = rouge_l_similarity(sentences[i], sentences[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    return sim_matrix


def compute_similarity_matrix(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences, normalize_embeddings=True)
    sim_matrix = np.dot(embeddings, embeddings.T)  # Cosine similarity
    return sim_matrix


def cluster_sentences(sentences, n_clusters=7):
    sim_matrix = compute_similarity_matrix(sentences)
    dist_matrix = 1 - sim_matrix  # Convert similarity to distance
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(dist_matrix)
    return labels


def select_representatives(sentences, labels):
    unique_labels = set(labels)
    selected = [0] * len(sentences)

    for label in unique_labels:
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        best_idx = max(indices, key=lambda i: sum(rouge_l_similarity(sentences[i], sentences[j]) for j in indices))
        cluster_size = len(indices)
        selected[best_idx] = cluster_size

    return selected
"""


def compute_distance_matrix(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    dist_matrix = squareform(pdist(embeddings, metric='euclidean'))
    return dist_matrix, embeddings

def cluster_sentences(sentences, n_clusters=7):
    dist_matrix, embeddings = compute_distance_matrix(sentences)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    labels = clustering.fit_predict(embeddings)  # Ward requires original feature space, not a precomputed matrix
    return labels, embeddings

def select_representatives(sentences, labels, embeddings):
    unique_labels = set(labels)
    selected = [0] * len(sentences)

    for label in unique_labels:
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        centroid = np.mean([embeddings[i] for i in indices], axis=0)
        best_idx = min(indices, key=lambda i: np.linalg.norm(embeddings[i] - centroid))
        cluster_size = len(indices)
        selected[best_idx] = cluster_size

    return selected


def get_scores(outputs, summary_key="good", metric="masked_out_KL-out"):
    """ if not summary_key in outputs[0]["metrics"]["approach_2"]:
        return None """
    if metric == "masked_out_KL":
        return np.array([o["metrics"]["approach_2"][summary_key]["masked_out_infilling"]["KL"] for o in outputs if "KL" in o["metrics"]["approach_2"][summary_key]["masked_out_infilling"].keys()])
    elif metric == "masked_out_logl":
        return np.array([o["metrics"]["approach_2"][summary_key]["masked_out_infilling"]["Log_prob"] for o in outputs if "Log_prob" in o["metrics"]["approach_2"][summary_key]["masked_out_infilling"].keys()])
    elif metric == "masked_out_correct_top_token":
        return - np.array([o["metrics"]["approach_2"][summary_key]["masked_out_infilling"]["correct_top_token"] for o in outputs if "Log_prob" in o["metrics"]["approach_2"][summary_key]["masked_out_infilling"].keys()])
    elif metric == "pmi":
        return np.array([o["metrics"]["approach_2"][summary_key]["PMI"] for o in outputs])
    elif metric == "approach_3_Score_3_1":
        return np.array([o["metrics"]["approach_3"][summary_key]["Score_3_1"] for o in outputs])
    elif metric == "approach_3_Score_3_2":
        return np.array([o["metrics"]["approach_3"][summary_key]["Score_3_2"] for o in outputs])
    elif metric == "approach_3_Score_3_3":
        return np.array([o["metrics"]["approach_3"][summary_key]["Score_3_3"] for o in outputs])
    elif metric == "approach_4_Score_4_1":
        return np.array([o["metrics"]["approach_4"][summary_key]["Score_4_1"] for o in outputs])
    elif metric == "approach_4_Score_4_2":
        return np.array([o["metrics"]["approach_4"][summary_key]["Score_4_2"] for o in outputs])
    elif metric == "approach_4_Score_4_3":
        return np.array([o["metrics"]["approach_4"][summary_key]["Score_4_3"] for o in outputs])
    else:
        raise NotImplementedError



def get_summary(outputs, metric="masked_out_KL-out"):
    scores_good = get_scores(outputs, "good", metric)
    scores_mid = get_scores(outputs, "mid", metric)
    scores_bad = get_scores(outputs, "bad", metric)
    scores_question = get_scores(outputs, "question_as_summary", metric)
    scores_empty = get_scores(outputs, "no_question_as_summary", metric)
    good_better_than_bad = (scores_bad > scores_good).mean() * 100 if scores_good is not None and scores_bad is not None else np.nan
    ranking_correct = ((scores_bad > scores_mid) * (scores_mid > scores_good)).mean() * 100 if scores_good is not None and scores_bad is not None and scores_mid is not None else np.nan
    mid_better_than_bad= (scores_bad > scores_mid).mean() * 100 if scores_bad is not None and scores_mid is not None else np.nan
    good_better_than_mid = (scores_mid > scores_good).mean() * 100 if scores_good is not None and scores_mid is not None else np.nan
    good_better_than_question = (scores_question > scores_good).mean() * 100 if scores_good is not None and scores_question is not None else np.nan
    mid_better_than_question = (scores_question > scores_mid).mean() * 100 if scores_mid is not None and scores_question is not None else np.nan
    bad_better_than_question = (scores_question > scores_bad).mean() * 100 if scores_bad is not None and scores_question is not None else np.nan
    good_better_than_empty = (scores_empty > scores_good).mean() * 100 if scores_good is not None and scores_empty is not None else np.nan
    n = len(scores_good)

    return n, good_better_than_bad, ranking_correct, good_better_than_question, mid_better_than_question, bad_better_than_question, mid_better_than_bad, good_better_than_mid, good_better_than_empty


def print_summary(outputs, metric="masked-out"):
    n, good_better_than_bad, ranking_correct, good_better_than_question, mid_better_than_question, bad_better_than_question, mid_better_than_bad, good_better_than_mid, good_better_than_empty = get_summary(outputs, metric)
    print(f"\n\nCalculated metric: {metric}")
    print(f"Number of tested questions: {n}")
    print(f"Good < Bad: {good_better_than_bad:.2f}%")
    print(f"Good < Mid: {good_better_than_mid:.2f}%")
    print(f"Mid < Bad: {mid_better_than_bad:.2f}%")
    print(f"Good < Mid < Bad: {ranking_correct:.2f}%")
    print("")
    print(f"Good < Question-as-summary: {good_better_than_question:.2f}%")
    print(f"Mid < Question-as-summary: {mid_better_than_question:.2f}%")
    print(f"Bad < Question-as-summary: {bad_better_than_question:.2f}%")
    print(f"Good < Empty: {good_better_than_empty:.2f}%")


def subsample_answers(outputs, pct=0.2):
    new_outputs = copy.deepcopy(outputs)

    # Recalculate scores based on only the first pct% of word tokens (so roughly only the first pct% of answers)
    for i, o in enumerate(outputs):
        # Prepare answers for later
        answers = o["answers"]
        answers = ["".join(text.split()) for text in answers]

        for summary_key, summary in o["metrics"]["approach_2"].items():
            if len(summary["masked_out_infilling"]) > 0:
                # Collect KL of the chosen sentences
                cur_sentence_idx = 0
                sentences = copy.deepcopy(answers)
                kl_sum = 0
                for answer in summary["masked_out_infilling"]["detailed_logging"]["answers"]:
                    # Detect if we are still in the current sentence
                    if len(sentences[cur_sentence_idx]) == 0:
                        cur_sentence_idx += 1
                    # Use only the first pct% of answers
                    if cur_sentence_idx >= int(pct * len(sentences)):
                        break
                    word = answer["masked_words"]["masked_word"]
                    assert sentences[cur_sentence_idx].startswith(word)
                    sentences[cur_sentence_idx] = sentences[cur_sentence_idx][len(word):]

                    kl_sum += answer["masked_words"]["KL_div_sum"]

                new_outputs[i]["metrics"]["approach_2"][summary_key]["masked_out_infilling"]["KL"] = kl_sum

    return new_outputs


def cluster_and_weight_sentences(outputs, pct=0.2):
    new_outputs = copy.deepcopy(outputs)

    # Recalculate scores based on only the first pct% of word tokens (so roughly only the first pct% of answers)
    for i, o in enumerate(outputs):
        # Cluster and select sentences
        answers = o["answers"]
        labels, embeddings = cluster_sentences(answers, n_clusters=int(pct * len(answers)))
        answer_weights = select_representatives(answers, labels, embeddings)

        # Prepare answers for later
        answers = ["".join(text.split()) for text in answers]

        for summary_key, summary in o["metrics"]["approach_2"].items():
            if len(summary["masked_out_infilling"]) > 0:
                # Collect KL of the chosen sentences
                cur_sentence_idx = 0
                sentences = copy.deepcopy(answers)
                kl_sum = 0
                for answer in summary["masked_out_infilling"]["detailed_logging"]["answers"]:
                    # Detect if we are still in the current sentence
                    if len(sentences[cur_sentence_idx]) == 0:
                        cur_sentence_idx += 1
                    word = answer["masked_words"]["masked_word"]
                    assert sentences[cur_sentence_idx].startswith(word)
                    sentences[cur_sentence_idx] = sentences[cur_sentence_idx][len(word):]

                    kl_sum += float((answer_weights[cur_sentence_idx] > 0)) * answer["masked_words"]["KL_div_sum"]  # don't weight because it performs worse
                    #kl_sum += answer_weights[cur_sentence_idx] * answer["masked_words"]["KL_div_sum"]

                new_outputs[i]["metrics"]["approach_2"][summary_key]["masked_out_infilling"]["KL"] = kl_sum

    return new_outputs


def subsample_words(outputs, pct=0.2, seed=29012025):
    new_outputs = copy.deepcopy(outputs)
    random.seed(seed)  # Set the seed for reproducibility

    # Recalculate scores based on a random subset of words
    for i, o in enumerate(outputs):
        for summary_key, summary in o["metrics"]["approach_2"].items():
            if len(summary["masked_out_infilling"]) > 0:
                # Collect KL per word
                n_words = len(summary["masked_out_infilling"]["detailed_logging"]["answers"])
                chosen_idxes = random.sample(range(n_words), int(pct * n_words))
                kl_sum = 0
                for answer_idx, answer in enumerate(summary["masked_out_infilling"]["detailed_logging"]["answers"]):
                    if answer_idx in chosen_idxes:
                        kl_sum += answer["masked_words"]["KL_div_sum"]

                new_outputs[i]["metrics"]["approach_2"][summary_key]["masked_out_infilling"]["KL"] = kl_sum

    return new_outputs


def subsample_words_tfidf(outputs, pct=0.2):
    new_outputs = copy.deepcopy(outputs)
    tf_idf = TFIDF()

    # Recalculate scores based on a random subset of words
    for i, o in enumerate(outputs):
        for summary_key, summary in o["metrics"]["approach_2"].items():
            if len(summary["masked_out_infilling"]) > 0:
                # Collect KL per word
                n_words = len(summary["masked_out_infilling"]["detailed_logging"]["answers"])
                words = [answer["masked_words"]["masked_word"] for answer in summary["masked_out_infilling"]["detailed_logging"]["answers"]]
                ranking = tf_idf(words)
                chosen_idxes = ranking[:int(pct * n_words)]
                #left_out_words = set([words[i] for i in ranking[int(pct * n_words):]])
                #print(set(left_out_words))
                kl_sum = 0
                for answer_idx, answer in enumerate(summary["masked_out_infilling"]["detailed_logging"]["answers"]):
                    if answer_idx in chosen_idxes:
                        kl_sum += answer["masked_words"]["KL_div_sum"]

                new_outputs[i]["metrics"]["approach_2"][summary_key]["masked_out_infilling"]["KL"] = kl_sum

    return new_outputs


def extract_word_kls(outputs):
    all_kls = []
    for i, o in enumerate(outputs):
        kls_per_question = []
        summary = o["metrics"]["approach_2"]["good"]
        if len(summary["masked_out_infilling"]) > 0:
            for answer_idx, answer in enumerate(summary["masked_out_infilling"]["detailed_logging"]["answers"]):
                kls_per_question += [answer["masked_words"]["KL_div_sum"]]
        if len(kls_per_question) > 0:
            all_kls.append(kls_per_question)

    return all_kls


def subsample_questions(outputs, n=10):
    return outputs[:n]


def subsample_answers_and_remove_stopwords(outputs, pct=0.2, stopword_thresh=0.8):
    new_outputs = copy.deepcopy(outputs)
    tf_idf = TFIDF()

    # Recalculate scores based on only the first pct% of word tokens (so roughly only the first pct% of answers)
    for i, o in enumerate(outputs):
        # Prepare answers for later
        answers = o["answers"]
        answers = ["".join(text.split()) for text in answers]

        for summary_key, summary in o["metrics"]["approach_2"].items():
            if len(summary["masked_out_infilling"]) > 0:
                # Collect KL of the chosen sentences
                cur_sentence_idx = 0
                sentences = copy.deepcopy(answers)
                kl_sum = 0
                for answer_idx, answer in enumerate(summary["masked_out_infilling"]["detailed_logging"]["answers"]):
                    # Detect if we are still in the current sentence
                    if len(sentences[cur_sentence_idx]) == 0:
                        cur_sentence_idx += 1
                    # Use only the first pct% of answers
                    if cur_sentence_idx >= int(pct * len(sentences)):
                        break
                    word = answer["masked_words"]["masked_word"]
                    assert sentences[cur_sentence_idx].startswith(word)
                    sentences[cur_sentence_idx] = sentences[cur_sentence_idx][len(word):]

                    if not tf_idf.is_stopword([answer["masked_words"]["masked_word"]])[0]:
                        kl_sum += answer["masked_words"]["KL_div_sum"]

                new_outputs[i]["metrics"]["approach_2"][summary_key]["masked_out_infilling"]["KL"] = kl_sum

    return new_outputs


def plot_subsample_questions(outputs):
    # Generate results using only the first n questions
    ns = np.arange(5, len(outputs) + 1)
    good_better_than_bads = []
    ranking_corrects = []
    for n in ns:
        _, good_better_than_bad, ranking_correct, _, _, _ = get_summary(subsample_questions(outputs, n))
        good_better_than_bads.append(good_better_than_bad / 100)
        ranking_corrects.append(ranking_correct / 100)

    # Plot
    plt.figure(figsize=(4, 3))
    plt.xlim((-2, 102))
    plt.plot(ns, good_better_than_bads, label="Good <= bad", c="tab:blue")
    plt.plot(ns, ranking_corrects, label="Good <= mid <= bad", c="tab:orange")
    plt.xlabel('(Reduced) Number of Questions')
    plt.ylabel('Correctness of our Metric')
    plt.tight_layout()
    plt.grid()
    plt.savefig("subsampling_questions.png")
    plt.close()


def segment_averages(data, n_segments=50):
    """
    Calculates 50 running averages, each corresponding to 2% of the input data.

    Args:
        data (list): A list of floats with length > 50.

    Returns:
        list: A list of 50 running averages.
    """
    if len(data) < n_segments:
        raise ValueError(f"The input data must have a length greater than {n_segments}.")

    length = len(data)
    segment_size = max(1, length // n_segments)  # Calculate the size of each 2% segment
    averages = []

    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size if i < (n_segments - 1) else length  # Include remainder in the last segment
        segment = data[start_idx:end_idx]
        segment_mean = sum(segment) / len(segment)
        averages.append(segment_mean)

    return averages


if __name__=="__main__":
    # Add multiple files into one list if you want them to be concatenated (two GPU runs on different question sets),
    # add multiple lists if you want them to be analyzed independently (individual models, individual edge cases, ...)
    """ different_experiment_files = [
        ["results/qwen_7b/output_evaluation_dataset_detailed.json"],
        ["results/qwen_7b/output_edgecase_dirac_verbosity_detailed.json"],
        ["results/qwen_7b/output_edgecase_majority_almostdirac_detailed.json"],
        ["results/qwen_7b/output_edgecase_number_ranges_detailed.json"],
        ["results/qwen_7b/output_edgecase_idk_cases_detailed.json"],
        ["results/qwen_7b/output_edgecase_wording_dirac_detailed.json"],
        ["results/qwen_7b/output_edgecase_wording_despite_majority_almostdirac_detailed.json"]
    ] """
    different_experiment_files = [
        #["results/metric_results_manual_summaries_0_to_333_approach_2_date_20_2_detailed.json"],
        #["results/metric_results_manual_summaries_0_to_333_approach_3_date_21_2_detailed.json"],
        ["results/metric_results_manual_summaries_0_to_333_approach_4_date_20_2_detailed.json"]
    ]

    for files in different_experiment_files:
        print("Results for " + ", ".join(files))
        outputs = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # If data is a dict, convert its values to a list
                if isinstance(data, dict):
                    outputs += list(data.values())
                else:
                    outputs += data

            print("Type of outputs:", type(outputs))
            """ if isinstance(outputs, dict):
                print("Dictionary keys:", list(outputs.keys()))
            elif isinstance(outputs, list):
                print("Length of outputs:", len(outputs))
                print("Type of first element:", type(outputs[0]))
                print("Content of first element:", outputs[0])
            else:
                pass """

        # Overall results
        test_overall = True
        if test_overall:
            print("Using all answers:")
            """ print_summary(outputs, metric="masked_out_KL")
            print_summary(outputs, metric="masked_out_logl")
            print_summary(outputs, metric="masked_out_correct_top_token")
            print_summary(outputs, metric="pmi")
            print_summary(outputs, metric="approach_3_Score_3_1")
            print_summary(outputs, metric="approach_3_Score_3_2")
            print_summary(outputs, metric="approach_3_Score_3_3")"""
            print_summary(outputs, metric="approach_4_Score_4_1")
            print_summary(outputs, metric="approach_4_Score_4_2")
            print_summary(outputs, metric="approach_4_Score_4_3")

        test_sentence_subsampling = False
        if test_sentence_subsampling:
            # Randomly subsample answers
            print("\nUsing first 5 answers:")
            print_summary(subsample_answers(outputs, 0.1))
            print("\nUsing first 10 answers:")
            print_summary(subsample_answers(outputs, 0.2))
            print("\nUsing first 20 answers:")
            print_summary(subsample_answers(outputs, 0.4))

        test_sentence_clustering = False
        if test_sentence_clustering:
            # Cluster answers
            print("\nUsing 10% of clustered sentences:")
            print_summary(cluster_and_weight_sentences(outputs, 0.1))
            print("\nUsing 20% of clustered sentences:")
            print_summary(cluster_and_weight_sentences(outputs, 0.2))
            print("\nUsing 40% of clustered sentences:")
            print_summary(cluster_and_weight_sentences(outputs, 0.4))

        test_word_subsampling = False
        if test_word_subsampling:
            # Random word subsampling
            print("\nUsing 10% of words:")
            print_summary(subsample_words(outputs, 0.1))
            print("\nUsing 20% of words:")
            print_summary(subsample_words(outputs, 0.2))
            print("\nUsing 40% of words:")
            print_summary(subsample_words(outputs, 0.4))

        test_tfidf_subsampling = False
        if test_tfidf_subsampling:
            print("\nUsing 10% rarest words:")
            print_summary(subsample_words_tfidf(outputs, 0.1))
            print("\nUsing 20% rarest words:")
            print_summary(subsample_words_tfidf(outputs, 0.2))
            print("\nUsing 40% rarest words:")
            print_summary(subsample_words_tfidf(outputs, 0.4))

        test_sentence_subsampling_and_stopword_removal = False
        if test_sentence_subsampling_and_stopword_removal:
            # Randomly subsample answers
            print("\nUsing first 5 answers:")
            print_summary(subsample_answers_and_remove_stopwords(outputs, 0.1))
            print("\nUsing first 10 answers:")
            print_summary(subsample_answers_and_remove_stopwords(outputs, 0.2))
            print("\nUsing first 20 answers:")
            print_summary(subsample_answers_and_remove_stopwords(outputs, 0.4))

        test_question_subsampling = False
        if test_question_subsampling:
            # Use only the first n questions
            plot_subsample_questions(outputs)

        test_order_bias = False
        if test_order_bias:
            # Test if earlier answers have harsher KL penalties because the model cannot recall them anymore
            all_kls = extract_word_kls(outputs)
            averaged_kls = [segment_averages(kls) for kls in all_kls if len(kls) >= 50]
            x = range(len(averaged_kls[0]))

            # Plot each list as a separate line
            for line in averaged_kls:
                plt.plot(x, line, alpha=0.2)
            plt.plot(x, np.array(averaged_kls).mean(0), linewidth=4, c="black")
            plt.xlabel('Answer index (lower = far back)')
            plt.ylabel('KL Penalty')
            plt.title('Plot of List of Lists')
            plt.tight_layout()
            plt.savefig("older_answers_worse_analysis.png")
            plt.close()
