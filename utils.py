import random
import torch
import json
import json
import re
import logging
import argparse
from pprint import pprint
import torch.nn.functional as F
from config import ExperimentConfig
from logging_config import setup_logging
from datasets import load_dataset # type: ignore
import gc
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.datasets import fetch_20newsgroups # type: ignore
import string
import numpy as np
from typing import Any, List
from config import ExperimentConfig
import os
from llm_wrapper import LLMWrapper
import datetime

logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    logger.info("Random seed set to: %s", seed)


def stitch_prompt(model="Qwen/Qwen2.5-7B-Instruct", system=None, user="", assistant=None):
    """
    Stitches together the messages list from prompts, while respecting different Llama / Gemma styles.

    Args:
        model: string, the huggingface model name
        system: string, the content of the system prompt
        user: string, the content of the user prompt
        assistant: string, the content of the assistant prompt

    Returns:
        List of messages, to be fed into apply_chat_template.
    """
    messages = []

    # Add system prompt
    if system is not None:
        if "gemma" in model.lower():
            # Gemma has no system prompt, needs to be added to user prompt
            # https://github.com/abetlen/llama-cpp-python/issues/1580
            user = "\n".join((system, user))
        else:
            messages.append({"role": "system", "content": system})

    # Add user prompt
    messages.append({"role": "user", "content": user})

    # Add assistant prompt
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})

    return messages


class TFIDF:
    def __init__(self):
        # Load a default corpus (20 Newsgroups dataset)
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        corpus = newsgroups.data

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = np.array(vectorizer.get_feature_names_out())

        # Compute average TF-IDF score per word
        tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        self.word_tfidf = dict(zip(feature_names, tfidf_scores))

    def __call__(self, word_list):
        """
        Sorts words by the rarest word first, example:
        print(TFIDF()(["christ", "you", "acidic", "telekfngieffh"]))
        > [2 0 1 3]
        Note that "telekfngieffh" is listed at the end,
        because it is not part of the vocabulary, so it's considered rare.
        """
        return np.argsort(self.get_tfidf_score(word_list))

    def _preprocess(self, word_list):
        return [word.strip(string.whitespace + string.punctuation).lower() for word in word_list]

    def get_tfidf_score(self, word_list):
        # For words in the vocabulary: Great, give their TF-IDF score
        # For words not in the vocabulary: Wow, probably super rare. Mark as rarer than the other words
        return [self.word_tfidf[word] if word in self.word_tfidf else -1 for word in self._preprocess(word_list)]

    def is_stopword(self, word_list, threshold=0.012):
        """
        Expects a list of words as input, outputs a list of booleans
        threshold is tuned manually to roughly exclude stopwords, but no important words
        """
        tfidf_scores = self.get_tfidf_score(word_list)
        return [s > threshold for s in tfidf_scores]


def generate_answers():
    logger.info("Generating answers for questions")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_questions",
        type=int,
        default=1000,
        help="Number of questions to generate answers for (default: 1000)"
    )
    parser.add_argument(
        "--num_answers",
        type=int,
        default=50,
        help="Number of generated answers per question (default=50)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Name of the LLaMA model to use"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="google-research-datasets/natural_questions",
        help="Name of the dataset to use"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_dataset.json",
        help="Name of the output file to save the questions and answers"
    )
    parser.add_argument(
        "--start_question",
        type=int,
        default=0,
        help="Index of the first question to process (default: 0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed (default that manual summaries have been generated for = 1)"
    )

    args = parser.parse_args()

    num_questions = args.num_questions
    num_answers = args.num_answers
    model_name = args.model_name
    dataset_name = args.dataset_name
    output_file = args.output_file
    start_question = args.start_question

    # For the answers to be random for each answer generation, comment this out
    set_seed(args.seed)

    llm_wrapper = LLMWrapper(model_name=model_name)

    # load split based on the dataset
    if  dataset_name == "basicv8vc/SimpleQA":
        ds = load_dataset(dataset_name, split="test")
        questions = ds['problem']
    elif dataset_name == "google-research-datasets/natural_questions":
        ds = load_dataset(dataset_name, split="validation")
        questions = ds['question']
    elif dataset_name == "mandarjoshi/trivia_qa":
        ds = load_dataset(dataset_name, "rc", split="validation")
        questions = ds['question']
    else:
        logger.critical("Dataset not found! Check dataset name or provide a different dataset.")


    output_data = {}
    for idx, item in enumerate(questions[start_question:start_question+num_questions]):
        logger.info("Generating answers for question %s/%s", idx+1, num_questions)

        system_content = ExperimentConfig.system_prompt

        if dataset_name == "google-research-datasets/natural_questions":
            question_text = item['text']
        elif dataset_name == "basicv8vc/SimpleQA" or dataset_name == "mandarjoshi/trivia_qa":
            question_text = item

        if question_text is not None:
            if question_text.endswith("?"):
                question_text += "\n"
            elif not question_text.endswith("?\n"):
                question_text += "?\n"

            if not question_text.startswith("Question: "):
                question_text = "Question: " + question_text
        else:
            question_text = ""
            
        user_content = f"{question_text}"

        messages = stitch_prompt(llm_wrapper.model.model.name_or_path, system=system_content, user=user_content)

        chat_prompt_str = llm_wrapper.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        logger.info("Chat prompt: %s", chat_prompt_str)

        answers = []
        for i in range(num_answers):
            logger.info("Generating answer %s/%s", i+1, num_answers)
            answer = llm_wrapper.generate(prompt_text=chat_prompt_str)
            logger.info("Generated answer: %s", answer)
            answers.append(answer)

        output_data[idx] = {
            "question_text": question_text,
            "answers": answers,
        }

        pprint(output_data[idx])

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    # -------------------------------------------------------------
    # NEW: Save the config for the generate_answers() method
    # -------------------------------------------------------------
    config_filename = output_file.replace(".json", "_generate_config.json")
    config_entry = {
            "num_questions": num_questions,
            "num_answers": num_answers,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "output_file": output_file
        }
    with open(config_filename, "w", encoding="utf-8") as cf:
        json.dump([config_entry], cf, indent=2)
    logger.info(f"Generation config data saved to {config_filename}")


def print_metrics(description, *metrics):
    """
    Prints a formatted description along with an arbitrary number of metric values.
    """
    print(f"\n{description}")
    for idx, metric in enumerate(metrics, 1):
        if metric is not None:
            print(f"  Metric {idx}: {metric}")
    print("=" * 50)


def print_token_probs(token_probs, tokens):
    #print("\nToken Probabilities:")
    for token, prob in zip(tokens, token_probs):
        cleaned_token = token[1:] if token.startswith('Ġ') else token
        
        # If prob is a float, print with {:.6f}. Otherwise, just print it as-is.
        if isinstance(prob, float):
            logger.info(f"  Token: '{cleaned_token}' | Probability: {prob:.6f}")
        else:
            # Fallback: prob is probably a list or something else we can't format numerically
            logger.info(f"  Token: '{cleaned_token}' | Probability: {prob}")



""" def print_top_k_token_probs(top_tokens_per_position, tokens, k=5):
    print(f"\nTop-{k} Token Probabilities per Position:")
    for idx, (token, top_tokens) in enumerate(zip(tokens, top_tokens_per_position)):
        cleaned_token = token[1:] if token.startswith('Ġ') else token
        print(f"  Token Position {idx + 1}: '{cleaned_token}'")
        for rank, (top_token, top_prob) in enumerate(top_tokens, start=1):
            clean_top_token = top_token[1:] if top_token.startswith('Ġ') else top_token
            print(f"    {rank}. Token: '{clean_top_token}' | Probability: {top_prob:.6f}") """

def print_top_k_token_probs(llm_wrapper, per_pos_log_probs, target_tokens, target_seq_len, top_n_debug=10):
    try:
        top_log_probs, top_indices = torch.topk(per_pos_log_probs, top_n_debug, dim=-1)
        top_probs = torch.exp(top_log_probs)
        top_probs_list = top_probs[0].tolist()
        top_indices_list = top_indices[0].tolist()

        top_tokens_per_position = []
        for pos_idx, (idx_list, prob_list) in enumerate(zip(top_indices_list, top_probs_list)):
            zipped = list(zip(llm_wrapper.tokenizer.convert_ids_to_tokens(idx_list), prob_list))
            top_tokens_per_position.append(zipped)
            logger.info("=== Top 10 tokens for target token index %s ('%s') ===", pos_idx, target_tokens[pos_idx])
            for i, (tok_str, prob_val) in enumerate(zipped):
                logger.info("   Rank %s: '%s' => prob %s", i + 1, tok_str, prob_val)
    except Exception as e:
        logger.error("Error computing top tokens: %s", e)
        top_tokens_per_position = [[] for _ in range(target_seq_len)]


def create_manual_summaries():
    with open('questions_and_answers.json', 'r') as f:
        questions_and_answers = json.load(f)

    manual_summaries = []
    for item in questions_and_answers.values():
        new_item = {
            "question": item['question_text'],
            "summary": {
                "good": "Good summary here",
                "bad": "Bad summary here",
                "mid": "Mid summary here"
            }
        }
        manual_summaries.append(new_item)

    with open('manual_summaries.json', 'w') as f:
        json.dump(manual_summaries, f, indent=4)


def pad_second_tensor(tensor1, tensor2):
    """
    Pads the second tensor (which is smaller in some of the dimensions) so that it has the same size as the first.

    Args:
        tensor1: torch tensor, larger or equal size in some dimensions
        tensor2: torch tensor, smaller or equal in some dimensions

    Returns:
        the padded second tensor that now has the size of the first.
    """
    # Get the size of each tensor
    size1 = tensor1.size()
    size2 = tensor2.size()

    # Compute the padding needed
    # Note: `torch.nn.functional.pad` pads in (last dimension, ..., first dimension) order
    padding = [0] * (2 * len(size2))  # Start with no padding
    for i in range(len(size2)):
        diff = size1[i] - size2[i]
        if diff > 0:
            padding[2 * i] = diff  # Pad only at the end of each dimension

    # Apply the padding to tensor2
    padding.reverse() # F.pad expects to go backwards
    padded_tensor2 = F.pad(tensor2, padding, mode='constant', value=-1)
    return padded_tensor2


def save_output_data(data, config, custom_filename=None):
    """
    Saves 'data' to a JSON file. The default is config.output_file + ".json"
    but if custom_filename is provided, use that instead.
    """

    if custom_filename is not None:
        output_file_path = custom_filename
    else:
        output_file_path = config.output_file + ".json"

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Provide the summaries per user input
def process_summaries_interactively():
    """
    Prompt the user to input summaries for different quality levels.

    Returns:
        Dict[str, str]: Dictionary containing 'good', 'mid', and 'bad' summaries.
    """
    logger.info("Prompting user to provide summaries.")
    good_summary = input("Please provide a good summary: ")
    logger.info("Provided good summary: %s", good_summary)
    mid_summary = input("Please provide a mid summary: ")
    logger.info("Provided mid summary: %s", mid_summary)
    bad_summary = input("Please provide a bad summary: ")
    logger.info("Provided bad summary: %s", bad_summary)
    return {
        "good": good_summary,
        "mid": mid_summary,
        "bad": bad_summary,
    }

# Load the existing summaries from the evaluation dataset file
def process_summaries_from_file(evaluation_dataset_file, question_idx):
    """
    Provide default summaries in non-interactive mode.

    Returns:
        Dict[str, str]: Dictionary containing 'good', 'mid', and 'bad' summaries.
    """
    logger.debug("Using summaries from file: %s", evaluation_dataset_file)
    with open(evaluation_dataset_file, "r") as json_file:
        data = json.load(json_file)
    try:
        question_data = data[question_idx]
        return question_data["summaries"]
    except KeyError:
        logger.warning("No summaries found for Question %s", question_idx)
        return {
            "good": "",
            "mid": "",
            "bad": "",
        }

def parse_elements_from_text(self, text):
    """
    Parses a numbered list from the generated text.
    """
    elements = re.findall(r'\d+\.\s*(.*)', text)
    return [element.strip() for element in elements]


def print_gpu_memory():
    size_model_mb = 0
    size_tensors_mb = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and str(obj.device) != "cpu" or (hasattr(obj, 'data') and torch.is_tensor(obj.data) and str(obj.data.device) != "cpu"):
                size_mb = obj.element_size() * obj.nelement() / 1e+6
                #print(type(obj), f"{size_mb:.2f}MB")
                if "parameter" in str(type(obj)).lower():
                    size_model_mb += size_mb
                else:
                    size_tensors_mb += size_mb
        except:
            pass

    print(f"Total size of model parameters: {(size_model_mb/1000):.2f}GB")
    print(f"Total size of other tensors: {(size_tensors_mb/1000):.2f}GB")



def create_summary_data(full_data):
    """
    Return a stripped-down version of 'full_data' that removes any
    large/detailed fields from each approach.
    Accepts either a list or dictionary of data.
    """
    # If a dictionary is passed, process its values
    if isinstance(full_data, dict):
        full_data_list = list(full_data.values())
    else:
        full_data_list = full_data
    
    summary_list = []

    for entry in full_data_list:
        # Ensure entry is a dict
        if not isinstance(entry, dict):
            logger.warning(f"Skipping non-dictionary entry: {entry}")
            continue

        summary_entry = {
            "prompt": entry.get("prompt"),
            "answers": entry.get("answers"),
            "summaries": entry.get("summaries"),
            "metrics": {}
        }

        metrics_dict = entry.get("metrics", {})
        for approach_name, approach_data in metrics_dict.items():
            summary_entry["metrics"][approach_name] = {}

            for summary_label, sub_data in approach_data.items():
                pruned_data = dict(sub_data)

                if approach_name == "approach_2" and "masked_out_infilling" in pruned_data:
                    masked_out_data = pruned_data["masked_out_infilling"]
                    pruned_data["masked_out_infilling"] = {
                        "Log_prob": masked_out_data.get("Log_prob"),
                        "KL": masked_out_data.get("KL")
                    }
                
                summary_entry["metrics"][approach_name][summary_label] = pruned_data

        summary_list.append(summary_entry)

    return summary_list


def get_unique_filename(path: str) -> str:
    """
    Returns a unique file path by appending an increasing counter if the file already exists.
    E.g., if 'results.json' exists, it will try 'results_1.json', 'results_2.json', etc.
    """
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 1
    new_path = f"{base}_{counter}{ext}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{ext}"
    return new_path

def save_all_results(all_approach_results: List[Any], config: ExperimentConfig, llm_wrapper: LLMWrapper) -> None:
    """
    Saves detailed results, summary results, and configuration data.
    """
    # Ensure the output folder exists
    os.makedirs(config.result_folder, exist_ok=True)

    if not config.custom_output_file:
        # Only keep the model name, remove everything before the last '/'
        evaluation_dataset_filename = os.path.basename(config.evaluation_dataset)
        evaluation_dataset_filename = evaluation_dataset_filename.replace(".json", "")
        current_date = datetime.date.today()

        file_name = (
            f"metric_results_{evaluation_dataset_filename}_"
            f"approach_{config.approach}_"
            f"date_{current_date.day}_{current_date.month}"
        )
    else:
        file_name = config.output_file

    # Generate base paths
    detailed_filename = os.path.join(config.result_folder, file_name + "_detailed.json")
    summary_filename = os.path.join(config.result_folder, file_name + "_summary.json")
    config_filename   = os.path.join(config.result_folder, file_name + "_config.json")

    # Get unique filenames to avoid overwriting
    detailed_filename = get_unique_filename(detailed_filename)
    summary_filename  = get_unique_filename(summary_filename)
    config_filename   = get_unique_filename(config_filename)

    # Save the "detailed" results
    save_output_data(all_approach_results, config, custom_filename=detailed_filename)
    print(f"Detailed data saved to {detailed_filename}")

    # Build and save a "summarized" version of the results
    summary_results = create_summary_data(all_approach_results)
    save_output_data(summary_results, config, custom_filename=summary_filename)
    print(f"Summary data saved to {summary_filename}")

    # Save the configuration to a JSON file
    config_entry = {"config": vars(config)}
    save_output_data([config_entry], config, custom_filename=config_filename)
    print(f"Config data saved to {config_filename}")


if __name__ == "__main__":
    from llm_wrapper import LLMWrapper

    try:
        setup_logging(log_level='INFO')
    except ValueError as e:
        print(f"Invalid log level: {e}")

    # **Initialize Module-Specific Logger**
    logger = logging.getLogger(__name__)

    generate_answers()

