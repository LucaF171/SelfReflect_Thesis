import argparse
import logging
import os
from tqdm import tqdm

from utils import (
    set_seed,
    save_all_results 
)
from cacher import CachedPromptModel
from logging_config import setup_logging
from config import ExperimentConfig
from llm_wrapper import LLMWrapper
from evaluator import Evaluator
from qa_manager import QAManager

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Number of questions to ask from the dataset/questions_and_answers file (default: 10)",
    )
    parser.add_argument(
        "--start_question",
        type=int,
        default=0,
        help="Index of the first question to process (default: 0)",
    )
    parser.add_argument(
        "--num_answers",
        type=int,
        default=50,
        help="Number of generated answers per question (default=50)",
    )
    parser.add_argument(
        "--num_answers_for_eval",
        type=int,
        default=10,
        help=("How many answers to evaluate the masked-out-task of our metric on. "
              "Default 10, to make it faster to compute. But set to a value >= num_answers to eval on all."),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Name of the LLaMA model to use (meta-llama/Llama-3.1-8B-Instruct|Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="google-research-datasets/natural_questions",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--generate_answers",
        type=bool,
        default=False,
        help="Whether to generate answers for the questions (default=False)",
    )
    parser.add_argument(
        "--evaluation_dataset",
        type=str,
        default="manual_summaries.json",
        help="File to import the questions, generated answers and curated summaries",
    )
    parser.add_argument(
        "--custom_output_file",
        type=bool,
        default=False,
        help="Whether to use a custom output file name (default=False)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results",
        help="File to save the output data (without .json extension)",
    )
    parser.add_argument(
        "--cache_device",
        type=str,
        default="auto",
        help="Which device to put the cache on. Auto uses the same as the model (probably GPU)."
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="all",
        help="Which approach to use (1, 2, 3, 4, all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default=None,
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--skip_edgecase_summaries",
        action="store_true",
        help="Not add the edge cases of empty summary and question as summary.",
    )
    parser.add_argument(
        "--provide_question",
        action="store_true",
        help="Whether to add the question to the masked out task. Default: No.",
    )
    parser.add_argument(
        "--post_hoc_temperature",
        type=float,
        default="1.",
        help="Temperature to scale the logits by to hopefully catch synonyms better",
    )

    args = parser.parse_args()
    return args


def main():
    # 1) Parse command-line arguments
    args = parse_args()

    # 2) Setup logging
    try:
        setup_logging(log_level=args.log_level)
    except ValueError as e:
        print(f"Error setting up logging level: {e}")
        return
    logger = logging.getLogger(__name__)

    # 3) Load the configuration
    try:
        config = ExperimentConfig.from_file(args.config)
        logger.info("Loaded configuration: %s", config)
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        return

    # Override config parameters with command-line args if provided
    config.update_from_args(args)
    logger.info("Current config: %s", config)

    # 4) Set random seed for reproducibility
    set_seed(config.seed)

    # 5) Initialize the model
    try:
        llm_wrapper = LLMWrapper(model_name=config.model_name)
        llm_wrapper.model = CachedPromptModel(llm_wrapper.model, device=args.cache_device)
        logger.info("Initialized LLMWrapper with model: %s", config.model_name)
    except Exception as e:
        logger.error("Failed to initialize LLaMA wrapper: %s", e)
        return

    # 6) Load the QA dataset (questions + answers + optional summaries)
    try:
        qa_manager = QAManager(os.path.join(config.evaluation_dataset))
        if not qa_manager.questions:
            logger.error("No questions loaded. Exiting...")
            return
    except Exception as e:
        logger.error("Failed to load QA data: %s", e)
        return

    # 7) Initialize the evaluator
    evaluator = Evaluator(llm_wrapper=llm_wrapper, config=config, qa_manager=qa_manager)

    # Ensure num_questions does not exceed available questions
    available_questions = len(qa_manager.questions)
    if args.num_questions > available_questions:
        logger.warning(
            f"The evaluation_dataset file only has {available_questions} questions, "
            f"but you asked for {args.num_questions}. Adjusting to {available_questions}."
        )
        config.num_questions = available_questions
    else:
        config.num_questions = args.num_questions

    logger.info("Processing %s questions.", config.num_questions)
    all_approach_results = {}

    # 8) Evaluate each question
    logger.info("Starting to iterate over the questions")
    for question_idx in tqdm(range(config.start_question, config.start_question + config.num_questions), desc="Processing Question"):
        try:
            print("Processing Question %s", question_idx + 1)
            collected_data = evaluator.process_question(question_idx=question_idx)
        except Exception as e:
            logger.error("Failed to process Question %s: %s", question_idx, e)
            continue

        if collected_data:
            all_approach_results[question_idx] = collected_data
        else:
            logger.warning(f"No data collected for Question {question_idx + 1}.")

    logger.info("Processing questions completed. Saving results...")
    # 9) Save the output data
    save_all_results(all_approach_results, config=config, llm_wrapper=llm_wrapper)


if __name__ == "__main__":
    main()