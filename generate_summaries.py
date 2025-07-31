import argparse
import logging

from config import ExperimentConfig
from llm_wrapper import LLMWrapper
from summary_generator import SummaryGenerator
from qa_manager import QAManager

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_questions",
        type=int,
        default=100,
        help="Number of questions to ask from the dataset/questions_and_answers file (default: 10)",
    )
    parser.add_argument(
        "--summary_strategy",
        type=str,
        default="greedy",
        choices=["greedy", "basic", "CoT", "beam_search", "answer_dist"],
        help="Which summary strategy to use."
    )
    parser.add_argument(
        "--questions_and_answers_file",
        type=str,
        default="questions_and_answers.json",
        help="Path to the file containing questions and answers."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Which model to load in the LLM Wrapper."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file for the experiment."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Setup logging
    logging.basicConfig(level=args.log_level)

    # 2) Load configuration
    config = ExperimentConfig.from_file(args.config)
    # Override config parameters with command-line args if provided
    config.update_from_args(args)
    logging.info("Loaded ExperimentConfig: %s", config)
    
    # 3) Override model_name in the config (if desired)
    config.model_name = args.model_name

    # 4) Initialize the model + wrapper
    llm_wrapper = LLMWrapper(model_name=config.model_name)

    # 5) Initialize QAManager
    qa_manager = QAManager(json_file=args.questions_and_answers_file)
    
    # 6) Initialize the SummaryGenerator
    summary_generator = SummaryGenerator(llm_wrapper=llm_wrapper, config=config, qa_manager=qa_manager)

    # 7) Call generate_all_summaries
    if args.summary_strategy == "all":
        for strategy in ["greedy", "basic", "CoT", "beam_search", "answer_dist"]:
            summary_generator.generate_summaries(
                summary_strategy=strategy
            )
    else:
        summary_generator.generate_summaries(
            summary_strategy=args.summary_strategy,
            questions_and_answers_file=args.questions_and_answers_file
        )

if __name__ == "__main__":
    main()
