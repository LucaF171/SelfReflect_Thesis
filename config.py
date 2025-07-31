# config.py
import json
import yaml  # Import PyYAML for YAML parsing # type: ignore
from dataclasses import dataclass
import logging
import argparse

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    approach: str = "all"
    num_answers: int = 10
    num_questions: int = 10
    num_answers_for_eval: int = 10
    start_question: int = 0
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "google-research-datasets/natural_questions"
    generate_answers: bool = False
    evaluation_dataset: str = "manual_summaries.json"
    result_folder: str = "results/"
    output_file: str = "output_data"
    custom_output_file: bool = False
    seed: int = 42  # Added seed parameter
    system_prompt: str = "You are a friendly chatbot who always responds with short and concise answers.\n"
    log_level: str = "INFO"
    skip_edgecase_summaries: bool = False
    provide_question: bool = False
    post_hoc_temperature: float = 1.

    @staticmethod
    def from_file(config_file: str) -> 'ExperimentConfig':
        """Load configuration from a YAML file."""
        #logger.debug(f"Loading configuration from {config_file}")
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    config_data = json.load(f)
                else:
                    raise ValueError("Unsupported configuration file format. Use .yaml or .json")
            #logger.debug(f"Configuration data loaded: {config_data}")
            return ExperimentConfig(**config_data)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_file} not found.")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise

    def update_from_args(self, args: argparse.Namespace):
        """
        Override configuration parameters with command-line arguments if they are provided.
        Only update if the argument is not None.
        """
        for arg, arg_value in args.__dict__.items():
            if arg_value is not None:
                setattr(self, arg, arg_value)
                logger.info(f"Overridden {arg} to: {arg_value}")

    def __str__(self):
        """Return a pretty-printed string representation of the configuration."""
        return json.dumps(self.__dict__, indent=4)
