import json
import logging
from typing import Dict, List

log = logging.getLogger(__name__)


class QASummary:
    def __init__(self, question_text: str, answers: List[str], summaries: Dict[str, str]):
        """
        Represents a single QA entry with its summaries.

        :param question_text: The text of the question.
        :param answers: A list of possible answers.
        :param summaries: A dictionary of summaries categorized as 'good', 'mid', and 'bad'.
        """
        self.question_text = question_text
        self.answers = answers
        self.summaries = summaries


class QAManager:
    def __init__(self, json_file: str):
        """
        Initializes the QAManager by loading data from a JSON file.
        """
        self.questions: Dict[str, QASummary] = {}
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            for key, value in self.data.items():
                # Provide fallback for missing 'summaries' instead of raising KeyError
                summaries_data = value.get('summaries', {})  # Use {} as default if not present
                
                self.questions[key] = QASummary(
                    question_text=value['question_text'],
                    answers=value['answers'],
                    summaries=summaries_data
                )
            log.info(f"Successfully loaded {len(self.questions)} questions from {json_file}.")
        
        except FileNotFoundError:
            log.error(f"The file {json_file} was not found.")
        except json.JSONDecodeError as e:
            log.error(f"Error decoding JSON from {json_file}: {e}")
        except KeyError as e:
            # If you still want question_text or answers to be required, you can handle it similarly:
            log.error(f"Missing required key in JSON data: {e}")

    def get_question_text(self, index: int) -> str:
        """
        Retrieves the question text for a given index.

        :param index: The index of the question (integer).
        :return: Question text as a string.
        """
        key = str(index)
        question = self.questions.get(key)
        if question:
            return question.question_text
        else:
            log.warning(f"No question found at index {index}.")
            return ""

    def get_answers(self, index: int) -> List[str]:
        """
        Retrieves the list of answers for a given index.

        :param index: The index of the question (integer).
        :return: List of answers.
        """
        key = str(index)
        question = self.questions.get(key)
        if question:
            return question.answers
        else:
            log.warning(f"No answers found at index {index}.")
            return []

    def get_summaries(self, index: int) -> Dict[str, str]:
        """
        Retrieves the summaries for a given index.

        :param index: The index of the question (integer).
        :return: Dictionary of summaries.
        """
        key = str(index)
        question = self.questions.get(key)
        if question:
            return question.summaries
        else:
            log.warning(f"No summaries found at index {index}.")
            return {}

    def list_all_indices(self) -> List[str]:
        """
        Lists all available question indices.

        :return: List of indices.
        """
        return list(self.questions.keys())
