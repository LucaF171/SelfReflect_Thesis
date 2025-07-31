import logging
import json
import os
from typing import List, Optional

from llm_wrapper import LLMWrapper
from qa_manager import QAManager
from config import ExperimentConfig
from utils import stitch_prompt
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SummaryGenerator:
    """
    Generates various summaries for questions and answers managed by QAManager.
    """

    def __init__(
        self,
        llm_wrapper: LLMWrapper,
        config: ExperimentConfig,
        qa_manager: QAManager
    ):
        """
        :param llm_wrapper: Your LLMWrapper instance (handles model calls).
        :param config: Experiment configuration (system prompt, etc.).
        :param qa_manager: QAManager instance for retrieving questions & answers.
        """
        self.llm_wrapper = llm_wrapper
        self.config = config
        self.qa_manager = qa_manager

    # -------------------------------------------------------------------------
    # INDIVIDUAL SUMMARY METHODS
    # -------------------------------------------------------------------------
        
    # Bottom Baseline: Don't ask the model to summarize, just take the greedy answer to the question as summary
    def greedy_summary(self, question: str) -> str:
        """
        1) Greedy approach: simply produce a direct answer without special instructions.
        """
        user_prompt = f"{question}"

        messages = stitch_prompt(
            self.llm_wrapper.model.model.name_or_path,
            system=self.config.system_prompt,
            user=user_prompt
        )
        prompt = self.llm_wrapper.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Use your LLMWrapper to generate the text (no sampling, so it's "greedy")
        summary = self.llm_wrapper.generate(
            prompt_text=prompt,
            do_sample=False
        )

        logger.info(f"[GREEDY] Generated summary: {summary}")
        return summary.strip()

    # Ask the model to "give a summary of answers" without providing the answers or CoT instructions
    def basic_summary(self, question: str) -> str:
        """
        2) Basic approach: ask for a concise summary of possible answers, no CoT.
        """
        user_prompt = (
            f"{question}"
            f"Please provide a short and concise summary of possible answers.\n"
        )

        messages = stitch_prompt(
            self.llm_wrapper.model.model.name_or_path,
            system=self.config.system_prompt,
            user=user_prompt
        )
        prompt = self.llm_wrapper.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        summary = self.llm_wrapper.generate(
            prompt_text=prompt,
            do_sample=False
        )

        logger.info(f"[BASIC] Summary: {summary}")
        return summary.strip()
    
    # Ask the model to CoT possible answers and summarize them
    def CoT_summary(self, question: str) -> str:
        """
        3) CoT approach: Provide minimal chain-of-thought instructions.
        """
        user_prompt = (
            f"{question}"
            f"Let's analyze step by step to see the important points:\n"
            f"1) Identify overlapping or conflicting points.\n"
            f"2) Determine the key information relevant to the question.\n"
            f"3) Summarize concisely.\n"
            f"Only return the Final Summary.\n"
            f"Final Summary:\n"
        )

        messages = stitch_prompt(
            self.llm_wrapper.model.model.name_or_path,
            system=self.config.system_prompt,
            user=user_prompt
        )
        prompt = self.llm_wrapper.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        summary = self.llm_wrapper.generate(
            prompt_text=prompt,
            do_sample=False
        )

        logger.info(f"[CoT] Summary: {summary}")
        return summary.strip()
    
    # Access the beam search answers, provide them back to the model and ask it to summarize it
    def beam_search_summary(self, question: str) -> str:

        # TODO set for now, can be adapted
        num_beams = 50
        
        # Step 1: Generate multiple candidate answers to the question using beam search
        # We will feed the same question to get multiple beams:
        
        prompt_for_beams = (
            f"{question}"
        )

        messages = stitch_prompt(self.llm_wrapper.model.model.name_or_path, system=self.config.system_prompt, user=prompt_for_beams)
        prompt = self.llm_wrapper.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        
        beam_outputs = []
        try:
            beam_candidates = self.llm_wrapper.generate(
                prompt_text=prompt,
                early_stopping=True,
                num_beams=num_beams,
                num_return_sequences=num_beams
            )
            if isinstance(beam_candidates, list):
                beam_outputs = beam_candidates
            else:
                beam_outputs = [beam_candidates]
        except Exception as e:
            logger.error(f"[BEAM SEARCH] Error generating multiple beams: {e}")
            # Fallback: single output
            single_output = self.llm_wrapper.generate(prompt_text=prompt_for_beams)
            beam_outputs = [single_output]

        # Summarize the multiple beams
            

        # print beams to console, line after line
        print("Beams:")
        for i, beam_output in enumerate(beam_outputs):
            print(f"{i+1}: {beam_output}")
        # Note that the prompt currently has the exact same structure as the answer distribution prompt
        beams_str = "\n".join([f"- {b}" for b in beam_outputs])
        

        prompt_summary = (
            f"{question}"
            f"Answers:\n"
            f"{beams_str}\n"
            f"Please produce a concise, single-paragraph summary of all relevant information from the answers, without referring to the format of the answers or explicitly mentioning differences, outliers, or headings.\n"
        )


        messages = stitch_prompt(
            self.llm_wrapper.model.model.name_or_path,
            system=self.config.system_prompt,
            user=prompt_summary
        )
        prompt = self.llm_wrapper.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        summary = self.llm_wrapper.generate(
            prompt_text=prompt,
            do_sample=False
        )

        logger.info(f"[BEAM SEARCH] Final summary: {summary}")
        return summary.strip()
    
    # Top Baseline: Provide the answer distribution and ask the model to summarize it
    def answer_distribution_summary(self, question: str, answers: List[str]) -> str:
        """
        5) Summarize an answer distribution or list.
        """
        answers_str = "\n".join([f"Answer: {a}" for a in answers])
        user_prompt = (
            f"{question}"
            f"Answers:\n"
            f"{answers_str}\n"
            f"Please summarize these answers.\n"
        )

        messages = stitch_prompt(
            self.llm_wrapper.model.model.name_or_path,
            system=self.config.system_prompt,
            user=user_prompt
        )
        prompt = self.llm_wrapper.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        summary = self.llm_wrapper.generate(
            prompt_text=prompt,
            do_sample=False
        )

        logger.info(f"[ANSWER DIST] Summary: {summary}")
        return summary.strip()

    # -------------------------------------------------------------------------
    # METHOD FOR GENERATING SUMMARIES FOR ALL QUESTIONS
    # -------------------------------------------------------------------------

    def generate_summaries(
        self,
        summary_strategy: str,
        questions_and_answers_file: str
    ):
        """
        Iterates over all questions in the QAManager, applies the chosen summary strategy,
        and optionally saves results to JSON.

        :param summary_strategy: One of ["greedy", "basic", "CoT", "beam_search", "answer_dist"].
        :param output_file: Path to save the JSON file. If None, defaults to a file in 'generated_summaries'.
        :param num_beams: Number of beams to use for beam_search_summary.
        :return: A list of dicts containing question, answers, and the generated summary.
        """
        logger.info(f"Generating summaries using strategy='{summary_strategy}' for all questions in QAManager.")
        # fallback path
        base_dir = "generated_summaries"
        os.makedirs(base_dir, exist_ok=True)
        model_name_for_file = getattr(self.llm_wrapper, "model_name", "model")
        # only keep the model name, remove everything before the last '/'
        questions_and_answers_file = questions_and_answers_file.split(".")[0]
        output_file = os.path.join(
            base_dir, f"{questions_and_answers_file}_{summary_strategy}.json"
        )
        logger.info(f"Output file will be: {output_file}")

        results: Dict[int, Dict[str, Any]] = {}
        all_indices = self.qa_manager.list_all_indices()
        # just take the first "num_questions" questions
        logger.info(f"Total number of questions to generate summaries for: {self.config.num_questions}")
        all_indices = all_indices[:self.config.num_questions]

        if not all_indices:
            logger.warning("No question indices found in QAManager. Exiting.")
            return results

        for key_str in all_indices:
            idx = int(key_str)
            question_text = self.qa_manager.get_question_text(idx)
            answers = self.qa_manager.get_answers(idx)
            if not question_text or not answers:
                logger.warning(f"Skipping Q#{idx}: no question text or no answers.")
                continue

            if question_text is not None:
                if question_text.endswith("?"):
                    question_text += "\n"
                elif not question_text.endswith("?\n"):
                    question_text += "?\n"

                if not question_text.startswith("Question: "):
                    question_text = "Question: " + question_text
            else:
                question_text = ""

            logger.info(f"Generating summary for Q#{idx}: {question_text}")

            # Switch on strategy
            if summary_strategy == "greedy":
                summary = self.greedy_summary(question_text)
            elif summary_strategy == "basic":
                summary = self.basic_summary(question_text)
            elif summary_strategy == "CoT":
                summary = self.CoT_summary(question_text)
            elif summary_strategy == "beam_search":
                summary = self.beam_search_summary(question_text)
            elif summary_strategy == "answer_dist":
                summary = self.answer_distribution_summary(question_text, answers)
            else:
                logger.error(f"Unknown summary strategy: {summary_strategy}. Skipping.")
                continue

            # idx corresponds to the question number in the questions_and_answers.json file, its not a counter
            results[idx] = {
                "question_text": question_text,
                "answers": answers,
                "summary_strategy": summary_strategy,
                "summaries": {
                    "summary": summary
                }
            }

        # Save to JSON
        logger.info(f"Writing summaries to {output_file}")
        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, indent=4, ensure_ascii=False)

        logger.info(f"Done generating summaries using '{summary_strategy}'.")
        return results
