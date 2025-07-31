import logging
from llm_wrapper import LLMWrapper
from qa_manager import QAManager
from typing import Any, Dict
from config import ExperimentConfig
from metrics import MetricsCalculator



class Evaluator:
    """
    Encapsulates the logic for evaluating summaries/answers using various approaches.
    """

    def __init__(self, llm_wrapper: LLMWrapper, config: ExperimentConfig, qa_manager: QAManager):
        self.llm_wrapper = llm_wrapper
        self.config = config
        self.qa_manager = qa_manager
        self.logger = logging.getLogger(__name__)
        self.metrics_calculator = MetricsCalculator(config=self.config, llm_wrapper=self.llm_wrapper)

    def process_question(self, question_idx: int) -> dict:
        """
        Retrieves a question, its answers, and summaries from QA Manager and computes metrics for each approach.
        """
        question = self.qa_manager.get_question_text(question_idx)
        answers = self.qa_manager.get_answers(question_idx)
        summaries = self.qa_manager.get_summaries(question_idx)
        self.logger.info("SUMMARIES: %s", summaries)

        if not question or not answers or not summaries:
            self.logger.error("Missing data for question index %s. Skipping...", question_idx)
            return {}

        answers = answers[:self.config.num_answers]

        metrics: Dict[str, Dict[str, Any]] = {
            "approach_1": {},
            "approach_2": {},
            "approach_3": {},
            "approach_4": {},
        }
        # Precompute for approach 2 if needed
        precomputed_2 = None
        if self.config.approach in ["all", "2"]:
            self.logger.info("APPROACH 2: Precomputing masked out infilling with answers.")
            self.llm_wrapper.model.reset_cache()
            """ precomputed_2 = self.metrics_calculator.precompute_masked_out_infilling_with_answers(
                question=question,
                answers=answers) """

        # Extend summaries with edge cases if not skipped
        extended_summaries = dict(summaries)
        # Dont need them for the main eval
        """ if not self.config.skip_edgecase_summaries:
            extended_summaries["question_as_summary"] = question
            extended_summaries["no_question_as_summary"] = ""
        """
        # Mapping approaches to functions
        approach_functions = {
            "1": self._compute_approach_1,
            "2": lambda q, a, s, label: self._compute_approach_2(q, a, s, label, precomputed_2),
            "3": self._compute_approach_3,
            "4": self._compute_approach_4,
        }

        if self.config.approach == "all":
            approaches_to_run = list(approach_functions.keys())
        else:
            approaches_to_run = [self.config.approach]

        for summary_label, summary in extended_summaries.items():
            self.logger.info("Processing '%s' summary", summary_label)
            for approach in approaches_to_run:
                self.logger.info("Running Approach %s for '%s'", approach, summary_label)
                result = approach_functions[approach](question, answers, summary, summary_label)
                metrics[f"approach_{approach}"][summary_label] = result

        return {
            "prompt": question,
            "answers": answers,
            "summaries": extended_summaries,
            "metrics": metrics
        }

    def _compute_approach_1(self, question: str, answers: list, summary: str, summary_label: str) -> dict:
        print(f"Computing Approach 1")
        self.logger.info("Computing approach 1")
        self.llm_wrapper.model.reset_cache()
        try:
            recall = self.metrics_calculator.calculate_recall(answers=answers, summary=summary)
            #precision = self.metrics_calculator.calculate_precision(answers=answers, summary=summary)
            precision = self.metrics_calculator.calculate_precision(answers=answers, summary=summary)
            self.logger.info("APPROACH 1 RESULTS for '%s': Recall=%s, Precision=%s", summary_label, recall, precision)

            f_score = self.metrics_calculator.calculate_f_score(recall=recall, precision=precision)
            return {"recall": recall, "precision": precision, "F_score": f_score}
        except Exception as e:
            self.logger.error("Failed to compute Approach 1 for '%s': %s", summary_label, e)
            return {}

    def _compute_approach_2(self, question: str, answers: list, summary: str, summary_label: str, precomputed_2) -> dict:
        # Computes PMI and optionally masked out infilling for Approach 2
        print("Computing approach 2")
        self.llm_wrapper.model.reset_cache()
        try:
            pmi_val = self.metrics_calculator.calculate_pmi(
                question=question,
                answers=answers,
                summary=summary
            )
        except Exception as e:
            self.logger.error("Failed PMI for summary '%s': %s", summary_label, e)
            pmi_val = 0.0

        """ if precomputed_2 is not None:
            try:
                self.llm_wrapper.model.reset_cache()
                infilling_results = self.metrics_calculator.calculate_masked_out_infilling_for_summary(
                    question=question,
                    summary=summary,
                    summary_label=summary_label,
                    precomputed_with_answers=precomputed_2
                )
            except Exception as e:
                self.logger.error("Failed masked out infilling for '%s': %s", summary_label, e)
                infilling_results = {}
        else:
            infilling_results = {} """
        
        self.logger.info("APPROACH 2 RESULTS for '%s': PMI=%s", summary_label, pmi_val)
        #return {"PMI": pmi_val, "masked_out_infilling": infilling_results}
        return {"PMI": pmi_val}

    def _compute_approach_3(self, question: str, answers: list, summary: str, summary_label: str) -> dict:
        # Computes Generative Distribution Discrepancy
        print("Computing approach 3")
        self.llm_wrapper.model.reset_cache()
        try:
            LH_3_0, LH_3_1, LH_3_2, LH_3_3, Score_3_1, Score_3_2, Score_3_3 = self.metrics_calculator.calculate_approach_3(
                question=question,
                answers=answers,
                summary=summary
            )
            result = {
                "LH_3_0 (x from P(x|s,q))": LH_3_0,
                "LH_3_1 (x from P(x|A,q))": LH_3_1,
                "LH_3_2 (x from P(x|(q,a_i)))": LH_3_2,
                "LH_3_3 (x from product of P(x|a_i,q))": LH_3_3,
                "Score_3_1": Score_3_1,
                "Score_3_2": Score_3_2,
                "Score_3_3": Score_3_3,
            }
            self.logger.info("APPROACH 3 RESULTS for '%s': %s", summary_label, result)
            return result
        except Exception as e:
            self.logger.error("Failed to compute Approach 3 for '%s': %s", summary_label, e)
            return {}

    def _compute_approach_4(self, question: str, answers: list, summary: str, summary_label: str) -> dict:
        # Computes Log Posterior Coverage Ratio
        print("Computing approach 4")
        self.llm_wrapper.model.reset_cache()

        try:
            # The metrics_calculator now returns the final scores directly
            score_4_1, score_4_2, score_4_3 = self.metrics_calculator.calculate_approach_4(
                question=question,
                answers=answers,
                summary=summary
            )
            
            result = {
                "Score_4_1": score_4_1,
                "Score_4_2": score_4_2,
                "Score_4_3": score_4_3,
            }
            self.logger.info("APPROACH 4 RESULTS for '%s': %s", summary_label, result)
            return result
        except Exception as e:
            self.logger.error("Failed to compute Approach 4 for '%s': %s", summary_label, e)
            return {}
