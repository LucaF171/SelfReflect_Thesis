import logging
from typing import List, Dict, Tuple
import numpy as np
import torch
from typing import Optional

from functools import lru_cache
from llm_wrapper import LLMWrapper

from utils import pad_second_tensor, stitch_prompt, TFIDF
from config import ExperimentConfig

log = logging.getLogger(__name__)


class MetricsCalculator:

    def __init__(self, llm_wrapper: LLMWrapper, config: ExperimentConfig):
        self.llm_wrapper = llm_wrapper
        self.config = config
            
    def calculate_recall(self, answers: List[str], summary: str) -> float:
        """
        Calculates the recall metric by checking all answers in a single batch.

        :param answers: List of answers.
        :param summary: The summary text.
        :return: Recall as a float.
        """
        if not answers:
            return 0.0

        # Create a numbered list of answers
        answer_list_str = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])

        # Create a prompt to ask the LLM to check all answers at once.
        prompt = (
            f"Summary: {summary}\n"
            f"Answers:\n{answer_list_str}\n"
            f"Which of the numbered answers in the list are directly supported or mentioned in the summary? "
            f"Please respond with a comma-separated list of the numbers of the covered answers (e.g., '1, 3, 4')."
        )

        system_content = self.config.system_prompt
        messages = stitch_prompt(self.llm_wrapper.model.model.name_or_path, system=system_content, user=prompt)
        chat_prompt_str = self.llm_wrapper.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        response_text = self.llm_wrapper.generate(prompt_text=chat_prompt_str)

        covered_count = 0
        try:
            # Parse the response
            if response_text:
                covered_indices = [int(i.strip()) for i in response_text.split(',') if i.strip().isdigit()]
                # Filter out-of-range indices and count unique answers
                valid_indices = {idx - 1 for idx in covered_indices if 0 < idx <= len(answers)}
                covered_count = len(valid_indices)
        except (ValueError, AttributeError) as e:
            log.error(f"Could not parse LLM response for recall: '{response_text}'. Error: {e}")
            covered_count = 0

        recall = covered_count / len(answers) if answers else 0
        return recall


    def calculate_precision(self, answers: List[str], summary: str) -> float:
        """
        Calculates the precision metric in a more efficient way.

        :param answers: List of answers.
        :param summary: The summary text.
        :return: Precision as a float.
        """
        log.info("Calculating precision")
        
        # This is still one LLM call, which is acceptable.
        summary_elements = self.llm_wrapper.extract_elements_from_summary(summary=summary)
        log.debug("Summary elements: %s", summary_elements)

        if not summary_elements:
            return 1.0 # If summary is empty, it makes no unsupported claims.

        # Create a numbered list of summary elements
        element_list_str = "\n".join([f"{i+1}. {elem}" for i, elem in enumerate(summary_elements)])
        answers_str = "\n".join(f"- {ans}" for ans in answers)

        # Create a prompt to ask the LLM to check all elements at once.
        prompt = (
            f"List of claims or elements:{element_list_str}\n\n"
            f"List of reference answers:{answers_str}\n\n"
            "Which of the numbered elements in the list are supported by at least one of the provided reference answers? "
            "An element is supported if the answers provide evidence for it. "
            "Please respond with a comma-separated list of the numbers of the supported elements (e.g., '1, 3, 4')."
        )

        system_content = self.config.system_prompt
        messages = stitch_prompt(self.llm_wrapper.model.model.name_or_path, system=system_content, user=prompt)
        chat_prompt_str = self.llm_wrapper.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        response_text = self.llm_wrapper.generate(prompt_text=chat_prompt_str)

        covered_count = 0
        try:
            # Parse the response
            if response_text:
                covered_indices = [int(i.strip()) for i in response_text.split(',') if i.strip().isdigit()]
                # Filter out-of-range indices and count unique elements
                valid_indices = {idx - 1 for idx in covered_indices if 0 < idx <= len(summary_elements)}
                covered_count = len(valid_indices)
        except (ValueError, AttributeError) as e:
            log.error(f"Could not parse LLM response for precision: '{response_text}'. Error: {e}")
            covered_count = 0

        precision = covered_count / len(summary_elements) if summary_elements else 1.0
        log.debug("Precision calculated: %s", precision)
        return precision

    
    def calculate_f_score(self, recall: float, precision: float) -> float:
        """
        Calculates the F-score based on recall and precision.

        :param recall: Recall value.
        :param precision: Precision value.
        :return: F-score as a float.
        """
        log.info("Calculating F-score")
        if recall + precision == 0:
            return 0.0
        # NOTE: Because we want to minimize the distance, we use 1-F-Score.
        # This way the value is also bounded between 0 and 1
        f_score = 1 - (2 * (precision * recall) / (precision + recall))
        log.debug("F-score calculated: %s", f_score)
        return f_score


    def calculate_pmi(self,
        question: str,
        answers: List[str],
        summary: str
    ) -> float:
        """
        Calculate the Pointwise Mutual Information (PMI) between the question and summary.

        :param question: The question string.
        :param answers: List of answer strings.
        :param summary: The summary string.
        :return: Average PMI as a float.
        """
        log.info("Calculating PMI for question: %s and summary: %s", question, summary)
        pmis = []

        for answer in answers[:min(len(answers), self.config.num_answers_for_eval)]:
            #log.debug("Calculating PMI for answer: %s", answer)

            prompt_with_summary = self.create_user_prompt(question=question, summary=summary, approach="2_1")
            prompt_without_summary = self.create_user_prompt(question=question, approach="2_1")
            """  print("---------")
            print("prompt_with_summary:", prompt_with_summary)
            print("prompt_without_summary:", prompt_without_summary)
            print("target:", answer)
            """
            try:
                (
                    sum_w_sum,
                    probs_w_sum,
                    tokens_w_sum,
                    dist_w_sum
                ) = self.compute_target_likelihood(
                    prompt=prompt_with_summary,
                    target=answer
                )
            except Exception as e:
                log.error("Error computing likelihood with summary for PMI: %s", e)
                continue

            try:
                (
                    sum_wo_sum,
                    probs_wo_sum,
                    tokens_wo_sum,
                    dist_wo_sum
                ) = self.compute_target_likelihood(
                    prompt=prompt_without_summary,
                    target=answer
                )
            except Exception as e:
                log.error("Error computing likelihood without summary for PMI: %s", e)
                continue

            if sum_w_sum == float('-inf') or sum_wo_sum == float('-inf'):
                log.warning("Skipping PMI calculation for answer '%s' due to -inf log probs.", answer)
                continue

            #print("sum_w_sum:", sum_w_sum, "sum_wo_sum:", sum_wo_sum)
            # PMI = log P(answer | summary, question) - log P(answer | question)
            pmi_val = - (sum_w_sum - sum_wo_sum) # changed this such that the distance function shall be minimized
            #print("PMI value for answer:", pmi_val)
            pmis.append(pmi_val)

        average_pmi = sum(pmis) / len(pmis) if pmis else 0
        log.info("All PMIs: %s", pmis)
        log.info("Average PMI: %s", average_pmi)
        return average_pmi


    # -----------------------------------------------------------------------------
    # 1) Precompute masked-out infilling "with answers" (once per question)
    # -----------------------------------------------------------------------------
    def precompute_masked_out_infilling_with_answers(
        self,
        question: str,
        answers: List[str]) -> Dict:
        """
        Precompute masked-out infilling *only for the 'prompt_with_answers'* (and optionally 
        'prompt_with_question' if you want) once per question. We will reuse this for every summary
        by calling 'calculate_masked_out_infilling_for_summary' later.

        Returns a dictionary mapping (answer_idx, word_idx) -> relevant log-probs, token distributions, etc.
        """
        log.info("[precompute_masked_out_infilling_with_answers] Starting...")

        precomputed_data = {}
        tf_idf = TFIDF()

        for ans_idx, answer in enumerate(answers):
            if ans_idx >= self.config.num_answers_for_eval:
                break

            answer_words = answer.split()
            if not answer_words:
                log.warning("Skipping empty answer at idx %d", ans_idx)
                continue

            for word_idx, masked_word in enumerate(answer_words):
                # Skip stopwords
                if tf_idf.is_stopword([masked_word])[0]:
                    continue

                # Create a masked version of the answer
                temp_words = answer_words[:]
                temp_words[word_idx] = "_"
                masked_out_answer = " ".join(temp_words)

                # 1) Prompt with answers
                prompt_with_answers = self.create_user_prompt(
                    answers=answers,
                    question=question if self.config.provide_question else None,
                    masked_out_answer=masked_out_answer,
                    approach="2_a"
                )

                # Optionally also do a "prompt_with_question" if you want to skip it for some 
                # reason. For demonstration, let's just store "with answers" here.
                # If you do want 'with question' once as well, you can do it similarly.

                try:
                    (
                        sum_w_ans,
                        probs_w_ans,
                        tokens_w_ans,
                        dist_w_ans
                    ) = self.compute_target_likelihood(
                        prompt=prompt_with_answers,
                        target=masked_word
                    )
                except Exception as e:
                    log.error("Error computing likelihood with answers: %s", e)
                    continue

                key = (ans_idx, word_idx)
                precomputed_data[key] = {
                    "answer_text": answer,
                    "masked_word": masked_word,
                    "masked_out_answer": masked_out_answer,
                    "sum_w_ans": sum_w_ans,
                    "probs_w_ans": probs_w_ans,
                    "tokens_w_ans": tokens_w_ans,
                    "dist_w_ans": dist_w_ans
                }

        log.info("[precompute_masked_out_infilling_with_answers] Done.")
        return precomputed_data


    # -----------------------------------------------------------------------------
    # 2) For each summary, we only compute "with summary" (or "with question", or "without question"),
    #    reusing the precomputed "with answers" data. 
    # -----------------------------------------------------------------------------
    def calculate_masked_out_infilling_for_summary(
        self,
        question: str,
        summary: str,
        summary_label: str,  # optional if you need to differentiate
        precomputed_with_answers: Dict
    ):
        """
        Given the precomputed "with answers" data, compute the masked-out infilling
        metrics for the specified summary. That is, we only compute the 'with summary'
        likelihood, do the log-prob difference, and compute KL.

        :param question: The question string.
        :param answers: List of answer strings.
        :param summary: The summary string.
        :param summary_label: The label for this summary (e.g., "bad", "mid", "good", "question_as_summary").
        :param precomputed_with_answers: The dictionary returned by 'precompute_masked_out_infilling_with_answers(...).'
        :return: Dictionary containing Log_prob and KL metrics (and detailed logging).
        """
        metrics = {
            "Log_prob": 0.0,
            "KL": 0.0,
            "correct_top_token": 0.0
            }
        

        metrics_logging = {"answers": []}
        total_tokens = 0

        def to_immutable(obj):
            """
            Recursively convert obj into something hashable (tuple, frozenset, etc.)
            """
            if isinstance(obj, list):
                # Convert list -> tuple of immutables
                return tuple(to_immutable(x) for x in obj)
            elif isinstance(obj, dict):
                # Convert dict -> sorted tuple of (key, value) with both sides immutable
                return tuple(
                    sorted((k, to_immutable(v)) for k, v in obj.items())
                )
            else:
                return obj  # int, float, str, etc. are already hashable

        def add_answer(answer_text, masked_dict):
            """
            if masked_dict is not None:
                masked_dict_immutable = to_immutable(masked_dict)
                entry = (answer_text, masked_dict_immutable)
            else:
                entry = (answer_text, None)

            if entry not in seen_entries:
                seen_entries.add(entry)
                metrics_logging["answers"].append({
                    "answer_text": answer_text,
                    "masked_words": masked_dict
                })
            """
            metrics_logging["answers"].append({
                "answer_text": answer_text,
                "masked_words": masked_dict
            })

        # This is the meaty bit of the calculation. We cache the outputs of this function, so whenever we
        # have already seen this exact masked_out_answer and this exact masked_word, (so basically when we
        # have an exact same answer twice) we return the recorded logl and kl scores.
        # It was important to thus define this as an inner function, because we can only wrap caches around
        # functions. It is also important that the inner function has no side effects, and only returns stuff.
        # Otherwise, when using the cache, the side effects would not be triggered.
        # Why put the cache here and not, e.g., around compute_target_likelihood ?
        # Because compute_target_likelihood returns tensors, so we'd have to cache very large objects,
        # whereas this function here only returns scalars, or at worst a list.
        # Also, since this is an inner function of calculate_masked_out_infilling_for_summary ,
        # it will be deleted once calculate_masked_out_infilling_for_summary ran through for this question,
        # so that the cache is reset after each question, which we want
        # (across questions, prompt_with_summary would be different)
        @lru_cache(maxsize=8192)
        def calc(masked_out_answer, masked_word):
            # Now compute "with summary" or "with question" for this masked word
            # with question as summary
            if summary_label == "question_as_summary":
                approach="2_q"
                provide_question = True
            # without question as summary, just empty
            elif summary_label == "no_question_as_summary":
                approach="2_nq"
                provide_question = self.config.provide_question
            # with summary as summary
            else:
                approach = "2_s"
                provide_question = self.config.provide_question

            prompt_with_summary = self.create_user_prompt(
                question=question if provide_question else None,
                summary=summary,
                masked_out_answer=masked_out_answer,
                approach=approach
            )


            (
                sum_w_sum,
                probs_w_sum,
                tokens_w_sum,
                dist_w_sum
            ) = self.compute_target_likelihood(
                prompt=prompt_with_summary,
                target=masked_word
            )

            # Compute log_prob_diff, e.g. (with_answers - with_summary)
            log_prob_diff = sum_w_ans - sum_w_sum

            # KL
            kl_val, kl_scores = self.calculate_KL_divergence(
                dist_w_ans,
                dist_w_sum,
                return_per_token=True
            )

            # Top token accuracy
            correct_top_token = (dist_w_ans.argmax(-1) == dist_w_sum.argmax(-1)).sum().float().item()

            return log_prob_diff, kl_val, dist_w_sum.size(0), kl_scores, correct_top_token

        log.info("[calculate_masked_out_infilling_for_summary] summary_label=%s", summary_label)

        # For each (answer_idx, word_idx) in the precomputed data
        for key, precomp in precomputed_with_answers.items():
            ans_idx, w_idx = key
            answer_text = precomp["answer_text"]
            masked_word = precomp["masked_word"]
            masked_out_answer = precomp["masked_out_answer"]

            # "with answers" data from the precomputation
            sum_w_ans = precomp["sum_w_ans"]
            dist_w_ans = precomp["dist_w_ans"]
            tokens_w_ans = precomp["tokens_w_ans"]

            try:
                log_prob_diff, kl_val, n_target_tokens, kl_scores, correct_top_token = calc(masked_out_answer, masked_word)
            except Exception as e:
                log.error("Error computing likelihood with summary for key=%s: %s", key, e)
                continue

            metrics["Log_prob"] += log_prob_diff
            metrics["KL"] += kl_val
            metrics["correct_top_token"] += correct_top_token

            total_tokens += n_target_tokens
            # For detailed logging
            kl_per_token_info = []
            for i, kl_token_val in enumerate(kl_scores):
                token_str = tokens_w_ans[i] if i < len(tokens_w_ans) else "<out_of_range>"
                kl_per_token_info.append({"token": token_str, "kl_value": kl_token_val.item()})

            masked_dict = {
                "answer_id": ans_idx,
                "masked_word": masked_word,
                "log_prob_diff_ans_minus_sum": log_prob_diff,
                "KL_div_sum": kl_val,
                "KL_div_per_token": kl_per_token_info,
                "correct_top_token": correct_top_token
            }
            add_answer(answer_text, masked_dict)

        # Normalize by total number of masked word tokens across all answers
        if total_tokens > 0:
            metrics["Log_prob"] /= total_tokens
            metrics["KL"] /= total_tokens
            metrics["correct_top_token"] /= total_tokens

        log.info("[calculate_masked_out_infilling_for_summary] Done for summary_label=%s", summary_label)
        return {
            **metrics,
            "detailed_logging": metrics_logging
        }


    def calculate_log_prob_diff(self, probs_1: torch.Tensor, probs_2: torch.Tensor) -> float:
        """
        Calculates the sum of (log_probs_1 - log_probs_2) across all tokens in the target.
        :param probs_1: shape (1, target_seq_len)
        :param probs_2: shape (1, target_seq_len)
        :return: float
        """
        log.info("Calculating log-prob difference")

        log_probs_1 = torch.log(probs_1)
        log_probs_2 = torch.log(probs_2)

        sum_of_log_prob_diff = 0.0
        if probs_1.shape == probs_2.shape:
            diff = log_probs_1[0] - log_probs_2[0]
            sum_of_log_prob_diff = diff.sum().cpu().item()
        else:
            log.error("Could not compute log-prob difference; mismatch in token probabilities.")

        return sum_of_log_prob_diff


    def calculate_KL_divergence(
        self,
        probs_1: torch.Tensor,
        probs_2: torch.Tensor,
        return_per_token: bool = False
    ):
        """
        Computes the Kullback-Leibler (KL) divergence between two probability distributions.

        Args:
            probs_1 (torch.Tensor): Probability tensor of shape [target_seq_len, vocab_size].
            probs_2 (torch.Tensor): Probability tensor of shape [target_seq_len, vocab_size].
            return_per_token (bool): If True, return (kl_val, kl_scores_per_token).
                                    Otherwise, return just kl_val as float.

        Returns:
            float or (float, torch.Tensor): The total KL divergence,
            and optionally a per-token 1D tensor if return_per_token=True.
        """
        log.info("Calculating KL divergence")

        if probs_1.shape != probs_2.shape:
            log.error("Cannot compute KL divergence: tensor shapes do not match. P1: %s, P2: %s", probs_1.shape, probs_2.shape)
            if return_per_token:
                return 0.0, torch.tensor([], device=probs_1.device if probs_1.numel() > 0 else 'cpu')
            else:
                return 0.0

        # Check for numerical errors: NaNs or infinite values.
        if torch.isnan(probs_1).any() or torch.isnan(probs_2).any():
            log.error("Input probability tensor contains NaNs.")
        if torch.isinf(probs_1).any() or torch.isinf(probs_2).any():
            log.error("Input probability tensor contains infinite values.")

        # Check that each row of the tensors is normalized (sums to 1 within a tolerance).
        tol = 1e-5
        if not torch.allclose(
                probs_1.sum(dim=-1),
                torch.ones(probs_1.size(0), device=probs_1.device, dtype=probs_1.dtype),
                atol=tol
        ):
            sum_across_vocab = probs_1.sum(dim=-1)
            log.error("Input tensor probs_1 is not correctly normalized. Each row must sum to 1. This row sums to: %s",
                      sum_across_vocab)
        if not torch.allclose(
                probs_2.sum(dim=-1),
                torch.ones(probs_2.size(0), device=probs_2.device, dtype=probs_2.dtype),
                atol=tol
        ):
            sum_across_vocab = probs_2.sum(dim=-1)
            log.error("Input tensor probs_2 is not correctly normalized. Each row must sum to 1. This row sums to: %s",
                      sum_across_vocab)

        # Clamp probabilities to prevent issues with log(0)
        epsilon = 1e-10  # Small value to prevent log(0)
        p = torch.clamp(probs_1, min=epsilon)
        q = torch.clamp(probs_2, min=epsilon)

        # Compute element-wise KL divergence
        kl = p * (torch.log(p) - torch.log(q))  # Shape: [target_seq_len, vocab_size]

        # Sum over the vocabulary dimension to get KL per token, then sum all tokens for total divergence
        kl_scores = kl.sum(dim=-1)  # Shape: [target_seq_len]
        kl_val = kl_scores.sum().item()
        log.info("KL divergence per token: %s", kl_scores)

        if return_per_token:
            return kl_val, kl_scores
        else:
            return kl_val


    def calculate_approach_3(
        self,
        question: str,
        answers: List[str],
        summary: str,
        num_samples: int = 5
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Calculates metrics for Approach 3, using multiple samples to reduce variance.

        :param question: The question string.
        :param answers: List of answer strings.
        :param summary: The summary string.
        :param num_samples: The number of samples to generate for each distribution.
        :return: Tuple containing log likelihoods and scores for sub-approaches.
        """
        system_content = ExperimentConfig.system_prompt

        # --- Helper function to generate samples and calculate average likelihood ---
        def get_avg_likelihood(user_prompt: str) -> float:
            messages = stitch_prompt(self.llm_wrapper.model.model.name_or_path, system=system_content, user=user_prompt)
            chat_prompt_str = self.llm_wrapper.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            samples = self.llm_wrapper.generate(
                prompt_text=chat_prompt_str,
                num_return_sequences=num_samples,
                do_sample=True
            )
            
            if isinstance(samples, str):
                samples = [samples]

            total_log_likelihood = 0.0
            total_tokens = 0

            for sample in samples:
                if not sample or not sample.strip():
                    continue
                
                lh, _, target_tokens, _ = self.compute_target_likelihood(
                    prompt=question,
                    target=sample
                )
                
                total_log_likelihood += lh
                if target_tokens:
                    total_tokens += len(target_tokens)
            
            return total_log_likelihood / total_tokens if total_tokens > 0 else 0.0

        # --- Calculate LH for approaches 3.0, 3.1, 3.2 ---
        prompt_with_summary = self.create_user_prompt(question=question, summary=summary, approach="3_0")
        LH_3_0 = get_avg_likelihood(prompt_with_summary)

        prompt_with_answers_3_1 = self.create_user_prompt(question=question, answers=answers, approach="3_1")
        LH_3_1 = get_avg_likelihood(prompt_with_answers_3_1)

        prompt_with_answers_3_2 = self.create_user_prompt(question=question, answers=answers, approach="3_2")
        LH_3_2 = get_avg_likelihood(prompt_with_answers_3_2)

        # --- Calculate LH for approach 3.3 ---
        output_3_3_total_likelihood = 0.0
        output_3_3_total_tokens = 0
        for answer in answers:
            prompt_with_answers_3_3 = self.create_user_prompt(question=question, answer=answer, approach="3_3")
            messages = stitch_prompt(self.llm_wrapper.model.model.name_or_path, system=system_content, user=prompt_with_answers_3_3)
            chat_prompt_str_3_3 = self.llm_wrapper.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            samples_3_3 = self.llm_wrapper.generate(
                prompt_text=chat_prompt_str_3_3, 
                num_return_sequences=num_samples,
                do_sample=True
            )
            
            if isinstance(samples_3_3, str):
                samples_3_3 = [samples_3_3]
            
            for sample in samples_3_3:
                if not sample or not sample.strip():
                    continue
                
                lh, _, target_tokens, _ = self.compute_target_likelihood(
                    prompt=question,
                    target=sample
                )
                output_3_3_total_likelihood += lh
                if target_tokens:
                    output_3_3_total_tokens += len(target_tokens)

        LH_3_3 = output_3_3_total_likelihood / output_3_3_total_tokens if output_3_3_total_tokens > 0 else 0.0

        log.info("LH_3_0 (avg over %d samples): %s", num_samples, LH_3_0)
        log.info("LH_3_1 (avg over %d samples): %s", num_samples, LH_3_1)
        log.info("LH_3_2 (avg over %d samples): %s", num_samples, LH_3_2)
        log.info("LH_3_3 (avg over %d answers, %d samples each): %s", len(answers), num_samples, LH_3_3)

        # Computing scores
        Score_3_1 = abs(LH_3_0 - LH_3_1)
        Score_3_2 = abs(LH_3_0 - LH_3_2)
        Score_3_3 = abs(LH_3_0 - LH_3_3)

        return LH_3_0, LH_3_1, LH_3_2, LH_3_3, Score_3_1, Score_3_2, Score_3_3


    # TODO finish
    # Compute P(A | s, q)
    def calculate_approach_4(
        self,
        question: str,
        answers: List[str],
        summary: str
    ) -> Tuple[float, float, float]:
        """
        Calculates the Log-Posterior Coverage Ratio for three different conditioning variants.
        The distance is calculated as: log P(S|Q) - log P(S|A,Q), ensuring a lower score is better.
        """
        # --- Baseline Likelihood: P(s|q) ---
        # This is calculated once and reused for each distance score.
        prompt_4_0 = self.create_user_prompt(question=question, approach="4_0")
        log.info("Prompt for P(s|q): %s", prompt_4_0)
        log_p_s_given_q, _, _, _ = self.compute_target_likelihood(prompt=prompt_4_0, target=summary)

        # --- Variant 4.1: Full Set Conditioning ---
        # Calculates distance based on P(s | A, q)
        prompt_4_1 = self.create_user_prompt(question=question, answers=answers, approach="4_1")
        log.info("4.1: Prompt for P(s|A,q): %s", prompt_4_1)
        log_p_s_given_A_q, _, _, _ = self.compute_target_likelihood(prompt=prompt_4_1, target=summary)
        distance_4_1 = log_p_s_given_q - log_p_s_given_A_q

        # --- Variant 4.2: Pairwise Conditioning ---
        # Calculates distance based on P(s | {(q, a_i)}, q)
        prompt_4_2 = self.create_user_prompt(question=question, answers=answers, approach="4_2")
        log.info("4.2: Prompt for P(s|(q,a_i)): %s", prompt_4_2)
        log_p_s_given_qa_pairs_q, _, _, _ = self.compute_target_likelihood(prompt=prompt_4_2, target=summary)
        distance_4_2 = log_p_s_given_q - log_p_s_given_qa_pairs_q

        # --- Variant 4.3: Factorized Conditioning ---
        # Calculates distance based on the average of P(s | q, a_i)
        log_probs_4_3 = []
        for answer in answers:
            user_prompt = self.create_user_prompt(question=question, answer=answer, approach="4_3")
            log.info("4.3: Prompt for P(s|q,a_i): %s", user_prompt)
            result, _, _, _ = self.compute_target_likelihood(prompt=user_prompt, target=summary)
            log_probs_4_3.append(result)
        
        avg_log_p_s_given_avg_a_q = np.mean(log_probs_4_3) if log_probs_4_3 else 0.0
        distance_4_3 = log_p_s_given_q - avg_log_p_s_given_avg_a_q

        log.info(f"P(s|q)={log_p_s_given_q:.4f}, P(s|A,q)={log_p_s_given_A_q:.4f}, P(s|(q,a_i))={log_p_s_given_qa_pairs_q:.4f}, P(s|avg(a_i))={avg_log_p_s_given_avg_a_q:.4f}")
        return distance_4_1, distance_4_2, distance_4_3
    
    # The "target" can either be an answer, a summary or a masked word
    def compute_target_likelihood(
        self,
        prompt,
        target
    ):
        if not target or not target.strip():
            log.warning("Target is empty or contains only whitespace, returning 0 likelihood.")
            vocab_size = self.llm_wrapper.model.model.config.vocab_size
            device = self.llm_wrapper.model.model.device
            # Return values with correct types and device for an empty target
            return 0.0, torch.tensor([], device=device), [], torch.empty((0, vocab_size), device=device)

        # 1) System prompt from config
        system_content = self.config.system_prompt

        # 2) If prompt is a list of messages, handle it. If it's a string, treat as user content
        if isinstance(prompt, list):
            user_content = str(prompt)
        else:
            user_content = str(prompt)

        # 3) Put `target` in the assistant role
        # 4) Construct the final list of messages
        messages = stitch_prompt(self.llm_wrapper.model.model.name_or_path, system=system_content, user=user_content, assistant=target)
        log.debug("Messages: %s", messages)

        # 5) Call apply_chat_template with tokenize=False, add_generation_prompt=False
        chat_prompt_str = self.llm_wrapper.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        log.debug("Target: %s", target)
        log.debug("Full chat-based prompt: %s", chat_prompt_str)

        # 6) Encode
        try:
            input_ids = self.llm_wrapper.tokenizer.encode(chat_prompt_str, return_tensors='pt')
        except Exception as e:
            log.error("Error encoding chat_prompt_str or target: %s", e)
            return float('-inf'), [], [], []

        # 7) Forward pass
        try:
            with torch.no_grad():
                outputs = self.llm_wrapper.model(input_ids.cuda() if self.llm_wrapper.model.model.device.type == "cuda" else input_ids)
                # Shift logits to align them with "current token"
                outputs.logits = outputs.logits.roll(1, 1)
                outputs.logits[:, 0, :] = 0
                logits = outputs.logits  # shape [1, seq_len, vocab_size]
        except Exception as e:
            log.error("Error during model inference: %s", e)
            return float('-inf'), [], [], []


        # Instead of computing log_softmax for the entire logits tensor, we compute it only for the target tokens.
        # 8) Figure out the target tokens, slice them out
        # Hand over the same prompts, but with empty assistant
        chat_prompt_str_empty_assistant = self.llm_wrapper.tokenizer.apply_chat_template(
            stitch_prompt(self.llm_wrapper.model.model.name_or_path, system=system_content, user=user_content, assistant=""),
            tokenize=False,
            add_generation_prompt=False
        )
        input_ids_empty_assistant = self.llm_wrapper.tokenizer.encode(chat_prompt_str_empty_assistant, return_tensors='pt')

        # Figure out how many tokens in the target
        target_seq_len = input_ids.shape[-1] - input_ids_empty_assistant.shape[-1]
        input_ids_empty_assistant = pad_second_tensor(input_ids, input_ids_empty_assistant)
        comparison = input_ids != input_ids_empty_assistant
        mismatch_indices = torch.nonzero(comparison)
        if mismatch_indices.numel() == 0:
            log.error("No valid target tokens found at the end of the chat prompt.")
            return float('-inf'), [], [], []

        target_start = mismatch_indices[0][-1].item()
        target_ids = input_ids[0, target_start : target_start + target_seq_len]

        # Extract only the logits for the target tokens and cast them to float32 for improved numerical precision.
        per_pos_logits = logits[:, target_start : target_start + target_seq_len, :]
        target_log_probs_full = torch.log_softmax(per_pos_logits.float() / self.config.post_hoc_temperature, dim=-1)
        log.debug("target_log_probs_full shape: %s", target_log_probs_full.shape)

        try:
            if self.llm_wrapper.model.model.device.type == "cuda":
                target_ids = target_ids.cuda()
            target_log_probs = target_log_probs_full.gather(
                2, target_ids.unsqueeze(0).unsqueeze(-1)
            ).squeeze(-1).squeeze(0)
            if self.llm_wrapper.model.model.device.type == "cuda":
                target_ids = target_ids.cpu()
        except Exception as e:
            log.error("Error gathering target log probabilities: %s", e)
            return float('-inf'), [], [], []

        # 10) Convert to probabilities
        probs_of_target_tokens = torch.exp(target_log_probs)  # shape: [target_seq_len]
        sum_of_target_token_log_probs = target_log_probs.sum().item()  # sum of log probs for the target tokens

        log.info("probs_of_target_tokens: %s", probs_of_target_tokens)
        log.info("probs_of_target_tokens shape: %s", probs_of_target_tokens.shape)

        target_tokens = self.llm_wrapper.tokenizer.convert_ids_to_tokens(target_ids)
        log.info("target_tokens: %s", target_tokens)  # target_tokens is a list

        # Compute probability distributions over target tokens (using the log_softmax computed in float32)
        prob_distributions_over_target_tokens = torch.exp(target_log_probs_full.squeeze(0))
        log.info(f"target_start: {target_start}")
        log.info(f"prob_distributions_over_target_tokens shape: {prob_distributions_over_target_tokens.shape}")

        return (
            sum_of_target_token_log_probs,  # a float
            probs_of_target_tokens,           # shape: torch.Size([target_seq_len])
            target_tokens,                    # a list of tokens
            prob_distributions_over_target_tokens  # shape: [target_seq_len, vocab_size]
        )


    def create_user_prompt(
        self,
        question: Optional[str] = None,
        summary: Optional[str] = None,
        masked_out_answer: Optional[str] = None,
        answer: Optional[str] = None,
        answers: Optional[List[str]] = None,
        approach: Optional[str] = None
        ) -> str:
            
        # Safely convert question, summary, and masked_out_answer to "" if they are None
        question = question or ""
        summary = summary or ""
        masked_out_answer = masked_out_answer or ""
        answers = answers or []

        if question != "":
            if question.endswith("?"):
                question += "\n"
            elif not question.endswith("?\n"):
                question += "?\n"

            if not question.startswith("Question: "):
                question = "Question: " + question

        # Prompts for Average Pointwise Mutual Information (PMI)
        if approach == "2_1":
            # Calculates P(A | S, Q)
            if summary:
                user_prompt = (
                    f"Question: {question}\n"
                    f"Summary: {summary}\n\n"
                    f"Provide an answer to the question."
                )

            # Calculates P(A | Q)
            else:
                user_prompt = (
                    f"Question: {question}\n\n"
                    f"Provide an answer to the question."
                )
        # SELF-REFLECT: Provide summary
        elif approach == "2_s":
            user_prompt = (
                f"{question}"
                f"Here's some background information for the following task:\n{summary}\n"
                f"We now show a text with a missing word \"_\". Fill in the missing word \"_\" only based on the above background information: {question + masked_out_answer}\n"
                f"Please provide only the missing word \"_\", not the whole sentence."
            )
        # SELF-REFLECT: Provide Answers
        elif approach == "2_a":
            if isinstance(answers, list):
                answers_str = "\n".join(answers)
            else:
                answers_str = str(answers)
            user_prompt = (
                f"{question}"
                f"Here's some background information the following task:\n{answers_str}\n"
                f"We now show a text with a missing word \"_\". Fill in the missing word \"_\" only based on the above background information: {question + masked_out_answer}\n"
                f"Please provide only the missing word \"_\", not the whole sentence."
            )
        # SELF-REFLECT: Side Case providing question
        elif approach == "2_q":
            user_prompt = (
                f"We now show a text with a missing word \"_\". Fill in the missing word \"_\": {question + masked_out_answer}\n"
                f"Please provide only the missing word \"_\", not the whole sentence."
            )
        # SELF-REFLECT: Side Case not providing question
        elif approach == "2_nq":
            user_prompt = (
                f"We now show a text with a missing word \"_\". Fill in the missing word \"_\": {masked_out_answer}\n"
                f"Please provide only the missing word \"_\", not the whole sentence."
            )
        # Generative Distribution Discrepancy: P(x | s, q)
        elif approach == "3_0":
            user_prompt = (
                f"Question: {question}\n"
                f"Summary: {summary}\n\n"
                f"Provide an answer to the question."
            )
        # Generative Distributio Discrepancy: P(x | a_1, ..., a_n, q)    
        elif approach == "3_1":
            answers_str = "\n".join([f"Answer: {a}" for a in answers])
            user_prompt = (
                f"Question: {question}\n"
                f"{answers_str}\n\n"
                f"Provide an answer to the question."
            )
        # Generative Distribution Discrepancy: P(x | (q, a_i)_i=1^n)
        elif approach == "3_2":
            questions_and_answers_str = ""
            for answer in answers:
                questions_and_answers_str += f"Question: {question}\n Answer: {answer}\n\n"
            user_prompt = (
                f"Question: {question}\n"
                f"{questions_and_answers_str}\n"
                f"Provide an answer to the question."
            )
        # Generative Distribution Discrepancy: P(x | q, a_i)
        elif approach == "3_3":
            user_prompt = (
                f"Question: {question}\n"
                f"Answer: {answer}\n\n"
                f"Provide an answer to the question."
            )
        # Log-Posterior Coverage Ratio: P(s | q)    
        elif approach == "4_0":
            user_prompt = (
                f"Question: {question}\n\n"
                "Provide a summary of possible answers to this question."
            )
        # Log-Posterior Coverage Ratio: P(s | q, a_1, ..., a_n)    
        elif approach == "4_1":
            answers_str = "\n".join([f"Answer: {a}" for a in answers])
            user_prompt = (
                f"Question: {question}\n"
                f"{answers_str}\n"
                f"Provide a summary of possible answers to this question."
            )
        # Log-Posterior Coverage Ratio: P(s | (q, a_i)_i=1^n)
        elif approach == "4_2":
            questions_and_answers_str = ""
            for answer in answers:
                questions_and_answers_str += f"Question: {question}\n Answer: {answer}\n"
            user_prompt = (
                f"{questions_and_answers_str}\n"
                f"Provide a summary of possible answers to this question."
            )
        # Log-Posterior Coverage Ratio: P(s | q, a_i)
        elif approach == "4_3":
            user_prompt = (
                f"Question: {question}\n"
                f"Answer: {answer}\n\n"
                f"Provide a summary of possible answers to this question."
            )
        else:
            log.error(f"Unknown approach: {approach}")
            raise ValueError(f"Unknown approach: {approach}")

        log.debug(f"Created user prompt for approach {approach}: {user_prompt}")
        return user_prompt