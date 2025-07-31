import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from typing import List

logger = logging.getLogger(__name__)


class LLMWrapper:
    """
    A wrapper class for language models. Handles tokenization,
    generation, prompts, and some convenience methods like generating summaries
    or answers with or without context.
    """

    def __init__(self, model_name: str, dtype="auto", device="auto"):
        """
        Initialize the LLMw
    rapper by loading a tokenizer, model, and pipeline.

        Args:
            model_name (str): The name or path of the model to load.
        """
        self.device = device
        self.model_name = model_name

        logger.info("Loading tokenizer for model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("Loading model: %s", model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=self.device)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.generation_config = self.model.generation_config
        self.framework = "pt"  # Could be "pt" or "tf"

    def generate(self, prompt_text: str, do_sample=True, **generate_kwargs):
        logger.debug("Generating text for prompt: %s", prompt_text)

        model_inputs = self.tokenizer(
            [prompt_text],
            return_tensors="pt"
        )
        model_inputs = model_inputs.to("cuda") if self.model.model.device.type == "cuda" else model_inputs
        #print("do_sample: ", do_sample)

        default_generation_params = {
            'do_sample': True,
            'top_p': 1,
            'temperature': 1,
            'max_new_tokens': 256,
        }

        default_generation_params.update(generate_kwargs)

        generated_ids = self.model.generate(
            **model_inputs,
            **default_generation_params
        )
        

        # Check if multiple sequences were generated for the single prompt
        if generated_ids.shape[0] > len(model_inputs.input_ids):
            input_length = model_inputs.input_ids.shape[-1]
            # For a single prompt with multiple sequences, generated_ids shape is (num_return_sequences, seq_len)
            generated_ids = generated_ids[:, input_length:]
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return responses


    def check_answer_in_summary(self, summary, answer):
        """
        Determine if an answer is included in the summary.
        """
        logger.info("Checking if the answer is included in the summary.")
        prompt = (
            "You are a helpful assistant that determines whether a given answer is included in a provided summary. "
            "Respond only with 'Yes' or 'No'.\n\n"
            "Example 1:\n"
            "Summary: The Eiffel Tower is located in Paris, France.\n"
            "Does the summary include the information: 'Paris is home to the Eiffel Tower'?\n"
            "Answer: Yes\n\n"
            "Example 2:\n"
            "Summary: Python is a popular programming language known for its readability.\n"
            "Does the summary include the information: 'Python is a type of snake'?\n"
            "Answer: No\n\n"
            f"Summary: {summary}\n"
            f"Does the summary include the information: '{answer}'?\n"
            "Answer:"
        )

        outputs = self.generate(prompt, do_sample=False)
        if outputs and isinstance(outputs, list) and len(outputs) > 0 and 'generated_text' in outputs[0]:
            response_text = outputs[0]['generated_text'].replace(prompt, '').strip().lower()
            logger.debug("Model Response: %s", response_text)
            if response_text.startswith('yes'):
                return True
            elif response_text.startswith('no'):
                return False
            else:
                logger.warning("Model gave an unexpected response. Using fallback method.")
                return answer.strip().lower() in summary.strip().lower()
        elif isinstance(outputs, str) and outputs:
            response_text = outputs.replace(prompt, "").strip().lower()
            logger.debug("Model Response: %s", response_text)
            if response_text.startswith('yes'):
                return True
            elif response_text.startswith('no'):
                return False
            else:
                logger.warning("Model gave an unexpected response. Using fallback method.")
                return answer.strip().lower() in summary.strip().lower()
        else:
            logger.error("Failed to generate response for checking answer in summary.")
            return False
        
    def check_element_in_answer(self, answer: str, element: str) -> bool:
        """
        Use the LLM to determine whether a given answer contains a specific element.
        The LLM is expected to respond with 'Yes' or 'No'.
        """
        prompt = (
            "You are a helpful assistant that determines whether a given answer contains a specific element.\n"
            "Respond only with 'Yes' or 'No'.\n\n"
            "Example 1:\n"
            "Answer: The Eiffel Tower is located in Paris, France.\n"
            "Does the answer contain the element: 'Paris'?\n"
            "Response: Yes\n\n"
            "Example 2:\n"
            "Answer: Python is a popular programming language.\n"
            "Does the answer contain the element: 'snake'?\n"
            "Response: No\n\n"
            f"Answer: {answer}\n"
            f"Does the answer contain the element: '{element}'?\n"
            "Response:"
        )

        outputs = self.generate(prompt, max_new_tokens=10, do_sample=False)
        response_text = ""
        if outputs:
            if isinstance(outputs, list):
                try:
                    if isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
                        response_text = outputs[0]["generated_text"].replace(prompt, "").strip().lower()
                    elif isinstance(outputs[0], str):
                        response_text = outputs[0].replace(prompt, "").strip().lower()
                except Exception as e:
                    logger.error("Error processing LLM output in check_element_in_answer: %s", e)
            elif isinstance(outputs, str):
                response_text = outputs.replace(prompt, "").strip().lower()

        logger.debug("LLM response for check_element_in_answer: '%s'", response_text)
        if response_text.startswith("yes"):
            return True
        elif response_text.startswith("no"):
            return False
        else:
            # Fallback: use a simple substring check if the LLM output is unexpected.
            return element.strip().lower() in answer.strip().lower()

    def check_element_in_answers(self, element: str, answers: List[str]) -> bool:
        """
        Returns True if any of the provided answers is determined by the LLM to contain the given element.
        """
        for answer in answers:
            if self.check_element_in_answer(answer, element):
                return True
        return False
    
    def extract_elements_from_summary(self, summary: str) -> list:
        """
        Use the LLM to extract key elements or phrases from the provided summary.
        The LLM is prompted to return a list of key points, one per line.
        """
        prompt = (
            "You are an assistant that extracts key elements or phrases from a given summary. "
            "Please list the most important elements or key phrases that capture the essence of the summary. "
            "Return each element on a separate line, without any extra commentary or numbering.\n\n"
            f"Summary: {summary}\n\n"
            "Key elements:"
        )
        outputs = self.generate(prompt, do_sample=False)
        if not outputs or not isinstance(outputs, str):
            logger.error("Failed to generate key elements from summary; returning empty list.")
            return []
        # Post-process the output by splitting into lines and cleaning up each element.
        elements = []
        for line in outputs.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                line = line[2:].strip()
            if line:
                elements.append(line)
        logger.debug("Extracted elements from summary: %s", elements)
        return elements

    