import torch
import copy
import transformers # type: ignore
import logging

log = logging.getLogger(__name__)

def get_start_overlap(reference_tensor, new_tensor):
    """
    Takes two tensors _with batchsize 1_ and returns at which index they start to differ.
    If they don't differ, it returns the length of the shorter tensor.

    Args:
        reference_tensor: Tensor of shape [1, tokens]
        new_tensor: Tensor of shape [1, tokens]

    Returns: integer, the position index where they start to differ

    """
    # Cut both tensors to the same length (no overlap can happen afterwards anyways)
    min_dims = tuple(min(reference_tensor.size(dim), new_tensor.size(dim)) for dim in range(len(new_tensor.shape)))
    reference_tensor = reference_tensor[tuple(slice(0, min_dim) for min_dim in min_dims)]
    new_tensor = new_tensor[tuple(slice(0, min_dim) for min_dim in min_dims)]

    # Find out where they start to differ
    comparison = reference_tensor != new_tensor
    mismatch_indices = torch.nonzero(comparison)
    if mismatch_indices.numel() == 0:
        start_mismatch = new_tensor.shape[-1]
    else:
        start_mismatch = mismatch_indices[0][-1].item()

    return start_mismatch


class CachedPromptModel(torch.nn.Module):
    """
    This is a wrapper around a huggingface LLM model. When its forward call is applied, it first checks
    if it has already cached the starting tokens of the query, and then does not need to recompute them.
    This saves time if you have a lot of prompts that have the same start, up to a certain point (which is auto detected).
    Currently only implemented for batchsize 1!
    Don't use this for anything but simple model() calls to calculate logits. It's not tested
    for any extra args or generate functions.
    """
    def __init__(self, model, device="auto"):
        """
        Args:
            model: a huggingface LLM
            device: string, which device to store the cache on. If "auto" uses the same as the model (probably GPU).
                    This increases speed, but requires VRAM and might run OOM.
                    If "cpu", always saves to RAM, but is slower because it has to copy it back to VRAM where the model is.
        """
        super().__init__()
        self.model = model
        self.reset_cache()
        self.device=device

    def reset_cache(self):
        log.info("Resetting cache.")
        self.cache = None
        self.cache_logits = None
        self.cache_tokens = None

    def __call__(self, input_ids, *args, past_key_values=None, **kwargs):
        """
        This call function replaces/wraps the call function of the LLM, to measure logits.
        It loads as much as possible from cache to avoid re-doing unnecessary work.
        Only works for batchsize 1!

        Args:
            input_ids: Tensor with integers
            past_key_values: optional, a cache object. If not None, we don't add our cache to not interfere.

        Returns: The output of a typical LLM call, so a CausalLMOutputWithPast output.
        """
        assert input_ids.shape[0] == 1
        n_input = input_ids.shape[-1]
        n_start_overlap = 0

        # See how many tokens from the start we already have in cache
        if self.cache is not None and past_key_values is None:
            # Get overlap between our cached prompt and the current input_ids
            n_start_overlap = get_start_overlap(self.cache_tokens, input_ids)

            past_key_values = copy.deepcopy(self.cache)  # Need to deepcopy because the cache will get overwritten
            past_key_values.crop(n_start_overlap)  # This is important, otherwise we get numerical issues + have to specify cache_position
            input_ids = input_ids[:,n_start_overlap:]

            if n_start_overlap == n_input:
                # We've cached everything, no need to recompute
                return transformers.modeling_outputs.CausalLMOutputWithPast(
                    logits = self.cache_logits[:,:n_start_overlap,:] if self.device == "auto" else self.cache_logits[:,:n_start_overlap,:].to(input_ids.device),
                    past_key_values=past_key_values if self.device == "auto" else past_key_values.to(input_ids.device)
                )

            if self.device != "auto":
                past_key_values = past_key_values.to(input_ids.device)


        # Decode the remaining tokens
        with torch.no_grad():
            out = self.model(input_ids=input_ids, past_key_values=past_key_values, *args, **kwargs)

        # Add our cached results to the decoded results
        if n_start_overlap > 0:
            if self.device != "auto":
                out.logits = out.logits.to(self.device)

            out.logits = torch.concatenate((self.cache_logits[:,:n_start_overlap,:], out.logits), dim=1)

            if self.device != "auto":
                out.logits = out.logits.to(input_ids.device)

        # If we don't have a cache yet, cache the current results
        if self.cache is None:
            # Update it with what we computed
            self.cache = copy.deepcopy(out.past_key_values if self.device == "auto" else out.past_key_values.to(self.device))
            self.cache_logits = copy.deepcopy(out.logits if self.device == "auto" else out.logits.to(self.device))
            self.cache_tokens = copy.deepcopy(input_ids)

        log.info(f"Caching saved {((float(n_start_overlap) / n_input) * 100):.2f}% of tokens")

        return out


    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


if __name__=="__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    import time

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    # You can change this to torch.bfloat16 to see the numerics issue
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cuda")
    model = CachedPromptModel(model)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    PREFIX = "You are a helpful assistant that gives short and concise responses. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent finibus efficitur lorem, a eleifend nisi condimentum eu. Cras tempor vel turpis tristique efficitur. Ut venenatis volutpat justo. Aliquam lorem lacus, fermentum id quam eget, cursus molestie tellus. Fusce vel dui vitae ipsum bibendum sollicitudin. Fusce lobortis orci ut lectus facilisis, id pulvinar leo tincidunt. Ut vitae dapibus massa. Duis a odio et enim scelerisque interdum. Proin mollis elit mauris, a lobortis leo gravida vel. Etiam ultricies, elit sed congue efficitur, purus nisi posuere risus, eu sagittis quam leo ac lacus. Mauris malesuada nulla dapibus dolor varius, id tristique sapien porttitor. Vestibulum nec augue viverra, tincidunt odio vel, commodo justo. Nulla sollicitudin imperdiet ultrices. Nullam quis sapien ac lectus mattis mattis eget in diam. Donec sodales orci sed tellus rutrum dapibus. Sed iaculis, ipsum vel tempor dapibus, purus justo molestie orci, at convallis velit purus ut nibh. Nunc at ultrices nunc. Etiam consequat quis ligula quis rhoncus. Ut porttitor diam purus, sit amet pulvinar risus vestibulum maximus. Nulla sed libero dui. Nunc pellentesque hendrerit pellentesque. Sed in urna tristique, accumsan ante quis, bibendum dolor. Curabitur sagittis, erat sit amet maximus fermentum, lectus mauris iaculis dui, eu commodo orci enim eu dui. Nullam quis pellentesque est. Ut ullamcorper efficitur sodales. Nam nec metus at libero pellentesque porttitor. Phasellus accumsan nulla eu urna finibus, id aliquam felis porta. Phasellus quis varius orci. Ut facilisis turpis id mauris gravida, non tincidunt tortor elementum. Fusce vehicula ligula non felis lobortis placerat. Quisque eget arcu nibh. Fusce posuere, orci eu luctus semper, diam neque pharetra lacus, non sodales massa dolor vel nunc. Nullam viverra lectus vel odio pharetra hendrerit sed ut tortor. Phasellus condimentum elementum malesuada. Nulla dui orci, varius at tortor ac, aliquam varius leo. Aliquam est velit, dictum non mollis vel, egestas vitae dui. Suspendisse ac porta eros. Pellentesque consequat vulputate est, ac tincidunt ligula auctor sit amet. Nam non quam a elit faucibus tempor vitae eget sapien. Vivamus ac ex non lacus luctus ullamcorper eu eu ante. Quisque volutpat felis ut elementum maximus. Aenean nulla turpis, bibendum id facilisis efficitur, ultricies ac mi. Curabitur tortor sapien, ultrices eu ultrices ultrices, pellentesque at enim. Nam justo mauris, tempus in magna quis, eleifend posuere erat. Donec efficitur arcu ipsum, ac molestie sem dictum quis. Vivamus convallis malesuada ipsum, eget luctus metus elementum vel. Integer ut scelerisque est, at ultricies eros. Cras lorem nisl, dignissim a. "
    prompt = "The capitol of France is Paris."

    # First call will create a cache, so here we cache the initial prompt
    inputs = tokenizer(PREFIX, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    start_time = time.time()
    model(**inputs)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Seconds taken to cache prefix: {end_time - start_time:.5f}")

    # Second call will use the cached PREFIX, so we only calculate logits for prompt (but still output for everything)
    inputs_with_prompt = tokenizer(PREFIX + prompt, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    start_time = time.time()
    out_prompt = model(**inputs_with_prompt)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Seconds taken to compute prompt after having cached the prefix: {end_time - start_time:.5f}")
    out_prompt = out_prompt.logits.cpu()

    # Make sure that the outputs are equal to calling the method normally
    torch.cuda.synchronize()
    start_time = time.time()
    out_normal = model.model(**inputs_with_prompt)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Seconds taken to compute prompt without caching: {end_time - start_time:.5f}")
    out_normal = out_normal.logits.cpu()
    diff = ((out_prompt - out_normal).abs() / out_normal).mean().item()
    print(f"Relative difference in logits with and without caching: {diff}%")
    # There are super slight numerical inaccuracies, which happen inside torch.nn.functional.scaled_dot_product_attention
    # Nothing we can do about this unfortunately, the caching is implemented correctly.
    if diff > 1e-3:
        print("Differences:")
        print((out_prompt - out_normal).abs() / out_normal)
