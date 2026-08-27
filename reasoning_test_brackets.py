import os
import sys
import logging
import time
import pickle

import sglang as sgl
from transformers import AutoTokenizer
from reasoning_task import BracketClosure
from huggingface_hub import login
import numpy as np
import torch

logger = logging.getLogger(__name__)

system_prompt = ""
model_name = str(sys.argv[6])

run_num = int(sys.argv[4])

if 'gemma' in model_name.lower():
    max_new_tokens = 31_000
    add_generation_prompt=True
elif 'Qwen3-0.6B' in model_name:
    max_new_tokens = 39_000
    add_generation_prompt=False
else:
    max_new_tokens = 75_000               # maximum number of new tokens to generate
    add_generation_prompt=False

#temperature = float(sys.argv[5])      # temperature for generation

num_instances = 100        # number of instances to sample
size = int(sys.argv[1])     # number of integers to add up 
min_value = 0               # smallest integer used in the addition
max_value = 100             # largest integer used in the addition
modulus = np.inf            # modulus for the arithmetic, set None for standard arithmetic

problem_seed = int(sys.argv[2]) + run_num # seed of the random number generator of the problem instance
model_seed = int(sys.argv[3]) + run_num   # seed of the random number generator of the model response


huggingface_token = ''

cache_dir = '/root/models/'

TP_DEGREE = int(os.environ.get("TP_DEGREE", "1"))

# Probability mass thresholds for p* token counts (must match patch_sglang.py).
# output_top_logprobs layout per token:
#   [0]: entropy (bits), [1..]: n_tokens for each threshold below.
P_STAR_THRESHOLDS = [1.0, 0.99, 0.90]
TOP_LOGPROBS_NUM = 1 + 2 * len(P_STAR_THRESHOLDS) + 4  # base(7) + surprisal + varentropy + p_max + rank = 11

def main():

    # login to hugging face
    login(token=huggingface_token)

    # seed the random number generators
    problem_rng = np.random.default_rng(seed=problem_seed)

    # send device to GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if torch.cuda.is_available() is False:
        print ("Device: ", str(device))
        print ("No GPU Assigned. Falling Back to CPU.")
        return -1

    # ── Load tokenizer (needed for chat template + token tracking) ──
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=cache_dir,
        local_files_only=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    # ── Pre-generate ALL problem instances and prompts ──
    print(f"Pre-generating {num_instances} problem instances...")
    raw_prompts = []
    correct_answers = []
    for i in range(num_instances):
        problem_instance = BracketClosure(num_pairs=size, rng=problem_rng)
        prompt_text = problem_instance.generate_prompt()
        if 'DeepSeek-R1' in model_name or 'Qwen' in model_name:
            prompt_text += '<think>\n'
        elif 'gemma' in model_name.lower():
            prompt_text += 'Reason step by step before arriving at your answer.\n'
        raw_prompts.append(prompt_text)
        correct_answers.append(problem_instance.solution)

    # ── Apply chat template to all prompts ──
    print("Applying chat templates...")
    formatted_prompts = []
    for p in raw_prompts:
        if 'deepseek-v4' in model_name.lower():
            _bos = "<｜begin▁of▁sentence｜>"
            _user = "<｜User｜>"
            _assistant = "<｜Assistant｜>"
            if system_prompt:
                formatted = f"{_bos}{system_prompt}\n{_user}{p}{_assistant}<think>"
            else:
                formatted = f"{_bos}{_user}{p}{_assistant}<think>"
        elif 'Llama-4' in model_name:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": p}]},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            )
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            )
        formatted_prompts.append(formatted)

    # ── Load model with SGLang Engine ──
    print("Loading model with SGLang (continuous batching + flash attention)...")
    engine = sgl.Engine(
        model_path=model_name,
        tp_size=TP_DEGREE,  # tensor parallelism across 4 GPUs
        dtype="bfloat16",
        mem_fraction_static=0.65,  # reserve more memory for KV-cache (less fragmentation with large models)
        trust_remote_code=True,
        disable_cuda_graph=False,
    )
    using_flash = True  # SGLang uses flash/paged attention by default

    # ── Generate across temperature settings ──
    print("Starting Runs...")

    for temperature in [0.6]:

        print("Starting Temperature:", temperature)
        start_time = time.time()

        result_dir = f'/root/models/new_results/{model_name}/{temperature:.1f}/'
        os.makedirs(result_dir, exist_ok=True)

        # Configure sampling parameters for SGLang
        sampling_params = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }

        # ── Batch-generate ALL prompts at once ──
        print(f"Generating {num_instances} instances via continuous batching...")
        outputs = engine.generate(
            formatted_prompts,
            sampling_params,
            return_logprob=True,
            top_logprobs_num=TOP_LOGPROBS_NUM,
        )

        generation_time = time.time() - start_time
        print(f"Batch generation completed in {generation_time:.1f}s")

        # ── Process results ──
        results = {
            'model': model_name,
            'system_prompt': system_prompt,
            'min_value': min_value,
            'max_value': max_value,
            'modulus': modulus,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'problem_seed': problem_seed,
            'model_seed': model_seed,
            'results': [],
        }

        for i, output in enumerate(outputs):
            output_text = output["text"]
            meta = output["meta_info"]

            # Debug: confirm the patched format on first output
            if i == 0:
                print(f"[DEBUG] meta_info keys: {list(meta.keys()) if isinstance(meta, dict) else type(meta)}")
                first_step = meta.get("output_top_logprobs", [[]])[0]
                print(f"[DEBUG] first token entry (7 values: entropy, n_tokens x3, entropy_bulk x3):")
                print(f"        {first_step}")

            # Extract exact per-token entropy, p* token counts, and bulk entropies.
            # output_top_logprobs[i] layout (N = len(P_STAR_THRESHOLDS) = 3):
            #   [0]:     entropy_full
            #   [1..N]:  n_tokens per threshold
            #   [N+1..]: entropy_bulk per threshold
            N = len(P_STAR_THRESHOLDS)
            output_top_logprobs = meta.get("output_top_logprobs", [])
            output_token_logprobs = meta.get("output_token_logprobs", [])
            num_tokens = len(output_top_logprobs)
            entropy = np.array([
                step[0][0] for step in output_top_logprobs if step
            ])
            p_star_n_tokens = {
                p: np.array([int(step[k + 1][0]) for step in output_top_logprobs if step])
                for k, p in enumerate(P_STAR_THRESHOLDS)
            }
            entropy_bulk = {
                p: np.array([step[N + k + 1][0] for step in output_top_logprobs if step])
                for k, p in enumerate(P_STAR_THRESHOLDS)
            }

            # Selected-token metrics:
            #   log p(token_t | prefix) from native output_token_logprobs
            #   p(token_t | prefix) = exp(log p)
            selected_token_logprob = np.array(
                [step[0] for step in output_token_logprobs if step],
                dtype=np.float32,
            )
            selected_token_probability = np.exp(selected_token_logprob, dtype=np.float32)

            # Optional metrics from patched sampler (when TOP_LOGPROBS_NUM large enough):
            #   [7]: sampled-token surprisal (nats), sentinel -8
            #   [8]: varentropy (bits^2), sentinel -9
            #   [9]: p_max (probability), sentinel -10
            #  [10]: rank (1-indexed), sentinel -11
            selected_token_surprisal = np.array(
                [step[7][0] for step in output_top_logprobs if step and len(step) > 7],
                dtype=np.float32,
            )
            varentropy = np.array(
                [step[8][0] for step in output_top_logprobs if step and len(step) > 8],
                dtype=np.float32,
            )
            p_max = np.array(
                [step[9][0] for step in output_top_logprobs if step and len(step) > 9],
                dtype=np.float32,
            )
            rank_sampled = np.array(
                [int(step[10][0]) for step in output_top_logprobs if step and len(step) > 10],
                dtype=np.int32,
            )

            results['results'].append({
                'input': [raw_prompts[i]],
                'output': output_text,
                'size': int(size),
                'correct_answer': int(correct_answers[i]),
                'num_tokens': num_tokens,
                'entropy': entropy,                  # np.ndarray, exact entropy (bits, full vocab)
                'p_star_n_tokens': p_star_n_tokens,  # dict: {p* -> np.ndarray of int counts}
                'entropy_bulk': entropy_bulk,         # dict: {p* -> np.ndarray, bulk entropy (bits)}
                'selected_token_logprob': selected_token_logprob,            # np.ndarray, natural log-probabilities
                'selected_token_surprisal': selected_token_surprisal,        # np.ndarray, nats
                'varentropy': varentropy,                                    # np.ndarray, bits^2
                'p_max': p_max,                                              # np.ndarray, probability
                'rank_sampled': rank_sampled,                                 # np.ndarray, 1-indexed int
                # 'selected_token_probability': selected_token_probability,      # np.ndarray, probabilities in [0,1]
                'run_time': time.time() - start_time,
            })

        if modulus == np.inf:
            mod_str = 'inf'
        else:
            mod_str = f"{modulus:.2f}"

        with open(f"{result_dir}result_size={size}_brackets-temp={temperature}-flash={using_flash}_sglang_batched_fp32.pkl", "wb") as f:
            pickle.dump(results, f)

        print("Completed Temperature:", temperature)
        end_time = time.time()
        print(f"Time Taken for Temperature {temperature}:", end_time - start_time)

    # Shutdown engine
    engine.shutdown()

if __name__ == "__main__":
    main()
