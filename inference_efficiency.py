import jax
import numpy as np

from data_utils import load_data, get_correct, get_length, group_length_correct
from stats_models import length_model_A, length_model_C
from stats_utils import infer_length_model, save_length_model_results
import sys

def main():

    ### PARAMETERS ###
    model_type = sys.argv[1]
    task = sys.argv[2]
    temp = float(sys.argv[3])
    ns = [2, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    Ns = [1.5e9, 7e9, 14e9, 32e9, 70e9]
    R = 100

    models = [
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
    ]

    seed = 5123
    num_warmup = 1000
    num_samples = 5000
    num_chains = 6
    target_accept_prob = 0.99
    max_tree_depth = 20

    ### DATA LOADING ###
    ns = np.asarray(ns)
    Ns = np.asarray(Ns)

    results = load_data(
        models,
        Ns,
        ns,
        [temp],
        task=task
    )

    correct_indicator = get_correct(
        results,
        Ns,
        ns,
        [temp],
        num_runs=R,
        task=task
    )

    length = get_length(
        results,
        Ns,
        ns,
        [temp],
        num_runs=R
    )

    group_result = group_length_correct(
        length,
        correct_indicator,
        Ns,
        ns,
        temp)
    
    Lmean, Lstd, Y, group_Ns_idx, group_ns_idx  = group_result

    ### SAMPLING ###
    if model_type == 'A':
        model = length_model_A
    elif model_type == 'C':
        model = length_model_C
    else:
        raise ValueError(f"Unknown model type: {model_type}")


    mcmc = infer_length_model(
        jax.random.PRNGKey(seed),
        model,
        (Ns, ns, group_Ns_idx, group_ns_idx, Y, Lstd, Lmean),
        num_warmup,
        num_samples,
        num_chains,
        target_accept_prob,
        max_tree_depth
    )

    # save the results
    _ = save_length_model_results(
        mcmc,
        Ns,
        ns,
        group_Ns_idx,
        group_ns_idx,
        Y,
        Lstd,
        R,
        temp,
        seed, 
        model_type,
        task,
        target_accept_prob,
        max_tree_depth
    )

if __name__ == "__main__":
    main()