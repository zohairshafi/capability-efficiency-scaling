import jax
import numpy as np
import sys

from data_utils import load_data, get_correct, count_correct
from stats_models import accuracy_model_A, accuracy_model_B
from stats_utils import infer_accuracy_model, save_accuracy_model_results

def main():

    ### PARAMETERS ###
    model_type = sys.argv[1]
    task = sys.argv[2]
    ns = [2, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    Ns = [1.5e9, 7e9, 14e9, 32e9, 70e9]
    R = 100
    temp = float(sys.argv[3])


    models = [
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
    ]

    seed = 1123 # 1123 #
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
        task=task)

    correct_indicator = get_correct(
        results,
        Ns,
        ns,
        [temp],
        num_runs=R,
        task=task
    )

    Y = count_correct(
        correct_indicator,
        Ns,
        ns,
        [temp]
    )
    
    Y = Y[temp]

    # reshape into indexed notation
    Ns_idx = []
    ns_idx = []
    Y_obs = []
    for i, N in enumerate(Ns):
        for j, n in enumerate(ns):
            Ns_idx.append(i)
            ns_idx.append(j)
            Y_obs.append(Y[i, j])
    Y_obs = np.asarray(Y_obs)
    Ns_idx = np.asarray(Ns_idx)
    ns_idx = np.asarray(ns_idx)

    ### SAMPLING ###
    if model_type == 'A':
        model = accuracy_model_A
    elif model_type == 'B':
        model = accuracy_model_B

    mcmc = infer_accuracy_model(
        jax.random.PRNGKey(seed),
        model,
        (Ns, ns, Ns_idx, ns_idx, R, Y_obs),
        num_warmup,
        num_samples,
        num_chains,
        target_accept_prob,
        max_tree_depth
    )

    ### SAVE RESULT ###
    _ = save_accuracy_model_results(
        mcmc,
        Ns,
        ns,
        Ns_idx,
        ns_idx,
        R,
        temp,
        seed,
        model_type,
        task,
        target_accept_prob,
        max_tree_depth
    )

## run main
if __name__ == "__main__":
    main()