import jax
import numpy as np
import sys

import numpyro.infer as infer
from data_utils import load_data, get_correct, count_correct
from stats_models import accuracy_model_A, accuracy_model_B
from stats_utils import infer_accuracy_model, logmeanexp

def main():

    idx = int(sys.argv[1])
    select = []
    for model_type in ["A", "B"]:
        for task in ["addition", "brackets", "index", "parity"]:
            for temp in [0.4, 0.6, 0.8]:
                select.append((model_type, task, temp))

    ### PARAMETERS ###
    model_type = select[idx][0]
    task = select[idx][1]
    temp = select[idx][2]
    ns = [2, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    Ns = [1.5e9, 7e9, 14e9, 32e9, 70e9]
    R = 100

    print(f"This run is for model={model_type}, task={task}, temp={temp}")

    models = [
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
    ]

    seed = 1123
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
    for i, _ in enumerate(Ns):
        for j, _ in enumerate(ns):
            Ns_idx.append(i)
            ns_idx.append(j)
            Y_obs.append(Y[i, j])
    Y_obs = np.asarray(Y_obs)
    Ns_idx = np.asarray(Ns_idx)
    ns_idx = np.asarray(ns_idx)

    key = jax.random.PRNGKey(seed)
    loo_likelihoods = []
    for out_idx in range(len(Ns_idx)):

        print(f"Processing: {out_idx}/{len(Ns_idx)}", end="\n")

        ### SAMPLING ###
        if model_type == 'A':
            model = accuracy_model_A
        elif model_type == 'B':
            model = accuracy_model_B
        
        # eliminate the leave out sample
        keep = np.arange(len(Y_obs)) != out_idx
        Ns_idx_keep = Ns_idx[keep]
        ns_idx_keep = ns_idx[keep]
        Y_obs_keep = Y_obs[keep]
        
        # fit the model
        key, key_ = jax.random.split(key)
        mcmc = infer_accuracy_model(
            key_,
            model,
            (Ns, ns, Ns_idx_keep, ns_idx_keep, R, Y_obs_keep),
            num_warmup,
            num_samples,
            num_chains,
            target_accept_prob,
            max_tree_depth
        )

        ll_test = infer.log_likelihood(
            model,
            mcmc.get_samples(),
            Ns=Ns,
            ns=ns,
            Ns_idx=np.asarray([Ns_idx[out_idx]]),
            ns_idx=np.asarray([ns_idx[out_idx]]),
            R=R,
            Y=np.asarray([Y_obs[out_idx]])
        )

        loo_likelihoods.append(
            logmeanexp(ll_test['num_correct'], axis=0).item()
        )

    outfile = f"./loo_accuracy/loo_likelihoods_model={model_type}_task={task}_temp={temp}.csv"
    np.savetxt(outfile, np.asarray(loo_likelihoods), delimiter=',')

## run main
if __name__ == "__main__":
    main()