import jax
import numpy as np
import numpyro.infer as infer

from data_utils import load_data, get_correct, get_length, group_length_correct
from stats_models import length_model_A, length_model_B, length_model_C, length_model_D
from stats_utils import infer_length_model, logmeanexp
import sys

def main():

    idx = int(sys.argv[1])
    select = []
    for model_type in ["A", "C"]:
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

    seed = 6123 # 5123
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
        temp
    )
    
    Lmean, Lstd, Y, group_Ns_idx, group_ns_idx  = group_result
    
    loo_likelihoods = []
    for out_idx in range(len(group_Ns_idx)):

        print(f"Processing: {out_idx}/{len(group_Ns_idx)}", end="\n")

        ### SAMPLING ###
        if model_type == 'A':
            model = length_model_A
        elif model_type == 'B':
            model = length_model_B
        elif model_type == 'C':
            model = length_model_C
        elif model_type == 'D':
            model = length_model_D
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # eliminate the leave out sample
        keep = np.arange(len(group_Ns_idx)) != out_idx
        Ns_idx_keep = group_Ns_idx[keep]
        ns_idx_keep = group_ns_idx[keep]
        Y_keep = Y[keep]
        Lstd_keep = Lstd[keep]
        Lmean_keep = Lmean[keep]

        mcmc = infer_length_model(
            jax.random.PRNGKey(seed),
            model,
            (Ns, ns, Ns_idx_keep, ns_idx_keep, Y_keep, Lstd_keep, Lmean_keep),
            num_warmup,
            num_samples,
            num_chains,
            target_accept_prob,
            max_tree_depth
        )

        if model_type == "A":
            latent_names = [
                "log_C_A",
                "log_C_alpha",
                "beta_A",
                "beta_alpha",
                "tau_log_A",
                "tau_log_alpha",
                "eps_A",
                "eps_alpha",
                "sigma_res_group",
            ]
        elif model_type == "C":
            latent_names = [
                "log_C_A",
                "log_C_alpha",
                "beta_A",
                "tau_log_A",
                "tau_log_alpha",
                "eps_A",
                "eps_alpha",
                "sigma_res_group",
            ]
        else:
            raise ValueError(f"Model type: {model_type} not supported.")
        
        samples = mcmc.get_samples()
        posterior_latents = {k: samples[k] for k in latent_names}

        ll_test = infer.log_likelihood(
            model,
            posterior_latents,
            Ns=Ns,
            ns=ns,
            group_Ns_idx=np.asarray([group_Ns_idx[out_idx]]),
            group_ns_idx=np.asarray([group_ns_idx[out_idx]]),
            Y=np.asarray([Y[out_idx]]),
            Lstd=np.asarray([Lstd[out_idx]]),
            Lmean=np.asarray([Lmean[out_idx]])
        )

        loo_likelihoods.append(
            logmeanexp(ll_test['log_Lmean_sample'], axis=0).item()
        )
    
    outfile = f"./loo_length/loo_likelihoods_model={model_type}_task={task}_temp={temp}.csv"
    np.savetxt(outfile, np.asarray(loo_likelihoods), delimiter=',')

if __name__ == "__main__":
    main()