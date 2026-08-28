import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
from typing import Optional
import arviz as az
import xarray as xr

def logmeanexp(x, axis=None):

    return jax.nn.logsumexp(x, axis=axis) - jnp.log(x.shape[axis])


def infer_accuracy_model(key, model, data, num_warmup, num_samples, num_chains, target_accept_prob, max_tree_depth):
    """Run inference for a accuracy model.

        Input
        key : jax.random.PRNGKey
        model : model function
        data : tuple of (Ns, ns, R, Y)
        num_warmup : number of warmup samples
        num_samples : number of posterior samples
        num_chains : number of MCMC chains
        target_accept_prob : target acceptance probability for NUTS sampler
        max_tree_depth : maximum tree depth for NUTS sampler

        Output
        post_pred : dictionary of posterior predictive samples
    """

    Ns, ns, Ns_idx, ns_idx, R, Y = data

    # define the NUTS kernel
    key, key_ = jax.random.split(key)

    kernel = infer.NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )

    mcmc = infer.MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )

    mcmc.run(
        key_,
        Ns=Ns,
        ns=ns,
        Ns_idx=Ns_idx,
        ns_idx=ns_idx,
        R=R,
        Y=Y
    )

    # sample the posterior
    mcmc.print_summary()

    return mcmc

def save_accuracy_model_results(mcmc, Ns, ns, Ns_idx, ns_idx, R, temp, seed, model_type, task, target_accept_prob, max_tree_depth):

    assert model_type in ["A", "B"], "model_type must be 'A' or 'B'"

    constant_data = {
        "Ns": np.asarray(Ns),
        "ns": np.asarray(ns),
        "Ns_idx": np.asarray(Ns_idx),
        "ns_idx": np.asarray(ns_idx),
    }

    coords = {
        "N": np.asarray(Ns),
        "n": np.asarray(ns),
        "obs" : np.arange(len(Ns_idx))
    }
    
    dims={
        "num_correct": ["obs"],
        "p" : ["N", "n"],
        "eps_nu" : ["N"],
        "log_nu" : ["N"],
        "log_nu_mean" : ["N"],
        "Ns" : ["N"],
        "ns" : ["n"]
    }

    if model_type == "B":
        dims["p_hard"] = ["N"]
        dims["eta_gap"] = ["N"]

    idata = az.from_numpyro(
    posterior=mcmc,
    constant_data=constant_data,
    coords=coords,
    dims=dims
    )

    idata.constant_data["temp"] = xr.DataArray(float(temp))
    idata.constant_data["R"] = xr.DataArray(int(R))

    idata.attrs["model"] = f"accuracy_model_{model_type}"
    idata.attrs["task"] = task
    idata.attrs["seed"] = seed
    idata.attrs["num_warmup"] = mcmc.num_warmup
    idata.attrs["num_samples"] = mcmc.num_samples
    idata.attrs["num_chains"] = mcmc.num_chains
    idata.attrs["target_accept_prob"] = target_accept_prob
    idata.attrs["max_tree_depth"] = max_tree_depth

    idata.to_netcdf(f"./samples_model={model_type}_task={task}_temp={temp:.1f}.nc")

    return idata

def infer_length_model(key, model, data, num_warmup, num_samples, num_chains, target_accept_prob, max_tree_depth):
    """Run inference for a length model.

        Input
        key : jax.random.PRNGKey
        model : model function
        data : tuple of (Ns, ns, R, Y)
        num_warmup : number of warmup samples
        num_samples : number of posterior samples
        num_chains : number of MCMC chains
        target_accept_prob : target acceptance probability for NUTS sampler
        max_tree_depth : maximum tree depth for NUTS sampler

        Output
        post_pred : dictionary of posterior predictive samples
    """

    Ns, ns, group_Ns_idx, group_ns_idx, Y, Lstd, Lmean = data

    # define the NUTS kernel
    key, key_ = jax.random.split(key)

    kernel = infer.NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )

    mcmc = infer.MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )

    mcmc.run(
        key_,
        Ns=Ns,
        ns=ns,
        group_Ns_idx=group_Ns_idx,
        group_ns_idx=group_ns_idx,
        Y=Y,
        Lstd=Lstd,
        Lmean=Lmean
    )

    # sample the posterior
    mcmc.print_summary()

    return mcmc

def save_length_model_results(mcmc, Ns, ns, group_Ns_idx, group_ns_idx, Y, Lstd, R, temp, seed, model_type, task, target_accept_prob, max_tree_depth):

    assert model_type in ["A", "C"], "model_type must be 'A' or 'C'."

    constant_data = {
        "Ns": np.asarray(Ns),
        "ns": np.asarray(ns),
        "group_Ns_idx": np.asarray(group_Ns_idx),
        "group_ns_idx": np.asarray(group_ns_idx),
        "Y" : np.asarray(Y),
        "Lstd" : np.asarray(Lstd),
    }

    coords = {
        "N": np.asarray(Ns),
        "n": np.asarray(ns),
        "obs" : np.arange(len(group_Ns_idx))
    }
    
    if model_type in ["A", "B", "C"]:
        dims={
            "alpha" : ["N"],
            "eps_A" : ["N"],
            "eps_alpha" : ["N"],
            "log_A" : ["N"],
            "log_alpha" : ["N"],
            "log_mu" : ["N", "n"],
            "Lmean_pred" : ["obs"],
            "log_mu_group" : ["obs"],
            "scale_group" : ["obs"],
            "var_group" : ["obs"],
            "Ns" : ["N"],
            "ns" : ["n"],
            "group_Ns_idx" : ["obs"],
            "group_ns_idx" : ["obs"],
            "Y" : ["obs"],
            "Lstd" : ["obs"],
            "log_Lmean_sample" : ["obs"]
        }
    elif model_type == "D":
        dims={
            "alpha" : ["N"],
            "eps_n50" : ["N"],
            "eps_alpha" : ["N"],
            "log_n50" : ["N"],
            "log_alpha" : ["N"],
            "log_mu" : ["N", "n"],
            "mu" : ["N", "n"],
            "Lmean_pred" : ["obs"],
            "log_mu_group" : ["obs"],
            "scale_group" : ["obs"],
            "var_group" : ["obs"],
            "Ns" : ["N"],
            "ns" : ["n"],
            "group_Ns_idx" : ["obs"],
            "group_ns_idx" : ["obs"],
            "Y" : ["obs"],
            "Lstd" : ["obs"],
            "log_Lmean_sample" : ["obs"]
        }

    idata = az.from_numpyro(
        posterior=mcmc,
        constant_data=constant_data,
        coords=coords,
        dims=dims
    )

    idata.constant_data["temp"] = xr.DataArray(float(temp))
    idata.constant_data["R"] = xr.DataArray(int(R))

    idata.attrs["model"] = f"accuracy_model_{model_type}"
    idata.attrs["task"] = task
    idata.attrs["seed"] = seed
    idata.attrs["num_warmup"] = mcmc.num_warmup
    idata.attrs["num_samples"] = mcmc.num_samples
    idata.attrs["num_chains"] = mcmc.num_chains
    idata.attrs["target_accept_prob"] = target_accept_prob
    idata.attrs["max_tree_depth"] = max_tree_depth

    idata.to_netcdf(f"./samples_model={model_type}_task={task}_temp={temp:.1f}.nc")

    return idata