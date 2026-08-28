import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
from typing import Optional
import arviz as az
import xarray as xr

def accuracy_model_A(Ns:jnp.ndarray, ns:jnp.ndarray, Ns_idx:jnp.ndarray, ns_idx:jnp.ndarray, R:int, Y:Optional[jnp.ndarray]=None):
    """Hierarchical model for number of correct responses with N-independent p_hard asymptote.

        Input
        Ns : array of model sizes
        ns : array of instance sizes
        Ns_idx : array of indices for model sizes
        ns_idx : array of indices for instance sizes
        R : number of different instances per size
        Y : number of correct responses (Ns.shape[0], ns.shape[0])
    """
    
    num_Ns = Ns.shape[0]       # number model sizes
    num_ns = ns.shape[0]       # number instance sizes
    num_obs = Ns_idx.shape[0]  # number of observations

    Ns = jnp.asarray(Ns)
    log_Ns = jnp.log(Ns)
    ns = jnp.asarray(ns)
    log_ns = jnp.log(ns)

    Ns_idx = jnp.asarray(Ns_idx)
    ns_idx = jnp.asarray(ns_idx)

    # scale model sizes by geometric mean
    x_N = log_Ns - jnp.mean(log_Ns)

    # put priors on the accuracy of easy (n->0) and hard (n->infty) problems
    # such that 1 >= p_easy > p_hard >= 0
    eta_easy = numpyro.sample(
        "eta_easy",
        dist.Normal(1.5, 1.0)
        )
    eta_gap = numpyro.sample(
        "eta_gap",
        dist.Normal(1.5, 1.0)
        )
    p_easy = numpyro.deterministic(
        "p_easy",
        jax.nn.sigmoid(eta_easy)
    )
    p_hard = numpyro.deterministic(
       "p_hard",
        p_easy * jax.nn.sigmoid(eta_gap)
    )

    # put a prior on the scaling exponent for the typical problem size that
    # a model of size N can solve.
    beta_nu = numpyro.sample(
        "beta_nu",
        dist.Normal(0.0, 1.5)
    )

    # put a prior on the typical problem size that a model of size of the geometric
    # mean of the model sizes can solve. 
    log_C = numpyro.sample(
        "log_C",
        dist.Normal(jnp.log(50), 1.0)
    )

    # put a prior on the variability of the problem size that a model of size N can solve.
    log_tau_nu = numpyro.sample(
        "log_tau_nu",
        dist.Normal(jnp.log(0.3), 0.5)
    )
    
    tau_nu = numpyro.deterministic(
        "tau_nu",
        jnp.exp(log_tau_nu)
    )

    # compute the expected size of the problem that model of size N can solve,
    # and then add the random effect for each model size.
    log_nu_mean = numpyro.deterministic(
        "log_nu_mean",
        log_C + beta_nu * x_N
    )

    with numpyro.plate("N_eps_nu", num_Ns, dim=-1):

        eps_nu = numpyro.sample(
            "eps_nu",
            dist.Normal(0.0, 1.0)
        )

    log_nu = numpyro.deterministic(
        "log_nu",
        log_nu_mean + tau_nu * eps_nu
    )

    # sample the number of correct responses for each model and instance size
    p = numpyro.deterministic(
        "p",
        p_hard + (p_easy - p_hard) * jnp.exp(-jnp.exp(log_ns[None, :] - log_nu[:, None]))
    )

    p_obs = p[Ns_idx, ns_idx]

    with numpyro.plate("obs", num_obs, dim=-1):
        numpyro.sample(
            "num_correct",
            dist.Binomial(probs=p_obs, total_count=R),
            obs=Y,
        )

def accuracy_model_B(Ns:jnp.ndarray, ns:jnp.ndarray, Ns_idx:jnp.ndarray, ns_idx:jnp.ndarray, R:int, Y:Optional[jnp.ndarray]=None):
    """Hierarchical model for number of correct responses with N-dependent p_hard asymptote.

        Input
        Ns : array of model sizes
        ns : array of instance sizes
        R : number of different instances per size
        Y : number of correct responses (Ns.shape[0], ns.shape[0])
    """
    
    num_Ns = Ns.shape[0]       # number model sizes
    num_ns = ns.shape[0]       # number instance sizes
    num_obs = Ns_idx.shape[0]  # number of observations

    Ns = jnp.asarray(Ns)
    log_Ns = jnp.log(Ns)
    ns = jnp.asarray(ns)
    log_ns = jnp.log(ns)

    Ns_idx = jnp.asarray(Ns_idx)
    ns_idx = jnp.asarray(ns_idx)

    # scale model sizes by geometric mean
    x_N = log_Ns - jnp.mean(log_Ns)

    # put priors on the accuracy of easy (n->0) and hard (n->infty) problems
    # such that 1 >= p_easy > p_hard >= 0
    eta_easy = numpyro.sample(
        "eta_easy",
        dist.Normal(1.5, 1.0)
        )
    p_easy = numpyro.deterministic(
        "p_easy",
        jax.nn.sigmoid(eta_easy)
    )
    
    with numpyro.plate("N_eta_hard", num_Ns, dim=-1):
        
        eta_gap = numpyro.sample(
            "eta_gap",
            dist.Normal(1.5, 1.0)
        )

    p_hard = numpyro.deterministic(
    "p_hard",
        p_easy * jax.nn.sigmoid(eta_gap)
    )

    # put a prior on the scaling exponent for the typical problem size that
    # a model of size N can solve.
    beta_nu = numpyro.sample(
        "beta_nu",
        dist.Normal(0.0, 1.5)
    )

    # put a prior on the typical problem size that a model of size of the geometric
    # mean of the model sizes can solve. 
    log_C = numpyro.sample(
        "log_C",
        dist.Normal(jnp.log(50), 1.0)
    )

    # put a prior on the variability of the problem size that a model of size N can solve.
    log_tau_nu = numpyro.sample(
        "log_tau_nu",
        dist.Normal(jnp.log(0.3), 0.5)
    )
    tau_nu = numpyro.deterministic(
        "tau_nu",
        jnp.exp(log_tau_nu)
    )

    # compute the expected size of the problem that model of size N can solve,
    # and then add the random effect for each model size.
    log_nu_mean = numpyro.deterministic(
        "log_nu_mean",
        log_C + beta_nu * x_N
    )

    with numpyro.plate("N_eps_nu", num_Ns, dim=-1):

        eps_nu = numpyro.sample(
            "eps_nu",
            dist.Normal(0.0, 1.0)
        )

    log_nu = numpyro.deterministic(
        "log_nu",
        log_nu_mean + tau_nu * eps_nu
    )

    # sample the number of correct responses for each model and instance size
    p = numpyro.deterministic(
        "p",
        p_hard[:, None] + (p_easy - p_hard[:, None]) * jnp.exp(-jnp.exp(log_ns[None, :] - log_nu[:, None]))
    )

    p_obs = p[Ns_idx, ns_idx]

    with numpyro.plate("obs", num_obs, dim=-1):
        numpyro.sample(
            "num_correct",
            dist.Binomial(probs=p_obs, total_count=R),
            obs=Y,
        )

def length_model_A(Ns:jnp.ndarray, ns:jnp.ndarray, group_Ns_idx:jnp.ndarray, group_ns_idx:jnp.ndarray, Y:jnp.ndarray, Lstd:jnp.ndarray, Lmean:Optional[jnp.ndarray]=None):
    """Hierarchical model for the grouped means of the output length."""

    # conversion
    Ns = jnp.asarray(Ns)
    ns = jnp.asarray(ns)
    group_Ns_idx = jnp.asarray(group_Ns_idx)
    group_ns_idx = jnp.asarray(group_ns_idx)
    Y = jnp.asarray(Y)

    Lstd = jnp.asarray(Lstd)

    if Lmean is not None:
        Lmean = jnp.asarray(Lmean)
        log_Lmean_obs = jnp.log(Lmean)
    else:
        log_Lmean_obs = None

    num_Ns = Ns.shape[0]
    num_groups = group_Ns_idx.shape[0]

    # centering with geometric means
    x_N = jnp.log(Ns) - jnp.mean(jnp.log(Ns))
    x_n = jnp.log(ns) - jnp.mean(jnp.log(ns))


    # Priors on the N-scaling parameters
    log_C_A = numpyro.sample(
        "log_C_A",
        dist.Normal(jnp.log(1e4), 0.75),
    )

    log_C_alpha = numpyro.sample(
        "log_C_alpha",
        dist.Normal(jnp.log(0.75), 0.35),
    )

    beta_A = numpyro.sample(
        "beta_A",
        dist.Normal(0.0, 0.5),
    )

    beta_alpha = numpyro.sample(
        "beta_alpha",
        dist.Normal(0.0, 0.5),
    )

    # excess variation in scaling
    tau_log_A = numpyro.sample(
        "tau_log_A",
        dist.HalfNormal(0.25),
    )

    tau_log_alpha = numpyro.sample(
        "tau_log_alpha",
        dist.HalfNormal(0.25),
    )

    with numpyro.plate("N_pars", num_Ns):
        eps_A = numpyro.sample(
            "eps_A",
            dist.Normal(0.0, 1.0),
        )

        eps_alpha = numpyro.sample(
            "eps_alpha",
            dist.Normal(0.0, 1.0),
        )

        log_A = numpyro.deterministic(
            "log_A",
            log_C_A + beta_A * x_N + tau_log_A * eps_A,
        )

        log_alpha = numpyro.deterministic(
            "log_alpha",
            log_C_alpha + beta_alpha * x_N + tau_log_alpha * eps_alpha,
        )

        alpha = numpyro.deterministic(
            "alpha",
            jnp.exp(log_alpha),
        )

    # compute the expected output length for each model and instance size
    log_mu = numpyro.deterministic(
        "log_mu",
        log_A[:, None] + alpha[:, None] * x_n[None, :]
    )

    # grouped likelihood
    log_mu_group = numpyro.deterministic(
        "log_mu_group",
        log_mu[group_Ns_idx, group_ns_idx]
    )

    # use CLT plus delta method: log(mu) ~ Normal(log(mu), (Lstd**2)/(R*mu**2)) and add residual variance to account for excess variation in the group means
    var_group = numpyro.deterministic(
        "var_group",
        (Lstd**2) / (Y * jnp.exp(log_mu_group)**2)
    )

    sigma_res_group = numpyro.sample(
        "sigma_res_group",
        dist.HalfNormal(0.5),
    )

    scale_group = numpyro.deterministic(
        "scale_group",
        jnp.sqrt(var_group + sigma_res_group**2)
    )

    with numpyro.plate("groups_obs", num_groups):
        log_Lmean_sample = numpyro.sample(
            "log_Lmean_sample",
            dist.Normal(
                log_mu_group,
                scale_group,
            ),
            obs=log_Lmean_obs,
        )

        numpyro.deterministic(
            "Lmean_pred",
            jnp.exp(log_Lmean_sample),
        )

def length_model_C(Ns:jnp.ndarray, ns:jnp.ndarray, group_Ns_idx:jnp.ndarray, group_ns_idx:jnp.ndarray, Y:jnp.ndarray, Lstd:jnp.ndarray, Lmean:Optional[jnp.ndarray]=None):
    """Hierarchical model for the grouped means of the output length."""

    # conversion
    Ns = jnp.asarray(Ns)
    ns = jnp.asarray(ns)
    group_Ns_idx = jnp.asarray(group_Ns_idx)
    group_ns_idx = jnp.asarray(group_ns_idx)
    Y = jnp.asarray(Y)

    Lstd = jnp.asarray(Lstd)

    if Lmean is not None:
        Lmean = jnp.asarray(Lmean)
        log_Lmean_obs = jnp.log(Lmean)
    else:
        log_Lmean_obs = None

    num_Ns = Ns.shape[0]
    num_groups = group_Ns_idx.shape[0]

    # centering with geometric means
    x_N = jnp.log(Ns) - jnp.mean(jnp.log(Ns))
    x_n = jnp.log(ns) - jnp.mean(jnp.log(ns))


    # Priors on the N-scaling parameters
    log_C_A = numpyro.sample(
        "log_C_A",
        dist.Normal(jnp.log(1e4), 0.75),
    )

    log_C_alpha = numpyro.sample(
        "log_C_alpha",
        dist.Normal(jnp.log(0.75), 0.35),
    )

    beta_A = numpyro.sample(
        "beta_A",
        dist.Normal(0.0, 0.5),
    )

    # excess variation in scaling
    tau_log_A = numpyro.sample(
        "tau_log_A",
        dist.HalfNormal(0.25),
    )

    tau_log_alpha = numpyro.sample(
        "tau_log_alpha",
        dist.HalfNormal(0.25),
    )

    with numpyro.plate("N_pars", num_Ns):
        eps_A = numpyro.sample(
            "eps_A",
            dist.Normal(0.0, 1.0),
        )

        eps_alpha = numpyro.sample(
            "eps_alpha",
            dist.Normal(0.0, 1.0),
        )

        log_A = numpyro.deterministic(
            "log_A",
            log_C_A + beta_A * x_N + tau_log_A * eps_A,
        )

        log_alpha = numpyro.deterministic(
            "log_alpha",
            log_C_alpha + tau_log_alpha * eps_alpha,
        )

        alpha = numpyro.deterministic(
            "alpha",
            jnp.exp(log_alpha),
        )

    # compute the expected output length for each model and instance size
    log_mu = numpyro.deterministic(
        "log_mu",
        log_A[:, None] + alpha[:, None] * x_n[None, :]
    )

    # grouped likelihood
    log_mu_group = numpyro.deterministic(
        "log_mu_group",
        log_mu[group_Ns_idx, group_ns_idx]
    )

    # use CLT plus delta method: log(mu) ~ Normal(log(mu), (Lstd**2)/(R*mu**2)) and add residual variance to account for excess variation in the group means
    var_group = numpyro.deterministic(
        "var_group",
        (Lstd**2) / (Y * jnp.exp(log_mu_group)**2)
    )

    sigma_res_group = numpyro.sample(
        "sigma_res_group",
        dist.HalfNormal(0.5),
    )

    scale_group = numpyro.deterministic(
        "scale_group",
        jnp.sqrt(var_group + sigma_res_group**2)
    )

    with numpyro.plate("groups_obs", num_groups):
        log_Lmean_sample = numpyro.sample(
            "log_Lmean_sample",
            dist.Normal(
                log_mu_group,
                scale_group,
            ),
            obs=log_Lmean_obs,
        )

        numpyro.deterministic(
            "Lmean_pred",
            jnp.exp(log_Lmean_sample),
        )