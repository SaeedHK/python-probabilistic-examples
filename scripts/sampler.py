import numpy as np
import pymc3 as pm
from timeit import default_timer
from scipy.stats import norm, halfnorm
import matplotlib.pyplot as plt
from pymc3.step_methods.hmc.nuts import NUTS
from pymc3.step_methods.hmc import quadpotential


print(f"Running on PyMC3 v{pm.__version__}")


alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma

basic_model = pm.Model()

with basic_model as m:

    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Expected value of outcome
    mu = alpha + beta[0] * X1 + beta[1] * X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

    # maximum a posteriori (MAP) estimate:
    # map_estimate = pm.find_MAP(model=basic_model)
    # print(f"Map Estimate:{map_estimate}")

    times_consumed = []

    N = 100
    alphas = norm.rvs(size=N)
    betas = halfnorm.rvs(size=(N, 2))
    sigmas = halfnorm.rvs(size=N)

    print("=========================================")
    print(f"Tuning NUTS")
    print("=========================================")

    n_chains = 4
    init_trace = pm.sample(draws=1000, tune=1000, cores=n_chains)
    cov = np.atleast_1d(pm.trace_cov(init_trace))
    start = list(np.random.choice(init_trace, n_chains))
    potential = quadpotential.QuadPotentialFull(cov)
    step_size = init_trace.get_sampler_stats("step_size_bar")[-1]
    size = m.bijection.ordering.size
    step_scale = step_size * (size ** 0.25)

    # with pm.Model() as model_new:  # reset model. If you use theano.shared you can also update the value of model1 above
    for i in range(N):
        # No-U-Turn Sampler NUTS
        print("=========================================")
        print(f"Turn {i}")
        print("=========================================")
        # start = {"alpha": alphas[i], "beta": betas[i], "sigma": sigmas[i]}
        time_zero = default_timer()
        step = pm.NUTS(potential=potential, adapt_step_size=False, step_scale=step_scale)
        step.tune = False
        trace = pm.sample(draws=100, step=step, tune=0, cores=n_chains, start=start)
        time_consumed = default_timer() - time_zero
        times_consumed.append(time_consumed)
        print("-----------------------------------------")
        print(f"Times consumed for turn {i}: {time_consumed}")
        print("-----------------------------------------")

    print("Plotting the time consumed and save it to sampler.png")
    print(times_consumed)
    plt.plot(times_consumed)
    plt.savefig("sampler.png")
