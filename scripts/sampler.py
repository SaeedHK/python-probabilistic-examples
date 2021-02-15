import numpy as np
import pymc3 as pm
from timeit import default_timer
from scipy.stats import norm, halfnorm
import matplotlib.pyplot as plt


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

with basic_model:

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

    N = 2
    alphas = norm.rvs(size=N)
    betas = halfnorm.rvs(size=(N, 2))
    sigmas = halfnorm.rvs(size=N)

    for i in range(N):
        # No-U-Turn Sampler NUTS
        start = {"alpha": alphas[i], "beta": betas[i], "sigma": sigmas[i]}
        print(start)
        time_zero = default_timer()
        trace = pm.sample(500, start=start, return_inferencedata=False)
        time_consumed = default_timer() - time_zero
        times_consumed.append(time_consumed)
        print(time_consumed)

    print(times_consumed)
    plt.plot(times_consumed)
    plt.show()
