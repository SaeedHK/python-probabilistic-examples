import numpy as np
import pymc3 as pm
from pymc3.step_methods.hmc import quadpotential
from pymc3.step_methods import step_sizes

n_chains = 4

with pm.Model() as m:
    x = pm.Normal("x", shape=10)
    # init == 'jitter+adapt_diag'
    start = []
    for _ in range(n_chains):
        mean = {var: val.copy() for var, val in m.test_point.items()}
        for val in mean.values():
            val[...] += 2 * np.random.rand(*val.shape) - 1
        start.append(mean)
    mean = np.mean([m.dict_to_array(vals) for vals in start], axis=0)
    var = np.ones_like(mean)
    potential = quadpotential.QuadPotentialDiagAdapt(m.ndim, mean, var, 10)
    step = pm.NUTS(potential=potential)
    trace1 = pm.sample(1000, step=step, tune=1000, cores=n_chains)

with m:  # need to be the same model
    step_size = trace1.get_sampler_stats("step_size_bar")[-1]

    step.tune = False
    step.step_adapt = step_sizes.DualAverageAdaptation(
        step_size, step.target_accept, 0.05, 0.75, 10
    )
    trace2 = pm.sample(draws=100, step=step, tune=0, cores=n_chains)
    print(trace2[-1])
