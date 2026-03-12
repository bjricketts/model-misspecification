import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

import emcee
import corner

def model_spectrum_bb(egrid, temp, norm):
    """Calculate the model spectrum."""
    renorm = 8.0525*norm/np.power(temp,4.)
    planck = np.exp(egrid/temp)-1.
    planck[planck>1e20] = 1e20
    model = renorm*np.power(egrid,2.)/planck
    return model

data = np.load("data/data.npy")
summed_spectrum = np.sum(data,axis=1)
fit_data = summed_spectrum[summed_spectrum>0]
fit_err = np.sqrt(fit_data)

egrid = np.logspace(-1, 1.5, 100)
fit_grid = egrid[summed_spectrum>0]

#optimization to start 
def log_likelihood(theta, x, y, yerr):
    kT, norm = theta
    model = model_spectrum_bb(x,kT,norm)
    loglike = (y-model)/yerr
    return loglike

np.random.seed(42)
initial = np.array([0.75, 1e3]) + 0.1 * np.random.randn(2)
soln = leastsq(log_likelihood, initial, args=(fit_grid, fit_data, fit_err))
ktBB, normBB = soln[0]

print("Maximum likelihood estimates:")
print("kT = {0:.2f}".format(ktBB))
print("norm = {0:.2f}".format(normBB))

def log_prior(theta):
    kT, norm = theta
    if 0.1 < kT < 3.5 and 0.0 < norm < 1e6:
        return 0.0
    return -np.inf
    
def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp - 0.5*np.sum(log_likelihood(theta, x, y, yerr)**2)
    
pos = soln[0] + 1e-4 * np.random.randn(32, 2)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(fit_grid, fit_data, fit_err)
)
sampler.run_mcmc(pos, 5000, progress=True)

tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=30, thin=15, flat=True)

fig = corner.corner(
    flat_samples, labels=["kT","norm"], truths=[1.0,1e4], range=[(0.96,1.0006),(9800,1.0004e4)]
)
fig.savefig("figures/corner.png")
