import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as gaussian
from scipy.optimize import leastsq

import emcee
import corner

import os
import sys
import copy

sys.path.append('/home/matteo/Software/nDspec/src/')
from ndspec.Response import ResponseMatrix


def model_spectrum_bb(egrid, temp, norm):
    """Calculate the model spectrum."""
    renorm = 8.0525*norm/np.power(temp,4.)
    planck = np.exp(egrid/temp)-1.
    planck[planck>1e20] = 1e20
    model = renorm*np.power(egrid,2.)/planck
    return model
    
def folded_model(model,response):
    folded = response.convolve_response(model,units_in="rate")
    return folded

def log_likelihood(theta, x, y, yerr, mask):
    kT, norm = theta
    model = model_spectrum_bb(x,kT,norm)
    model = folded_model(model,nicer_rm)
    model_masked = np.extract(mask,model)
    loglike = (y-model_masked)/yerr
    return loglike
    
def log_prior(theta):
    kT, norm = theta
    if 0.1 < kT < 3.5 and 0.0 < norm < 1e6:
        return 0.0
    return -np.inf
    
def log_probability(theta, x, y, yerr, mask):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp - 0.5*np.sum(log_likelihood(theta, x, y, yerr, mask)**2)

nicer_rm = ResponseMatrix('/home/matteo/Data/1820_spectral_fits/nicer/ni1200120197mpu7.rmf')
nicer_rm.load_arf('/home/matteo/Data/1820_spectral_fits/nicer/ni1200120197mpu7.arf')

wiggle_response = copy.copy(nicer_rm)

temp = 2.3
norm = 5.
print(f"True values: kTbb {temp}, normalization {norm}")
grid = 0.5*(nicer_rm.energ_hi+nicer_rm.energ_lo)
changrid = 0.5*(nicer_rm.emax+nicer_rm.emin)
bb_model = model_spectrum_bb(grid,temp,norm)

#autoregressive process to mess up the arf 
autoreg = np.ones(nicer_rm.n_energs)
for i in range(0,nicer_rm.n_energs):
    autoreg[i] = autoreg[i-1] + gaussian.rvs(0,scale=0.005)

test_model = folded_model(bb_model,nicer_rm)

wiggle_response.resp_matrix = wiggle_response.resp_matrix*autoreg.reshape(nicer_rm.n_energs,1)   
fold_model = folded_model(bb_model,wiggle_response)
simulated_spectrum = np.random.poisson(fold_model)

plt.figure(1)
plt.errorbar(changrid,simulated_spectrum,yerr=np.sqrt(simulated_spectrum),zorder=1)
plt.loglog(changrid,test_model,label="Correct resp",lw=3,zorder=2)
plt.loglog(changrid,fold_model,label="Wiggle resp",lw=3,zorder=3)
plt.legend(loc="best")
plt.xlim([0.5,11.])
plt.ylim([3e1,8e3])
plt.savefig("figures/simulate_wiggle.png")

#now select only data between 0.5 and 10 keV
mask = ((nicer_rm.emin>0.5)&(nicer_rm.emax<10.0))

fit_data = np.extract(mask,simulated_spectrum)
fit_err = np.sqrt(fit_data)

np.random.seed(42)
initial = np.array([2.0, 3.0]) + 0.1 * np.random.randn(2)
soln = leastsq(log_likelihood, initial, args=(grid, fit_data, fit_err, mask))
ktBB, normBB = soln[0]

print("Maximum likelihood estimates:")
print("kT = {0:.2f}".format(ktBB))
print("norm = {0:.2f}".format(normBB))

plt.figure(2)
plt.errorbar(changrid,simulated_spectrum,yerr=np.sqrt(simulated_spectrum),zorder=1)
plt.loglog(changrid,folded_model(model_spectrum_bb(grid,ktBB,normBB),wiggle_response),zorder=2,lw=3)
plt.legend(loc="best")
plt.xlim([0.5,11.])
plt.ylim([3e1,8e3])
plt.savefig("figures/MLE.png")

pos = soln[0] + 1e-4 * np.random.randn(32, 2)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(grid, fit_data, fit_err, mask)
)
sampler.run_mcmc(pos, 10000, progress=True)

tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=200, thin=50, flat=True)

fig = corner.corner(
    flat_samples, labels=["kT","norm"],nbins=40
)
fig.savefig("figures/corner_response.png")
