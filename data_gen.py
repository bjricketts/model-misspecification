import numpy as np
import matplotlib.pyplot as plt

"""
The following functions are plotting functions for visualisation of data generated.
"""

def parameter_visualisation(t,gamma,norm,data,label):
    fig = plt.figure(figsize=(12, 12))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, gamma)
    ax1.set_ylabel(label)
    ax1.set_xlabel("Time (s)")

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t, norm)
    ax2.set_ylabel("Norm")
    ax2.set_xlabel("Time (s)")

    ax3 = plt.subplot(2, 1, 2)
    pcm = ax3.pcolormesh(t, egrid, np.log10(data), shading="auto")
    fig.colorbar(pcm, ax=ax3, label="log10(Counts)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Energy (keV)")
    ax3.set_title("2D Histogram of Counts")
    ax3.set_yscale("log")
    plt.tight_layout()
    plt.savefig("figures/model_variation.png")
    plt.close()

def model_spectra_visualisation(egrid, data, model, gamma, norm):
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    for i in data.T:
        ax1.scatter(egrid, i, color="red", alpha=0.1, s=10, marker="x", zorder=2)
    ax1.plot(egrid, model*dt, color="blue", ls= "--", zorder=1)
    ax1.set_xlabel("Energy (keV)")
    ax1.set_ylabel("Counts")
    ax1.set_title("Model Spectra")
    ax1.loglog()

    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(egrid, np.sum(data,axis=1), color="red", alpha=0.5, s=10, marker="x", zorder=2)
    ax2.plot(egrid, model_spectrum_bb(egrid, np.mean(gamma), np.mean(norm))*time_length, color="blue", ls= "--", zorder=1)
    ax2.fill_between(egrid, model_spectrum_bb(egrid, np.mean(gamma)-np.std(gamma), np.mean(norm)-np.std(norm))*time_length,
                     model_spectrum_bb(egrid, np.mean(gamma)+np.std(gamma), np.mean(norm)+np.std(norm))*time_length, color="blue", alpha=0.2)
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel("Counts")
    ax2.set_title("Total data and averaged model spectrum")
    ax2.loglog()

    plt.tight_layout()
    plt.savefig("figures/model_spectra.png")
    plt.close()

"""
The following functions are data generation functions for simulating data.
"""

def gen_noise(time, x, fmin=1e-3, fmax=1e2, nfreq=1e2, beta=1.0, norm=1):
    """
    Adding some noise à la Timmer & Koenig
    
    Parameters
    ----------
    time : array
        The time array for the simulation.
    x : array
        The input signal to which noise will be added.
    fmin : float
        The minimum frequency for the noise.
    fmax : float
        The maximum frequency for the noise.
    nfreq : int
        The number of frequency bins for the noise.
    beta : float
        The spectral index for the noise.
    norm : float
        The normalization factor for the noise.
    """

    freqs = 10**((np.log10(fmax) - np.log10(fmin)) *  np.linspace(0, 1, int(nfreq)) + np.log10(fmin))

    lc0 = np.mean(x)
    lc = x - lc0

    for ifreq,freq in enumerate(freqs):
        lc += norm * (1./freq)**beta * np.cos(2*np.pi*freq*time - np.random.rand()*2*np.pi) / nfreq

    return lc - np.mean(lc) + lc0

def model_spectrum_bb(egrid, temp, norm):
    """Generate the model spectrum."""
    renorm = 8.0525*norm/np.power(temp,4.)
    planck = np.exp(egrid/temp)-1.
    planck[planck>1e20] = 1e20
    model = renorm*np.power(egrid,2.)/planck
    return model

def model_spectrum_pl(egrid, gamma, norm):
    """
    Generate the model spectrum. Simple powerlaw.
    
    Parameters
    ----------
    egrid : array
        The energy grid for the simulation.
    gamma : float
        The spectral index for the simulation.
    norm : float
        The normalization factor for the simulation.
    """
    return norm * egrid**-gamma

def generate_data(time_length, dt, egrid, temp, temp_scatter, norm):
    """
    Generate synthetic data for the model.

    Parameters
    ----------
    time_length : int
        The total length of time for the simulation.
    dt : int
        The time resolution of the simulation.
    egrid : array
        The energy grid for the simulation.
    gamma : float
        The spectral index for the simulation.
    temp_scatter : float
        The amount of scatter in the gamma values.
    norm : float
        The normalization factor for the simulation.
    """
    t = np.linspace(0, time_length, time_length//dt)
    model = np.zeros(shape=(len(egrid), len(t)))
    temp = np.ones(shape=len(t)) * temp
    norm = np.ones(shape=len(t)) * norm
    temp = gen_noise(t, temp, beta=0.5, norm=temp_scatter)
    norm = gen_noise(t, norm, beta=0.25, norm=norm*0.2)
    for i in range(len(t)):
        y = model_spectrum_bb(egrid, temp[i], norm[i])
        model[:, i] = y
    data = np.random.poisson(model*dt, size=model.shape)
    return data, model, t, temp, norm

"""
Main body of the code.
"""

time_length = 1000
dt = 10
egrid = np.logspace(-1, 1.5, 100)
data, model, t, gamma, norm = generate_data(time_length, dt, egrid, 1.0, 0.25, 10)

parameter_visualisation(t, gamma, norm, data, label="temperature")
model_spectra_visualisation(egrid, data, model, gamma, norm)

np.save("data/data.npy", data)
np.save("data/model.npy", model)
np.save("data/parameters.npy", np.array([gamma, norm]))

print("Finished")
