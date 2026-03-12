import numpy as np
import matplotlib.pyplot as plt

def parameter_visualisation(t,gamma,norm,data):
    fig = plt.figure(figsize=(12, 12))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, gamma)
    ax1.set_ylabel("Gamma")
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
    ax2.plot(egrid, model_spectrum(egrid, np.mean(gamma), np.mean(norm))*time_length, color="blue", ls= "--", zorder=1)
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel("Counts")
    ax2.set_title("Total data and averaged model spectrum")
    ax2.loglog()
    plt.savefig("figures/model_spectra.png")
    plt.close()

def gen_noise(time, x, fmin=1e-3, fmax=1e2, nfreq=1e2, beta=1.0, norm=1):
    """Adding some noise à la Timmer & Koenig"""

    freqs = 10**((np.log10(fmax) - np.log10(fmin)) *  np.linspace(0, 1, int(nfreq)) + np.log10(fmin))

    lc0 = np.mean(x)
    lc = x - lc0

    for ifreq,freq in enumerate(freqs):
        lc += norm * (1./freq)**beta * np.cos(2*np.pi*freq*time - np.random.rand()*2*np.pi) / nfreq

    return lc - np.mean(lc) + lc0

def model_spectrum(egrid, gamma, norm):
    """Generate the model spectrum."""
    return norm * egrid**-gamma

def generate_data(time_length, dt, egrid, gamma, gamma_scatter, norm):
    t = np.linspace(0, time_length, time_length//dt)
    model = np.zeros(shape=(len(egrid), len(t)))
    gamma = np.ones(shape=len(t)) * gamma
    norm = np.ones(shape=len(t)) * norm
    gamma = gen_noise(t, gamma, beta=0.5, norm=gamma_scatter)
    norm = gen_noise(t, norm, beta=0.25, norm=norm*0.2)
    for i in range(len(t)):
        y = model_spectrum(egrid, gamma[i], norm[i])
        model[:, i] = y
    data = np.random.poisson(model*dt, size=model.shape)
    return data, model, t, gamma, norm

time_length = 1000
dt = 10
egrid = np.logspace(-1, 2, 100)
data, model, t, gamma, norm = generate_data(time_length, dt, egrid, 2, 0.25, 10)

parameter_visualisation(t, gamma, norm, data)
model_spectra_visualisation(egrid, data, model, gamma, norm)

print("Finished")