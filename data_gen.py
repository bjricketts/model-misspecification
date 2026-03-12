import numpy as np
import matplotlib.pyplot as plt

def generate_data(time_length, dt, egrid, gamma, gamma_scatter, norm):
    t = np.linspace(0, time_length, time_length//dt)
    model = np.zeros(shape=(len(egrid), len(t)))
    for i in range(len(t)):
        gamma_t = np.random.normal(loc=gamma, scale=gamma_scatter)
        y = norm * egrid**-gamma_t
        model[:, i] = y
    data = np.random.poisson(model*dt, size=model.shape)
    return data, model

time_length = 1000
dt = 100
egrid = np.logspace(-1, 2, 100)
data, model = generate_data(time_length, dt, egrid, 2, 0.1, 10000)

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 2, 1)
for i in data.T:
    ax1.scatter(egrid, i, color="red", alpha=0.5, s=10, marker="x", zorder=2)
ax1.plot(egrid, model*dt, color="blue", ls= "--", zorder=1)
ax1.set_xlabel("Energy (keV)")
ax1.set_ylabel("Counts")
ax1.set_title("Model Spectra")
ax1.loglog()

ax2 = plt.subplot(1, 2, 2)
ax2.scatter(egrid, np.sum(data,axis=1), color="red", alpha=0.5, s=10, marker="x", zorder=2)
ax2.plot(egrid, np.mean(model,axis=1)*time_length, color="blue", ls= "--", zorder=1)
ax2.set_xlabel("Energy (keV)")
ax2.set_ylabel("Counts")
ax2.set_title("Total data and averaged model spectrum")
ax2.loglog()
plt.savefig("figures/model_spectra.png")
plt.close()
print("Finished")