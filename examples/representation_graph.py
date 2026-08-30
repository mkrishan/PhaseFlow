"""Build a multithreshold feature graph and measure an intervention."""

import numpy as np

from phaseflow.analysis import intervention_effect, representation_percolation_curve


rng = np.random.default_rng(17)
latent = rng.normal(size=(500, 3))
mixing = rng.normal(size=(3, 12))
activations = latent @ mixing + 0.1 * rng.normal(size=(500, 12))

curve = representation_percolation_curve(activations, np.linspace(0.2, 0.9, 8))
for row in curve:
    print(row)

readout = rng.normal(size=(12, 4))
baseline = activations @ readout
intervened_activations = activations.copy()
intervened_activations[:, 0] = 0.0
intervened = intervened_activations @ readout
print(intervention_effect(baseline, intervened))
