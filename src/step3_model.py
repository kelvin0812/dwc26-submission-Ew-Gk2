"""Step 3: the models we tried as a proxy for the simulator.

In: the 4 parameters. Out: 11 PCA scores (oil + water). kNN is just a baseline, the GP won in step 4.
"""
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 42


def make_gp(kernel_name):
    ard = np.ones(4)  # separate length-scale for each input so the GP can tell which ones matter
    base = {"matern15": Matern(ard, nu=1.5), "matern25": Matern(ard, nu=2.5), "rbf": RBF(ard)}[kernel_name]
    base.length_scale_bounds = (1e-2, 1e3)
    # simulator has no noise, the small white noise is just to keep the fit stable
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(1e-4, (1e-9, 1e-1))
    # one GP per score, PC1 is mostly porosity but the later ones depend more on perm/aquifer
    return MultiOutputRegressor(GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3,
                                                         random_state=SEED))


# settings we tried for each model, step 4 picks one with CV
CANDIDATES = {
    "kNN (baseline)": {f"k={k}": KNeighborsRegressor(k, weights="distance") for k in (2, 3, 5)},
    "Polynomial ridge": {f"degree {d}": make_pipeline(PolynomialFeatures(d), RidgeCV(np.logspace(-8, 2, 21)))
                         for d in (2, 3, 4)},
    "Gaussian Process": {k: make_gp(k) for k in ("matern15", "matern25", "rbf")},
    "LightGBM": {f"{n} leaves": MultiOutputRegressor(LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=n,
                                                                   min_child_samples=4, subsample=0.9, subsample_freq=1,
                                                                   random_state=SEED, verbose=-1))
                 for n in (4, 8)},
    "MLP (neural net)": {f"{h}": MLPRegressor(hidden_layer_sizes=h, alpha=1e-3, solver="lbfgs", max_iter=5000,
                                              random_state=SEED)
                         for h in ((32, 32), (64, 64))},
}


def with_scaled_targets(model):
    # PC1 is ~1000x bigger than PC5, without scaling the model basically ignores the small ones
    return TransformedTargetRegressor(regressor=clone(model), transformer=StandardScaler())
