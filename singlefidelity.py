import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import r2_score
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Load Data
def load_data(files, fidelities):
    """Load multiple Excel files and append a fidelity column."""
    dfs = [pd.read_excel(file, sheet_name='Sheet1').assign(fidelity=fid) for file, fid in zip(files, fidelities)]
    return pd.concat(dfs, ignore_index=True)

# File paths and corresponding fidelity levels
files = [
    'micropillar_sz75 - Copy.xlsx',
    'micropillar_sz100 - Copy.xlsx',
    'micropillar_sz125 - Copy.xlsx',
    'micropillar_sz150 - Copy.xlsx'
]
fidelities = [1, 2, 3, 4]

# Load dataset
full_df = load_data(files, fidelities)

# Define features and target variables
input_cols = ['radius (um)', 'offset (um)', 'height (um)', 'Kx (um^2)', 'Ky (um^2)', 'fidelity']
output_cols = ['kcondz', 'dPuc']

# Data normalization
scaler_x, scaler_y = StandardScaler(), StandardScaler()
X = scaler_x.fit_transform(full_df[input_cols])
y = scaler_y.fit_transform(full_df[output_cols])

# Gaussian Process Model Class
class MultiFidelityGP:
    def __init__(self):
        """Initialize a multi-fidelity Gaussian Process model."""
        self.kernel = (C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X.shape[1]),
                    length_scale_bounds=(1e-5, 1e5)) + WhiteKernel())
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-5,
                                           normalize_y=True, n_restarts_optimizer=10)

    def fit(self, X, y):
        self.gp.fit(X, y)
        return self

    def predict(self, X):
        return self.gp.predict(X)

# Train model
model = MultiFidelityGP().fit(X, y)

# Grid generation
def create_grid(n=20):
    """Generate a 3D grid with fixed features for prediction."""
    data_min, data_max = full_df[input_cols].min(), full_df[input_cols].max()
    grid = np.meshgrid(
        np.linspace(data_min['radius (um)'], data_max['radius (um)'], n),
        np.linspace(data_min['offset (um)'], data_max['offset (um)'], n),
        np.linspace(data_min['height (um)'], data_max['height (um)'], n)
    )
    grid_array = np.vstack([g.ravel() for g in grid]).T
    fixed_features = np.column_stack([
        np.ones(grid_array.shape[0]) * 1,  # lattice
        np.ones(grid_array.shape[0]) * 75, # size
        np.ones(grid_array.shape[0]) * 4   # highest fidelity
    ])
    return np.hstack([grid_array, fixed_features])

# Generate and scale prediction grid
X_grid = create_grid(15)
X_grid_scaled = scaler_x.transform(X_grid)

# Predictions
pred_y = scaler_y.inverse_transform(model.predict(X_grid_scaled))

# Pareto front extraction
def get_pareto_front(objectives):
    """Compute non-dominated Pareto front."""
    nds = NonDominatedSorting().do(-objectives, only_non_dominated_front=True)
    return objectives[nds]

# Visualization
def plot_results(true_obj, pred_obj):
    """Plot original data, predicted Pareto front, and validation results."""
    plt.figure(figsize=(15, 5))

    # Original Data
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(true_obj[:, 0], true_obj[:, 1], c=full_df['fidelity'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Fidelity Level')
    plt.xlabel("Thermal Performance: kcond")
    plt.ylabel("Hydraulic Performance: dPuc")
    plt.title("Original Data")

    # Predicted Pareto Front
    plt.subplot(1, 3, 2)
    plt.scatter(pred_obj[:, 0], pred_obj[:, 1], c='g', alpha=0.3)
    pareto = get_pareto_front(pred_obj)
    plt.scatter(pareto[:, 0], pareto[:, 1], c='r', s=50, edgecolors='k', label='Pareto Front')
    plt.xlabel("Predicted Thermal Performance: kcond")
    plt.ylabel("Predicted Hydraulic Performance: dPuc")
    plt.title("Predicted Pareto Front")
    plt.legend()

    # Validation
    plt.subplot(1, 3, 3)
    true_vals = scaler_y.inverse_transform(y)
    pred_vals = scaler_y.inverse_transform(model.predict(X))
    plt.scatter(true_vals[:, 0], pred_vals[:, 0], alpha=0.3, label="Thermal: kcond")
    plt.scatter(true_vals[:, 1], pred_vals[:, 1], alpha=0.3, label="Hydraulic: dPuc")
    plt.plot([min(true_vals[:, 0]), max(true_vals[:, 0])],
             [min(true_vals[:, 0]), max(true_vals[:, 0])], 'k--')  # Identity line
    r2_val = r2_score(true_vals, pred_vals)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Validation (R² = {r2_val:.2f})")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Execute plotting
plot_results(scaler_y.inverse_transform(y), pred_y)
