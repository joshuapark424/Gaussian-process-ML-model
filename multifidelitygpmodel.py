import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Load Data
def load_data(files, fidelities):
    dfs = []
    for file, fid in zip(files, fidelities):
        df = pd.read_excel(file, sheet_name='Sheet1')
        df['fidelity'] = fid
        dfs.append(df)
    return pd.concat(dfs)

# Remember to store these four files to ./content in colab
files = [
    'micropillar_sz75 - Copy.xlsx',
    'micropillar_sz100 - Copy.xlsx',
    'micropillar_sz125 - Copy.xlsx',
    'micropillar_sz150 - Copy.xlsx'
]

fidelities = [1, 2, 3, 4]

full_df = load_data(files, fidelities)

# Define X and y
input_cols = ['radius (um)', 'offset (um)', 'height (um)', 'Kx (um^2)', 'Ky (um^2)', 'fidelity']
output_cols = ['kcondz', 'dPuc']

# Normalization
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X = scaler_x.fit_transform(full_df[input_cols])
y = scaler_y.fit_transform(full_df[output_cols])

# GP Model Construction
class MultiFidelityGP:
    def __init__(self):
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=np.ones(X.shape[1]),
            length_scale_bounds=(1e-5, 1e5)) + WhiteKernel()
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-5,
            normalize_y=True,
            n_restarts_optimizer=10
        )

    def fit(self, X, y):
        self.gp.fit(X, y)
        return self

    def predict(self, X):
        return self.gp.predict(X)

model = MultiFidelityGP()
model.fit(X, y)

# Grid generation
def create_grid(n=20):
    # Range of training set
    data_min = full_df[input_cols].min()
    data_max = full_df[input_cols].max()

    # Generate 3D grid (radius, offset, height)
    radius = np.linspace(data_min['radius (um)'], data_max['radius (um)'], n)
    offset = np.linspace(data_min['offset (um)'], data_max['offset (um)'], n)
    height = np.linspace(data_min['height (um)'], data_max['height (um)'], n)

    grid = np.meshgrid(radius, offset, height)
    grid_array = np.vstack([g.ravel() for g in grid]).T

    # Add fixed features
    fixed_features = np.column_stack([
        np.ones(grid_array.shape[0]) * 1,  # lattice=1
        np.ones(grid_array.shape[0]) * 75, # size=75
        np.ones(grid_array.shape[0]) * 4   # fidelity=4 (highest fidelity)
    ])

    return np.hstack([grid_array, fixed_features])

# Generate prediction grid
X_grid = create_grid(15)  # Reduce grid density for speed
X_grid_scaled = scaler_x.transform(X_grid)

# Make predictions
pred_y = scaler_y.inverse_transform(model.predict(X_grid_scaled))

# Visualization
def get_pareto_front(objectives):
    F = -objectives
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return objectives[nds]

def plot_results(true_obj, pred_obj):
    
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.scatter(true_obj[:,0], true_obj[:,1], c=full_df['fidelity'], cmap='viridis', alpha=0.6)
    plt.colorbar(label='Fidelity Level')
    plt.xlabel("Thermal Performance: kcond")
    plt.ylabel("Hydraulic Performance: dPuc")
    plt.title("Original Data Pareto Front")

    plt.subplot(1,3,2)
    plt.scatter(pred_obj[:,0], pred_obj[:,1], c='g', alpha=0.3)
    pareto = get_pareto_front(pred_obj)
    plt.scatter(pareto[:,0], pareto[:,1], c='r', s=50, edgecolors='k')
    plt.xlabel("Predicted Thermal Performance: kcond")
    plt.ylabel("Predicted Hydraulic Performance: dPuc")
    plt.title("Predicted Pareto Front")

    plt.subplot(1,3,3)
    plt.scatter(scaler_y.inverse_transform(y)[:,0], scaler_y.inverse_transform(model.predict(X))[:,0], alpha=0.3, label="Thermal: kcond")
    plt.scatter(scaler_y.inverse_transform(y)[:,1], scaler_y.inverse_transform(model.predict(X))[:,1], alpha=0.3, label="Hydraulic: dPuc")
    plt.plot([0, 0], [250, 250], 'k--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Validation Performance")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_results(scaler_y.inverse_transform(y), pred_y)