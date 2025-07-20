import numpy as np
import pandas as pd
import pymoo  # Confirm version
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import r2_score, mean_squared_error

def load_data(files, assign_fidelity=False, fidelities=None):
    dfs = []
    for i, file in enumerate(files):
        df = pd.read_csv(file)
        if assign_fidelity and fidelities is not None:
            df['fidelity'] = fidelities[i]
        dfs.append(df)
    return pd.concat(dfs)

files = [
    'stock_data_1.csv',
    'stock_data_2.csv',
    'stock_data_3.csv',
    'stock_data_4.csv' #however many files you want to include for stock data
]
fidelities = [1, 2, 3, 4]

single_df = load_data(files, assign_fidelity=False)
multi_df = load_data(files, assign_fidelity=True, fidelities=fidelities)

def prepare_data(df, include_fidelity=False):
    input_cols = ['open', 'high', 'low', 'close', 'volume', 'moving_avg_10', 'rsi'] # Adjust these columns based on your dataset
    if 'fidelity' in df.columns and include_fidelity:
        input_cols += ['fidelity']
    if include_fidelity:
        input_cols += ['fidelity']
    if 'fidelity' in df.columns:
        df = df.drop(columns=['fidelity'])
    df = df.dropna(subset=input_cols + ['future_return'])  # Ensures no NaN values in input and output columns
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1  # Example target variable, adjust as needed
    df = df.dropna(subset=['future_return'])  # Drop rows where future_return is NaN   

    output_cols = ['future_return']

    X = df[input_cols].values
    y = df[output_cols].values

    return X, y, input_cols

X_single, y_single, single_cols = prepare_data(single_df, include_fidelity=False)
X_single_train, X_single_test, y_single_train, y_single_test = train_test_split(
    X_single, y_single, test_size=0.2, random_state=42
)

X_multi, y_multi, multi_cols = prepare_data(multi_df, include_fidelity=True)
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

scaler_single_x = StandardScaler().fit(X_single_train)
scaler_single_y = StandardScaler().fit(y_single_train)

scaler_multi_x = StandardScaler().fit(X_multi_train)
scaler_multi_y = StandardScaler().fit(y_multi_train)

class GPModel:
    def __init__(self, input_dim):
        self.kernel = C(1.0) * RBF(length_scale=np.ones(input_dim)) + WhiteKernel()
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-5,
            n_restarts_optimizer=10
        )

    def fit(self, X, y):
        self.gp.fit(X, y)
        return self

    def predict(self, X, return_std=False):
        return self.gp.predict(X, return_std=return_std)

single_gp = GPModel(X_single_train.shape[1]).fit(
    scaler_single_x.transform(X_single_train),
    scaler_single_y.transform(y_single_train)
)

multi_gp = GPModel(X_multi_train.shape[1]).fit(
    scaler_multi_x.transform(X_multi_train),
    scaler_multi_y.transform(y_multi_train)
)

def cross_validate(model, X, y, scaler_x, scaler_y, n_splits=5):
    kf = KFold(n_splits=n_splits)
    r2_scores = []
    rmse_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_scaled = scaler_x.transform(X_train)
        y_train_scaled = scaler_y.transform(y_train)
        X_test_scaled = scaler_x.transform(X_test)

        model.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(scaler_y.transform(y_test))

        r2_scores.append(r2_score(y_true, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_true, y_pred)))

    return np.mean(r2_scores), np.mean(rmse_scores)

single_r2, single_rmse = cross_validate(
    GPModel(X_single_train.shape[1]),
    X_single, y_single, scaler_single_x, scaler_single_y
)
multi_r2, multi_rmse = cross_validate(
    GPModel(X_multi_train.shape[1]),
    X_multi, y_multi, scaler_multi_x, scaler_multi_y
)

print("Cross-Validation Results:")
print(f"{'Model':<15} | {'R²':^10} | {'RMSE':^10}")
print("-"*35)
print(f"{'Single-Fidelity':<15} | {single_r2:>10.4f} | {single_rmse:>10.4f}")
print(f"{'Multi-Fidelity':<15} | {multi_r2:>10.4f} | {multi_rmse:>10.4f}")

def plot_feature_importance(model, feature_names):
    length_scales = model.gp.kernel_.k1.k2.length_scale
    plt.barh(feature_names, 1 / length_scales)
    plt.xlabel('Feature Importance (1 / Length Scale)')
    plt.title('Feature Importance Analysis')
    plt.show()

plot_feature_importance(multi_gp, multi_cols)

def plot_uncertainty(model, scaler_x, scaler_y, X_test, y_test):
    X_test_scaled = scaler_x.transform(X_test)
    y_pred_scaled, y_std_scaled = model.predict(X_test_scaled, return_std=True)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(scaler_y.transform(y_test))

    plt.figure(figsize=(8, 5))
    plt.errorbar(y_true[:, 0], y_pred[:, 0], yerr=y_std_scaled[:, 0], fmt='o', alpha=0.5, label='future_return')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', label='Ideal')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Uncertainty Visualization')
    plt.xlim(y_true.min(), y_true.max())
    plt.ylim(y_true.min(), y_true.max())
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(np.arange(y_true.min(), y_true.max(), step=0.1))
    plt.yticks(np.arange(y_true.min(), y_true.max(), step=0.1))
    plt.axhline(0, color='gray', lw=0.5, ls='--')
    plt.axvline(0, color='gray', lw=0.5, ls='--')
    plt.title('Uncertainty in Predictions')
    plt.xlabel('True Future Return')
    plt.ylabel('Predicted Future Return')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_uncertainty(multi_gp, scaler_multi_x, scaler_multi_y, X_multi_test, y_multi_test)
