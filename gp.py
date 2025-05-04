import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import r2_score
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# 1. Generate sample data (replace with your actual data)
def generate_data(n_samples=100, noise=0.1):
    X = np.random.rand(n_samples, 2)
    f1 = X[:, 0] + X[:, 1] + np.random.normal(0, noise, n_samples)  # Thermal performance
    f2 = -X[:, 0] + X[:, 1] + np.random.normal(0, noise, n_samples)  # Hydraulic performance
    return X, np.vstack((f1, f2)).T

X, y = generate_data()

# 2. Data preprocessing
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled = x_scaler.transform(X_val)

y1_scaler = StandardScaler()
y2_scaler = StandardScaler()

y_train_f1 = y1_scaler.fit_transform(y_train[:, 0].reshape(-1, 1))
y_train_f2 = y2_scaler.fit_transform(y_train[:, 1].reshape(-1, 1))

# 3. Train Gaussian Process model
kernel = RBF() + WhiteKernel()
gp_f1 = GaussianProcessRegressor(kernel=kernel, random_state=0)
gp_f2 = GaussianProcessRegressor(kernel=kernel, random_state=0)

gp_f1.fit(X_train_scaled, y_train_f1)
gp_f2.fit(X_train_scaled, y_train_f2)

# 4. Model validation
y_val_f1_pred = y1_scaler.inverse_transform(gp_f1.predict(X_val_scaled).reshape(-1, 1))
y_val_f2_pred = y2_scaler.inverse_transform(gp_f2.predict(X_val_scaled).reshape(-1, 1))

r2_f1 = r2_score(y_val[:, 0], y_val_f1_pred)
r2_f2 = r2_score(y_val[:, 1], y_val_f2_pred)
print(f"Validation R² - Thermal: {r2_f1:.3f}, Hydraulic: {r2_f2:.3f}")

def create_grid(n=50):
    x = np.linspace(0, 1, n)
    X1, X2 = np.meshgrid(x, x)
    return np.vstack([X1.ravel(), X2.ravel()]).T

X_grid = create_grid(50)
X_grid_scaled = x_scaler.transform(X_grid)

f1_pred = y1_scaler.inverse_transform(gp_f1.predict(X_grid_scaled).reshape(-1, 1))
f2_pred = y2_scaler.inverse_transform(gp_f2.predict(X_grid_scaled).reshape(-1, 1))
predictions = np.hstack([f1_pred, f2_pred])

# 6. Calculate Pareto front
def get_pareto_front(objectives):
    F = -objectives  # Convert to minimization problem
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return objectives[nds]

original_pareto = get_pareto_front(y)
predicted_pareto = get_pareto_front(predictions)

# 7. Visualization
plt.figure(figsize=(15, 5))


# Fit a line of best fit for the original Pareto front
pareto_fit = np.polyfit(original_pareto[:, 0], original_pareto[:, 1], 1)  # Linear fit (y = mx + b)
pareto_line_x = np.linspace(min(original_pareto[:, 0]), max(original_pareto[:, 0]), 100)
pareto_line_y = np.polyval(pareto_fit, pareto_line_x)

# Compute R² value for the original Pareto front
pareto_y_pred = np.polyval(pareto_fit, original_pareto[:, 0])
r2_pareto = r2_score(original_pareto[:, 1], pareto_y_pred)

# Fit a line of best fit for the predicted Pareto front
predicted_pareto_fit = np.polyfit(predicted_pareto[:, 0], predicted_pareto[:, 1], 1)
predicted_pareto_line_x = np.linspace(min(predicted_pareto[:, 0]), max(predicted_pareto[:, 0]), 100)
predicted_pareto_line_y = np.polyval(predicted_pareto_fit, predicted_pareto_line_x)

# Compute R² value for the predicted Pareto front
predicted_pareto_y_pred = np.polyval(predicted_pareto_fit, predicted_pareto[:, 0])
r2_predicted_pareto = r2_score(predicted_pareto[:, 1], predicted_pareto_y_pred)

# Original data and Pareto front with best-fit line and R² value
plt.subplot(1, 3, 1)
plt.scatter(y[:, 0], y[:, 1], c='b', alpha=0.5, label="Original Data")
plt.scatter(original_pareto[:, 0], original_pareto[:, 1], c='r', label="Pareto Front")
plt.plot(pareto_line_x, pareto_line_y, 'k--', label="Best Fit Line")  # Add best fit line
plt.xlabel("Thermal Performance")
plt.ylabel("Hydraulic Performance")
plt.title(f"Original Data Pareto Front\nR² = {r2_pareto:.3f}")  # Display R² value in the title
plt.legend()

# Prediction results and Pareto front with best-fit line and R² value
plt.subplot(1, 3, 2)
plt.scatter(predictions[:, 0], predictions[:, 1], c='g', alpha=0.3, label="Predictions")
plt.scatter(predicted_pareto[:, 0], predicted_pareto[:, 1], c='r', label="Predicted Pareto")
plt.plot(predicted_pareto_line_x, predicted_pareto_line_y, 'k--', label="Best Fit Line")  # Add best fit line
plt.xlabel("Thermal Performance")
plt.ylabel("Hydraulic Performance")
plt.title(f"Predicted Pareto Front\nR² = {r2_predicted_pareto:.3f}")  # Display R² value in the title
plt.legend()

# Validation set prediction performance
plt.subplot(1, 3, 3)
plt.scatter(y_val[:, 0], y_val_f1_pred, c='b', alpha=0.5, label="Thermal")
plt.scatter(y_val[:, 1], y_val_f2_pred, c='g', alpha=0.5, label="Hydraulic")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Validation Performance")
plt.legend()


plt.tight_layout()
plt.show()
