import os
import numpy as np
import pickle
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras

# ------------------------------
# Ensure models folder exists
# ------------------------------
os.makedirs("models", exist_ok=True)

# ------------------------------
# Load California Housing Dataset
# ------------------------------
data = fetch_california_housing()
X, y = data.data, data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Linear Regression
# ------------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

print("Linear Regression R²:", r2_score(y_test, y_pred_lin))
print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred_lin))
print("Linear Regression RMSE:", mean_squared_error(y_test, y_pred_lin)**0.5)

joblib.dump(lin_model, "models/linear.pkl")

# ------------------------------
# Decision Tree
# ------------------------------
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("Decision Tree R²:", r2_score(y_test, y_pred_dt))
print("Decision Tree MAE:", mean_absolute_error(y_test, y_pred_dt))
print("Decision Tree RMSE:", mean_squared_error(y_test, y_pred_dt)**0.5)

joblib.dump(dt_model, "models/decision_tree.pkl")

# ------------------------------
# Random Forest
# ------------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))
print("Random Forest RMSE:", mean_squared_error(y_test, y_pred_rf)**0.5)

joblib.dump(rf_model, "models/random_forest.pkl")

# ------------------------------
# Gradient Boosting
# ------------------------------
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

print("Gradient Boosting R²:", r2_score(y_test, y_pred_gb))
print("Gradient Boosting MAE:", mean_absolute_error(y_test, y_pred_gb))
print("Gradient Boosting RMSE:", mean_squared_error(y_test, y_pred_gb)**0.5)

joblib.dump(gb_model, "models/gradient_boosting.pkl")

# ------------------------------
# Neural Network
# ------------------------------
nn_model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1)
])

nn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 loss="mse",
                 metrics=["mae"])

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = nn_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop]
)

nn_pred = nn_model.predict(X_test).flatten()
print("Neural Network R²:", r2_score(y_test, nn_pred))
print("Neural Network MAE:", mean_absolute_error(y_test, nn_pred))
print("Neural Network RMSE:", mean_squared_error(y_test, nn_pred)**0.5)

nn_model.save("models/neural_network.keras")
