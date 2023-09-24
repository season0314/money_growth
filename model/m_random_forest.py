import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



data = np.load("data/saved.npy", allow_pickle=True).item()
X = data["feature"]
y = data["target"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)


y_pred = reg.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

