from sklearn.datasets import fetch_california_housing
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


import ssl
import urllib.request


ssl._create_default_https_context = ssl._create_unverified_context

housing = fetch_california_housing()
x_train, x_test, y_train, y_test = train_test_split(
    housing.data, housing.target, random_state=42)

mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], early_stopping=True, verbose=True, random_state=42)

pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(x_train, y_train)


mlp_reg.best_validation_score_
y_pred = pipeline.predict(x_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(rmse)
