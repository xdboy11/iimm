import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


pd.set_option('display.max_columns', None)
df = pd.read_csv("./data/processed_train.csv")

df2 = df.drop(['Cabin', 'Name'], axis=1)

X = df2.drop('Age', axis=1)
y = df2['Age']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_train = linear_model.predict(X_train)
y_pred_test = linear_model.predict(X_test)

print("-----Оценка-----")

mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)

print(f"\nОбучающая выборка:")
print(f"  MSE: {mse_train:.4f}")
print(f"  MAE: {mae_train:.4f}")

print(f"Тестовая выборка:")
print(f"  MSE: {mse_test:.4f}")
print(f"  RMSE: {rmse_test:.4f}")
print(f"  MAE: {mae_test:.4f}")