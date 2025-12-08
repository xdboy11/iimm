import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('data/processed_data.csv')

df_reg = df.drop(columns=['id_number'])

y = df_reg['charges']

X = df_reg.drop(columns=['charges'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = DecisionTreeRegressor(max_depth=4, random_state=42)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²:  {r2:.4f}")

plt.figure(figsize=(20,10))
plot_tree(regressor, feature_names=X.columns, filled=True, max_depth=2, fontsize=10)
plt.title("Дерево решений для регрессии (charges)")
plt.savefig('tree_regression.png', dpi=100)
plt.close()

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Реальные значения charges')
plt.ylabel('Предсказанные значения charges')
plt.title('Реальные vs Предсказанныgе значения')
plt.grid(True)
plt.savefig('regression_scatter.png', dpi=100)
plt.close()

print("Графики сохранены в tree_regression.png и regression_scatter.png")