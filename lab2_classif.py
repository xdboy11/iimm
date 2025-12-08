import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, precision_score,
                           recall_score, f1_score)

df = pd.read_csv("./data/processed_train.csv")
df2 = df.drop(['Cabin', 'Name'], axis=1)

X = df2.drop('Transported', axis=1)
y = df2['Transported']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logreg_model = LogisticRegression(max_iter=500)
logreg_model.fit(X_train, y_train)

y_pred = logreg_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Точность: {accuracy:.4f} ({accuracy:.1%})")

print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
