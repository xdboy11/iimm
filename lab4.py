import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Загрузка данных
df = pd.read_csv('data/train.lab1.new.csv')

# Целевая переменная и признаки
y = df['Survived']
X = df.drop(columns=[
    'Survived',
    'PassengerId',
    'Name',
    'Ticket'
])

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# random forest
print("random forest")

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]

# Метрики качества
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"Accuracy:  {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall:    {recall_rf:.4f}")
print(f"F1 score:  {f1_rf:.4f}")
print()

# OOB оценка
print(f"OOB Accuracy: {rf_clf.oob_score_:.4f}")
print(f"OOB Error:    {1 - rf_clf.oob_score_:.4f}")

# ROC кривая для Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
print(f"ROC-AUC: {roc_auc_rf:.4f}")
print()

# ada bust
print("AdaBoost")

ada_clf = AdaBoostClassifier(
    n_estimators=50,
    random_state=42
)

ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_test)
y_proba_ada = ada_clf.predict_proba(X_test)[:, 1]

# Метрики качества для AdaBoost
accuracy_ada = accuracy_score(y_test, y_pred_ada)
precision_ada = precision_score(y_test, y_pred_ada)
recall_ada = recall_score(y_test, y_pred_ada)
f1_ada = f1_score(y_test, y_pred_ada)

print(f"Accuracy:  {accuracy_ada:.4f}")
print(f"Precision: {precision_ada:.4f}")
print(f"Recall:    {recall_ada:.4f}")
print(f"F1 score:  {f1_ada:.4f}")
print()

# ROC кривая для AdaBoost
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_proba_ada)
roc_auc_ada = auc(fpr_ada, tpr_ada)
print(f"ROC-AUC: {roc_auc_ada:.4f}")
print()

#градиентный бустинг
print("градиентный бустинг")

gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

gb_clf.fit(X_train, y_train)
y_pred_gb = gb_clf.predict(X_test)
y_proba_gb = gb_clf.predict_proba(X_test)[:, 1]

# Метрики качества
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)

print(f"Accuracy:  {accuracy_gb:.4f}")
print(f"Precision: {precision_gb:.4f}")
print(f"Recall:    {recall_gb:.4f}")
print(f"F1 score:  {f1_gb:.4f}")
print()

# ROC кривая
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)
print(f"ROC-AUC: {roc_auc_gb:.4f}")
print()

#Roc кривая
plt.figure(figsize=(10, 8))

# ROC кривые для всех моделей на одном графике
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2,
         label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
plt.plot(fpr_ada, tpr_ada, color='blue', lw=2,
         label=f'AdaBoost (AUC = {roc_auc_ada:.4f})')
plt.plot(fpr_gb, tpr_gb, color='green', lw=2,
         label=f'Gradient Boosting (AUC = {roc_auc_gb:.4f})')

# Диагональ случайной модели
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random classifier (AUC = 0.5)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
plt.title('ROC-кривые: Сравнение ансамблевых методов', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig('all_models_roc_curve.png', dpi=100, bbox_inches='tight')
plt.close()

print("ROC-кривая для всех моделей сохранена в 'all_models_roc_curve.png'")
print()

#матрицы ошибок
models = [('Random Forest', y_pred_rf),
          ('AdaBoost', y_pred_ada),
          ('Gradient Boosting', y_pred_gb)]

for model_name, y_pred in models:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='bwr',
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png',
                dpi=100, bbox_inches='tight')
    plt.close()
