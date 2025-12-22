import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import label_binarize

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

# Создание модели случайного леса с OOB оценкой
rf_clf = RandomForestClassifier(
    n_estimators=100,        # количество деревьев
    max_depth=10,            # ограничение глубины
    min_samples_split=5,     # минимальное число объектов для разделения
    min_samples_leaf=5,      # минимальное число объектов в листе
    oob_score=True,          # включить OOB оценку
    random_state=42,
    n_jobs=-1                # использовать все ядра
)

# Обучение модели
rf_clf.fit(X_train, y_train)

# Предсказания на тестовой выборке
y_pred = rf_clf.predict(X_test)
y_proba = rf_clf.predict_proba(X_test)[:, 1]

# Метрики качества
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=" * 50)
print("СЛУЧАЙНЫЙ ЛЕС: МЕТРИКИ КАЧЕСТВА")
print("=" * 50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1:.4f}")
print()

# OOB оценка
print("=" * 50)
print("ОЦЕНКА ЧЕРЕЗ OOB ДАННЫЕ")
print("=" * 50)
print(f"OOB Accuracy: {rf_clf.oob_score_:.4f}")
print(f"OOB Error:    {1 - rf_clf.oob_score_:.4f}")
print()

# ROC кривая
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print("=" * 50)
print("ОЦЕНКА ПО ROC-КРИВОЙ")
print("=" * 50)
print(f"ROC-AUC: {roc_auc:.4f}")

# Визуализация ROC кривой
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'Random Forest ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random classifier (AUC = 0.5)')
plt.plot([0, 0, 1], [0, 1, 1], color='green', lw=1, linestyle=':',
         alpha=0.5, label='Perfect classifier')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
plt.title('ROC-кривая: Случайный лес', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig('random_forest_roc_curve.png', dpi=100, bbox_inches='tight')
plt.close()

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix: Random Forest')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('random_forest_confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.close()

# Важность признаков
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['feature'], feature_importances['importance'])
plt.xlabel('Важность признака')
plt.title('Важность признаков в случайном лесе')
plt.gca().invert_yaxis()
plt.savefig('random_forest_feature_importance.png', dpi=100, bbox_inches='tight')
plt.close()

print("=" * 50)
print("ВАЖНОСТЬ ПРИЗНАКОВ")
print("=" * 50)
print(feature_importances.to_string(index=False))
print()
print("Графики сохранены:")
print("- random_forest_roc_curve.png")
print("- random_forest_confusion_matrix.png")
print("- random_forest_feature_importance.png")