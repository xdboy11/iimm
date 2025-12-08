import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize

df = pd.read_csv('data/train.lab1.new.csv')

y = df['Survived']

X = df.drop(columns=[
    'Survived',
    'PassengerId',
    'Name',
    'Ticket'
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(
    max_depth=4,          # ограничение глубины
    min_samples_split=5,  # минимальное число объектов для разделения
    min_samples_leaf=5,   # минимальное число объектов в листе
    random_state=42
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=" * 50)
print("МОДЕЛЬ КЛАССИФИКАЦИИ: ДЕРЕВО РЕШЕНИЙ")
print("=" * 50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1 score:             {f1:.4f}")
print()

# вычисление FPR, TPR и трешхолдов
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print("ОЦЕНКА ПО ROC-КРИВОЙ:")
print(f"ROC-AUC: {roc_auc:.4f}")

plt.figure(figsize=(10, 8))

# ROC кривая
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.4f})')

# диагональ случайной модели
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random classifier (AUC = 0.5)')

# настройка графика
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
plt.title('ROC-кривая для классификации "Survived"', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)

plt.plot([0, 0, 1], [0, 1, 1], color='green', lw=1, linestyle=':',
         alpha=0.5, label='Perfect classifier')

plt.savefig('roc_curve_classification.png', dpi=100, bbox_inches='tight')
plt.close()

print("ROC кривая сохранена в файл: roc_curve_classification.png")


plt.figure(figsize=(20, 12))
plot_tree(clf,
          feature_names=X.columns,
          class_names=['Not Survived', 'Survived'],
          filled=True,
          max_depth=2,
          fontsize=10,
          impurity=False,
          rounded=True)
plt.title("Дерево решений для классификации (первые 2 уровня)", fontsize=14)
plt.savefig('decision_tree_classification.png', dpi=100, bbox_inches='tight')
plt.close()

print("Визуализация дерева сохранена в файл: decision_tree_classification.png")