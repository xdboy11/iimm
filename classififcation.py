import pandas as pd
import numpy as np
import matplotlib
# Используем неинтерактивный backend для избежания ошибок Tkinter
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize

# Загрузка данных
df = pd.read_csv('data/train.lab1.new.csv')

# 1. ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАССИФИКАЦИИ
# Целевая переменная — Survived
y = df['Survived']

# Признаки — удаляем нерелевантные колонки
X = df.drop(columns=[
    'Survived',          # целевая переменная
    'PassengerId',       # ID пассажира
    'Name',              # имя
    'Ticket',            # номер билета
    # 'Age'              # можно оставить или удалить, решим оставить
])

# Преобразуем булевы колонки в int
X = X.astype(int)

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ
clf = DecisionTreeClassifier(
    max_depth=4,          # ограничиваем глубину для избежания переобучения
    min_samples_split=5,  # минимальное число объектов для разделения
    min_samples_leaf=5,   # минимальное число объектов в листе
    random_state=42
)

clf.fit(X_train, y_train)

# 3. ПРЕДСКАЗАНИЯ И ОЦЕНКА МОДЕЛИ
# Предсказанные метки классов
y_pred = clf.predict(X_test)

# Вероятности для положительного класса (Survived=1)
# Для деревьев решений используем predict_proba
y_proba = clf.predict_proba(X_test)[:, 1]

# Базовые метрики классификации
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=" * 50)
print("МОДЕЛЬ КЛАССИФИКАЦИИ: ДЕРЕВО РЕШЕНИЙ")
print("=" * 50)
print(f"Accuracy (точность):  {accuracy:.4f}")
print(f"Precision (точность): {precision:.4f}")
print(f"Recall (полнота):     {recall:.4f}")
print(f"F1-score:             {f1:.4f}")
print()

# 4. ПОСТРОЕНИЕ ROC-КРИВОЙ
# Вычисляем FPR, TPR и пороги
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print("ОЦЕНКА ПО ROC-КРИВОЙ:")
print(f"ROC-AUC: {roc_auc:.4f}")
print("(Чем ближе к 1, тем лучше модель разделяет классы)")
print()

# 5. ВИЗУАЛИЗАЦИЯ ROC-КРИВОЙ
plt.figure(figsize=(10, 8))

# ROC-кривая
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.4f})')

# Диагональ случайной модели
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random classifier (AUC = 0.5)')

# Настройка графика
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
plt.title('ROC-кривая для классификации "Survived"', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)

# Добавляем сетку и аннотацию для идеального классификатора
plt.plot([0, 0, 1], [0, 1, 1], color='green', lw=1, linestyle=':',
         alpha=0.5, label='Perfect classifier')

# Сохраняем ROC-кривую
plt.savefig('roc_curve_classification.png', dpi=100, bbox_inches='tight')
plt.close()

print("ROC-кривая сохранена в файл: roc_curve_classification.png")

# 6. ВЫВОД ИНФОРМАЦИИ О МОДЕЛИ
print("\n" + "=" * 50)
print("ИНФОРМАЦИЯ О МОДЕЛИ:")
print("=" * 50)
print(f"Число признаков: {X.shape[1]}")
print(f"Названия признаков: {list(X.columns)}")
print(f"Размер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")

# 7. ДОПОЛНИТЕЛЬНО: ТОЧКИ НА ROC-КРИВОЙ
print("\n" + "=" * 50)
print("ПРИМЕРЫ ТОЧЕК НА ROC-КРИВОЙ:")
print("=" * 50)
print("Порог |   FPR   |   TPR   ")
print("-" * 30)

# Выводим несколько точек с разными порогами
for i in range(0, len(thresholds), max(1, len(thresholds)//10)):
    if i < len(thresholds):
        print(f"{thresholds[i]:.3f} | {fpr[i]:.4f}  | {tpr[i]:.4f}")

# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В ФАЙЛ
results = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'n_features': X.shape[1],
    'train_size': X_train.shape[0],
    'test_size': X_test.shape[0]
}

results_df = pd.DataFrame([results])
results_df.to_csv('classification_results.csv', index=False)
print("\nРезультаты сохранены в файл: classification_results.csv")

# 9. ВИЗУАЛИЗАЦИЯ ДЕРЕВА (ПЕРВЫЕ 2 УРОВНЯ)
from sklearn.tree import plot_tree

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
#plt.savefig('decision_tree_classification.png', dpi=100, bbox_inches='tight')
plt.close()

print("Визуализация дерева сохранена в файл: decision_tree_classification.png")

print("\n" + "=" * 50)
print("АНАЛИЗ РЕЗУЛЬТАТОВ:")
print("=" * 50)
