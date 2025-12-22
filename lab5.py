import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score)


# Task 5
def show_info_about_model(y_test, y_pred_test, FPR, TPR, auc, model_name=""):
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="bwr")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(f'Confusion Matrix{"" if not model_name else ": " + model_name}')

    # Сохранение матрицы ошибок в файл
    cm_filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(cm_filename, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Матрица ошибок сохранена в файл: {cm_filename}")

    # ROC-кривая
    plt.figure(figsize=(5, 5))
    plt.plot(FPR, TPR, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Random classifier (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve{"" if not model_name else ": " + model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Сохранение ROC-кривой в файл
    roc_filename = f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(roc_filename, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"ROC-кривая сохранена в файл: {roc_filename}")

    return cm_filename, roc_filename


# Task 1
def generate_data(rows_amount):
    print(f"Генерация {rows_amount} строк данных...")

    np.random.seed(42)

    # Генерация признаков
    X = np.random.randint(0, 2, size=(rows_amount, 12))
    np.savetxt("data/dataIn.txt", X, fmt="%d")
    print(f"Признаки сохранены в data/dataIn.txt")

    # Генерация меток
    Y = np.random.randint(0, 2, rows_amount)
    y_ohe = np.zeros((rows_amount, 2))
    y_ohe[np.arange(rows_amount), Y] = 1
    np.savetxt("data/dataOut.txt", y_ohe, fmt="%d")

    print(f"Метки сохранены в data/dataOut.txt")
    print(f"Размерность X: {X.shape}")
    print(f"Размерность y_ohe: {y_ohe.shape}")
    print()

    return X, y_ohe


#генерация данных
X, y = generate_data(100)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Разделение данных:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_test:  {y_test.shape}")
print()

# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Масштабирование признаков выполнено")
print()

# Преобразование меток из one-hot encoding в обычный формат
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

print(f"Метки преобразованы из one-hot encoding:")
print(f"  y_train shape: {y_train.shape}")
print(f"  y_test shape:  {y_test.shape}")
print()

#Создание и обучение модели MLP

mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(50,),  # один скрытый слой с 50 нейронами
    activation="logistic",  # сигмоидная функция активации
    solver="adam",  # алгоритм оптимизации
    random_state=42,  # для воспроизводимости
    max_iter=1000,  # максимальное количество итераций
    verbose=False  # без подробного вывода
)

mlp_classifier.fit(X_train, y_train)

#оценка модели

# Предсказания на тестовой выборке
y_pred_test = mlp_classifier.predict(X_test)
y_pred_proba = mlp_classifier.predict_proba(X_test)[:, 1]

# Расчет метрик
FPR, TPR, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred_test)

# Вывод метрик
print(f"  AUC:        {auc:.4f}")
print(f"  Accuracy:   {accuracy:.4f}")
print()

cm_file, roc_file = show_info_about_model(y_test, y_pred_test, FPR, TPR, auc, "MLP_Classifier")
