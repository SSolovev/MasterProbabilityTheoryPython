import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.model_selection import cross_val_score
import numpy as np

# ---------- 1. Загрузка данных ----------
data = pd.read_csv('dataset_diseases.csv')
print(data.head())

test_counts = data['Test'].value_counts()
print(test_counts)

status_counts = data['Status'].value_counts()
print(status_counts)

# ---------- 2. Анализ распределения классов ----------
print("\nРаспределение классов по статусу болезни:")
status_counts = data["Status"].value_counts()
print(status_counts)
print("\nВ процентах:")
class_percent = data["Status"].value_counts(normalize=True) * 100
print(status_counts)

plt.figure(figsize=(5, 4))
plt.bar(status_counts.index.astype(str), status_counts.values, color=['gold', 'lightseagreen'])
plt.title('Распределение по статусу болезни')
plt.xlabel('Статус')
plt.ylabel('Количество людей')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

print("\nРаспределение классов по статусу теста:")
test_counts = data["Test"].value_counts()
print(test_counts)
print("\nВ процентах:")
test_percent = data["Test"].value_counts(normalize=True) * 100
print(test_percent)

plt.figure(figsize=(5, 4))
plt.bar(test_counts.index.astype(str), test_counts.values, color=['blue', 'salmon'])
plt.title('Распределение по статусу теста')
plt.xlabel('Статус')
plt.ylabel('Количество людей')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# ---------- 3. Кодирование категориальных данных ----------
le_test = LabelEncoder()
le_age = LabelEncoder()
le_status = LabelEncoder()

data["Test_enc"] = le_test.fit_transform(data["Test"])
data["Age_enc"] = le_age.fit_transform(data["Age_Group"])
data["Status_enc"] = le_status.fit_transform(data["Status"])

print("\nКодировка категориальных данных:")
print("Test", dict(zip(le_test.classes_, le_test.transform(le_test.classes_))))
print("Age_Group", dict(zip(le_age.classes_, le_age.transform(le_age.classes_))))
print("Status", dict(zip(le_status.classes_, le_status.transform(le_status.classes_))))

# ---------- 4. Разделение данных ----------
X = data[["Test_enc", "Age_enc"]]
y = data["Status_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("\nРазмер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)

# ---------- 5. Обучение модели (Наивный Байес со сглаживанием Лапласа) ----------
alpha_value = 1.0
model = CategoricalNB(alpha=alpha_value)
model.fit(X_train, y_train)

# ---------- 6. Предсказания ----------
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ---------- 7. Метрики ----------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n=== Оценка качества модели ===")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall (чувствительность): {rec:.3f}")
print(f"F1-score:  {f1:.3f}")

print("\nКлассификационный отчёт:\n", classification_report(y_test, y_pred, target_names=le_status.classes_))
print("Матрица ошибок:\n", cm)
false_positives = cm[0, 1]
true_positives = cm[1, 1]
print(f"\nЛожные срабатывания (FP): {false_positives}")
print(f"Истинные срабатывания (TP): {true_positives}")
print(f"ROC‑AUC: {roc_auc:.3f}")

# ---------- 8. ROC‑кривая ----------
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC‑кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate (ложные срабатывания)')
plt.ylabel('True Positive Rate (чувствительность)')
plt.title('ROC‑кривая для CategoricalNB (Laplace smoothing)')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# ---------- 9. Проверка устойчивости через кросс‑валидацию ----------
alphas = [0.01, 0.1, 0.5, 1, 5]
print("\nСредние F1‑результаты при различных alpha:")
for a in alphas:
    model = CategoricalNB(alpha=a)
    scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    print(f"alpha={a:<5} → Среднее F1: {np.mean(scores):.4f}")