# откройте данные: ваш код здесь
import pandas as pd
import gdown
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# # подключим drive, если нужно работать напрямую с Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# или скачиваем файл по прямой ссылке через gdown:
# pip install gdown


url = 'https://drive.google.com/uc?id=18fFIiCadWQHk0_wCRWmzGXVXW32CyVFV'
output = 'emails.csv'
gdown.download(url, output, quiet=False)

# читаем датасет
data = pd.read_csv('emails.csv')
print('DATA=',data['email'] )
# рассчитайте частоты для классов : ваш код здесь

class_counts = data['label'].value_counts()  # или 'spam', если столбец так называется
print(class_counts)

# переведём в проценты для наглядности
class_percent = data['label'].value_counts(normalize=True) * 100
print(class_percent)

plt.figure(figsize=(5, 4))
plt.bar(class_counts.index.astype(str), class_counts.values, color=['cornflowerblue', 'salmon'])
plt.title('Распределение писем по классам')
plt.xlabel('Класс (0 — не спам, 1 — спам)')
plt.ylabel('Количество писем')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()


# Заменим пустые строки и строки из пробелов на NaN, затем удалим строки с пропусками
# data['email'].replace(r'^\s*$', pd.NA, regex=True, inplace=True)
data['email'] = data['email'].replace(r'^\s*$', pd.NA, regex=True)
data.dropna(subset=['email'], inplace=True)

# Проверим, что пропусков больше нет
print("Количество пропусков в колонке email:", data['email'].isna().sum())

# Переводим данные в векторный вид
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["email"])

# Определяем, сколько признаков (слов/токенов) теперь у нас в наборе данных
num_features = len(vectorizer.get_feature_names_out())
print("Количество признаков (уникальных слов):", num_features)
# print("Vector:", X)
print("Уникальные слова 10шт):",vectorizer.get_feature_names_out()[:10])
# целевая переменная (y) — это метка "спам / не спам"
y = data['label']
print("Размер матрицы признаков:", X.shape)
print("Размер вектора ответов:", y.shape)

from sklearn.model_selection import train_test_split

# делим выборку: 25% — тест, 75% — обучение
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,        # чтобы доли классов (спам / не спам) сохранялись
    random_state=42    # фиксируем случайную генерацию для воспроизводимости
)

print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)

mean_target = y.mean()
print("Среднее значение целевой переменной (доля спама):", mean_target)

# ----------
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, RocCurveDisplay

# создаём и обучаем модель
model = MultinomialNB(alpha=0.01)
model.fit(X_train, y_train)

# предсказания на тестовой выборке
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]   # вероятности для ROC‑кривой

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1‑score:", f1_score(y_test, y_pred))
print("\nКлассификационный отчёт:\n", classification_report(y_test, y_pred))
print("\nМатрица ошибок:\n", confusion_matrix(y_test, y_pred))
print("ROC‑AUC:", roc_auc_score(y_test, y_pred_proba))

import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', label='ROC‑кривая')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate (ложные срабатывания)')
plt.ylabel('True Positive Rate (чувствительность)')
plt.title('ROC‑кривая для MultinomialNB (alpha = 0.01)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

from sklearn.model_selection import cross_val_score
import numpy as np

alphas = [0.001, 0.01, 0.1, 0.5, 1, 5, 10]

for a in alphas:
    model = MultinomialNB(alpha=a)
    scores = cross_val_score(model, X, y, cv=5, scoring='f1')  # используем F1 для сравнения
    print(f"alpha={a:<6}  Среднее F1: {np.mean(scores):.4f}")