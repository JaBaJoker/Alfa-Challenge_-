import numpy as np # для линейной алгебры
import pandas as pd # для обработки данных и работы с CSV файлами

# Вывод списка всех файлов в директории входных данных
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Загрузка данных
train_df = pd.read_parquet('/kaggle/input/alfa-challenge/train.pa')
df_transactions = pd.read_parquet('/kaggle/input/alfa-challenge/df_transaction.pa')

# Объединение данных по клиентам
merged_data = pd.merge(df_transactions, train_df, on='client_num')

# Преобразование данных
merged_data['date_time'] = pd.to_datetime(merged_data['date_time'])
merged_data['day'] = merged_data['date_time'].dt.day
merged_data['month'] = merged_data['date_time'].dt.month
merged_data['year'] = merged_data['date_time'].dt.year

# Вычисление агрегированных характеристик для каждого клиента
client_features = merged_data.groupby('client_num').agg({
    'amount': ['sum', 'mean', 'max', 'min', 'std'],
    'day': ['min', 'max'],
    'month': ['min', 'max'],
    'year': ['min', 'max']
}).reset_index()

# Переименование столбцов
client_features.columns = ['client_num', 'total_amount', 'avg_amount', 'max_amount', 'min_amount', 'std_amount', 
                          'min_day', 'max_day', 'min_month', 'max_month', 'min_year', 'max_year']

# Объединение с целевой переменной
final_data = pd.merge(client_features, train_df[['client_num', 'target']], on='client_num')

# Разделение данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
X = final_data.drop(['client_num', 'target'], axis=1)
y = final_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Построение модели
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Оценка модели
y_pred = model.predict(X_test_scaled)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Оценка WMAE
wmae = 0
for i in range(len(y_test)):
    wmae += abs(y_test.iloc[i] - y_pred[i]) * (y_test.iloc[i] + 1)
wmae /= len(y_test)
print("WMAE:", wmae)

# Предсказание результатов для тестовой выборки
test_pred = model.predict(scaler.transform(final_data.drop(['client_num', 'target'], axis=1)))
submission_df = pd.DataFrame({'client_num': final_data['client_num'], 'target': test_pred})
submission_df.to_csv('submission.csv', index=False)