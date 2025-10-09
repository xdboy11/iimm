import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

pd.set_option('display.max_columns', None)
df = pd.read_csv("./data/train.csv")


print(df.head(10)) # вывод первых 10 элементов датасета

nan_matrix_sum_before = df.isnull().sum()

Age_median = df['Age'].median()
RoomService_median = df['RoomService'].median()

median_values = {
    'Age': df['Age'].median(),
    'RoomService': df['RoomService'].median(),
    'FoodCourt': df['FoodCourt'].median(),
    'ShoppingMall': df['ShoppingMall'].median(),
    'Spa': df['Spa'].median(),
    'VRDeck': df['VRDeck'].median()
}
mode_values = {
    'HomePlanet': df['HomePlanet'].mode()[0],
    'CryoSleep': df['CryoSleep'].mode()[0],
    'Cabin': df['Cabin'].mode()[0],
    'Destination': df['Destination'].mode()[0],
    'VIP': df['VIP'].mode()[0],
    'Name': "unknown"
}

df.fillna(median_values, inplace = True)
df.fillna(mode_values, inplace = True)

nan_matrix_sum_after = df.isnull().sum()

print("кол-во пропусков ДО  ПОСЛЕ")
for col in df.columns:
    print(f"{col:<15} {nan_matrix_sum_before[col]:>3} -> {nan_matrix_sum_after[col]}")

#Создание списка для Z-оценки
expense_columns = list(median_values.keys())
expense_columns.remove('Age')

standard_scaler = StandardScaler()
df[expense_columns] = standard_scaler.fit_transform(df[expense_columns])

#Возраст нормализуем мин-макс масштабированием
age_scaler = MinMaxScaler()
df[['Age']] = age_scaler.fit_transform(df[['Age']])

#Категориальные колонки с малым количеством вариантов преобразуем с помощью OHE
OHE_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
df_final = pd.get_dummies(df, columns=OHE_columns, drop_first=True)

#У Cabin много значений, преобразуем последовательностью чисел
Cabin_encoder = LabelEncoder()
df_final['Cabin_encoded'] = Cabin_encoder.fit_transform(df['Cabin'])

#Колонку PassengerId не преобразуем по очевидным причинам
#Колонка Name, теоретически не должно иметь никакой кореляции с тем, пропал ли человек, поэтому ее также не преобразуем

#df_final.to_csv("./data/processed_train.csv", index=False)



