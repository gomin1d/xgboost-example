import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

# load titanic.csv
df = pd.read_csv('titanic.csv')
print("Дані:")
print(df)

# видаляю непотрібні колонки
df = df.reset_index(drop=True)
df = df.drop('PassengerId', axis=1)
df = df.drop('Name', axis=1)
df = df.drop('Ticket', axis=1)
df = df.drop('Cabin', axis=1)
print("Дані без непотрібних колонок:")
print(df)

# видаляємо рядки з відсутніми значеннями
df = df.dropna()

# перемішуємо дані
df = df.sample(frac=1, random_state=0)

# Перетворюю колонку Embarked в числа
mapping = {k: v for v, k in enumerate(df.Embarked.unique())}
df['Embarked'] = df.Embarked.map(mapping)

# Перетворюю колонку Sex в числа
mapping = {k: v for v, k in enumerate(df.Sex.unique())}
df['Sex'] = df.Sex.map(mapping)

print("Дані після конвертацій:")
print(df)

# зберігаємо у "X" усі колонки, окрім Survived
# зберігаємо у "y" тільки колонку Survived

X = df.drop('Survived', axis=1)
y = df['Survived']

# розділяємо дані на дві частини
# X_train, y_train - для навчання
# X_test, y_test - для перевірки ефективності
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# створюємо модель xgboost, навчаємо модель
xgb_clf = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.15)
xgb_clf.fit(X_train, y_train)

# крос валідація
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(estimator=xgb_clf, X=X, y=y,
                         cv=skf, scoring='roc_auc', n_jobs=-1)
print('Крос валідація = {:.5f} +/- {:.5f}'.format(scores.mean(), scores.std()))

new_data = pd.DataFrame({
    'Pclass': [3],
    'Sex': [1],
    'Age': [7],
    'SibSp': [1],
    'Parch': [1],
    'Fare': [300],
    'Embarked': [0],
})
new_result = xgb_clf.predict(new_data)
print("Чи виживе новий пасажир: ")
print(new_result)
