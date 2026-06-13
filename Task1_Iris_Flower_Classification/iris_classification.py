import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("Iris.csv")


print("Dataset Preview:")
print(df.head())

print("\nColumns:")
print(df.columns)

if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])


X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)


sns.pairplot(df, hue='Species')
plt.show()