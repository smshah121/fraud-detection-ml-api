import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

print("Loading dataset...")





df = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully")


X = df.drop("Class", axis=1)
y = df["Class"]

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")



model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained successfully")


predictions = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

joblib.dump(model, "fraud_model.pkl")

print("\nModel saved as fraud_model.pkl")