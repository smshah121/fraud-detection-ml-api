import requests
import pandas as pd

# Load dataset

df = pd.read_csv("creditcard.csv")

# Take first row sample
sample = df.drop("Class", axis=1).iloc[0].tolist()

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "features": sample
    }
)

print(response.json())