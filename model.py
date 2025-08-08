import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("kidney_disease.csv")

df.drop(columns=["id", "classification"], inplace=True)

categorical_map = {
    "normal": 1,
    "abnormal": 0,
    "present": 1,
    "notpresent": 0,
    "yes": 1,
    "no": 0,
    "good": 1,
    "poor": 0
}

df.replace(categorical_map, inplace=True)

df = df.apply(pd.to_numeric, errors='coerce')

df.fillna(df.mean(numeric_only=True), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

with open("pca_ckd.pkl", "wb") as f:
    pickle.dump((scaler, pca, components), f)

plt.figure(figsize=(8, 6))
plt.scatter(components[:, 0], components[:, 1], c='green', alpha=0.6)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Chronic Kidney Disease Dataset")
plt.grid(True)
plt.savefig("static/pca_plot.png")
plt.close()
