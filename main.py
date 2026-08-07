# ==========================================
# Codomax AI/ML Internship - Day 23
# Topic: K-Means Clustering
# Author: Akash Kumar Jha
# ==========================================

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("=" * 65)
print("             K-MEANS CLUSTERING - DAY 23")
print("=" * 65)

# Load dataset
df = pd.read_csv("student_data.csv")

print("\nOriginal Dataset\n")
print(df)

# Select features
X = df[["Study_Hours", "Attendance", "Previous_Score"]]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means model
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Fit model
df["Cluster"] = model.fit_predict(X_scaled)

print("\nClustered Dataset\n")
print(df)

# Display cluster centers
centers = scaler.inverse_transform(model.cluster_centers_)

cluster_centers = pd.DataFrame(
    centers,
    columns=["Study_Hours", "Attendance", "Previous_Score"]
)

print("\nCluster Centers\n")
print(cluster_centers.round(2))

# Cluster counts
print("\nStudents in Each Cluster\n")
print(df["Cluster"].value_counts().sort_index())

print("\nClustering Completed Successfully ✅")
print("=" * 65)
