import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import pickle

# Load dataset
df = pd.read_csv("teamStats_needed.csv")

# Separate numeric columns for imputation
numeric_columns = df.select_dtypes(include=["number"]).columns

# Impute missing values for numeric columns (mean imputation)
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# ❗ Replace this with the actual name of your target column
target_column = "wins"

# Check if the target column exists
if target_column not in df.columns:
    print(f"Error: Target column '{target_column}' not found!")
else:
    # Prepare features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # One-Hot Encoding for categorical columns
    X = pd.get_dummies(X)

    # Train the CART model (Decision Tree Classifier)
    cart_clf = DecisionTreeClassifier()
    cart_clf.fit(X, y)

    # Save the CART model to cart_model.pkl
    with open("cart_model.pkl", "wb") as f:
        pickle.dump(cart_clf, f)
    print("✅ cart_model.pkl saved!")

    # Train the KMeans model (Clustering)
    kmeans = KMeans(n_clusters=3, random_state=42)  # Set number of clusters (e.g., 3)
    kmeans.fit(X)

    # Save the KMeans model to kmeans_model.pkl
    with open("kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    print("✅ kmeans_model.pkl saved!")
