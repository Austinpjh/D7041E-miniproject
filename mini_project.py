import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score, classification_report

#Load and preprocess datasets
def load_and_preprocess_iris():
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['label'] = data.target
    return df, 'label'

def load_and_preprocess_wine():
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
    features = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
                "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
                "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
    data.columns = ["label"] + features
    return data, 'label'

def load_and_preprocess_breast_cancer():
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)
    features = [f"feature_{i}" for i in range(2, 32)]
    data.columns = ["id", "label"] + features
    data['label'] = LabelEncoder().fit_transform(data['label'])
    return data.drop(columns=['id']), 'label'

def load_and_preprocess_seeds():
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt", delim_whitespace=True, header=None)
    features = [f"feature_{i}" for i in range(1, 8)]
    data.columns = features + ['label']
    return data, 'label'

def load_and_preprocess_heart():
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header=None)
    data.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "label"]
    data = data.replace("?", np.nan).dropna()
    data['label'] = (data['label'] > 0).astype(int)  # Binary classification
    return data, 'label'

# Dataset handlers
datasets = [load_and_preprocess_iris, load_and_preprocess_wine, load_and_preprocess_breast_cancer, 
            load_and_preprocess_seeds, load_and_preprocess_heart]

results = []

for dataset in datasets:
    # Load and preprocess dataset
    data, target_col = dataset()
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Unsupervised Learning (K-Means)
    kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42)
    kmeans.fit(X_train)
    y_kmeans = kmeans.predict(X_test)
    unsupervised_acc = adjusted_rand_score(y_test, y_kmeans)

    # Supervised Learning (Random Forest)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    supervised_acc = accuracy_score(y_test, y_pred)

    # Store results
    results.append({
        "Dataset": dataset.__name__,
        "Unsupervised Accuracy (ARI)": unsupervised_acc,
        "Supervised Accuracy": supervised_acc
    })

# Display results
results_df = pd.DataFrame(results)
print(results_df)
