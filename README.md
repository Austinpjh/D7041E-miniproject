# **Mini-Project: Performance Comparison of Supervised and Unsupervised Learning Models**

## **Overview**
This project evaluates the performance of supervised (Random Forest) and unsupervised (K-Means) learning models on five datasets from the UCI Machine Learning Repository. The goal is to compare the models’ effectiveness across diverse datasets and analyze their strengths and limitations.

---

## **Datasets**
The following datasets were selected for analysis:

| Dataset               | Instances | Features | Target Variable         | Description                                            |
|-----------------------|-----------|----------|-------------------------|-------------------------------------------------------|
| **Iris**             | 150       | 4        | Species (3 classes)     | Measures of iris flowers (setosa, versicolor, virginica). |
| **Wine**             | 178       | 13       | Wine Type (3 classes)   | Chemical properties of wines for classification.      |
| **Breast Cancer**    | 569       | 30       | Diagnosis (Benign/Malignant) | Diagnostic information for cancer classification.     |
| **Seeds**            | 210       | 7        | Seed Type (3 classes)   | Physical attributes of seeds for clustering/classification. |
| **Heart Disease**    | 303       | 13       | Disease (Yes/No)        | Medical data for predicting heart disease presence.   |

---

## **Models**
### **Unsupervised Learning**
- **Model**: K-Means Clustering
- **Objective**: Group data into clusters and evaluate clustering performance using Adjusted Rand Index (ARI).
- **Parameters**:
  - Number of clusters: Equal to the number of unique target labels.
  - Initialization: k-means++ with `random_state=42`.

### **Supervised Learning**
- **Model**: Random Forest Classifier
- **Objective**: Predict class labels using labeled data and evaluate using accuracy.
- **Parameters**:
  - Default hyperparameters with `random_state=42`.

---

## **Methodology**
1. **Data Preprocessing**:
   - Features were standardized using `StandardScaler`.
   - Missing values were removed for datasets with incomplete data (e.g., `Heart Disease`).
   - Labels were encoded as integers for compatibility with the models.

2. **Data Splitting**:
   - Each dataset was split into training (80%) and testing (20%) subsets.

3. **Performance Metrics**:
   - **Unsupervised Learning**: Adjusted Rand Index (ARI).
   - **Supervised Learning**: Accuracy score.

---

## **Results**
The performance of the models is summarized below:

| Dataset               | Unsupervised Accuracy (ARI) | Supervised Accuracy |
|-----------------------|-----------------------------|---------------------|
| **Iris**             | 0.810485                    | 1.000000            |
| **Wine**             | 0.924551                    | 1.000000            |
| **Breast Cancer**    | 0.735098                    | 0.964912            |
| **Seeds**            | 0.628850                    | 0.833333            |
| **Heart Disease**    | 0.579596                    | 0.883333            |

---

## **Conclusions**
### **Performance Insights**:
- **K-Means Clustering**:
  - Performed well on datasets with clear feature separations (e.g., `Iris` and `Wine`).
  - Struggled with overlapping or complex distributions (e.g., `Heart Disease` and `Seeds`).

- **Random Forest**:
  - Achieved high accuracy across all datasets, demonstrating its robustness and ability to handle diverse data distributions.

### **Dataset Characteristics**:
- **Iris** and **Wine**: Highly separable feature spaces, favorable for both supervised and unsupervised methods.
- **Heart Disease** and **Seeds**: Complex feature spaces with overlaps, challenging for clustering models.

### **Limitations**:
- K-Means is sensitive to scaling and assumes spherical clusters, limiting its performance on complex datasets.
- Random Forest, while robust, requires labeled data and could benefit from hyperparameter optimization.

### **Future Work**:
- Experiment with advanced clustering techniques (e.g., DBSCAN, hierarchical clustering).
- Tune hyperparameters of Random Forest for improved performance.
- Expand analysis to include more datasets for greater generalizability.

---

## **Usage**
1. Clone the repository and install the necessary libraries:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   pip install -r requirements.txt
   ```

2. Run the project script:
   ```bash
   python mini_project.py
   ```

3. Results will be printed to the console.

