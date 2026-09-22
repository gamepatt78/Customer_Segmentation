# Customer Segmentation Using K-Means Clustering

## EncoderX Remote Internship — Data Science | Week 03

> **Task:** Customer Segmentation
> **Domain:** Data Science
> **Primary Area:** Customer Analytics
> **Method:** K-Means Clustering
> **Dataset:** Mall Customer Segmentation Data

---

## 📌 Project Overview

Customer segmentation is a data science technique used to divide customers into groups based on similarities in their characteristics and spending behavior.

This project applies **K-Means Clustering**, an unsupervised machine learning algorithm, to identify meaningful customer segments using **Annual Income** and **Spending Score**.

The objective is to discover groups of customers with similar characteristics and derive business insights that can support:

* Targeted marketing
* Customer engagement
* Personalized offers
* Product recommendations
* Customer retention strategies

This project was completed as part of the **EncoderX Remote Internship — Data Science, Week 03**.

---

## 🎯 Objectives

The main objectives of this project are:

1. Understand the customer dataset.
2. Clean and preprocess the data.
3. Perform Exploratory Data Analysis.
4. Select relevant clustering features.
5. Normalize numerical features.
6. Apply K-Means clustering.
7. Determine an appropriate number of clusters.
8. Evaluate clustering using the Elbow Method and Silhouette Score.
9. Visualize customer segments.
10. Profile and interpret the identified customer groups.
11. Develop business recommendations.

---

## 📊 Dataset

### Mall Customer Segmentation Data

The project uses the **Mall Customer Segmentation Data** dataset available on Kaggle.

**Kaggle Dataset:**

https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

### Dataset Features

| Feature                  | Description                           |
| ------------------------ | ------------------------------------- |
| `CustomerID`             | Unique customer identifier            |
| `Gender`                 | Customer gender                       |
| `Age`                    | Customer age                          |
| `Annual Income (k$)`     | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Spending behavior score               |

The dataset contains **200 customers**.

---

## 🛠️ Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

### Machine Learning Technique

**K-Means Clustering**

### Additional Techniques

* Data preprocessing
* Feature scaling
* Exploratory Data Analysis
* Elbow Method
* Silhouette Score
* PCA visualization
* Cluster profiling

---

## 📁 Project Structure

```text
Customer-Segmentation-KMeans/
│
├── Customer_Segmentation_EncoderX_Week03.ipynb
├── Mall_Customers.csv
├── customer_segments.csv
├── README.md
│
├── visualizations/
│   ├── age_distribution.png
│   ├── income_distribution.png
│   ├── spending_distribution.png
│   ├── correlation_heatmap.png
│   ├── elbow_method.png
│   ├── silhouette_score.png
│   ├── customer_segments.png
│   └── pca_visualization.png
│
└── presentation/
    └── Customer_Segmentation_Presentation.pdf
```

---

# 🔄 Project Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Feature Scaling
   ↓
Elbow Method
   ↓
Silhouette Score
   ↓
K-Means Clustering
   ↓
Cluster Visualization
   ↓
Cluster Profiling
   ↓
Segment Interpretation
   ↓
Business Recommendations
```

---

# 🧹 Data Preprocessing

### 1. Missing Value Analysis

The dataset was checked for missing values using:

```python
df.isnull().sum()
```

The dataset was examined to ensure that missing values would not affect the clustering analysis.

### 2. Duplicate Analysis

Duplicate records were checked before performing clustering.

### 3. Identifier Handling

`CustomerID` was treated as an identifier and was not used as a clustering feature.

### 4. Feature Selection

The final K-Means model used:

```text
Annual Income (k$)
Spending Score (1-100)
```

These features were selected because they directly represent customer purchasing capacity and spending behavior.

### 5. Feature Scaling

The selected numerical features were standardized using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Scaling is important because K-Means uses distance calculations to assign customers to clusters.

---

# 🔎 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer characteristics and relationships between important variables.

The analysis included:

* Age distribution
* Gender distribution
* Annual income distribution
* Spending score distribution
* Age vs. spending score
* Annual income vs. spending score
* Correlation analysis

The relationship between:

```text
Annual Income (k$)
        vs.
Spending Score (1-100)
```

was particularly important because these variables were used for the final clustering model.

---

# 🤖 K-Means Clustering

K-Means is an unsupervised machine learning algorithm that divides observations into a predefined number of clusters.

The algorithm works by:

1. Selecting the number of clusters.
2. Initializing cluster centroids.
3. Assigning customers to the nearest centroid.
4. Updating the centroid positions.
5. Repeating the process until the clusters stabilize.

The model was trained using the standardized Annual Income and Spending Score features.

---

# 📐 Determining the Number of Clusters

Two evaluation methods were used:

## Elbow Method

The Elbow Method evaluates the clustering inertia for different values of `K`.

```python
inertia = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
```

The inertia curve was examined to identify a suitable number of clusters.

## Silhouette Score

The Silhouette Score was calculated for different values of `K`.

The scores obtained were:

| K     | Silhouette Score |
| ----- | ---------------: |
| 2     |           0.3213 |
| 3     |           0.4666 |
| 4     |           0.4939 |
| **5** |       **0.5547** |
| 6     |           0.5399 |
| 7     |           0.5281 |
| 8     |           0.4552 |
| 9     |           0.4571 |
| 10    |           0.4432 |

The final project used **K = 5**.

The final Silhouette Score was:

**0.5547**

---

# 📊 Final K-Means Model

The final model was configured as:

```python
optimal_k = 5

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters
```

### Final Results

```text
Number of Customers: 200
Number of Clusters: 5
Final Silhouette Score: 0.5547
```

---

# 👥 Customer Segmentation Results

The final dataset contains five customer segments.

| Cluster | Customer Segment               | Avg. Age | Avg. Income (k$) | Avg. Spending Score | Customers |
| ------: | ------------------------------ | -------: | ---------------: | ------------------: | --------: |
|       0 | Standard Customers             |    42.72 |            55.30 |               49.52 |        81 |
|       1 | High-Value Customers           |    32.69 |            86.54 |               82.13 |        39 |
|       2 | Careful Spenders               |    25.27 |            25.73 |               79.36 |        22 |
|       3 | Premium Low-Spending Customers |    41.11 |            88.20 |               17.11 |        35 |
|       4 | Young High-Spending Customers  |    45.22 |            26.30 |               20.91 |        23 |

---

# 📌 Segment Interpretation

### Cluster 0 — Standard Customers

* Average age: **42.72**
* Average income: **55.30k**
* Average spending score: **49.52**
* Customers: **81**

This is the largest customer segment and represents customers with relatively moderate income and spending behavior.

### Cluster 1 — High-Value Customers

* Average age: **32.69**
* Average income: **86.54k**
* Average spending score: **82.13**
* Customers: **39**

This segment combines relatively high income with high spending activity.

### Cluster 2 — Careful Spenders

* Average age: **25.27**
* Average income: **25.73k**
* Average spending score: **79.36**
* Customers: **22**

This segment has relatively lower income but a high spending score.

### Cluster 3 — Premium Low-Spending Customers

* Average age: **41.11**
* Average income: **88.20k**
* Average spending score: **17.11**
* Customers: **35**

This segment has relatively high income but a low spending score.

### Cluster 4 — Young High-Spending Customers

* Average age: **45.22**
* Average income: **26.30k**
* Average spending score: **20.91**
* Customers: **23**

This segment has relatively lower income and lower spending activity based on the final cluster profile.

> **Note:** Segment names are descriptive labels assigned to the project clusters based on their observed average characteristics.

---

# 📈 Visualizations

The project includes visualizations for:

### Customer Demographics

* Age distribution
* Gender distribution

### Customer Behavior

* Annual income distribution
* Spending score distribution
* Age vs. spending score
* Annual income vs. spending score

### Clustering Analysis

* Elbow Method
* Silhouette Score
* Customer segment scatter plot
* Cluster centers
* PCA visualization
* Correlation heatmap

These visualizations help demonstrate the differences between the identified customer groups.

---

# 💡 Business Insights

The segmentation demonstrates that customers can have significantly different combinations of income and spending behavior.

### Targeted Marketing

Marketing campaigns can be customized according to the characteristics of each customer segment.

### Personalized Offers

Customers with lower spending scores can be targeted with relevant promotional offers.

### Customer Engagement

Higher-spending customer groups can be considered for loyalty and engagement programs.

### Product Recommendations

Customer segments can be used to develop more relevant product recommendations.

### Customer Retention

Segment-level analysis can help organizations identify customer groups that may require additional engagement.

---

# 📌 Key Findings

### Clustering Results

```text
Number of Customers: 200
Number of Clusters: 5
Final Silhouette Score: 0.5547
```

### Most Representative Segment Profiles

| Segment                        | Income Level | Spending Level | Customers |
| ------------------------------ | ------------ | -------------- | --------: |
| Standard Customers             | Moderate     | Moderate       |        81 |
| High-Value Customers           | High         | High           |        39 |
| Careful Spenders               | Low          | High           |        22 |
| Premium Low-Spending Customers | High         | Low            |        35 |
| Young High-Spending Customers  | Low          | Low            |        23 |

---

# 📈 Model Evaluation

The clustering solution was evaluated using three approaches.

### Elbow Method

Used to examine how clustering inertia changes as the number of clusters increases.

### Silhouette Score

The final clustering configuration achieved a **Silhouette Score of 0.5547**.

### Cluster Visualization

Scatter plots and PCA visualization were used to inspect the separation and distribution of the customer segments.

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/gamepatt78/Customer_Segmentation.git
```

## 2. Navigate to the Project

```bash
cd Customer_Segmentation
```

## 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

## 5. Open the Notebook

```text
Customer_Segmentation_EncoderX_Week03.ipynb
```

## 6. Run All Cells

Execute the notebook from beginning to end to reproduce the analysis and clustering results.

---

# 📦 Deliverables

* Customer dataset
* Data preprocessing
* Missing-value analysis
* Numerical feature normalization
* Exploratory Data Analysis
* Feature analysis
* K-Means clustering
* Optimal cluster investigation
* Elbow Method
* Silhouette Score
* Cluster visualizations
* Customer segment interpretation
* Business recommendations
* Jupyter Notebook
* Source code
* README documentation
* Final customer segmentation CSV

---

# 🎥 Project Demonstration

A **3–5 minute project demonstration video** is included as part of the EncoderX submission requirements.

The demonstration covers:

* Project objective
* Dataset
* Data preprocessing
* Exploratory Data Analysis
* Feature selection
* K-Means clustering
* Elbow Method
* Silhouette Score
* Customer segments
* Visualizations
* Business insights

**Demo Video:**

https://drive.google.com/file/d/1GNWcJIBnLicvGdGSiRBX0PQj-wFeKccB/view?usp=sharing

---

# 🔗 Project Links

| Resource              | Link                                                                                                                               |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **GitHub Repository** | https://github.com/gamepatt78/Customer_Segmentation                                                                                |
| **Dataset**           | https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python                                              |
| **Demo Video**        | https://drive.google.com/file/d/1GNWcJIBnLicvGdGSiRBX0PQj-wFeKccB/view?usp=sharing                                                 |
| **LinkedIn Post**     | https://www.linkedin.com/posts/athwin-videsh-b40723125_encoderx-datascience-customersegmentation-activity-7507988758647066624-Kv4x |

---

# 🏁 Conclusion

This project demonstrates the application of **unsupervised machine learning for customer segmentation** using K-Means clustering.

The analysis used customer **Annual Income** and **Spending Score** as the primary clustering features. After feature scaling and evaluation using the Elbow Method and Silhouette Score, a **5-cluster solution** was used.

The final model achieved a **Silhouette Score of 0.5547** and identified five distinct customer segments with different income and spending characteristics.

These segments can provide useful insights for targeted marketing, customer engagement, personalized recommendations, and retention strategies.

---

## 👨‍💻 Author

**Athwin Videsh M**

MCA Graduate | Data Science

**GitHub:**
https://github.com/gamepatt78

**LinkedIn:**
https://linkedin.com/in/athwin-videsh-b40723125

---

## 🏷️ Tags

```text
#EncoderX
#DataScience
#CustomerSegmentation
#MachineLearning
#DataAnalytics
#KMeans
#Clustering
#Internship
#LearningInPublic
```

---

## 📜 Internship

**EncoderX Remote Internship — Data Science**

**Batch:** 02
**Week:** 03
**Task:** Customer Segmentation
