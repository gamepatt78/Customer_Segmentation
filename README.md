# Customer Segmentation Using K-Means Clustering

## EncoderX Remote Internship — Data Science | Week 03

> **Task:** Customer Segmentation
> **Domain:** Data Science
> **Primary Area:** Customer Analytics
> **Method:** K-Means Clustering
> **Dataset:** Mall Customer Segmentation Data

---

## 📌 Project Overview

Customer segmentation is a data science technique used to divide customers into groups based on similarities in their demographic and behavioral characteristics.

This project applies **K-Means Clustering**, an unsupervised machine learning algorithm, to identify meaningful customer segments using customer income and spending behavior.

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

1. Understand the characteristics of the customer dataset.
2. Clean and preprocess the data.
3. Perform exploratory data analysis.
4. Normalize numerical features.
5. Apply K-Means clustering.
6. Determine an appropriate number of customer clusters.
7. Evaluate the clustering using the Elbow Method and Silhouette Score.
8. Visualize the identified customer segments.
9. Interpret the characteristics of each segment.
10. Develop meaningful business recommendations.

---

## 📊 Dataset

### Mall Customer Segmentation Data

The project uses the **Mall Customer Segmentation Data** dataset available on Kaggle.

**Kaggle Dataset:**
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

### Dataset Features

| Feature                  | Description                                      |
| ------------------------ | ------------------------------------------------ |
| `CustomerID`             | Unique customer identifier                       |
| `Gender`                 | Customer gender                                  |
| `Age`                    | Customer age                                     |
| `Annual Income (k$)`     | Annual income in thousands of dollars            |
| `Spending Score (1-100)` | Spending behavior score assigned to the customer |

The dataset contains customer demographic and spending information suitable for customer segmentation.

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

The following preprocessing steps were performed:

### 1. Missing Value Analysis

The dataset was checked for missing values.

```python
df.isnull().sum()
```

Where necessary, missing numerical values were handled using median imputation and categorical values using the mode.

### 2. Duplicate Analysis

Duplicate records were checked before performing clustering.

### 3. Identifier Removal

`CustomerID` was excluded from the clustering features because it is an identifier rather than a behavioral or demographic characteristic.

### 4. Categorical Encoding

The `Gender` feature was converted into a numerical representation where required.

### 5. Feature Scaling

Numerical features were standardized using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Feature scaling is important for clustering because features with larger numerical ranges can otherwise have a greater influence on distance calculations.

---

# 🔎 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the characteristics of the customers and identify relationships between important variables.

The analysis includes:

* Age distribution
* Annual income distribution
* Spending score distribution
* Gender distribution
* Age vs. spending score
* Annual income vs. spending score
* Feature correlation analysis

### Key Visualization

The relationship between:

```text
Annual Income (k$)
        vs.
Spending Score (1-100)
```

was examined because these variables provide useful information for distinguishing customer spending behavior.

---

# 🤖 K-Means Clustering

K-Means is an unsupervised machine learning algorithm that divides observations into a predefined number of clusters.

The algorithm works by:

1. Selecting the number of clusters.
2. Initializing cluster centroids.
3. Assigning customers to their nearest centroid.
4. Updating the centroid positions.
5. Repeating the process until the clusters stabilize.

In this project, K-Means was applied to the selected customer features after normalization.

---

# 📐 Determining the Number of Clusters

Two evaluation approaches were used.

## Elbow Method

The Elbow Method evaluates the within-cluster sum of squares, represented by inertia, for different values of `K`.

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

The resulting curve was analyzed to identify an appropriate cluster count.

---

## Silhouette Score

The Silhouette Score measures how well each customer fits within its assigned cluster compared with other clusters.

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(
    X_scaled,
    labels
)

print(score)
```

The Elbow Method and Silhouette Score were considered together when selecting the final clustering configuration.

---

# 📊 Customer Segmentation

After selecting the clustering configuration, K-Means was applied to the standardized customer features.

The resulting clusters were analyzed based on:

* Average age
* Average annual income
* Average spending score
* Customer count
* Gender distribution

A cluster profile table was generated to understand the characteristics of each segment.

---

# 📈 Visualizations

The project includes the following visualizations:

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
* Cluster distribution
* Customer segment scatter plot
* Cluster centers
* PCA-based visualization
* Correlation heatmap

These visualizations help communicate the differences between customer segments.

---

# 👥 Customer Segment Interpretation

The identified clusters are interpreted using their average demographic and spending characteristics.

A typical interpretation framework is:

| Segment Characteristic              | Business Interpretation                             |
| ----------------------------------- | --------------------------------------------------- |
| High income + high spending         | Potential high-value customers                      |
| High income + low spending          | Customers with potential for targeted engagement    |
| Low income + high spending          | Customers showing strong spending behavior          |
| Low income + low spending           | Customers with relatively lower purchasing activity |
| Moderate income + moderate spending | Standard or average customer group                  |

> **Note:** The final segment names and descriptions should be based on the actual cluster profile generated by the notebook.

---

# 💡 Business Insights

Customer segmentation can help organizations understand that customers do not have identical purchasing behavior.

Potential business applications include:

### Targeted Marketing

Different customer groups can receive different marketing campaigns based on their spending behavior and demographic characteristics.

### Personalized Offers

Customers with lower spending activity can receive targeted discounts or promotional offers.

### Customer Engagement

High-spending customers can be targeted with loyalty programs and personalized campaigns.

### Product Recommendations

Customer segments can be used to develop more relevant product recommendations.

### Customer Retention

Segment-level analysis can help identify groups that may require additional engagement to improve retention.

---

# 📌 Key Findings

The final findings should be updated after executing the notebook.

### Clustering Results

```text
Number of clusters: [INSERT FINAL K]

Silhouette Score: [INSERT SCORE]

Number of customers: [INSERT COUNT]
```

### Segment Summary

| Cluster   | Average Age | Average Income | Average Spending Score | Customer Count |
| --------- | ----------: | -------------: | ---------------------: | -------------: |
| Cluster 0 |     [Value] |        [Value] |                [Value] |        [Value] |
| Cluster 1 |     [Value] |        [Value] |                [Value] |        [Value] |
| Cluster 2 |     [Value] |        [Value] |                [Value] |        [Value] |
| Cluster 3 |     [Value] |        [Value] |                [Value] |        [Value] |
| Cluster 4 |     [Value] |        [Value] |                [Value] |        [Value] |

---

# 📈 Model Evaluation

The clustering solution was evaluated using:

### Elbow Method

Used to examine the relationship between the number of clusters and clustering inertia.

### Silhouette Score

Used to evaluate how well-separated and internally consistent the clusters are.

### Cluster Visualization

Scatter plots and PCA visualization were used to visually inspect the separation between customer segments.

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Customer-Segmentation-KMeans.git
```

## 2. Navigate to the Project

```bash
cd Customer-Segmentation-KMeans
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

Open:

```text
Customer_Segmentation_EncoderX_Week03.ipynb
```

## 6. Run All Cells

Execute the notebook from beginning to end to reproduce the analysis and clustering results.

---

# 📦 Deliverables

This project includes the deliverables required for the EncoderX Week 03 task:

* [x] Customer-related dataset
* [x] Data preprocessing
* [x] Missing-value analysis
* [x] Numerical feature normalization
* [x] Exploratory Data Analysis
* [x] Feature analysis
* [x] K-Means clustering
* [x] Optimal cluster investigation
* [x] Elbow Method
* [x] Silhouette Score
* [x] Cluster visualizations
* [x] Customer segment interpretation
* [x] Business recommendations
* [x] Jupyter Notebook
* [x] Source code
* [x] README documentation

---

# 🎥 Project Demonstration

A **3–5 minute project demonstration video** is included as part of the EncoderX submission requirements.

The demonstration covers:

* Project objective
* Dataset
* Data preprocessing
* Exploratory Data Analysis
* K-Means clustering
* Elbow Method
* Silhouette Score
* Customer segments
* Visualizations
* Business insights

**Demo Video:** `[ADD VIDEO LINK]`

---

# 📊 Presentation

**Presentation:** `[ADD PRESENTATION LINK OR FILE]`

The presentation summarizes:

* Problem statement
* Dataset
* Methodology
* EDA
* Clustering
* Evaluation
* Customer segments
* Business insights
* Recommendations
* Conclusion

---

# 🔗 Project Links

| Resource          | Link                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------- |
| GitHub Repository | `[ADD GITHUB LINK]`                                                                                     |
| Jupyter Notebook  | `[ADD NOTEBOOK LINK]`                                                                                   |
| Dataset           | [Kaggle Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) |
| Demo Video        | `[ADD VIDEO LINK]`                                                                                      |
| LinkedIn Post     | `[ADD LINKEDIN POST LINK]`                                                                              |
| Presentation      | `[ADD PRESENTATION LINK]`                                                                               |

---

# 🏁 Conclusion

This project demonstrates the use of **unsupervised machine learning for customer segmentation**.

By combining data preprocessing, exploratory analysis, feature scaling, K-Means clustering, clustering evaluation, visualization, and business interpretation, the project identifies groups of customers with similar characteristics.

The resulting segmentation can provide useful insights for targeted marketing, customer engagement, personalized recommendations, and retention strategies.

---

## 👨‍💻 Author

**Athwin Videsh M**

MCA Graduate | Data Science

GitHub:
https://github.com/gamepatt78

LinkedIn:
https://linkedin.com/in/athwin-videsh-b40723125

---

## 🏷️ Tags

```text
#EncoderX
#DataScience
#CustomerSegmentation
#MachineLearning
#DataAnalytics
#Internship
#LearningInPublic
```

---

## 📜 Internship

**EncoderX Remote Internship — Data Science**
**Batch:** 02
**Week:** 03
**Task:** Customer Segmentation
