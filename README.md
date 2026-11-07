# Credit Portfolio Segmentation and Risk Pattern Analysis

**Machine Learning-Based Borrower Segmentation and Credit Risk Analysis**

An end-to-end machine learning project that uses unsupervised learning techniques to identify borrower segments, analyse historical default patterns, and explore credit portfolio risk through an interactive Streamlit dashboard.

---

## Project Overview

Financial institutions work with borrowers who have different financial backgrounds, credit scores, employment histories, and loan requirements.

This project applies machine learning clustering techniques to group borrowers with similar financial and credit characteristics. It also analyses historical default rates across the identified segments to understand differences in portfolio risk.

The project includes data preprocessing, exploratory data analysis, clustering, dimensionality reduction, anomaly detection, and an interactive dashboard.

## Key Features

- **Data Cleaning and Preprocessing:** Handled missing values, duplicate records, suspicious values, categorical encoding, and numerical scaling.
- **Exploratory Data Analysis:** Analysed borrower income, FICO scores, loan amounts, debt-to-income ratios, and default patterns.
- **Borrower Segmentation:** Applied K-Means clustering to identify four distinct borrower groups.
- **Clustering Comparison:** Experimented with K-Means, Hierarchical Clustering, and DBSCAN.
- **Dimensionality Reduction:** Used PCA and t-SNE to explore and visualize borrower clusters.
- **Anomaly Detection:** Identified unusual borrower profiles for further investigation.
- **Risk Pattern Analysis:** Compared historical default rates across borrower segments.
- **Interactive Dashboard:** Developed a Streamlit application with portfolio insights, cluster profiles, and individual borrower segmentation.

## Dataset

The project uses the `loans_granting_model_dataset.csv` dataset containing historical loan and borrower information.

The dataset includes features such as:

- Annual income
- Debt-to-income ratio (DTI)
- Loan amount
- FICO credit score
- Employment length
- Loan purpose
- Home ownership
- Geographical information
- Loan issue date
- Historical default status

**Note:** The original dataset is approximately 159 MB and is not included in this GitHub repository. Place the dataset in the project root directory before running the notebook or application.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Clustering | K-Means, MiniBatchKMeans, Hierarchical Clustering, DBSCAN |
| Dimensionality Reduction | PCA, t-SNE |
| Anomaly Detection | Isolation Forest |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Version Control | Git, GitHub |

## Project Workflow

1. Data loading and understanding
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Feature engineering and selection
5. Categorical encoding and numerical scaling
6. Clustering algorithm implementation and comparison
7. Cluster evaluation and selection
8. Dimensionality reduction and visualization
9. Borrower segment profiling
10. Historical default pattern analysis
11. Anomaly detection
12. Model serialization and Streamlit dashboard development

## Clustering Results

The final borrower segmentation model uses four clusters.

| Cluster | Borrowers | Average Income | Average FICO | Average Loan | Default Rate |
|---|---:|---:|---:|---:|---:|
| 0 | 346,536 | $78,363 | 700.70 | $12,435 | 19.42% |
| 1 | 313,535 | $125,622 | 706.12 | $25,838 | 20.30% |
| 2 | 391,544 | $50,899 | 690.48 | $9,474 | 22.78% |
| 3 | 296,066 | $60,115 | 696.91 | $11,139 | 16.59% |

**Overall Portfolio Default Rate: 19.98%**

Cluster 2 has the highest historical default rate, while Cluster 3 has the lowest among the four identified segments.

These results represent historical portfolio patterns, not predictions of an individual borrower's probability of default.

## Interactive Dashboard

The Streamlit dashboard contains three main sections:

### 1. Portfolio Overview
- Total borrower count
- Estimated historical defaults
- Overall portfolio default rate
- Borrower distribution across clusters

### 2. Borrower Segmentation
- Cluster summary and borrower profiles
- Historical default rate comparison
- Financial characteristics of each segment

### 3. Individual Borrower Explorer
- Enter borrower financial information
- Assign the borrower to a segment using the trained K-Means model
- Display the corresponding cluster profile

## Project Structure

```text
credit-portfolio-segmentation-risk-analysis/
│
├── Credit_Portfolio_Analysis.ipynb
├── app.py
├── credit_portfolio_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

The original dataset is excluded from the repository because of its file size.


## Model Information

The final segmentation model is saved as:

`credit_portfolio_model.pkl`

It contains the trained clustering model and preprocessing objects required to transform borrower inputs and assign them to the corresponding cluster.

## Important Notes and Limitations

- Clustering is an unsupervised learning task; cluster labels do not represent default predictions.
- Historical default rates are used after clustering to interpret portfolio risk patterns.
- Hierarchical Clustering and DBSCAN were evaluated using representative samples due to computational requirements.
- Anomaly detection identifies unusual records that may require further investigation; it does not establish fraud or creditworthiness.
- Results depend on the dataset and preprocessing choices.

## Future Improvements

- Explore additional clustering algorithms and distance metrics.
- Improve cluster separation and stability.
- Add more detailed geographical risk analysis.
- Integrate a supervised default prediction model as a separate module.
- Deploy the dashboard to a cloud platform.

## Author

**Umesh Reddy**

GitHub: [UmeshReddy27](https://github.com/UmeshReddy27)

---

*Developed as a machine learning project focused on borrower segmentation, credit portfolio analysis, and historical risk pattern discovery.*
