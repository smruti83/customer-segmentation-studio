# Customer Segmentation Studio
#LIVE URL - https://customer-segmentation-studio-f8k2z4gxlacvogjrs8dycu.streamlit.app/

A practical, portfolio-ready Streamlit application for customer segmentation with K-Means, the Elbow Method, Silhouette Score, preprocessing for mixed numerical/categorical data, PCA, cluster profiling, and CSV downloads.

> The bundled customer file is **synthetic practice data**, not real customer information.

## Features

- Upload your own customer CSV or use the bundled sample data.
- Select which columns are model features. Likely ID columns are excluded from the default selection.
- Fill missing numeric values using the median and categorical values using the most frequent value.
- Scale numerical features and one-hot encode categorical features.
- Optionally run PCA before clustering and select a target explained-variance percentage.
- Compare candidate K values with Elbow/Inertia and Silhouette Score charts.
- Get a suggested K from the tested Silhouette Scores or select K manually.
- Explore cluster sizes, a 2D PCA visualization, and cluster profiles.
- Download customer records with cluster labels and a cluster-profile CSV.

## 1. Requirements

- Python 3.10 or newer recommended
- Internet access for the first-time package installation

## 2. Install and run

Open a terminal in this project folder and run:

### Windows

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

## 3. Use your own CSV

1. Prepare one row per customer and one column per feature.
2. Open the app and upload the CSV from the left sidebar.
3. Choose relevant columns under **Features**.
4. Do not use customer IDs, names, email addresses, or phone numbers as similarity features.
5. Review missing values and high-cardinality warnings.
6. Inspect the Elbow and Silhouette charts. Treat the suggested K as evidence, not as an unquestionable answer.
7. Review customer counts and cluster profiles before naming the segments.
8. Download `customer_segments.csv` or `cluster_profile.csv`.

### Suggested columns

A customer table may include `Age`, `AnnualIncome`, `SpendingScore`, `PurchaseFrequency`, `AverageOrderValue`, `RecencyDays`, `Gender`, `City`, and `MembershipType`. Use only columns that are appropriate and permitted for your business use case.


## Deploy on Railway (public URL)

1. Push this whole project folder to a GitHub repository. Keep `Dockerfile`, `requirements.txt`, `app.py`, and `data/sample_customers.csv` in the repository root/expected paths.
2. In Railway, create a new project and choose **Deploy from GitHub Repo**, then select the repository. Railway detects and builds the included `Dockerfile`.
3. Wait for the deployment to finish and review the deployment logs if it fails.
4. Open the deployed service's **Settings → Networking → Public Networking → Generate Domain**.
5. Open the generated `https://...up.railway.app` domain to use the app. The Dockerfile reads Railway's runtime `PORT` variable.

Railway usage is metered; check its current plan and usage before leaving the app online. Do not upload sensitive customer information to a public demo without appropriate access controls and authorization.

## Optional deployment with Docker

If Docker is installed, run from the project folder:

```bash
docker build -t customer-segmentation-studio .
docker run --rm -p 8501:8501 customer-segmentation-studio
```

Then open `http://localhost:8501`. The included `render.yaml` and `Dockerfile` can also be used as a starting point for hosting on a Docker-capable web service. Configure secrets/access controls before using non-public data.

## Methodology

- Numerical columns: median imputation + standard scaling.
- Categorical columns: most-frequent imputation + one-hot encoding.
- Clustering: K-Means with a fixed random seed for reproducibility.
- K evaluation: inertia for the Elbow Method and Silhouette Score for separation.
- PCA: optional for clustering; an independent PCA projection is used for the 2D visualization.
- Profiles: numerical means and most frequent categorical values per cluster.

## Important limitations

- Clustering is exploratory; it does not prove that a segment is commercially meaningful.
- Silhouette Score can favor compact, well-separated groups and may not capture business value.
- The Elbow Method can be ambiguous.
- PCA can discard information. Compare results with and without PCA.
- For a production deployment, add authentication, access controls, logging, monitoring, data retention rules, tests, and a reviewed privacy/security process. Do not upload personal data unless you have authorization.

## Project structure

```text
customer_segmentation_product/
├── app.py
├── requirements.txt
├── README.md
├── Dockerfile
├── render.yaml
├── .streamlit/
│   └── config.toml
├── .gitignore
└── data/
    └── sample_customers.csv
```
