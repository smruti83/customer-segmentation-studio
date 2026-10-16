from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_TITLE = "Customer Segmentation Studio"
RANDOM_STATE = 42
DATA_DIR = Path(__file__).parent / "data"
SAMPLE_PATH = DATA_DIR / "sample_customers.csv"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    """Load the bundled synthetic customer dataset."""
    return pd.read_csv(SAMPLE_PATH)


def looks_like_identifier(column: str, series: pd.Series) -> bool:
    """Return True for likely ID columns so they are not selected by default."""
    normalized = column.lower().replace(" ", "").replace("-", "_")
    id_names = {"id", "customerid", "customer_id", "userid", "user_id", "clientid", "client_id"}
    if normalized in id_names or normalized.endswith("_id") or normalized.endswith("id"):
        return True
    non_null = series.dropna()
    if len(non_null) > 0 and non_null.nunique() / len(non_null) > 0.95:
        return True
    return False


def default_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if not looks_like_identifier(column, frame[column])]


def build_preprocessor(frame: pd.DataFrame, features: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(frame[c])]
    categorical_features = [c for c in features if c not in numeric_features]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("Choose at least one usable feature.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_features, categorical_features


def to_float_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Preprocessed data must be a two-dimensional matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("The feature matrix still has missing or infinite values. Check the selected columns.")
    return matrix


def evaluate_cluster_counts(matrix: np.ndarray, max_k: int) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    n_samples = matrix.shape[0]
    for k in range(2, min(max_k, n_samples - 1) + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(matrix)
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= n_samples:
            score = np.nan
        else:
            score = float(silhouette_score(matrix, labels))
        rows.append({"Clusters (K)": k, "Inertia": float(model.inertia_), "Silhouette Score": score})
    return pd.DataFrame(rows)


def profile_clusters(original: pd.DataFrame, features: list[str], labels: np.ndarray) -> pd.DataFrame:
    profile_data = original[features].copy()
    profile_data["Cluster"] = labels
    aggregations: dict[str, Any] = {}
    for column in features:
        if pd.api.types.is_numeric_dtype(profile_data[column]):
            aggregations[column] = "mean"
        else:
            aggregations[column] = lambda values: values.mode(dropna=True).iloc[0] if not values.mode(dropna=True).empty else "N/A"
    grouped = profile_data.groupby("Cluster", dropna=False).agg(aggregations).reset_index()
    sizes = profile_data.groupby("Cluster").size().rename("Customer Count").reset_index()
    return sizes.merge(grouped, on="Cluster", how="left")


def convert_csv_download(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def main() -> None:
    st.title("🛍️ Customer Segmentation Studio")
    st.caption("Explore customer groups with preprocessing, Elbow Method, Silhouette Score, PCA visualization, and K-Means.")

    with st.sidebar:
        st.header("1. Dataset")
        uploaded_file = st.file_uploader("Upload a customer CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_data = pd.read_csv(uploaded_file)
                source_name = uploaded_file.name
            except Exception as exc:
                st.error(f"Could not read this CSV: {exc}")
                st.stop()
        else:
            raw_data = load_sample_data()
            source_name = "Bundled synthetic sample"
            st.info("No file uploaded, so the included sample data is being used.")

        st.write(f"**Source:** {source_name}")

    if raw_data.empty:
        st.error("The dataset has no rows.")
        st.stop()
    if len(raw_data) < 4:
        st.error("Please provide at least 4 customer rows to compare multiple cluster choices reliably.")
        st.stop()
    if raw_data.shape[1] < 2:
        st.error("The dataset needs a customer identifier/description column and at least one feature column.")
        st.stop()

    # Fully empty fields cannot provide information; drop them before feature selection.
    empty_columns = [c for c in raw_data.columns if raw_data[c].isna().all()]
    data = raw_data.drop(columns=empty_columns).copy()
    if empty_columns:
        st.warning("Dropped fully empty columns: " + ", ".join(map(str, empty_columns)))

    with st.sidebar:
        st.header("2. Features")
        suggested = default_feature_columns(data)
        if not suggested:
            suggested = list(data.columns)
        features = st.multiselect(
            "Choose columns to use for segmentation",
            options=list(data.columns),
            default=suggested,
            help="Exclude customer IDs, names, phone numbers, or other unique identifiers. They usually should not determine similarity.",
        )
        use_pca = st.checkbox("Use PCA before clustering", value=False)
        variance_percent = st.slider(
            "PCA variance to retain (%)",
            min_value=70,
            max_value=99,
            value=90,
            step=1,
            disabled=not use_pca,
            help="PCA will retain enough components to reach approximately this cumulative explained variance.",
        )
        st.header("3. Cluster search")
        maximum_allowed_k = min(10, len(data) - 1)
        max_k = st.slider(
            "Test K values up to",
            min_value=2,
            max_value=maximum_allowed_k,
            value=min(6, maximum_allowed_k),
        )

    top_left, top_middle, top_right = st.columns(3)
    top_left.metric("Customer rows", f"{len(data):,}")
    top_middle.metric("CSV columns", f"{data.shape[1]:,}")
    top_right.metric("Selected features", f"{len(features):,}")

    with st.expander("Preview dataset", expanded=False):
        st.dataframe(data.head(20), use_container_width=True)
        st.write("**Missing values by column**")
        missing = data.isna().sum().rename("Missing Values").to_frame()
        st.dataframe(missing, use_container_width=True)

    if not features:
        st.warning("Choose at least one feature in the sidebar.")
        st.stop()
    if len(features) == 1 and not pd.api.types.is_numeric_dtype(data[features[0]]):
        st.warning("One categorical feature by itself usually does not provide meaningful customer segmentation. Choose more useful features.")

    # High-cardinality text columns often produce thousands of dummy features.
    high_cardinality = [
        c for c in features
        if not pd.api.types.is_numeric_dtype(data[c]) and data[c].nunique(dropna=True) > min(100, max(30, int(len(data) * 0.5)))
    ]
    if high_cardinality:
        st.warning(
            "High-cardinality categorical feature(s) selected: " + ", ".join(high_cardinality)
            + ". Consider excluding names, emails, free text, or IDs to avoid noisy clusters."
        )

    try:
        preprocessor, numeric_features, categorical_features = build_preprocessor(data, features)
        with st.spinner("Cleaning and scaling selected features..."):
            feature_matrix = to_float_matrix(preprocessor.fit_transform(data[features]))
    except Exception as exc:
        st.error(f"Could not prepare the selected features: {exc}")
        st.stop()

    if feature_matrix.shape[1] < 1:
        st.error("No usable features remained after preprocessing.")
        st.stop()

    cluster_matrix = feature_matrix
    fitted_pca = None
    if use_pca:
        try:
            fitted_pca = PCA(n_components=variance_percent / 100.0, svd_solver="full")
            cluster_matrix = to_float_matrix(fitted_pca.fit_transform(feature_matrix))
            st.success(
                f"PCA retained {cluster_matrix.shape[1]} component(s) and "
                f"{fitted_pca.explained_variance_ratio_.sum() * 100:.1f}% of variance."
            )
        except Exception as exc:
            st.error(f"PCA could not be applied to this dataset: {exc}")
            st.stop()

    if len(data) <= max_k:
        max_k = len(data) - 1
    results = evaluate_cluster_counts(cluster_matrix, max_k)
    if results.empty or results["Silhouette Score"].notna().sum() == 0:
        st.error("Unable to evaluate K values for this dataset. Try a larger dataset or different features.")
        st.stop()

    st.header("4. Choose the number of clusters")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Elbow Method")
        elbow_fig = go.Figure()
        elbow_fig.add_trace(go.Scatter(
            x=results["Clusters (K)"],
            y=results["Inertia"],
            mode="lines+markers",
            name="Inertia",
        ))
        elbow_fig.update_layout(
            xaxis_title="Number of clusters (K)",
            yaxis_title="Inertia (lower is tighter)",
            margin=dict(l=20, r=20, t=25, b=20),
        )
        st.plotly_chart(elbow_fig, use_container_width=True)
        st.caption("Look for the bend where additional clusters provide diminishing improvement. Some datasets have no clear elbow.")
    with right:
        st.subheader("Silhouette Score")
        silhouette_fig = go.Figure()
        silhouette_fig.add_trace(go.Scatter(
            x=results["Clusters (K)"],
            y=results["Silhouette Score"],
            mode="lines+markers",
            name="Silhouette Score",
        ))
        silhouette_fig.update_layout(
            xaxis_title="Number of clusters (K)",
            yaxis_title="Silhouette score (-1 to 1)",
            margin=dict(l=20, r=20, t=25, b=20),
        )
        st.plotly_chart(silhouette_fig, use_container_width=True)
        st.caption("A higher score generally means clusters are better separated, but business usefulness matters too.")

    st.dataframe(
        results.style.format({"Inertia": "{:.3f}", "Silhouette Score": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    valid_scores = results.dropna(subset=["Silhouette Score"])
    suggested_k = int(valid_scores.loc[valid_scores["Silhouette Score"].idxmax(), "Clusters (K)"])
    k_options = results["Clusters (K)"].astype(int).tolist()
    choice_col1, choice_col2 = st.columns([1, 2])
    with choice_col1:
        selection_mode = st.radio("K selection", ["Use best Silhouette suggestion", "Choose manually"], index=0)
    with choice_col2:
        if selection_mode == "Use best Silhouette suggestion":
            selected_k = suggested_k
            st.info(f"Suggested K = {suggested_k} (highest Silhouette Score in the tested range). Review the graphs before finalizing.")
        else:
            selected_k = st.selectbox("Final number of clusters (K)", options=k_options, index=k_options.index(suggested_k))

    final_model = KMeans(n_clusters=int(selected_k), random_state=RANDOM_STATE, n_init=10)
    final_labels = final_model.fit_predict(cluster_matrix)
    output_data = raw_data.copy()
    output_data["Cluster"] = final_labels

    st.header("5. Customer segments")
    counts = pd.Series(final_labels, name="Cluster").value_counts().sort_index().rename_axis("Cluster").reset_index(name="Customer Count")
    metric_cols = st.columns(min(4, len(counts)))
    for i, row in counts.iterrows():
        metric_cols[i % len(metric_cols)].metric(f"Cluster {int(row['Cluster'])}", f"{int(row['Customer Count']):,} customers")

    count_fig = px.bar(counts, x="Cluster", y="Customer Count", title="Customers per Cluster", text="Customer Count")
    count_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(count_fig, use_container_width=True)

    # Independent 2D PCA is used for visualization, regardless of the clustering space.
    st.subheader("2D visualization")
    try:
        if feature_matrix.shape[1] >= 2 and len(data) >= 2:
            visual_pca = PCA(n_components=2, svd_solver="full")
            coords = visual_pca.fit_transform(feature_matrix)
            plot_data = pd.DataFrame({
                "PC1": coords[:, 0],
                "PC2": coords[:, 1],
                "Cluster": final_labels.astype(str),
            })
            scatter_fig = px.scatter(
                plot_data,
                x="PC1",
                y="PC2",
                color="Cluster",
                title="Customer clusters projected into 2D (PCA)",
                hover_data={"PC1": ":.2f", "PC2": ":.2f"},
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
            st.caption("The plot is a 2D projection for visual inspection; some differences can be lost when high-dimensional data is projected into 2D.")
        else:
            st.info("A 2D PCA plot needs at least two processed feature dimensions. Cluster counts and the downloadable labels are still available.")
    except Exception as exc:
        st.warning(f"Could not draw the PCA visualization: {exc}")

    st.subheader("Cluster profile")
    profile = profile_clusters(data, features, final_labels)
    st.dataframe(profile.round(2), use_container_width=True, hide_index=True)
    st.caption("Numeric values are cluster averages. Categorical values are the most common category within each cluster.")

    st.subheader("Segmented customer records")
    st.dataframe(output_data.head(100), use_container_width=True, hide_index=True)
    st.download_button(
        label="Download segmented customers (CSV)",
        data=convert_csv_download(output_data),
        file_name="customer_segments.csv",
        mime="text/csv",
        type="primary",
    )
    st.download_button(
        label="Download cluster profile (CSV)",
        data=convert_csv_download(profile),
        file_name="cluster_profile.csv",
        mime="text/csv",
    )

    st.header("6. How to use this result")
    st.markdown(
        """
        - Use cluster profiles to describe segments from their actual behavior (for example, high-spend frequent buyers).
        - Do not name a segment using its cluster number alone; inspect its averages and common categories first.
        - Compare more than one K value. The highest Silhouette Score is a suggestion, not an automatic business decision.
        - Exclude identifiers such as customer IDs, names, phone numbers, and email addresses from the modeling features.
        - This demonstration runs clustering on the dataset you supply. Review data permissions and remove personal identifiers before sharing results.
        """
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
