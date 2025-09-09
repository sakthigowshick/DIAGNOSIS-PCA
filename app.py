import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

# ---------------------------
# Load dataset
# ---------------------------
data = pd.read_csv("New_file.csv")

# Drop target column (Diagnosis)
X = data.drop("Diagnosis", axis=1).select_dtypes(include=['float64', 'int64']).values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Train base models
# ---------------------------
hierarchical_model = AgglomerativeClustering(n_clusters=3).fit(X_scaled)
dbscan_model = DBSCAN(eps=1.5, min_samples=5).fit(X_scaled)

if len(dbscan_model.core_sample_indices_) > 0:
    dbscan_core_samples = dbscan_model.components_
    dbscan_core_labels = dbscan_model.labels_[dbscan_model.core_sample_indices_]
else:
    dbscan_core_samples, dbscan_core_labels = np.array([]), np.array([])

# ---------------------------
# Prediction + Visualization
# ---------------------------
def predict_cluster(algorithm, k, *features):
    features_scaled = scaler.transform([features])

    if algorithm == "KMeans":
        model = KMeans(n_clusters=int(k), random_state=42).fit(X_scaled)
        cluster = model.predict(features_scaled)[0]
        labels = model.labels_
        title = f"KMeans (k={k})"
  
    elif algorithm == "Hierarchical":
        new_data = np.vstack([X_scaled, features_scaled])
        labels = AgglomerativeClustering(n_clusters=int(k)).fit_predict(new_data)
        cluster = labels[-1]
        title = f"Hierarchical (k={k})"

        # PCA Visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(new_data)

        plt.figure(figsize=(6, 5))
        plt.scatter(X_pca[:-1, 0], X_pca[:-1, 1], c=labels[:-1], cmap="tab10", alpha=0.6)
        plt.scatter(X_pca[-1, 0], X_pca[-1, 1], c="red", marker="*", s=200, edgecolors="black", label="New Sample")
        plt.title(f"{title} — New sample → Cluster {cluster}")
        plt.legend()
    
        return f"✅ Belongs to Cluster {cluster} ({title})", plt

    elif algorithm == "DBSCAN":
        if dbscan_core_samples.size == 0:
            return "⚠ DBSCAN found no clusters.", None
        dists = euclidean_distances(features_scaled, dbscan_core_samples)
        nearest_idx = np.argmin(dists)
        nearest_dist = dists[0, nearest_idx]
        cluster = dbscan_core_labels[nearest_idx]
        if nearest_dist > dbscan_model.eps:
            return f"🚨 OUTLIER (noise, dist={nearest_dist:.2f})", None
        labels = dbscan_model.labels_
        title = f"DBSCAN (eps={dbscan_model.eps})"

    # ---- Visualization (PCA 2D) ----
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    new_point = pca.transform(features_scaled)

    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.scatter(new_point[:, 0], new_point[:, 1], c="red", marker="*", s=200, edgecolors="black", label="New Sample")
    plt.title(f"{title} — New sample → Cluster {cluster}")
    plt.legend()
    
    return f"✅ Belongs to Cluster {cluster} ({title})", plt

# ---------------------------
# Login Validation
# ---------------------------
def login(username, password):
    if username == "admin" and password == "1234":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(value="❌ Invalid login. Try again."), gr.update(visible=False)

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks() as demo:
    # Page 1: Login
    with gr.Row(visible=True) as login_page:
        with gr.Column():
            gr.Markdown("## 🔑 Login to Access Breast Cancer Clustering App")
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_msg = gr.Textbox(label="Login Status")

    # Page 2: Clustering App
    with gr.Row(visible=False) as app_page:
        with gr.Column():
            gr.Markdown("## 🧬 Breast Cancer Clustering App")

            algorithm = gr.Dropdown(["KMeans", "Hierarchical", "DBSCAN"], label="Select Algorithm")
            k_value = gr.Number(label="Number of Clusters (k for KMeans)", value=3)

            inputs = []
            with gr.Accordion("Enter Feature Values", open=False):
                for col in data.drop("Diagnosis", axis=1).select_dtypes(include=['float64', 'int64']).columns:
                    inputs.append(gr.Number(label=col, value=float(data[col].median())))

            btn = gr.Button("Find Cluster")
            output_text = gr.Textbox(label="Result")
            output_plot = gr.Plot()

    # Button Actions
    login_btn.click(fn=login, inputs=[username, password], outputs=[login_msg, app_page])
    btn.click(fn=predict_cluster, inputs=[algorithm, k_value] + inputs, outputs=[output_text, output_plot])

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch()
