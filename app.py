import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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
# PCA Function
# ---------------------------
def run_pca(n_components):
    pca = PCA(n_components=int(n_components))
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    explained_var = pca.explained_variance_ratio_

    # Plot PCA
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c="blue", alpha=0.6)
    plt.title(f"PCA with {n_components} Components")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)

    return f"Explained Variance Ratio: {explained_var}", plt

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
            gr.Markdown("## 🔑 Login to Access Breast Cancer PCA App")
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_msg = gr.Textbox(label="Login Status")

    # Page 2: PCA App
    with gr.Row(visible=False) as app_page:
        with gr.Column():
            gr.Markdown("## 🧬 Breast Cancer PCA Visualization")

            n_components = gr.Slider(2, min(len(data.columns)-1, 10), step=1, value=2, label="Number of PCA Components")

            btn = gr.Button("Run PCA")
            output_text = gr.Textbox(label="Explained Variance")
            output_plot = gr.Plot()

    # Button Actions
    login_btn.click(fn=login, inputs=[username, password], outputs=[login_msg, app_page])
    btn.click(fn=run_pca, inputs=[n_components], outputs=[output_text, output_plot])

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch()
