import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="NLP Word/Text Similarity", page_icon="🔍", layout="wide")

st.title("🔍 NLP Text Similarity Explorer")
st.markdown("**Model:** `all-MiniLM-L6-v2` (Free pretrained — SentenceTransformers)")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.sidebar.header("Input")
default_texts = "king\nqueen\nman\nwoman\ndoctor\nnurse\ncar\nbicycle"
raw_input = st.sidebar.text_area("Enter words or phrases (one per line):", value=default_texts, height=200)
texts = [t.strip() for t in raw_input.strip().split("\n") if t.strip()]

query = st.sidebar.text_input("Query word/phrase for similarity ranking:", value="king")

if len(texts) < 2:
    st.warning("Please enter at least 2 words or phrases.")
    st.stop()

embeddings = model.encode(texts, convert_to_tensor=True)
embeddings_np = embeddings.cpu().numpy()

# Compute pairwise similarity
from sentence_transformers import util as st_util
sim_matrix = st_util.cos_sim(embeddings, embeddings).cpu().numpy()

# Query similarity
if query.strip():
    query_emb = model.encode([query.strip()], convert_to_tensor=True)
    query_scores = st_util.cos_sim(query_emb, embeddings).cpu().numpy()[0]
else:
    query_scores = sim_matrix[0]

# ─── Critical Thinking: Clarity ───
st.markdown("---")
st.subheader("📌 Clarity — What was given and what does output mean?")
st.info(
    f"**Input:** {len(texts)} text items were given: `{', '.join(texts)}`.\n\n"
    f"**Query:** `{query}` is compared against all inputs using cosine similarity of their sentence embeddings.\n\n"
    "**Output:** Cosine similarity scores range from -1 to 1. A score closer to **1** means the texts are semantically very similar."
)

# ─── Graph 1: Bar Chart ───
st.markdown("---")
st.subheader("📊 Graph 1 — Bar Chart: Similarity Scores vs Query")

sorted_idx = np.argsort(query_scores)[::-1]
sorted_labels = [texts[i] for i in sorted_idx]
sorted_scores = [query_scores[i] for i in sorted_idx]

fig1, ax1 = plt.subplots(figsize=(8, 4))
colors = ["#4CAF50" if s > 0.7 else "#2196F3" if s > 0.4 else "#FF5722" for s in sorted_scores]
bars = ax1.barh(sorted_labels[::-1], sorted_scores[::-1], color=colors[::-1])
ax1.set_xlabel("Cosine Similarity Score")
ax1.set_title(f"Similarity of each item to: '{query}'")
ax1.set_xlim(0, 1)
for bar, score in zip(bars, sorted_scores[::-1]):
    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f"{score:.3f}", va='center', fontsize=9)
plt.tight_layout()
st.pyplot(fig1)

st.markdown("**Precision & Relevance:** Exact scores shown. Top result is the most semantically similar to your query according to the model.")

# ─── Graph 2: Heatmap ───
st.markdown("---")
st.subheader("🌡️ Graph 2 — Heatmap: Pairwise Similarity Matrix")

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(
    sim_matrix,
    xticklabels=texts,
    yticklabels=texts,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    ax=ax2,
    vmin=0, vmax=1
)
ax2.set_title("Pairwise Cosine Similarity Heatmap")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig2)

st.markdown("**Logic:** The heatmap shows all pairwise similarities. Diagonal is always 1.0 (identical). Semantically related pairs should show higher scores.")

# ─── Graph 3: 2D PCA Embedding Plot ───
st.markdown("---")
st.subheader("🗺️ Graph 3 — 2D PCA Embedding Plot")

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings_np)

fig3, ax3 = plt.subplots(figsize=(8, 6))
scatter = ax3.scatter(reduced[:, 0], reduced[:, 1], c=range(len(texts)), cmap="tab10", s=100, zorder=3)
for i, label in enumerate(texts):
    ax3.annotate(label, (reduced[i, 0], reduced[i, 1]), textcoords="offset points", xytext=(8, 4), fontsize=10)
ax3.set_title("2D PCA Projection of Sentence Embeddings")
ax3.set_xlabel("PCA Component 1")
ax3.set_ylabel("PCA Component 2")
ax3.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
st.pyplot(fig3)

st.markdown("**Significance:** Words that appear close together in 2D space share similar semantic meaning in the model's high-dimensional embedding space.")

# ─── Critical Thinking Summary ───
st.markdown("---")
st.subheader("🧠 Paul's Critical Thinking Standards — Summary")

top_match = sorted_labels[0]
top_score = sorted_scores[0]

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
| Standard | Application |
|---|---|
| **Clarity** | Input: `{', '.join(texts[:4])}...` → output is cosine similarity |
| **Accuracy** | Model: `all-MiniLM-L6-v2` (verified, open-source HuggingFace) |
| **Precision** | Scores shown to 3 decimal places (e.g. `{top_score:.3f}`) |
| **Relevance** | All 3 graphs directly visualize the same similarity scores |
""")
with col2:
    st.markdown(f"""
| Standard | Application |
|---|---|
| **Logic** | `{top_match}` scores highest (`{top_score:.3f}`) against `{query}` — semantically expected |
| **Significance** | Most important result: `{top_match}` is the closest semantic match |
| **Fairness** | **Limitation:** Model may reflect training data biases (e.g. gender stereotypes in word associations) |
""")

st.markdown("---")
st.caption("Built with SentenceTransformers · all-MiniLM-L6-v2 · No preprocessing · No training · Free model only")
