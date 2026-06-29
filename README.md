# NLP Text Similarity Explorer

A Streamlit web app that uses the free pretrained model `all-MiniLM-L6-v2` (SentenceTransformers) to compute and visualize text/word similarity.

## Model
**`all-MiniLM-L6-v2`** — Free, open-source sentence embedding model from HuggingFace/SentenceTransformers. No training, no preprocessing, no paid API used.

## App Purpose
Enter words or short phrases and explore:
- How similar they are to a query word (bar chart)
- Pairwise similarity between all inputs (heatmap)
- How they cluster in 2D semantic space (PCA plot)

Critical thinking notes (Paul's Standards) are shown throughout.

## Streamlit App Link
> *(Add your Streamlit Community Cloud link here after deployment)*

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Screenshots
> *(Add screenshots after deployment)*

## Files
- `app.py` — Main Streamlit application
- `requirements.txt` — Required libraries
- `README.md` — This file
