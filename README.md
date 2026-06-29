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
> https://z9afsuyathmt78maa2wdxi.streamlit.app

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run Theapp.py
```
## Screenshots
<img width="1460" height="719" alt="1" src="https://github.com/user-attachments/assets/b9a233af-cf5e-4dae-be2b-e420ba38e50d" />
<img width="1460" height="1169" alt="2" src="https://github.com/user-attachments/assets/8e38d55b-27f7-42e9-9b5b-c4f0084b1754" />
<img width="1460" height="1094" alt="3" src="https://github.com/user-attachments/assets/60afc47a-f49b-4bb7-9bac-133f24b81a84" />


## Files
- `Theapp.py` — Main Streamlit application
- `requirements.txt` — Required libraries
- `README.md` — This file
