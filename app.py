import numpy as np
import streamlit as st

from prediction import predict

st.set_page_config(page_title="Iris Classifier", page_icon="🌿", layout="wide")

# Theming with better contrast and icon fonts
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');
        html, body, [class*="css"] {
            font-family: 'Space Grotesk', system-ui, -apple-system, sans-serif;
        }
        .main {
            background: linear-gradient(180deg, #eef2f7 0%, #f9fbff 30%, #ffffff 100%);
        }
        .hero {
            padding: 1.5rem 1.75rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0b3954, #166699);
            color: #f8fbff;
            margin-bottom: 1.25rem;
            box-shadow: 0 15px 40px rgba(12, 64, 120, 0.25);
        }
        .hero h1 { color: #fefefe; }
        .hero p { color: #fefefe; }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.15);
            font-size: 0.85rem;
            letter-spacing: 0.02em;
        }
        .card {
            background: #ffffff;
            border: 1px solid #e5e9f2;
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 12px 40px rgba(11, 57, 84, 0.08);
        }
        .stSlider label, .stMarkdown p, .stMarkdown li, .stText {
            color: #1a2733;
        }
        .stSlider > div[data-baseweb="slider"] {
            padding-top: 0.35rem;
        }
        .section-title {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin-bottom: 0.35rem;
        }
        .section-title i {
            color: #166699;
        }
        .result-box {
            font-size: 1.1rem;
            padding: 0.95rem 1.05rem;
            border-radius: 12px;
            background: #0b3954;
            border: 1px solid #082739;
            color: #f3f7fb;
            box-shadow: 0 10px 30px rgba(11, 57, 84, 0.25);
        }
        .accent-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(22, 102, 153, 0.08);
            color: #0b3954;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .tip-list li { margin-bottom: 0.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="badge">Week 13 · Iris KNN</div>
        <h1 style="margin:0.4rem 0 0.2rem;">Classify Iris Flowers</h1>
        <p style="margin:0;">Set sepal and petal measurements to predict whether the flower is setosa,
        versicolor, or virginica. Model artifacts load automatically; if missing, a fallback model is trained.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title"><i class="fa-solid fa-leaf"></i><h3 style="margin:0;">Sepal characteristics</h3></div>', unsafe_allow_html=True)
        with st.container():
            sepal_l = st.slider("Sepal length (cm)", 1.0, 8.0, 5.0, 0.1)
            sepal_w = st.slider("Sepal width (cm)", 2.0, 4.4, 3.0, 0.1)

    with col2:
        st.markdown('<div class="section-title"><i class="fa-solid fa-seedling"></i><h3 style="margin:0;">Petal characteristics</h3></div>', unsafe_allow_html=True)
        with st.container():
            petal_l = st.slider("Petal length (cm)", 1.0, 7.0, 4.0, 0.1)
            petal_w = st.slider("Petal width (cm)", 0.1, 2.5, 1.2, 0.1)

st.markdown("")

cta_col, info_col = st.columns([0.55, 0.45])
with cta_col:
    st.markdown("#### Prediction")
    if st.button("Predict Iris species", use_container_width=True):
        result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
        st.markdown(f"<div class='result-box'>Prediction: <strong>{result[0]}</strong></div>", unsafe_allow_html=True)
    else:
        st.caption("Adjust the sliders, then click predict to see the model output.")

with info_col:
    st.markdown('<div class="section-title"><i class="fa-solid fa-lightbulb"></i><h4 style="margin:0;">Tips</h4></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul class="tip-list">
            <li>Typical sepal length ranges 4–7 cm, petal length 1–6.5 cm.</li>
            <li>Setosa usually has very small petals (&lt; 2 cm).</li>
            <li>Versicolor tends to sit in the middle; Virginica often has longer petals.</li>
            <li>Model loads saved KNN artifacts (<code>knn_model.sav</code> / <code>modelKNN1.pkl</code>) or trains a fallback automatically.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
