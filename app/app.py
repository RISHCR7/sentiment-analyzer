import re
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# Config and model loading
# ------------------------------------------------------------------
st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="wide")

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "models" / "sentiment_pipeline.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"
LABELS = ["positive", "neutral", "negative"]
COLORS = {"positive": "#2e9e5b", "neutral": "#e0b100", "negative": "#d64545"}
EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😠"}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def clean(text):
    """Same cleaning used in preprocess.py (must match training)."""
    text = str(text).lower()
    text = re.sub(r"http\S+|[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict_with_confidence(model, texts):
    """Returns (labels, confidence) for a list of raw texts."""
    cleaned = [clean(t) for t in texts]
    labels = model.predict(cleaned)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(cleaned)
    else:  # LinearSVC has no probabilities, so use a softmax of the decision scores
        d = model.decision_function(cleaned)
        e = np.exp(d - d.max(axis=1, keepdims=True))
        scores = e / e.sum(axis=1, keepdims=True)

    return labels, scores, list(model.classes_)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.info("Run `python src/train.py` first, and make sure the scikit-learn version matches.")
    st.stop()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This app classifies customer reviews as **positive**, **neutral** or "
        "**negative** using a TF-IDF + machine learning pipeline trained on "
        "real app-store reviews."
    )
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        st.subheader("Model info")
        st.write(f"**Model:** {m['best_model']}")
        st.write(f"**Test accuracy:** {m['test_accuracy']:.2%}")
        st.write(f"**Test macro F1:** {m['test_macro_f1']:.2f}")
    st.caption("Built as an MCA end-to-end ML project.")

st.title("💬 Customer Review Sentiment Analyzer")
tab1, tab2 = st.tabs(["Single review", "Bulk CSV analysis"])

# ------------------------------------------------------------------
# Tab 1: single review
# ------------------------------------------------------------------
with tab1:
    examples = {
        "(write your own)": "",
        "Positive example": "Great app, delivery is always on time and the food arrives hot.",
        "Negative example": "Worst experience ever. Order was late and support never replied.",
        "Mixed example": "Food was good but the delivery took too long and the app keeps crashing.",
    }
    choice = st.selectbox("Try an example or write your own:", list(examples.keys()))
    text = st.text_area("Review text", value=examples[choice], height=150,
                        placeholder="Paste a customer review here...")

    if st.button("Analyze", type="primary"):
        if len(clean(text).split()) < 2:
            st.warning("Please enter a longer review (at least a couple of words).")
        else:
            labels, scores, classes = predict_with_confidence(model, [text])
            label = labels[0]
            conf = scores[0][classes.index(label)]

            st.markdown(f"### {EMOJI[label]} Sentiment: **{label.upper()}**")
            st.progress(float(conf), text=f"Confidence: {conf:.1%}")

            prob_df = pd.DataFrame({"sentiment": classes, "probability": scores[0]})
            fig = px.bar(prob_df, x="probability", y="sentiment", orientation="h",
                         color="sentiment", color_discrete_map=COLORS,
                         range_x=[0, 1], title="Class probabilities")
            fig.update_layout(showlegend=False, height=250)
            st.plotly_chart(fig, use_container_width=True)

            if conf < 0.5:
                st.info("The model is not very sure about this review. "
                        "It may be mixed, sarcastic, or very short.")

# ------------------------------------------------------------------
# Tab 2: bulk CSV
# ------------------------------------------------------------------
with tab2:
    st.write("Upload a CSV file containing a column of review text.")
    file = st.file_uploader("CSV file", type="csv")

    if file:
        try:
            data = pd.read_csv(file)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()

        text_cols = [c for c in data.columns if data[c].dtype == object]
        if not text_cols:
            st.error("No text column found in this CSV.")
            st.stop()

        default = next((i for i, c in enumerate(text_cols)
                        if c.lower() in ("review", "content", "text", "comment")), 0)
        col = st.selectbox("Which column has the review text?", text_cols, index=default)

        if st.button("Run bulk analysis", type="primary"):
            data = data.dropna(subset=[col]).copy()
            if len(data) > 20000:
                st.warning("Large file: using the first 20,000 rows.")
                data = data.head(20000)

            with st.spinner("Analyzing reviews..."):
                labels, scores, classes = predict_with_confidence(model, data[col].astype(str).tolist())
            data["sentiment"] = labels
            data["confidence"] = scores.max(axis=1).round(3)
            st.session_state["bulk"] = data
            st.session_state["bulk_col"] = col

    if "bulk" in st.session_state:
        data = st.session_state["bulk"]
        col = st.session_state["bulk_col"]

        counts = data["sentiment"].value_counts().reindex(LABELS).fillna(0).astype(int)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total reviews", len(data))
        c2.metric("😊 Positive", f"{counts['positive']} ({counts['positive']/len(data):.0%})")
        c3.metric("😐 Neutral", f"{counts['neutral']} ({counts['neutral']/len(data):.0%})")
        c4.metric("😠 Negative", f"{counts['negative']} ({counts['negative']/len(data):.0%})")

        left, right = st.columns(2)
        with left:
            pie = px.pie(names=counts.index, values=counts.values, color=counts.index,
                         color_discrete_map=COLORS, title="Sentiment split", hole=0.4)
            st.plotly_chart(pie, use_container_width=True)
        with right:
            data["length"] = data[col].astype(str).str.split().str.len()
            box = px.box(data, x="sentiment", y="length", color="sentiment",
                         color_discrete_map=COLORS,
                         category_orders={"sentiment": LABELS},
                         title="Review length by sentiment")
            box.update_layout(showlegend=False)
            st.plotly_chart(box, use_container_width=True)

        st.subheader("Results")
        pick = st.multiselect("Filter by sentiment", LABELS, default=LABELS)
        view = data[data["sentiment"].isin(pick)].drop(columns=["length"], errors="ignore")
        st.dataframe(view, use_container_width=True, height=350)

        st.download_button("Download results as CSV",
                           view.to_csv(index=False).encode("utf-8"),
                           "sentiment_results.csv", "text/csv")