import streamlit as st
import pandas as pd
import joblib
import os

# First Streamlit call must be configuration
st.set_page_config(page_title="🎬 Movie Genre Predictor", layout="wide")

def main():
    if 'predict_clicked' not in st.session_state:
        st.session_state.predict_clicked = False

    st.title("🎬 Movie Genre Predictor")

    # ── Load model & vectorizer ──────────────────────────────────────────────
    @st.cache_resource
    def load_model():
        base = os.path.dirname(os.path.abspath(__file__))
        vec_path   = os.path.join(base, 'Model files', 'tfidf_vectorizer.pkl')
        model_path = os.path.join(base, 'Model files', 'ensemble_model.pkl')
        vect  = joblib.load(vec_path)
        model = joblib.load(model_path)
        return vect, model

    vectorizer, ensemble = load_model()

    # ── Layout ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "Upload text file (ID ::: Title ::: Description)",
            type=["txt"],
            key="file_uploader"
        )

    with col2:
        single_desc = st.text_area("Or enter a movie description and click Submit:")
        submit_single = st.button("Submit", key="single_submit")

    # ── Parse uploaded file ──────────────────────────────────────────────────
    def parse_uploaded_file(uploaded):
        ids, titles, descs = [], [], []
        content = uploaded.getvalue().decode('utf-8', errors='ignore')
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('ID :::'):
                continue
            parts = [p.strip() for p in line.split(' ::: ')]
            if len(parts) >= 3:
                ids.append(parts[0])
                titles.append(parts[1])
                descs.append(parts[-1])
        return pd.DataFrame({'id': ids, 'title': titles, 'description': descs})

    # ── Batch prediction ─────────────────────────────────────────────────────
    if uploaded_file and not st.session_state.predict_clicked:
        df_file = parse_uploaded_file(uploaded_file)
        if not df_file.empty:
            X = vectorizer.transform(df_file['description'])
            df_file['PredictedGenre'] = ensemble.predict(X)
            st.subheader("Batch Predictions")
            st.dataframe(df_file[['id', 'title', 'PredictedGenre']])
        else:
            st.warning("Uploaded file contains no valid entries.")

    # ── Single description prediction ────────────────────────────────────────
    if submit_single and single_desc:
        X_single = vectorizer.transform([single_desc])
        genre_single = ensemble.predict(X_single)[0]
        st.session_state.predict_clicked = True
        st.subheader("Prediction Result")
        st.success(f"**Predicted Genre:** {genre_single}")

if __name__ == '__main__':
    main()
