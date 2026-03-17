import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="🎬 Movie Genre Predictor", layout="wide")

# ── Load model & vectorizer ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base       = os.path.dirname(os.path.abspath(__file__))
    vec_path   = os.path.join(base, 'Model files', 'tfidf_vectorizer.pkl')
    model_path = os.path.join(base, 'Model files', 'ensemble_model.pkl')
    vect  = joblib.load(vec_path)
    model = joblib.load(model_path)
    return vect, model

@st.cache_data
def load_default_test():
    base = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(base, 'Data', 'test_data.txt')
    ids, titles, descs = [], [], []
    if not os.path.exists(test_path):
        return pd.DataFrame()
    with open(test_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('ID :::'):
                continue
            parts = [p.strip() for p in line.split(' ::: ')]
            if len(parts) == 3:
                ids.append(parts[0]); titles.append(parts[1]); descs.append(parts[2])
            elif len(parts) >= 4:
                ids.append(parts[0]); titles.append(parts[1]); descs.append(parts[3])
    return pd.DataFrame({'ID': ids, 'Title': titles, 'Description': descs})

def parse_txt(uploaded):
    ids, titles, descs = [], [], []
    content = uploaded.getvalue().decode('utf-8', errors='ignore')
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('ID :::'):
            continue
        parts = [p.strip() for p in line.split(' ::: ')]
        if len(parts) == 3:
            ids.append(parts[0]); titles.append(parts[1]); descs.append(parts[2])
        elif len(parts) >= 4:
            ids.append(parts[0]); titles.append(parts[1]); descs.append(parts[3])
    return pd.DataFrame({'ID': ids, 'Title': titles, 'Description': descs})

def parse_csv(uploaded):
    df = pd.read_csv(uploaded)
    df.columns = [c.strip().lower() for c in df.columns]
    id_col    = next((c for c in df.columns if c in ['id','movie_id','film_id']), None)
    title_col = next((c for c in df.columns if c in ['title','movie_title','name','film']), None)
    desc_col  = next((c for c in df.columns if c in ['description','plot','summary','desc','overview']), None)
    if desc_col is None:
        st.error("CSV must have a column named: description, plot, summary, or overview")
        return pd.DataFrame()
    result = pd.DataFrame()
    result['ID']          = df[id_col].astype(str)    if id_col    else pd.RangeIndex(1, len(df)+1).astype(str)
    result['Title']       = df[title_col].astype(str) if title_col else ['—'] * len(df)
    result['Description'] = df[desc_col].astype(str)
    return result

def predict(df, vectorizer, model):
    X = vectorizer.transform(df['Description'])
    df = df.copy()
    df['Predicted Genre'] = model.predict(X)
    return df

def main():
    vectorizer, ensemble = load_model()

    st.title("🎬 Movie Genre Predictor")
    st.markdown("Predict the genre of movies from plot descriptions — upload a file or enter a description manually.")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📂 Upload File", "✏️ Single Description", "📊 Default Test Data"])

    with tab1:
        st.subheader("Upload a TXT or CSV file")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info("**TXT format:**\n```\n1 ::: Title ::: Description\n```")
        with col_info2:
            st.info("**CSV format:**\n```\nid, title, description\n```")

        uploaded = st.file_uploader("Choose a .txt or .csv file", type=["txt", "csv"])
        if uploaded:
            ext = uploaded.name.split('.')[-1].lower()
            df_upload = parse_txt(uploaded) if ext == 'txt' else parse_csv(uploaded)
            if not df_upload.empty:
                st.success(f"✅ Loaded **{len(df_upload)}** records from `{uploaded.name}`")
                result = predict(df_upload, vectorizer, ensemble)
                st.dataframe(result[['ID', 'Title', 'Predicted Genre']], use_container_width=True)
                csv_out = result[['ID', 'Title', 'Predicted Genre']].to_csv(index=False)
                st.download_button("⬇️ Download Predictions CSV", csv_out,
                                   file_name="predictions.csv", mime="text/csv")

    with tab2:
        st.subheader("Enter a Movie Description")
        single_desc = st.text_area("Paste a plot summary below:", height=180,
                                   placeholder="e.g. A young lion prince flees his kingdom after the murder of his father...")
        if st.button("🔍 Predict Genre", type="primary"):
            if single_desc.strip():
                X = vectorizer.transform([single_desc])
                genre = ensemble.predict(X)[0]
                st.success(f"🎬 **Predicted Genre: {genre.upper()}**")
            else:
                st.warning("Please enter a description first.")

    with tab3:
        st.subheader("Predictions on Default Test Data")
        st.caption("Predictions run on the bundled Data/test_data.txt file.")
        df_test = load_default_test()
        if df_test.empty:
            st.warning("test_data.txt not found in the Data folder.")
        else:
            result_test = predict(df_test, vectorizer, ensemble)
            st.info(f"📄 **{len(result_test)} records** from test_data.txt")
            genres = sorted(result_test['Predicted Genre'].unique())
            selected = st.multiselect("Filter by genre:", genres, default=genres)
            filtered = result_test[result_test['Predicted Genre'].isin(selected)]
            st.dataframe(filtered[['ID', 'Title', 'Predicted Genre']], use_container_width=True)
            st.bar_chart(filtered['Predicted Genre'].value_counts())
            csv_out = result_test[['ID', 'Title', 'Predicted Genre']].to_csv(index=False)
            st.download_button("⬇️ Download Full Predictions CSV", csv_out,
                               file_name="test_predictions.csv", mime="text/csv")

if __name__ == '__main__':
    main()
