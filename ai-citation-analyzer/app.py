import streamlit as st
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Citation Analyzer",
    page_icon="🔎",
    layout="wide"
)

# ---------------------------------------------------
# STYLE
# ---------------------------------------------------

st.markdown(
"""
<style>

.block-container {
    padding-top: 2rem;
}

.metric-box {
    padding:20px;
    border-radius:10px;
    background:#f6f7fb;
}

.result-card{
    padding:20px;
    border-radius:10px;
    background:#ffffff;
    box-shadow:0 4px 10px rgba(0,0,0,0.05);
    margin-bottom:15px;
}

</style>
""",
unsafe_allow_html=True
)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("🔎 AI Citation Analyzer")

st.markdown(
"""
Detect which **sentences and sources** influenced an AI-generated answer.

Paste an AI answer and source URLs to analyze **semantic similarity and attribution**.
"""
)

st.divider()

# ---------------------------------------------------
# INPUT AREA
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    ai_answer = st.text_area(
        "🧠 AI Answer",
        height=220,
        placeholder="Paste the AI-generated answer here..."
    )

with col2:

    source_urls = st.text_area(
        "🌐 Source URLs (one per line)",
        height=220,
        placeholder="https://example.com/article"
    )

st.divider()

analyze = st.button("🚀 Analyze Sources", use_container_width=True)

# ---------------------------------------------------
# LOAD MODEL (CACHED)
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------------------------------
# HIGHLIGHT FUNCTION
# ---------------------------------------------------

def highlight_overlap(ai_sentence, source_sentence):

    ai_words = set(re.findall(r'\w+', ai_sentence.lower()))

    highlighted = []

    for word in source_sentence.split():

        clean = re.sub(r'\W+', '', word.lower())

        if clean in ai_words:
            highlighted.append(f"**{word}**")
        else:
            highlighted.append(word)

    return " ".join(highlighted)

# ---------------------------------------------------
# ANALYSIS
# ---------------------------------------------------

if analyze:

    with st.spinner("Analyzing sources..."):

        urls = source_urls.split("\n")
        results = []

        ai_sentences = re.split(r'(?<=[.!?]) +', ai_answer)
        ai_sentences = [s.strip() for s in ai_sentences if len(s) > 20]

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})

        for url in urls:

            url = url.strip()

            if not url:
                continue

            try:

                domain = urlparse(url).netloc

                response = session.get(url, timeout=15)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                main = soup.find("article") or soup.find("main") or soup.body

                content = main.find_all(["h1","h2","h3","p"])

                paragraphs = []
                section_map = []

                current_section = "Introduction"

                for tag in content:

                    if tag.name in ["h1","h2","h3"]:
                        current_section = tag.get_text().strip()

                    if tag.name == "p":

                        text = tag.get_text().strip()

                        if text:
                            paragraphs.append(text)
                            section_map.append(current_section)

                sentences = []
                sentence_section_map = []

                for paragraph, section in zip(paragraphs, section_map):

                    split_sentences = re.split(r'(?<=[.!?]) +', paragraph)

                    clean = [s.strip() for s in split_sentences if len(s) > 20]

                    for s in clean:
                        sentences.append(s)
                        sentence_section_map.append(section)

                    for i in range(len(clean)-1):
                        sentences.append(clean[i]+" "+clean[i+1])
                        sentence_section_map.append(section)

                unique = list(dict.fromkeys(zip(sentences,sentence_section_map)))
                sentences, sentence_section_map = zip(*unique)

                article_embeddings = model.encode(sentences)

                for ai_sentence in ai_sentences:

                    ai_embedding = model.encode([ai_sentence])

                    similarity_scores = cosine_similarity(
                        ai_embedding,
                        article_embeddings
                    )

                    best_index = similarity_scores.argmax()

                    best_sentence = sentences[best_index]
                    best_section = sentence_section_map[best_index]

                    score = similarity_scores[0][best_index]

                    results.append({
                        "AI Sentence": ai_sentence,
                        "Domain": domain,
                        "Source URL": url,
                        "Section": best_section,
                        "Similarity (%)": round(score*100,2),
                        "Matched Sentence": best_sentence
                    })

            except Exception as e:

                results.append({
                    "AI Sentence":"Error",
                    "Domain":"-",
                    "Source URL":url,
                    "Section":"Error",
                    "Similarity (%)":0,
                    "Matched Sentence":str(e)
                })

# ---------------------------------------------------
# RESULTS
# ---------------------------------------------------

        if results:

            df = pd.DataFrame(results)

            df = df.sort_values(by="Similarity (%)",ascending=False)

            st.divider()
            st.subheader("📊 Citation Analysis Results")

            colA, colB, colC = st.columns(3)

            with colA:
                st.metric("AI Sentences", len(ai_sentences))

            with colB:
                st.metric("Sources Analyzed", len(urls))

            with colC:
                st.metric("Matches Found", len(df))

            st.dataframe(df,width="stretch",height=450)

            st.divider()

            st.subheader("🧩 Highlighted Matches")

            for _,row in df.head(10).iterrows():

                highlighted = highlight_overlap(
                    row["AI Sentence"],
                    row["Matched Sentence"]
                )

                st.markdown(
                f"""
                <div class="result-card">

                <b>AI Sentence</b><br>
                {row['AI Sentence']}<br><br>

                <b>Matched Source</b><br>
                {highlighted}<br><br>

                <b>Similarity:</b> {row['Similarity (%)']}%

                </div>
                """,
                unsafe_allow_html=True
                )

            st.divider()

            st.subheader("🌍 Domain Influence")

            summary = (
                df.groupby("Domain")["Similarity (%)"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )

            st.bar_chart(summary.set_index("Domain"))

            st.download_button(
                "📥 Download CSV",
                df.to_csv(index=False),
                "ai_citation_results.csv",
                "text/csv"
            )
