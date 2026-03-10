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
# CLEAN UI STYLE
# ---------------------------------------------------

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

/* result cards */

.result-card{
    padding:22px;
    border-radius:12px;
    background:#ffffff;
    border:1px solid #e5e7eb;
    box-shadow:0 5px 16px rgba(0,0,0,0.06);
    margin-bottom:18px;
}

/* headings */

.card-title{
    font-weight:600;
    color:#111827;
    margin-bottom:5px;
}

/* ai text */

.ai-text{
    color:#1f2937;
}

/* source text */

.source-text{
    color:#4b5563;
}

/* similarity badge */

.similarity-badge{
    display:inline-block;
    margin-top:10px;
    padding:6px 12px;
    border-radius:8px;
    background:#eef2ff;
    color:#3730a3;
    font-weight:600;
    font-size:13px;
}

/* highlight matched words */

strong{
    color:#2563eb;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("🔎 AI Citation Analyzer")

st.markdown(
"""
Identify which **web sources likely influenced an AI-generated answer** using **semantic similarity analysis**.
"""
)

st.divider()


# ---------------------------------------------------
# INPUT
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    ai_answer = st.text_area(
        "🧠 AI Answer",
        height=220,
        placeholder="Paste AI generated answer..."
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
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ---------------------------------------------------
# WORD HIGHLIGHT FUNCTION
# ---------------------------------------------------

def highlight_overlap(ai_sentence, source_sentence):

    ai_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', ai_sentence.lower()))

    highlighted = []

    for word in source_sentence.split():

        clean = re.sub(r'\W+', '', word.lower())

        if clean in ai_words:
            highlighted.append(f"<strong>{word}</strong>")
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
        session.headers.update({"User-Agent":"Mozilla/5.0"})


        for url in urls:

            url = url.strip()

            if not url:
                continue

            try:

                domain = urlparse(url).netloc

                response = session.get(url,timeout=15)
                response.raise_for_status()

                soup = BeautifulSoup(response.text,"html.parser")

                main = soup.find("article") or soup.find("main") or soup.body

                content = main.find_all(["h1","h2","h3","p"])

                paragraphs=[]
                section_map=[]

                current_section="Introduction"

                for tag in content:

                    if tag.name in ["h1","h2","h3"]:
                        current_section = tag.get_text().strip()

                    if tag.name=="p":

                        text = tag.get_text().strip()

                        if text:
                            paragraphs.append(text)
                            section_map.append(current_section)


                sentences=[]
                sentence_section_map=[]


                for paragraph,section in zip(paragraphs,section_map):

                    split_sentences = re.split(r'(?<=[.!?]) +',paragraph)

                    clean=[s.strip() for s in split_sentences if len(s)>20]

                    for s in clean:
                        sentences.append(s)
                        sentence_section_map.append(section)


                # remove duplicates
                sentences = list(dict.fromkeys(sentences))

                article_embeddings = model.encode(sentences)


                for ai_sentence in ai_sentences:

                    ai_embedding = model.encode([ai_sentence])

                    similarity_scores = cosine_similarity(
                        ai_embedding,
                        article_embeddings
                    )

                    best_index = similarity_scores.argmax()

                    best_sentence = sentences[best_index]

                    score = similarity_scores[0][best_index]

                    results.append({

                        "AI Sentence":ai_sentence,
                        "Domain":domain,
                        "Source URL":url,
                        "Similarity (%)":round(score*100,2),
                        "Matched Sentence":best_sentence

                    })


            except Exception as e:

                results.append({

                    "AI Sentence":"Error",
                    "Domain":"-",
                    "Source URL":url,
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

            colA,colB,colC = st.columns(3)

            colA.metric("AI Sentences",len(ai_sentences))
            colB.metric("Sources",len(urls))
            colC.metric("Matches",len(df))

            st.dataframe(df,width="stretch",height=450)


            st.divider()
            st.subheader("🧩 Highlighted Matches")


            for _,row in df.head(10).iterrows():

                highlighted = highlight_overlap(
                    row["AI Sentence"],
                    row["Matched Sentence"]
                )

                st.markdown(f"""
<div class="result-card">

<div class="card-title">AI Sentence</div>
<div class="ai-text">{row['AI Sentence']}</div>

<br>

<div class="card-title">Matched Source</div>
<div class="source-text">{highlighted}</div>

<span class="similarity-badge">
Similarity: {row['Similarity (%)']}%
</span>

</div>
""",unsafe_allow_html=True)


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
