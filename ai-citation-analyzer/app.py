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

st.markdown("""
<style>

.block-container{padding-top:2rem;}

.result-card{
padding:22px;
border-radius:12px;
background:#ffffff;
border:1px solid #e5e7eb;
box-shadow:0 5px 16px rgba(0,0,0,0.06);
margin-bottom:18px;
}

.card-title{font-weight:600;color:#111827;margin-bottom:5px;}
.ai-text{color:#1f2937;}
.source-text{color:#4b5563;}

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

strong{color:#2563eb;}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("🔎 AI Citation Analyzer")

st.markdown("""
Identify which **web sources likely influenced an AI-generated answer**
using **semantic + keyword similarity analysis**.
""")

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
    return SentenceTransformer("all-mpnet-base-v2")

model = load_model()

# ---------------------------------------------------
# SIMPLE CLAIM EXTRACTION
# ---------------------------------------------------

def extract_claim(sentence):
    words = sentence.split()
    if len(words) < 5:
        return sentence
    return " ".join(words[:8])  # simple heuristic

# ---------------------------------------------------
# KEYWORD OVERLAP
# ---------------------------------------------------

def keyword_overlap(a, b):

    a_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', a.lower()))
    b_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', b.lower()))

    if not a_words:
        return 0

    overlap = a_words.intersection(b_words)

    return len(overlap) / len(a_words)

# ---------------------------------------------------
# HIGHLIGHT
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
# SCRAPER
# ---------------------------------------------------

def scrape_article(url):

    headers = {
        "User-Agent":"Mozilla/5.0",
        "Accept":"text/html",
        "Accept-Language":"en-US,en;q=0.9",
        "Referer":"https://google.com"
    }

    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text,"html.parser")

    main = soup.find("article") or soup.find("main") or soup.body

    paragraphs = []

    for p in main.find_all("p"):
        text = p.get_text().strip()
        if len(text) > 80:
            paragraphs.append(text)

    return paragraphs

# ---------------------------------------------------
# ANALYSIS
# ---------------------------------------------------

if analyze:

    with st.spinner("Analyzing sources..."):

        urls = source_urls.split("\n")
        results = []

        ai_sentences = re.split(r'(?<=[.!?]) +', ai_answer)
        ai_sentences = [s.strip() for s in ai_sentences if len(s) > 40]

        used_sources = set()

        for url in urls:

            url = url.strip()

            if not url:
                continue

            try:

                domain = urlparse(url).netloc

                paragraphs = scrape_article(url)

                sentences = []

                for para in paragraphs:
                    split = re.split(r'(?<=[.!?]) +', para)
                    clean = [s.strip() for s in split if len(s) > 40]
                    sentences.extend(clean)

                sentences = list(dict.fromkeys(sentences))

                if not sentences:
                    continue

                para_embeddings = model.encode(paragraphs)
                sent_embeddings = model.encode(sentences)

                for ai_sentence in ai_sentences:

                    claim = extract_claim(ai_sentence)

                    ai_emb = model.encode([claim])

                    best_score = 0
                    best_match = ""

                    # paragraph matching

                    p_scores = cosine_similarity(ai_emb, para_embeddings)[0]
                    top_p = p_scores.argsort()[-3:][::-1]

                    for i in top_p:

                        para = paragraphs[i]

                        sem = p_scores[i]
                        key = keyword_overlap(ai_sentence, para)

                        score = (sem*0.7)+(key*0.3)

                        if score > best_score:
                            best_score = score
                            best_match = para

                    # sentence matching

                    s_scores = cosine_similarity(ai_emb, sent_embeddings)[0]
                    top_s = s_scores.argsort()[-5:][::-1]

                    for i in top_s:

                        sent = sentences[i]

                        if sent in used_sources:
                            continue

                        sem = s_scores[i]
                        key = keyword_overlap(ai_sentence, sent)

                        score = (sem*0.7)+(key*0.3)

                        if score > best_score:
                            best_score = score
                            best_match = sent

                    if best_score > 0.45:

                        used_sources.add(best_match)

                        results.append({
                            "AI Sentence": ai_sentence,
                            "Domain": domain,
                            "Source URL": url,
                            "Similarity (%)": round(best_score*100,2),
                            "Matched Sentence": best_match
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

            df = df.sort_values(by="Similarity (%)", ascending=False)

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
""", unsafe_allow_html=True)

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
