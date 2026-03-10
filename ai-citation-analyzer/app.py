import streamlit as st
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="AI Citation Analyzer",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 AI Citation Analyzer")

st.write(
"Find which source sentences are most similar to an AI generated answer."
)

# ------------------------------------------------
# MODELS
# ------------------------------------------------

@st.cache_resource
def load_models():

    embed_model = SentenceTransformer("all-mpnet-base-v2")

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return embed_model, reranker


embed_model, reranker = load_models()


# ------------------------------------------------
# INPUT
# ------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    ai_answer = st.text_area(
        "AI Answer",
        height=220
    )

with col2:
    source_urls = st.text_area(
        "Source URLs (one per line)",
        height=220
    )

analyze = st.button("Analyze")


# ------------------------------------------------
# SCRAPER
# ------------------------------------------------

def scrape(url):

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }

    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    main = soup.find("article") or soup.find("main") or soup.body

    paragraphs = []

    for p in main.find_all("p"):
        t = p.get_text().strip()

        if len(t) > 80:
            paragraphs.append(t)

    return paragraphs


# ------------------------------------------------
# SENTENCE SPLIT
# ------------------------------------------------

def split_sentences(text):

    s = re.split(r'(?<=[.!?]) +', text)

    return [x.strip() for x in s if len(x) > 40]


# ------------------------------------------------
# ANALYSIS
# ------------------------------------------------

if analyze:

    urls = source_urls.split("\n")

    ai_sentences = split_sentences(ai_answer)

    results = []

    used_sources = set()

    for url in urls:

        url = url.strip()

        if not url:
            continue

        try:

            domain = urlparse(url).netloc

            paragraphs = scrape(url)

            sentences = []

            for p in paragraphs:
                sentences += split_sentences(p)

            sentences = list(dict.fromkeys(sentences))

            if not sentences:
                continue

            embeddings = embed_model.encode(sentences)

            for ai_sentence in ai_sentences:

                ai_embed = embed_model.encode([ai_sentence])

                sim = cosine_similarity(ai_embed, embeddings)[0]

                top_k = sim.argsort()[-10:][::-1]

                candidates = [sentences[i] for i in top_k]

                pairs = [(ai_sentence, c) for c in candidates]

                scores = reranker.predict(pairs)

                best_index = scores.argmax()

                best_sentence = candidates[best_index]

                best_score = scores[best_index]

                if best_sentence in used_sources:
                    continue

                if best_score > 0.3:

                    used_sources.add(best_sentence)

                    results.append({
                        "AI Sentence": ai_sentence,
                        "Matched Source": best_sentence,
                        "Domain": domain,
                        "Score": round(float(best_score), 3)
                    })

        except Exception as e:

            results.append({
                "AI Sentence": "Error",
                "Matched Source": str(e),
                "Domain": url,
                "Score": 0
            })

    if results:

        df = pd.DataFrame(results)

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "results.csv"
        )
