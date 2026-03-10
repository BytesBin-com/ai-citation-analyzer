import streamlit as st
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Citation Tracker",
    layout="wide"
)

st.title("AI Citation Tracker for SEO")

st.write(
"Paste an AI generated answer and page URLs. "
"The tool identifies which **page sections (headers + sentences)** "
"are most similar to the AI response."
)


# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ---------------------------------------------------
# INPUT
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    ai_text = st.text_area(
        "AI Answer",
        height=260
    )

with col2:
    urls_text = st.text_area(
        "Source URLs (one per line)",
        height=260
    )

run = st.button("Analyze")


# ---------------------------------------------------
# TEXT SPLIT
# ---------------------------------------------------

def split_sentences(text):

    sentences = re.split(r'(?<=[.!?]) +', text)

    return [s.strip() for s in sentences if len(s) > 30]


# ---------------------------------------------------
# HIGHLIGHT WORD MATCHES
# ---------------------------------------------------

def highlight(ai_sentence, source_sentence):

    ai_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', ai_sentence.lower()))

    words = []

    for w in source_sentence.split():

        clean = re.sub(r'\W+', '', w.lower())

        if clean in ai_words:
            words.append(f"<strong>{w}</strong>")
        else:
            words.append(w)

    return " ".join(words)


# ---------------------------------------------------
# SCRAPE PAGE
# ---------------------------------------------------

def scrape_page(url):

    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    main = soup.find("article") or soup.find("main") or soup.body

    current_header = "Introduction"

    sections = []

    for tag in main.find_all(["h1","h2","h3","p"]):

        if tag.name in ["h1","h2","h3"]:

            current_header = tag.get_text().strip()

        if tag.name == "p":

            paragraph = tag.get_text().strip()

            sentences = split_sentences(paragraph)

            for s in sentences:

                sections.append({
                    "header": current_header,
                    "sentence": s
                })

    return sections


# ---------------------------------------------------
# ANALYSIS
# ---------------------------------------------------

if run:

    urls = urls_text.split("\n")

    ai_sentences = split_sentences(ai_text)

    results = []

    for url in urls:

        url = url.strip()

        if not url:
            continue

        try:

            domain = urlparse(url).netloc

            sections = scrape_page(url)

            source_sentences = [s["sentence"] for s in sections]

            embeddings = model.encode(source_sentences)

            for ai_sentence in ai_sentences:

                ai_embedding = model.encode([ai_sentence])

                scores = cosine_similarity(ai_embedding, embeddings)[0]

                best_index = scores.argmax()

                best_score = scores[best_index]

                if best_score > 0.45:

                    results.append({

                        "AI Sentence": ai_sentence,

                        "Domain": domain,

                        "URL": url,

                        "Header": sections[best_index]["header"],

                        "Matched Sentence": sections[best_index]["sentence"],

                        "Similarity (%)": round(best_score*100,2)

                    })

        except Exception as e:

            results.append({

                "AI Sentence": "Error",

                "Domain": "-",

                "URL": url,

                "Header": "-",

                "Matched Sentence": str(e),

                "Similarity (%)": 0

            })


# ---------------------------------------------------
# RESULTS TABLE
# ---------------------------------------------------

    if results:

        df = pd.DataFrame(results)

        df = df.sort_values(by="Similarity (%)", ascending=False)

        st.subheader("Citation Matches")

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "citation_results.csv"
        )


# ---------------------------------------------------
# MATCH DETAILS
# ---------------------------------------------------

        st.subheader("Match Details")

        for _, row in df.head(10).iterrows():

            highlighted = highlight(
                row["AI Sentence"],
                row["Matched Sentence"]
            )

            st.markdown(f"""
**AI Sentence**

{row['AI Sentence']}

**Matched Header**

{row['Header']}

**Matched Sentence**

{highlighted}

**Similarity:** {row['Similarity (%)']}%
---
""", unsafe_allow_html=True)

    else:

        st.warning("No matches found.")
