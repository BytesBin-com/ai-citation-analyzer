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
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 AI Citation Tracker (SEO Analysis Tool)")

st.write(
"""
Paste an **AI generated answer** and **source URLs**.

The tool identifies which **sections of those pages (headers + paragraphs)** 
are most similar to the AI answer so SEO teams can understand 
what type of content AI systems surface.
"""
)

# ---------------------------------------------------
# LOAD MODEL
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
        height=260,
        placeholder="Paste AI generated answer..."
    )

with col2:
    urls_text = st.text_area(
        "Source URLs (one per line)",
        height=260,
        placeholder="https://example.com/article"
    )

run = st.button("Analyze")


# ---------------------------------------------------
# WORD HIGHLIGHT
# ---------------------------------------------------

def highlight_overlap(ai_sentence, paragraph):

    ai_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', ai_sentence.lower()))

    highlighted = []

    for word in paragraph.split():

        clean = re.sub(r'\W+', '', word.lower())

        if clean in ai_words:
            highlighted.append(f"<strong>{word}</strong>")
        else:
            highlighted.append(word)

    return " ".join(highlighted)


# ---------------------------------------------------
# SCRAPE PAGE SECTIONS
# ---------------------------------------------------

def scrape_page(url):

    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    main = soup.find("article") or soup.find("main") or soup.body

    sections = []

    current_header = "Introduction"

    for tag in main.find_all(["h1", "h2", "h3", "p"]):

        if tag.name in ["h1", "h2", "h3"]:
            current_header = tag.get_text().strip()

        if tag.name == "p":

            text = tag.get_text().strip()

            if len(text) > 60:

                sections.append({
                    "header": current_header,
                    "paragraph": text
                })

    return sections


# ---------------------------------------------------
# SPLIT AI TEXT INTO SENTENCES
# ---------------------------------------------------

def split_sentences(text):

    sentences = re.split(r'(?<=[.!?]) +', text)

    sentences = [s.strip() for s in sentences if len(s) > 30]

    return sentences


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

            paragraphs = [s["paragraph"] for s in sections]

            if not paragraphs:
                continue

            para_embeddings = model.encode(paragraphs)

            for ai_sentence in ai_sentences:

                ai_embedding = model.encode([ai_sentence])

                similarity_scores = cosine_similarity(
                    ai_embedding,
                    para_embeddings
                )[0]

                top_matches = similarity_scores.argsort()[-3:][::-1]

                for idx in top_matches:

                    score = similarity_scores[idx]

                    if score < 0.45:
                        continue

                    results.append({

                        "AI Sentence": ai_sentence,
                        "Domain": domain,
                        "URL": url,
                        "Header": sections[idx]["header"],
                        "Paragraph": sections[idx]["paragraph"],
                        "Similarity (%)": round(score * 100, 2)

                    })

        except Exception as e:

            results.append({

                "AI Sentence": "Error",
                "Domain": "-",
                "URL": url,
                "Header": "-",
                "Paragraph": str(e),
                "Similarity (%)": 0

            })


# ---------------------------------------------------
# RESULTS TABLE
# ---------------------------------------------------

    if results:

        df = pd.DataFrame(results)

        df = df.sort_values(by="Similarity (%)", ascending=False)

        st.subheader("📊 Citation Matches")

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "ai_citation_matches.csv"
        )


# ---------------------------------------------------
# DETAILED MATCH VIEW
# ---------------------------------------------------

        st.subheader("🔎 Match Details")

        for _, row in df.head(10).iterrows():

            highlighted = highlight_overlap(
                row["AI Sentence"],
                row["Paragraph"]
            )

            st.markdown(f"""
**AI Sentence**

{row['AI Sentence']}

**Matched Header**

{row['Header']}

**Matched Paragraph**

{highlighted}

**Similarity:** {row['Similarity (%)']}%
---
""", unsafe_allow_html=True)

    else:

        st.warning("No strong matches detected.")
