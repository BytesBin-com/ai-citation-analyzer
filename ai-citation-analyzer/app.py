import streamlit as st
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse

st.title("AI Citation Analyzer")

st.write(
    "Paste an AI answer and source URLs to detect which section and sentence AI likely used."
)

ai_answer = st.text_area("Paste AI Answer")
source_urls = st.text_area("Paste Source URLs (one per line)")

# Load semantic model
model = SentenceTransformer("all-MiniLM-L6-v2")


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


if st.button("Analyze"):

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

            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")

            soup = BeautifulSoup(response.text, "html.parser")

            main = soup.find("article") or soup.find("main") or soup.body

            content = main.find_all(["h1", "h2", "h3", "p"])

            paragraphs = []
            section_map = []

            current_section = "Introduction"

            for tag in content:

                if tag.name in ["h1", "h2", "h3"]:
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

                for i in range(len(clean) - 1):
                    sentences.append(clean[i] + " " + clean[i+1])
                    sentence_section_map.append(section)

                for i in range(len(clean) - 2):
                    sentences.append(clean[i] + " " + clean[i+1] + " " + clean[i+2])
                    sentence_section_map.append(section)

            # Remove duplicates
            unique = list(dict.fromkeys(zip(sentences, sentence_section_map)))
            sentences, sentence_section_map = zip(*unique)

            for ai_sentence in ai_sentences:

                texts = [ai_sentence] + list(sentences)

                embeddings = model.encode(texts)

                similarity_scores = cosine_similarity(
                    [embeddings[0]],
                    embeddings[1:]
                )

                best_index = similarity_scores.argmax()

                best_sentence = sentences[best_index]
                best_section = sentence_section_map[best_index]

                score = similarity_scores[0][best_index]

                if score >= 0.7:
                    confidence = "High"
                elif score >= 0.5:
                    confidence = "Medium"
                elif score >= 0.3:
                    confidence = "Low"
                else:
                    confidence = "Very Low"

                results.append({
                    "AI Sentence": ai_sentence,
                    "Domain": domain,
                    "Source URL": url,
                    "Section": best_section,
                    "Similarity (%)": round(score * 100, 2),
                    "Confidence": confidence,
                    "Matched Sentence": best_sentence
                })

        except Exception as e:

            results.append({
                "AI Sentence": "Error",
                "Domain": "-",
                "Source URL": url,
                "Section": "Error",
                "Similarity (%)": 0,
                "Confidence": "Error",
                "Matched Sentence": str(e)
            })

    if results:

        df = pd.DataFrame(results)

        df = df.sort_values(by="Similarity (%)", ascending=False)

        st.subheader("Citation Analysis Results")

        st.dataframe(
            df,
            use_container_width=True,
            height=600
        )

        st.subheader("Highlighted Matches")

        for _, row in df.iterrows():

            highlighted = highlight_overlap(
                row["AI Sentence"],
                row["Matched Sentence"]
            )

            st.markdown(f"**AI Sentence:** {row['AI Sentence']}")
            st.markdown(f"**Matched Sentence:** {highlighted}")
            st.write("---")

        st.subheader("Domain Influence Summary")

        summary = (
            df.groupby("Domain")["Similarity (%)"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        st.dataframe(summary, use_container_width=True)

        st.download_button(
            "Download Results CSV",
            df.to_csv(index=False),
            "ai_citation_results.csv",
            "text/csv"
        )