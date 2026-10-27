"""
astro-rag-chatbot - by Lesego Mokoena
Team UmbraOnxy | NASA Space Apps 2026
Databricks Accredited Gen AI Engineer - ID 198237963

Simple RAG Chatbot - retrieves answers from NASA docs
Tech: Python + TF-IDF (no API key needed for v1)
"""
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_docs():
    docs = []
    kb_path = "knowledge_base"
    if not os.path.exists(kb_path):
        os.makedirs(kb_path)
        with open(f"{kb_path}/nasa_moon_guide.txt", "w") as f:
            f.write("""NASA Moon Mission Guide - Artemis Program
The Artemis program aims to return humans to the Moon and establish presence at lunar South Pole.
Lunar South Pole has water ice in permanently shadowed regions - critical for life support and fuel.
Challenges: 14 days sunlight then 14 days darkness, dust mitigation, radiation, 1.3 sec comms delay.
NASA Open Data: LRO lunar maps, Artemis specs, Moon Mineralogy Mapper.""")
    
    if os.path.exists(kb_path):
        for file in os.listdir(kb_path):
            with open(os.path.join(kb_path, file), 'r', encoding='utf-8') as f:
                docs.append(f.read())
    return docs

class AstroRAG:
    def __init__(self, documents):
        self.chunks = []
        for doc in documents:
            self.chunks.extend([p.strip() for p in doc.split('\n') if len(p.strip()) > 25])
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(self.chunks)

    def ask(self, question):
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.vectors).flatten()
        best_idx = scores.argmax()
        return self.chunks[best_idx], scores[best_idx]

if __name__ == "__main__":
    print("🚀 Astro RAG Chatbot - NASA Space Apps 2026 | Lesego Mokoena")
    print("Ask about Moon/Mars missions. Type 'exit' to quit.\n")
    docs = load_docs()
    bot = AstroRAG(docs)
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            print("Bot: Good luck on the Moon! 🌙")
            break
        answer, score = bot.ask(q)
        print(f"Bot (confidence {score:.2f}): {answer}\n")
