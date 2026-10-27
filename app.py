import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("🚀 Astro RAG Chatbot - NASA Space Apps 2026 | by Lesego Mokoena")
print("Ask about Moon/Mars missions. Type 'exit' to quit.\n")

files = glob.glob("knowledge_base/*.txt")
docs = []
for f in files:
    with open(f, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read()
        chunks = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
        docs.extend(chunks)

print(f"Loaded {len(docs)} knowledge chunks\n")

vectorizer = TfidfVectorizer().fit(docs)
doc_vectors = vectorizer.transform(docs)

while True:
    q = input("You: ")
    if q.lower() == 'exit':
        break
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, doc_vectors)[0]
    idx = sims.argmax()
    conf = sims[idx]
    answer = docs[idx][:700]
    print(f"Bot (confidence {conf:.2f}): {answer}\n")
