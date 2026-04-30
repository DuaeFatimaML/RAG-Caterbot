import os, json, sys, math
from sklearn.feature_extraction.text import TfidfVectorizer

DOCS_FOLDER = r"G:\LLM Chatbot\Document"
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_store.json")
CHUNK_SIZE  = 400
OVERLAP     = 50
 
def load_documents():
    docs = []
    print("📂 Reading documents...")
    for fname in os.listdir(DOCS_FOLDER):
        if fname.endswith(".txt"):
            path = os.path.join(DOCS_FOLDER, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"filename": fname, "text": text})
            print(f"   ✅ {fname} ({len(text)} chars)")
    return docs
 
def make_chunks(docs):
    chunks = []
    print("\n✂️  Splitting into chunks...")
    for doc in docs:
        text, fname = doc["text"], doc["filename"]
        start, i = 0, 0
        while start < len(text):
            chunk = text[start : start + CHUNK_SIZE].strip()
            if chunk:
                chunks.append({
                    "id":        f"{fname}_chunk{i}",
                    "source":    fname,
                    "text":      chunk,
                    "embedding": []
                })
                i += 1
            start += CHUNK_SIZE - OVERLAP
    print(f"   Total chunks: {len(chunks)}")
    return chunks
 
def embed_all(chunks):
    print(f"\n🔢 Creating TF-IDF embeddings (pure Python, no DLL issues)...")
    texts = [c["text"] for c in chunks]
 
    vectorizer = TfidfVectorizer(max_features=512)
    matrix = vectorizer.fit_transform(texts).toarray()
 
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = matrix[i].tolist()
        print(f"   [{i+1}/{len(chunks)}] {chunk['id']}")
 
    # Save vectorizer vocabulary so rag_app can use same embedding
    vocab = vectorizer.vocabulary_
    idf   = vectorizer.idf_.tolist()
    return chunks, vocab, idf
 
def save(chunks, vocab, idf):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
 
    meta_file = OUTPUT_FILE.replace("vector_store.json", "tfidf_meta.json")
    vocab_serializable = {k: int(v) for k, v in vocab.items()}  # fix int64
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({"vocabulary": vocab_serializable, "idf": idf}, f)
 
    print(f"\n💾 Saved {len(chunks)} chunks → {OUTPUT_FILE}")
    print(f"💾 Saved TF-IDF metadata → tfidf_meta.json")
 
if __name__ == "__main__":
    print("=" * 50)
    print("  BUILDING RAG KNOWLEDGE BASE")
    print("  (Pure Python TF-IDF — zero DLL issues)")
    print("=" * 50)
    docs = load_documents()
    if not docs:
        print("❌ No .txt files found.")
        sys.exit()
    chunks = make_chunks(docs)
    chunks, vocab, idf = embed_all(chunks)
    save(chunks, vocab, idf)
    print("\n✅ Knowledge base ready!")
    print("=" * 50)