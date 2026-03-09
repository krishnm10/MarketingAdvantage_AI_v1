import chromadb

CHROMA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("ingested_content_local")

print("Total records:", collection.count())

# View sample data
results = collection.peek(limit=5)
for i, doc in enumerate(results["documents"]):
    print(f"\n--- Document {i+1} ---")
    print("ID:", results["ids"][i])
    print("Metadata:", results["metadatas"][i])
    print("Text Snippet:", doc[:500])
