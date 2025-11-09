from api.services.rag_service import _collection

# See how many documents are stored in vector DB
print("Total documents in ChromaDB:", _collection.count())

# Peek into a few entries
results = _collection.get(include=["metadatas", "documents"], limit=3)
for i, doc in enumerate(results["documents"]):
    print(f"\n--- Vector {i+1} ---")
    print("Metadata:", results["metadatas"][i])
    print("Document Preview:", doc[:300], "...")
