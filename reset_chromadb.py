from chromadb import PersistentClient

client = PersistentClient(path="./data/rag_db")
for collection in client.list_collections():
    print(f"🗑️ Deleting collection: {collection.name}")
    client.delete_collection(collection.name)

print("✅ ChromaDB fully reset.")