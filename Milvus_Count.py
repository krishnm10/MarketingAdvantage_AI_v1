from pymilvus import connections, utility, Collection

# connect to Milvus
connections.connect(host="13.204.84.105", port="19530", token="root:Milvus")

# get collections
collections = utility.list_collections()

print("Total collections:", len(collections))

for col_name in collections:
    col = Collection(col_name)
    print(f"Collection: {col_name} | Records: {col.num_entities}")
