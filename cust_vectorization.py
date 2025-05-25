import csv
from text_normalize import normalize_text1, normalize_text2
import chromadb
from chromadb.utils import embedding_functions

# Load data from the CSV file
with open('customer_insurance_policies.csv') as file:
    reader = csv.reader(file)

    # Skip header
    next(reader)

    documents = []
    metadatas = []
    ids = []

    id = 1
    for line in reader:
        customer_id = line[0]
        cust_name = normalize_text1(line[1])
        policy_text = normalize_text2(line[2])

        # You can combine customer name and policy text as the document if desired
        doc = f"{cust_name}. {policy_text}"

        documents.append(doc)
        metadatas.append({"customer_id": customer_id, "cust_name": cust_name})
        ids.append(str(id))
        id += 1

# Set up ChromaDB with persistent storage
chroma_client = chromadb.PersistentClient(path="my_vectordb")

# Optional: Delete the existing collection if needed
chroma_client.delete_collection(name="customer_policies")

# Initialize embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

# Create or get the vector collection
collection = chroma_client.get_or_create_collection(
    name="customer_policies",
    embedding_function=sentence_transformer_ef
)

# Add documents to the collection
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

# Query the collection
results = collection.query(
    query_texts=["What is the date of issue for personal accident policy?"],
    n_results=1
)

# Print the most relevant document
print(results['documents'])

# Flatten and join for additional context
additional_context = " ".join([doc for sublist in results["documents"] for doc in sublist])
print(additional_context)
