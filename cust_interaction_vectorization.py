import csv
from text_normalize import normalize_text1,normalize_text2

# Load sample data (a restaurant menu of items)
with open('customer_interactions.csv') as file:
    lines = csv.reader(file)

    # Store the name of the menu items in this array. In Chroma, a "document" is a string i.e. name, sentence, paragraph, etc.
    documents = []

    # Store the corresponding menu item IDs in this array.
    metadatas = []

    # Each "document" needs a unique ID. This is like the primary key of a relational database. We'll start at 1 and increment from there.
    ids = []
    id = 1

    # Loop thru each line and populate the 3 arrays.
    for i, line in enumerate(lines):
        if i==0:
            # Skip the first row (the column headers)
            continue
        # Normalize the text
        # doc = normalize_text2(line[1])
        doc = line[1]
        print(doc)
        documents.append(doc)
        metadatas.append({"customer_id": line[0]})
        ids.append(str(id))
        id+=1


import chromadb
from chromadb.utils import embedding_functions

# Instantiate chromadb instance. Data is stored in memory only.
# chroma_client = chromadb.Client()

# Instantiate chromadb instance. Data is stored on disk (a folder named 'my_vectordb' will be created in the same folder as this file).
chroma_client = chromadb.PersistentClient(path="my_vectordb")


# Select the embedding model to use.
# List of model names can be found here https://www.sbert.net/docs/pretrained_models.html
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

# Use this to delete the database
chroma_client.delete_collection(name="customer_interaction")

# Create the collection, aka vector database. Or, if database already exist, then use it. Specify the model that we want to use to do the embedding.
collection = chroma_client.get_or_create_collection(name="customer_interaction", embedding_function=sentence_transformer_ef)

# Add all the data to the vector database. ChromaDB automatically converts and stores the text as vector embeddings. This may take a few minutes.
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)


results = collection.query(
        query_texts=[" claim for accidental hospitalization"],
        n_results=1  # Retrieve top 3 relevant results
    )
print(results['documents'])
# additional_context = " ".join([doc for doc in results["documents"]])

# Flatten the nested list and join the elements
additional_context = " ".join([doc for sublist in results["documents"] for doc in sublist])
print(additional_context)