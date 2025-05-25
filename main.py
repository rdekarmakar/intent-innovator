import os
from intent_cost_evaluator import classify_and_get_cost
from message_router import MessageRouter
from text_normalize import normalize_text2

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import uuid

# Setup ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="my_vectordb")

# Define the embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

# Uncomment to reset collection
chroma_client.delete_collection(name="customer_interaction")

# Get or create the collection
collection = chroma_client.get_or_create_collection(name="customer_interaction", embedding_function=embedding_fn)

def process_and_route_messages(messages):
    labels = []
    processing_costs = []
    routing_info = []
    message_ids = []

    documents = []
    metadatas = []
    ids = []

    for i, (channel, message_content) in enumerate(messages):
        # Normalize and prepare for Chroma
        norm_msg = normalize_text2(message_content)
        doc_id = str(uuid.uuid4())
        documents.append(norm_msg)
        metadatas.append({"channel": channel})
        ids.append(doc_id)
        message_ids.append(doc_id)

        # Classify and compute cost
        label, cost = classify_messages(channel, message_content)
        labels.append(label)
        processing_costs.append(cost)

        # Routing
        router = MessageRouter(label)
        routing_info.append(router.display_routing())

    # Add to ChromaDB
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    return labels, routing_info, processing_costs, message_ids

def classify_messages(channel, message_content):
    classification, total_cost = classify_and_get_cost(message_content)
    label = classification.model_dump_json(indent=2)
    return label, total_cost

def classify_csv(input_file):
    input_path = os.path.join("input", input_file)
    df = pd.read_csv(input_path, encoding='ISO-8859-1')

    # Classify based on 'channel' and 'message_content'
    messages = list(zip(df["channel"], df["message_content"]))
    labels, routing_info, processing_costs, chroma_ids = process_and_route_messages(messages)

    # Append results
    df["target_label"] = labels
    df["routing_info"] = routing_info
    df["processing_cost"] = processing_costs
    df["chroma_vector_id"] = chroma_ids

    print(df.head())

    output_path = os.path.join("output", f"output_with_chroma.csv")
    df.to_csv(output_path, index=False)

    return output_path

if __name__ == '__main__':
    classify_csv("test.csv")
