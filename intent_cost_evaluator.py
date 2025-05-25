from intent_predictor import (
    count_tokens,
    calculate_token_cost,
    build_combined_input,
    classify_request_from_input,
    calculate_total_input_cost,
)
import chromadb

def classify_and_get_cost(request_text: str):
    # Initialize ChromaDB
    # collection_customer_interaction=""
    chroma_client = chromadb.PersistentClient(path="my_vectordb")
    collection_customer_interaction = chroma_client.get_or_create_collection(name="customer_interaction")
    collection_customer_policies = chroma_client.get_or_create_collection(name="customer_policies")

    # Build combined input
    combined_input = build_combined_input(request_text, collection_customer_interaction, collection_customer_policies)

    # Classify request
    classification = classify_request_from_input(combined_input)

    # Input Token cost
    token_stats = calculate_total_input_cost(combined_input)
    input_cost = token_stats['total_cost']

    # Output Token cost
    output = classification.model_dump_json(indent=2)
    output_tokens = count_tokens(output)
    output_cost = calculate_token_cost(output_tokens, 0.60)

    # Total cost
    total_cost = input_cost + output_cost

    return classification, total_cost
