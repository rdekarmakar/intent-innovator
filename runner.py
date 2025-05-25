from intent_cost_evaluator import classify_and_get_cost

request_text = """
George : Personal Accident policy claim for my accident on March 15th.
"""

classification, total_cost = classify_and_get_cost(request_text)

# Print or process as needed
print("Classification Result:")
print(classification.model_dump_json(indent=2))
print(f"Total Cost: ${total_cost:.6f}")
