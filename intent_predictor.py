from typing import List
import tiktoken
from pydantic import BaseModel, Field
from enum import Enum
import instructor
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# Enums and Pydantic models
# -------------------------------
class RequestCategory(str, Enum):
    POLICY_INQUIRY = "policy_inquiry"
    CLAIM_STATUS = "claim_status"
    BILLING_PAYMENTS = "billing_payments"
    POLICY_UPDATE = "policy_update"
    COVERAGE_INQUIRY = "coverage_inquiry"
    PROVIDER_SEARCH = "provider_search"
    PRE_AUTHORIZATION = "pre_authorization"
    IDCARD_REQUEST = "idcard_request"
    TECHNICAL_SUPPORT = "technical_support"
    POLICY_CANCELLATION = "policy_cancellation"
    CLAIM_DENIAL = "claim_denial"
    DEPENDENT_COVERAGE = "dependent_coverage"
    OTHER = "other"

class CustomerSentiment(str, Enum):
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"

class RequestUrgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RequestClassification(BaseModel):
    category: RequestCategory
    urgency: RequestUrgency
    sentiment: CustomerSentiment
    confidence: float = Field(ge=0, le=1, description="Confidence score for the classification")
    key_information: List[str] = Field(description="List of key points extracted from the request")
    suggested_action: str = Field(description="Brief suggestion for handling the request")

# -------------------------------
# Utilities
# -------------------------------
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def calculate_token_cost(token_count: int, cost_per_million_tokens: float) -> float:
    return (token_count * cost_per_million_tokens) / 1_000_000

def build_combined_input(request_text: str, interaction_collection='', policy_collection='') -> str:
    results = interaction_collection.query(query_texts=[request_text], n_results=1)
    interaction_context = " ".join([doc for sublist in results["documents"] for doc in sublist])

    results = policy_collection.query(query_texts=[request_text], n_results=1)
    policy_context = " ".join([doc for sublist in results["documents"] for doc in sublist])
    user_request_delimiter = "###"
    additional_context = f"{interaction_context} {policy_context}".strip()
    return f"{user_request_delimiter}{request_text}{user_request_delimiter}\n\nAdditional Context:\n{additional_context}"

# -------------------------------
# Intent Predictor function
# -------------------------------

# SYSTEM_PROMPT = """
# You are an AI assistant for a large health insurance customer support team.
# Your role is to analyze incoming customer support requests and provide structured information to help our team respond quickly and effectively.
# Business Context:
# - We handle thousands of requests daily across various categories (claim, coverage, policy, provider,dependent, technical issues, billing).
# - Quick and accurate classification is crucial for customer satisfaction and operational efficiency.
# - We prioritize based on urgency and customer sentiment.
# Your tasks:
# 1. Categorize the requests into the most appropriate category.
# 2. Assess the urgency of the issue (low, medium, high, critical).
# 3. Determine the customer's sentiment.
# 4. Extract key information that would be helpful for our support team.
# 5. Suggest an initial action for handling the request.
# 6. Provide a confidence score for your classification.
# Remember:
# - Be objective and base your analysis solely on the information provided in the request.
# - If you're unsure about any aspect, reflect that in your confidence score.
# - For 'key_information', extract specific details like Policy numbers, product names,current issues or brief from previous customer interactions.
# - The 'suggested_action' should be a brief, actionable step for our support team.
# Analyze the following customer support requests and provide the requested information in the specified format.
# As additional context, you can use the customer interaction history and customer policies if you found any exact match.
# """

SYSTEM_PROMPT = """
You are an AI assistant for a large health insurance customer support team. 
Your role is to analyze incoming *health insurance-related* customer support requests and provide structured information to help our team respond quickly and effectively.

IMPORTANT SCOPE LIMITATION:
- Do NOT answer any questions or process requests unrelated to **health insurance** (e.g., life insurance, auto insurance, banking, tech products, general inquiries). 
- If a request falls outside the health insurance domain, respond with: 
  "This request falls outside the scope of our health insurance support system."

Business Context:
- We handle thousands of health insurance-related requests daily across categories such as claims, coverage, policy, providers, dependents, technical issues, and billing.
- Quick and accurate classification is crucial for customer satisfaction and operational efficiency.
- We prioritize based on urgency and customer sentiment.

Your tasks:
1. **Categorize** the request into the most appropriate health insurance category.
2. **Assess the urgency** of the issue (low, medium, high, critical).
3. **Determine the customer's sentiment** (positive, neutral, negative).
4. **Extract key information** helpful for our support team (e.g., policy numbers, product names, current issues, references to prior interactions).
5. **Suggest an initial action** to help the support team respond effectively.
6. **Provide a confidence score** (0 to 1) indicating how confident you are in your classification and assessment.

Guidelines:
- Be objective and base your analysis **solely on the provided information** in the request.
- If any detail is unclear or missing, reflect that in your confidence score.
- User request is delimited by '###'.
- Use customer interaction or policy data only if an **exact match** is found for name or policy number between the user request and the additional context.
- Keep all recommendations **specific to health insurance only**.
"""


# Patch instructor to the Groq client
groq_client = instructor.from_groq(Groq())

def classify_request_from_input(combined_input: str) -> RequestClassification:
    response = groq_client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        response_model=RequestClassification,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": combined_input}
        ]
    )
    return response

def get_system_prompt() -> str:
    return SYSTEM_PROMPT

def calculate_total_input_cost(combined_input: str, model: str = "gpt-3.5-turbo", cost_per_million_tokens: float = 0.15) -> dict:
    system_prompt = get_system_prompt()
    system_prompt_tokens = count_tokens(system_prompt, model)
    input_tokens = count_tokens(combined_input, model)
    total_tokens = input_tokens + system_prompt_tokens
    total_cost = calculate_token_cost(total_tokens, cost_per_million_tokens)
    return {
        "system_prompt_tokens": system_prompt_tokens,
        "input_tokens": input_tokens,
        "total_tokens": total_tokens,
        "total_cost": total_cost
    }
