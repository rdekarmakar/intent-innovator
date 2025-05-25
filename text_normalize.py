import re

def normalize_text1(text: str) -> str:
    """
    Normalize the input text by converting to lowercase, removing special characters,
    and reducing multiple spaces to a single space.

    Args:
        text (str): The input text to normalize.

    Returns:
        str: The normalized text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove leading and trailing whitespace
    text = text.strip()
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text



import string
from nltk.corpus import stopwords
import nltk

def normalize_text2(text, remove_stopwords=False):
    """
    Normalize input text by:
    - Lowercasing
    - Removing punctuation
    - Removing extra whitespace
    - Optionally removing stopwords

    Parameters:
    - text (str): The text to normalize.
    - remove_stopwords (bool): Whether to remove stopwords.

    Returns:
    - str: The normalized text.
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if remove_stopwords:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        words = text.split()
        text = ' '.join([word for word in words if word not in stop_words])

    return text

ticket7 = """
Hi Team,

I wanted to check if I'm eligible for the free annual health check-up mentioned in the policy.

Could you please let me know how to book it and if there’s a specific hospital or clinic I need to visit?

Thanks in advance for your help!

Best,
John Doe
"""


# print("normalize_text1",normalize_text1(ticket7))
#
# print("normalize_text2",normalize_text2(ticket7, remove_stopwords=True))