# import requests
# import json

# def sentiment_analyzer(text_to_analyse):
    
    
#     # Define the URL for the sentiment analysis API
#     url = 'https://sn-watson-sentiment-bert.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict'

#     # Create the payload with the text to be analyzed
#     myobj = { "raw_document": { "text": text_to_analyse } }

#     # Set the headers with the required model ID for the API
#     header = {"grpc-metadata-mm-model-id": "sentiment_aggregated-bert-workflow_lang_multi_stock"}

#     # Make a POST request to the API with the payload and headers
#     response = requests.post(url, json=myobj, headers=header)

#     # Parse the response from the API
#     formatted_response = json.loads(response.text)

#     # If the response status code is 200, extract the label and score from the response
#     if response.status_code == 200:
#         label = formatted_response['documentSentiment']['label']
#         score = formatted_response['documentSentiment']['score']
#     # If the response status code is 500, set label and score to None
#     elif response.status_code == 500:
#         label = None
#         score = None

#     # Return the label and score in a dictionary
#     return {'label': label, 'score': score}

# from ibm_watson import NaturalLanguageUnderstandingV1
# from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
# from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions
# import os
# import logging
# from dotenv import load_dotenv

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables from .env file
# load_dotenv()

# def sentiment_analyzer(text_to_analyse):
#     """
#     Analyze sentiment of the given text using IBM Watson Natural Language Understanding
    
#     Args:
#         text_to_analyse (str): Text to analyze
    
#     Returns:
#         dict: Dictionary containing 'label' (str) and 'score' (float) of the sentiment
#               Returns {'label': None, 'score': None} if analysis fails
#     """
    
#     # Validate input
#     if not text_to_analyse or not isinstance(text_to_analyse, str):
#         logger.warning("Invalid input text provided")
#         return {'label': None, 'score': None}
    
#     try:
#         # Get credentials from environment variables
#         api_key = os.getenv('WATSON_API_KEY')
#         service_url = os.getenv('WATSON_SERVICE_URL')
        
#         if not api_key or not service_url:
#             raise ValueError("Missing Watson API credentials in environment variables")
        
#         # Initialize Watson NLU service
#         authenticator = IAMAuthenticator(api_key)
#         service = NaturalLanguageUnderstandingV1(
#             version='2022-04-07',
#             authenticator=authenticator
#         )
#         service.set_service_url(service_url)
        
#         # Analyze sentiment
#         response = service.analyze(
#             text=text_to_analyse,
#             features=Features(sentiment=SentimentOptions())
#         ).get_result()
        
#         # Extract and return results
#         label = response['sentiment']['document']['label']
#         score = response['sentiment']['document']['score']
        
#         logger.info(f"Successfully analyzed sentiment: {label} (score: {score})")
#         return {'label': label, 'score': score}
        
#     except Exception as e:
#         logger.error(f"Error in sentiment analysis: {str(e)}")
#         return {'label': None, 'score': None}

from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def sentiment_analyzer(text_to_analyse):
    """
    Analyze sentiment of text using IBM Watson NLU
    
    Args:
        text_to_analyse (str): Text to analyze (minimum 3 words recommended)
    
    Returns:
        dict: {'label': 'positive'|'neutral'|'negative', 'score': float}
              or {'label': None, 'score': None} if analysis fails
    """
    # Validate input
    if not text_to_analyse or not isinstance(text_to_analyse, str):
        logger.warning("Invalid input text")
        return {'label': None, 'score': None}
    
    # Check minimum text length (Watson requires at least 3 words)
    if len(text_to_analyse.split()) < 3:
        logger.warning("Text too short for accurate analysis")
        return {'label': None, 'score': None}
    
    try:
        # Initialize service
        authenticator = IAMAuthenticator(os.getenv('WATSON_API_KEY'))
        service = NaturalLanguageUnderstandingV1(
            version='2022-04-07',
            authenticator=authenticator
        )
        service.set_service_url(os.getenv('WATSON_SERVICE_URL'))
        
        # Analyze sentiment
        response = service.analyze(
            text=text_to_analyse,
            features=Features(sentiment=SentimentOptions())
        ).get_result()
        
        return {
            'label': response['sentiment']['document']['label'],
            'score': response['sentiment']['document']['score']
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return {'label': None, 'score': None}