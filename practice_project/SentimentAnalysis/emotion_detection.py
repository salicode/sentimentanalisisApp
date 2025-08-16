from asyncio.log import logger
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
# from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, EmotionOptions
import requests
import json
import os
import logging
from dotenv import load_dotenv
# def emotion_detector(text_to_analyze):
#     """
#     Analyze emotions in text using Watson NLP Emotion Predict
    
#     Args:
#         text_to_analyze (str): Text to analyze for emotions
        
#     Returns:
#         dict: Contains emotion scores and dominant emotion in the specified format
#               Returns None for all fields if analysis fails
#     """
#     # Watson NLP Emotion Predict endpoint
#     url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
#     headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
#     payload = { "raw_document": { "text": text_to_analyze } }

#     try:
#         # Make the POST request
#         response = requests.post(url, json=payload, headers=headers)
        
#         # Convert response text to dictionary
#         response_dict = json.loads(response.text)
        
#         # Check for successful response
#         if response.status_code == 200:
#             # Extract emotion scores
#             emotions = response_dict['emotionPredictions'][0]['emotion']
            
#             # Find dominant emotion (emotion with highest score)
#             dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
#             # Return in specified format
#             return {
#                 'anger': emotions['anger'],
#                 'disgust': emotions['disgust'],
#                 'fear': emotions['fear'],
#                 'joy': emotions['joy'],
#                 'sadness': emotions['sadness'],
#                 'dominant_emotion': dominant_emotion
#             }
#         else:
#             # Return None values for error cases
#             return {
#                 'anger': None,
#                 'disgust': None,
#                 'fear': None,
#                 'joy': None,
#                 'sadness': None,
#                 'dominant_emotion': None
#             }
            
#     except Exception as e:
#         # Handle any exceptions (network errors, JSON decode errors, etc.)
#         print(f"Error in emotion detection: {str(e)}")
#         return {
#             'anger': None,
#             'disgust': None,
#             'fear': None,
#             'joy': None,
#             'sadness': None,
#             'dominant_emotion': None
#         }

def emotion_detector(text_to_analyze):
    """
    Analyze emotions using IBM Watson NLU service
    Requires: WATSON_API_KEY and WATSON_SERVICE_URL in .env
    """
    try:
        authenticator = IAMAuthenticator(os.getenv('WATSON_API_KEY'))
        service = NaturalLanguageUnderstandingV1(
            version='2022-04-07',
            authenticator=authenticator
        )
        service.set_service_url(os.getenv('WATSON_SERVICE_URL'))
        
        response = service.analyze(
            text=text_to_analyze,
            features=Features(emotion=EmotionOptions())
        ).get_result()
        
        emotions = response['emotion']['document']['emotion']
        dominant = max(emotions.items(), key=lambda x: x[1])[0]
        
        return {
            'anger': emotions['anger'],
            'disgust': emotions['disgust'],
            'fear': emotions['fear'],
            'joy': emotions['joy'],
            'sadness': emotions['sadness'],
            'dominant_emotion': dominant
        }
    except Exception as e:
        logger.error(f"Emotion analysis failed: {str(e)}")
        return {e: None for e in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'dominant_emotion']}