''' Executing this function initiates the application of sentiment
    analysis to be executed over the Flask channel and deployed on
    localhost:5000.
'''
from flask import Flask, jsonify, render_template, request

from SentimentAnalysis.sentiment_analysis import sentiment_analyzer
from SentimentAnalysis.emotion_detection import emotion_detector

# app = Flask("Sentiment Analyzer")
app = Flask("Sentiment and Emotion Analyzer")

@app.route("/sentimentAnalyzer")
def sent_analyzer():
    text_to_analyze = request.args.get('textToAnalyze')
    
    if not text_to_analyze or len(text_to_analyze.split()) < 3:
        return jsonify({
            'error': 'Please enter at least 3 words for accurate analysis'
        }), 400
    
    response = sentiment_analyzer(text_to_analyze)
    
    if response['label'] is None:
        return jsonify({
            'error': 'Could not analyze sentiment. Please try with different text.'
        }), 500
    
    return jsonify({
        'label': response['label'],
        'score': response['score'],
        'message': f"The given text has been identified as {response['label']} with a score of {response['score']:.2f}."
    })

@app.route("/emotionAnalyzer")
def emotion_analysis():
    text_to_analyze = request.args.get('textToAnalyze')
    
    if not text_to_analyze or len(text_to_analyze.split()) < 3:
        return jsonify({
            'error': 'Please enter at least 3 words for accurate analysis'
        }), 400
    
    response = emotion_detector(text_to_analyze)
    
    if None in response.values():
        return jsonify({
            'error': 'Could not analyze emotions. Please try with different text.'
        }), 500
    
    # Format the dominant emotion for display
    dominant_emotion = response['dominant_emotion'].capitalize()
    
    return jsonify({
        'emotions': {
            'anger': response['anger'],
            'disgust': response['disgust'],
            'fear': response['fear'],
            'joy': response['joy'],
            'sadness': response['sadness']
        },
        'dominant_emotion': dominant_emotion,
        'message': f"Dominant emotion is {dominant_emotion} "
                   f"(Anger: {response['anger']:.2f}, "
                   f"Disgust: {response['disgust']:.2f}, "
                   f"Fear: {response['fear']:.2f}, "
                   f"Joy: {response['joy']:.2f}, "
                   f"Sadness: {response['sadness']:.2f})"
    })
@app.route("/")
def render_index_page():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
