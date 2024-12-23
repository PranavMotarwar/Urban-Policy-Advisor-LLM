from flask import Flask, request, jsonify
import pandas as pd
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Load your preprocessed traffic data
data = pd.read_csv('daily_traffic_by_boro.csv')

# Load a pre-trained question answering model
question_answerer = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Helper function to get the most congested areas
def get_most_congested_areas():
    congestion_by_borough = data.groupby('Boro')['Vol'].sum().sort_values(ascending=False)
    top_congested_areas = congestion_by_borough.head(3).to_dict()
    return top_congested_areas

# Helper function to get daily average traffic volume per borough
def get_average_traffic_per_borough():
    daily_avg_traffic = data.groupby('Boro')['Vol'].mean().to_dict()
    return daily_avg_traffic

# Helper function to recommend road construction
def recommend_road_construction():
    congestion_by_borough = data.groupby('Boro')['Vol'].sum().sort_values(ascending=False)
    recommendations = list(congestion_by_borough.head(2).index)
    return recommendations

# Helper function for a 5-year infrastructure plan
def five_year_plan():
    return "Focus on road expansion in areas like Brooklyn and Queens, and consider investments in public transportation to alleviate congestion."

# Endpoint to handle incoming questions
@app.route('/ask', methods=['POST'])
def ask():
    question = request.get_json().get("question")
    
    # Example context from your traffic data
    context = f"""
    The most congested areas in NYC are {', '.join(get_most_congested_areas().keys())}.
    Average daily traffic volumes are as follows: {get_average_traffic_per_borough()}.
    We recommend building new roads in {', '.join(recommend_road_construction())}.
    Our 5-year plan suggests {five_year_plan()}.
    """

    # Use the question answering model to find an answer in the context
    answer = question_answerer(question=question, context=context)
    
    response = {
        "response": answer['answer']
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
