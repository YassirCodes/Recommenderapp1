from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import requests
from fuzzywuzzy import process
import pandas as pd

# Load precomputed data for different methods
with open('precomputed_glove.pkl', 'rb') as f_glove:
    data_glove = pickle.load(f_glove)
with open('precomputed_word2vec.pkl', 'rb') as f_word2vec:
    data_word2vec = pickle.load(f_word2vec)
with open('precomputed_transformer.pkl', 'rb') as f_transformer:
    data_transformer = pickle.load(f_transformer)

# Mapping method names to data
method_data_map = {
    'glove': data_glove,
    'word2vec': data_word2vec,
    'transformer': data_transformer
}

app = Flask(__name__)

# Function to get recommendations based on the selected method
def get_recommendations(anime_name, method, top_n=5):
    anime_name = anime_name.lower()
    data = method_data_map[method]  # Load the right precomputed data
    df = data['df']
    similarity_matrix = data['similarity_matrix']
    
    matches = process.extract(anime_name, df['title_english'].str.lower(), limit=1)
    if matches and matches[0][1] > 80:  # 80% similarity threshold
        matched_title = matches[0][0]
        anime_index = df[df['title_english'].str.lower() == matched_title].index
        if not anime_index.empty:
            anime_index = anime_index[0]
            similar_anime = list(enumerate(similarity_matrix[anime_index]))
            similar_anime = sorted(similar_anime, key=lambda x: x[1], reverse=True)
            l = []
            j = 0
            for i, _ in similar_anime[1:]:
                if j == top_n:
                    break
                if pd.notna(df.iloc[i]["title_english"]) and df.iloc[i]["title_english"].strip() != "":
                    l.append(i)
                    j += 1
            return df.iloc[l]
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/autocomplete')
def autocomplete():
    query = request.args.get('q', '').lower()
    data = method_data_map['glove']  # Default to GloVe for autocomplete
    df = data['df']
    matches = process.extract(query, df['title_english'].str.lower(), limit=10)
    results = [{'id': str(i), 'text': title} for title, score, i in matches if score > 60]
    return jsonify(results)

@app.route('/recommend', methods=['GET'])
def recommend():
    anime_name = request.args.get('name')
    method = request.args.get('method', 'glove')  # Default method is GloVe
    if not anime_name:
        return jsonify({'error': 'Please provide an anime name'}), 400
    
    recommendations = get_recommendations(anime_name, method)
    
    if recommendations is None:
        return jsonify({'error': 'Anime not found'}), 404

    required_columns = ['title_english', 'score', 'id']
    for col in required_columns:
        if col not in recommendations.columns:
            return jsonify({'error': f'Missing column: {col}'}), 500
    
    recommendations_list = recommendations[required_columns].to_dict(orient='records')
    return jsonify(recommendations_list)

if __name__ == '__main__':
    app.run(debug=False)
