import streamlit as st
import pickle
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Load the df
with open('df.pkl', 'rb') as f:
    df = pickle.load(f)

# Load the combined_text (preprocessed text data)
with open('combined_text.pkl', 'rb') as f:
    combined_text = pickle.load(f)

# Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf.fit_transform(combined_text)

# Dimensionality Reduction using TruncatedSVD
svd = TruncatedSVD(n_components=50)
svd_matrix = svd.fit_transform(tfidf_matrix)

# Function to preprocess text
nlp = spacy.load("en_core_web_sm")
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Function to make predictions
def get_recipe_recommendations(query, tfidf_vectorizer, svd_model, data):
    # Preprocess the query
    query = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([query])
    query_svd = svd_model.transform(query_vector)
    # Calculate cosine similarity
    similarity_scores = cosine_similarity(query_svd, svd_matrix)
    # Get index of the most similar recipe
    top_recipe_index = np.argmax(similarity_scores)
    # Return the most similar recipe
    return data.iloc[top_recipe_index]

# Streamlit app layout
st.title("Recipe Recommender")

# User input
user_input = st.text_area("What do you want to make?")

# Recommendation
if st.button("Get Recommendation"):
    if user_input:
        recommended_recipe = get_recipe_recommendations(user_input, tfidf, svd, df)
        st.write("Recommended Recipe:")
        st.write("Title:", recommended_recipe['title'])
        st.write("Ingredients:", recommended_recipe['ingredients'])
        st.write("Instructions:", recommended_recipe['instructions'])
        #st.write("Cooking Time:", recommended_recipe['cooking_time'])
    else:
        st.warning("Please enter a question.")
