import streamlit as st
import pandas as pd
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model
data, vectorizer, feature_matrix = load("restaurant_recommendation_model.joblib")

# CSS for background image
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://images8.alphacoders.com/134/thumb-350-1343322.webp");
    background-size: cover;
    background-attachment: fixed;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Streamlit app
st.title("Restaurant Recommendation System")


# User inputs
preferred_cuisine = st.text_input("Enter your preferred cuisine (e.g., Japanese, Italian):")
city = st.text_input("Enter your city (e.g., Mumbai, Bangalore):")
top_n = st.slider("Number of recommendations:", min_value=1, max_value=20, value=5)

# Get recommendations
if st.button("Get Recommendations"):
    if preferred_cuisine and city:
        try:
            # Prepare the query vector
            preferred_features = preferred_cuisine.lower() + ' ' + city.lower()
            preferred_vector = vectorizer.transform([preferred_features])
            
            # Compute similarity
            similarities = cosine_similarity(preferred_vector, feature_matrix).flatten()
            top_indices = similarities.argsort()[-top_n:][::-1]
            
            if similarities.max() == 0:
                st.write("No matching recommendations found.")
            else:
                # Display recommendations
                recommendations = data.iloc[top_indices][['Restaurant Name', 'Cuisines', 'City', 'Aggregate rating']]
                st.write("Top Recommendations:")
                st.dataframe(recommendations)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please fill out both cuisine and city fields.")
