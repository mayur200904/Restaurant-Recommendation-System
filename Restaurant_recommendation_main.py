import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump

# Load and preprocess the dataset
data = pd.read_csv(r"/Users/mayursantoshtarate/Desktop/project/restaurant  recommendation /task 2/Dataset.csv")
data.dropna(subset=['Cuisines'], inplace=True)  # Drop rows with missing cuisines
data['Cuisines'] = data['Cuisines'].str.lower()  # Normalize cuisines to lowercase

# Combine features for recommendation
data['Features'] = data['Cuisines'] + ' ' + data['City'].str.lower()

# Convert text data into feature vectors
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(data['Features'])

# Save the preprocessed data and vectorizer
dump((data, vectorizer, feature_matrix), "restaurant_recommendation_model.joblib")

# Recommendation function
def recommend_restaurants(preferred_cuisine, city, top_n=5):
    # Prepare the query vector using the same vectorizer
    preferred_features = preferred_cuisine.lower() + ' ' + city.lower()
    preferred_vector = vectorizer.transform([preferred_features])
    
    # Check dimensions of vectors to ensure alignment
    if preferred_vector.shape[1] != feature_matrix.shape[1]:
        raise ValueError("Mismatch in feature dimensions between query and dataset vectors.")
    
    # Compute similarity with all restaurants
    similarities = cosine_similarity(preferred_vector, feature_matrix).flatten()
    
    # Get indices of top matching restaurants
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    if similarities.max() == 0:
        return "No matching recommendations found."
    
    # Fetch recommended restaurants
    recommendations = data.iloc[top_indices][['Restaurant Name', 'Cuisines', 'City', 'Aggregate rating']]
    return recommendations
