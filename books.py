# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Load and clean data
books = pd.read_csv('BX-Books.csv', delimiter=';', encoding='latin-1')
ratings = pd.read_csv('BX-Book-Ratings.csv', delimiter=';', encoding='latin-1')

# Filtering users and books with sufficient ratings
books_ratings_counts = ratings.groupby('ISBN').size()
users_ratings_counts = ratings.groupby('User-ID').size()

ratings = ratings[ratings['ISBN'].isin(books_ratings_counts[books_ratings_counts >= 100].index)]
ratings = ratings[ratings['User-ID'].isin(users_ratings_counts[users_ratings_counts >= 200].index)]

# Create user-item matrix
user_item = ratings.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

# Get the Cosine Similarity matrix for user-item matrix
cosine_sim = cosine_similarity(user_item.T)

# Converting book titles to numerical indexes
le = LabelEncoder()
books['ISBN'] = le.fit_transform(books['ISBN'])

# Get book recommendations
def get_recommends(title, cosine_sim=cosine_sim):
    idx = le.transform([title])
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    return books['Title'].iloc[book_indices], cosine_sim[idx].flatten()[1:6]

recommendations = get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
print(recommendations)
recommendations = get_recommends("The Catcher in the Rye")
print(recommendations)