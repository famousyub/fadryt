import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sample candidate skills and job offers
candidate_skills = [
    "python programming", "machine learning", "data analysis", "nlp"
]

job_offers = [
    "data scientist position with expertise in python and machine learning",
    "looking for a python developer experienced in data analysis",
    "nlp engineer role for natural language processing projects"
]

# Preprocess skills and job offers
stop_words = set(stopwords.words("english"))

def preprocess(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

candidate_skills = [preprocess(skill) for skill in candidate_skills]
job_offers = [preprocess(offer) for offer in job_offers]

# Create a CountVectorizer to convert text to a matrix of token counts
vectorizer = CountVectorizer().fit_transform(candidate_skills + job_offers)

# Calculate cosine similarity between candidate skills and job offers
cosine_similarities = cosine_similarity(vectorizer[:len(candidate_skills)], vectorizer[len(candidate_skills):])

# Find the best job offer for each candidate skill
for i, skill in enumerate(candidate_skills):
    best_match_index = np.argmax(cosine_similarities[i])
    print(f"Candidate skill: {skill}")
    print(f"Best matching job offer: {job_offers[best_match_index]}\n")
