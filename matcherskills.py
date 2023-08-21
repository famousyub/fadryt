import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector

# Initialize NLTK and MySQL connection
nltk.download('punkt')
mysql_conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="myarticles"
)
cursor = mysql_conn.cursor()

# Example job descriptions and candidate profiles
jobs = ["Python developer with NLTK and Scikit-learn skills",
        "Data Scientist with Python, machine learning, and SQL expertise"]

candidates = ["Experienced Python developer with NLTK and Scikit-learn proficiency",
              "Data Scientist with strong skills in Python and machine learning"]

# Preprocess job descriptions and candidate profiles
vectorizer = TfidfVectorizer()
job_features = vectorizer.fit_transform(jobs)
candidate_features = vectorizer.transform(candidates)

# Calculate cosine similarity
similarity_scores = cosine_similarity(candidate_features, job_features)

# Threshold for considering a match
threshold = 0.6

# Save matching results to MySQL
for i, candidate in enumerate(candidates):
    for j, job in enumerate(jobs):
        if similarity_scores[i, j] > threshold:
            cursor.execute("INSERT INTO matching_results (candidate_id, job_id, score) VALUES (%s, %s, %s)",
                           (i + 1, j + 1, similarity_scores[i, j]))

mysql_conn.commit()
cursor.close()
mysql_conn.close()
