import nltk
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="myarticles"
)

# Define candidate skills and requirements
candidate_skills = ["nltk", "sklearn", "python", "algorithms"]
required_skills = ["nltk", "sklearn", "python"]

# Calculate similarity score
similarity_score = cosine_similarity([candidate_skills], [required_skills])

print(similarity_score)
gh = similarity_score[0][0]
# Set a threshold for a match
threshold = 0.7

# Check if the similarity score meets the threshold
if similarity_score >= threshold:
    # Save matching result to the database
    cursor = db.cursor()
    query = "INSERT INTO matches (candidate_name, match_score) VALUES (%s, %s)"
    values = ("Candidate Name", similarity_score)
    cursor.execute(query, values)
    db.commit()
    cursor.close()

# Close the database connection
db.close()
