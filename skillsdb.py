import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="username",
    password="password",
    database="job_matching_db"
)

cursor = db.cursor()

# Create a table for job offers
cursor.execute("""
    CREATE TABLE IF NOT EXISTS job_offers (
        id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(255),
        skills TEXT
    )
""")

# Create a table for candidate skills
cursor.execute("""
    CREATE TABLE IF NOT EXISTS candidate_skills (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255),
        skills TEXT
    )
""")

db.commit()
