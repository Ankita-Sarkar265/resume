import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to load resumes from text files (or any other source)
def load_resumes(resume_dir):
    resumes = []
    resume_files = os.listdir(resume_dir)
    for file in resume_files:
        if file.endswith(".txt"):  # Assuming the resumes are in text format
            with open(os.path.join(resume_dir, file), 'r') as f:
                resumes.append(f.read())
    return resumes

# Function to process and vectorize the job description and resumes using TF-IDF
def vectorize_text(job_description, resumes):
    # Combine the job description with the resumes
    texts = [job_description] + resumes
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(texts)
    return vectors

# Function to calculate similarity between job description and resumes
def calculate_similarity(job_description, resumes):
    vectors = vectorize_text(job_description, resumes)
    job_desc_vector = vectors[0]
    resume_vectors = vectors[1:]
    
    # Calculate cosine similarity between the job description and each resume
    similarities = cosine_similarity(job_desc_vector, resume_vectors)
    return similarities.flatten()  # Flatten to a 1D array for easier processing

# Function to rank and select resumes based on similarity
def rank_resumes(job_description, resumes):
    similarity_scores = calculate_similarity(job_description, resumes)
    
    # Create a list of tuples with resume index and similarity score
    resume_scores = [(index, score) for index, score in enumerate(similarity_scores)]
    
    # Sort resumes based on the similarity score (higher score = better match)
    sorted_resumes = sorted(resume_scores, key=lambda x: x[1], reverse=True)
    
    # Select the top N resumes (e.g., top 5)
    top_resumes = sorted_resumes[:5]
    
    return top_resumes, similarity_scores

# Main function to interact with the user and display the results
def main():
    # Dynamic input for job description
    job_description = input("Enter the job description:\n")
    
    # Specify the directory where resumes are stored (assumed in .txt format for simplicity)
    resume_dir = 'resumes/'  # Change this to your resumes directory path

    # Load resumes from the directory
    resumes = load_resumes(resume_dir)
    
    # Rank resumes based on their similarity to the job description
    top_resumes, similarity_scores = rank_resumes(job_description, resumes)
    
    # Display the ranked resumes with their similarity scores
    print("\nTop 5 matching resumes (Ranked):")
    for rank, (index, score) in enumerate(top_resumes, start=1):
        print(f"\nRank {rank}:")
        print(f"Resume {index + 1} - Similarity Score: {score:.4f}")
        print(f"Resume Content: {resumes[index][:200]}...")  # Display the first 200 characters of the resume

# Run the script
if __name__ == "__main__":
    main()
