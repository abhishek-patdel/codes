"""
Can be used in removing sklearn pairwise cosine similarity.
"""
import numpy as np

def calculate_cosine_similarity(vector1, vector2):
    """
    Calculate the cosine similarity between two vectors without using sklearn.
    
    Parameters:
    - vector1: First vector (list or numpy array)
    - vector2: Second vector (list or numpy array)
    
    Returns:
    - similarity: Cosine similarity score (float)
    """
    # Convert to numpy arrays if they are not already
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    # Check for empty vectors
    if vector1.size == 0 or vector2.size == 0:
        return 0.0
    
    # Check for zero vectors
    if np.all(vector1 == 0) or np.all(vector2 == 0):
        return 0.0
    
    # Check for different lengths
    if vector1.shape != vector2.shape:
        return 0.0
    
    # Calculate the dot product
    dot_product = np.dot(vector1, vector2)
    
    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Check for zero magnitudes to avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    
    return similarity

# Example usage
vector1 = [0.112, 0.123, 0.1454]
vector2 = [0.911, 0.113, 0.0454]
similarity = calculate_cosine_similarity(vector1, vector2)
print(f"Similarity: {similarity}")






