import os
import csv

def write_scores_to_file(scores: dict, filename: str):
    """
    Write the scores dictionary to a .dat file.
    """

    # Type check(s)
    if not isinstance(scores, dict) or not all(isinstance(doc_id, str) and isinstance(score, (int, float)) for doc_id, score in scores.items()):
        raise TypeError("scores: value must be a dict of str keys and int or float values")
    if not isinstance(filename, str):
        raise TypeError("filename: value must be a string.")
    
    # Combine the directory and filename to form the full path
    directory = 'RankingOutputs'
    filepath = os.path.join(directory, filename)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')  # Using tab delimiter for .dat format
        for doc_id, score in scores.items():
            writer.writerow([doc_id, score])

def score_normalisation(scores: dict, mode: str = 'minmax') -> dict:
    """
    Normalises the numerical scores for documents using Min-Max scaling.
    
    The function takes a dictionary of document scores and scales each score
    linearly between 0 and 1. If all scores are identical, they are set to 0 to reflect
    no variation in significance among them.
    """
    
    # Type check(s)
    if not isinstance(scores, dict) or not all(isinstance(doc_id, str) and isinstance(score, (int, float)) for doc_id, score in scores.items()):
        raise TypeError("scores: value must be a dict of str keys and int or float values.")
    if not isinstance(mode, str) or mode not in ['minmax', 'zscore']:
        raise TypeError("mode: value must be 'minmax' or 'zscore'.")

    # Min-Max normalization
    if mode == 'minmax':
        min_score = min(scores.values())
        max_score = max(scores.values())
        if (max_score - min_score) == 0:  # to avoid division by zero if all scores are the same,
            return {doc_id: 0 for doc_id in scores}  # return 0 (opted for this rather than 1 because if everything is relevant, nothing it)
        
        normalized_scores = {doc_id: (score - min_score) / (max_score - min_score) for doc_id, score in scores.items()}

    # Z-score normalization
    elif mode == 'zscore':
        
        mean_score = sum(scores.values()) / len(scores)
        std_dev = (sum((score - mean_score) ** 2 for score in scores.values()) / len(scores)) ** 0.5

        # Handle case where standard deviation is zero (all scores are the same)
        if std_dev == 0:
            return {doc_id: 0 if scores[doc_id] == 0 else 1 for doc_id in scores}  # Normalize non-zero scores to 1

        normalized_scores = {doc_id: (score - mean_score) / std_dev for doc_id, score in scores.items()}

        # Shift and scale scores to ensure non-negativity and preserve zero scores
        min_score = min(normalized_scores.values())
        if min_score < 0:
            normalized_scores = {doc_id: score - min_score for doc_id, score in normalized_scores.items()}
        
        # Ensure scores that were originally zero remain zero
        normalized_scores = {doc_id: 0 if scores[doc_id] == 0 else score for doc_id, score in normalized_scores.items()}
        
    return normalized_scores