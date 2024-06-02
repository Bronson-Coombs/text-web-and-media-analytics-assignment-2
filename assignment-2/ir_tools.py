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

def score_normalisation(scores: dict) -> dict:
    """
    Normalises the numerical scores for documents using zscore normaliseation with shift/scaling applied for interpretability.
    """
    
    # Type check(s)
    if not isinstance(scores, dict) or not all(isinstance(doc_id, str) and isinstance(score, (int, float)) for doc_id, score in scores.items()):
        raise TypeError("scores: value must be a dict of str keys and int or float values.")
        
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

def get_top_15(model_results: dict):
    """
    Takes the model results, prints out the top-15 sorted by weights.
    """

    # Iterate terating over each set of {query:predictions}, where predictions is a dictionary of {doc_id : document weight}
    for(query, predictions) in model_results.items():
        print('Query' + str(query) + ' (DocID Weight):')  # print result header information

        # For the given result set, sort the document weights and take the top 15 scores ("up to n" indexing doesn't break for lists shorter than n)
        sorted_weights_top15 = {doc_id:doc_score for doc_id,doc_score in sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:15]}

        # Iterate over each doc_id:weight for the predictions
        for (doc_id, weight) in sorted_weights_top15.items():
            print(doc_id + ': ' + str(weight))  # print results data

        print()  # print linebreak for readability