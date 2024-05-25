import os
import csv

from data_structures import bow_document_collection

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

def term_specificity(collection: bow_document_collection, query: dict, evaluations: dict, theta_1: float | int, theta_2: float | int) -> dict:
    """
    Calculates term specificity based on the coverage of terms in relevant and non-relevant documents,
    then adjusts query term weights based on their calculated specificity thresholds.

    The function differentiates terms into positive, general, and negative categories based on
    their specificity values exceeding, falling within, or not meeting defined thresholds (theta_1 and theta_2).
    It adjusts the weights of terms in the query based on their categorization to enhance the effectiveness
    of information retrieval tasks.
    """

    # Type check(s)
    if not isinstance(collection, bow_document_collection):
        raise TypeError("collection: must be a bow_document_collection object.")
    if not isinstance(query, dict):
        raise TypeError("query: must be a dict object.")
    if not isinstance(theta_1, (float, int)):
        raise TypeError("theta_1: must be a float or int value.")
    if not isinstance(theta_2, (float, int)):
        raise TypeError("theta_2: must be a float or int value.")
    if not isinstance(evaluations, dict) or \
    not all(isinstance(doc_id, str) and isinstance(relevance, int) and relevance in [0, 1]
                for doc_id, relevance in evaluations.items()):
        raise TypeError("evaluations: must be a dict with str keys and with values 0 or 1.")

    # If no collection contains no documents, raise attribute error
    if len(collection.docs) == 0:
        raise AttributeError("bow_document_collectionection: object contains no documents (Rcv1Doc objects).")

    # Boundary check
    if not theta_2 > theta_1:
        raise ValueError("theta_2: must be greater than theta_1.")
    
    # Pulling list of relevant and non-relevant documents for each query
    relevant_docs = {doc_id:relevance for doc_id, relevance in evaluations.items() if relevance == 1}
    non_relevant_docs = {doc_id:relevance for doc_id, relevance in evaluations.items() if relevance == 0}

    # Ensure we have at least one relevant and at least one non-relevant document
    if not len(relevant_docs) > 0 or not len(non_relevant_docs) > 0:
        raise AttributeError("evaluations: each query must contain at least one relevant document and at least one non-relevant document.")
                
    specificity = {}

    D = len(relevant_docs)
    
    # Loop through each term in the query.
    for query_term in query.keys():
        positive_coverage = 0
        negative_coverage = 0
        for doc_id, doc in collection.docs.items():
            if doc_id in relevant_docs:
                positive_coverage += doc.terms.get(query_term, 0)
            elif doc_id in non_relevant_docs:
                negative_coverage += doc.terms.get(query_term, 0)
        
        specificity[query_term] = (positive_coverage - negative_coverage) / D

    positive_terms = [term for term in specificity if specificity[term] >= theta_2]
    general_terms = [term for term in specificity if theta_1 < specificity[term] < theta_2]
    negative_terms = [term for term in specificity if specificity[term] <= theta_1]

    weighted_terms = {}

    for query_term, query_term_frequency in query.items():
        if query_term in positive_terms:
            weighted_terms[query_term] = query_term_frequency + query_term_frequency * specificity[query_term]
        elif query_term in general_terms:
            weighted_terms[query_term] = query_term_frequency
        elif query_term in negative_terms:
            weighted_terms[query_term] = query_term_frequency - abs(query_term_frequency * specificity[query_term])

    return weighted_terms

def score_normalisation(scores: dict) -> dict:
    """
    Normalises the numerical scores for documents using Min-Max scaling.
    
    The function takes a dictionary of document scores and scales each score
    linearly between 0 and 1. If all scores are identical, they are set to 0 to reflect
    no variation in significance among them.
    """
    
    # Type check(s)
    if not isinstance(scores, dict) or not all(isinstance(doc_id, str) and isinstance(score, (int, float)) for doc_id, score in scores.items()):
        raise TypeError("scores: value must be a dict of str keys and int or float values")

    # Min-Max normalisation
    min_score = min(scores.values())
    max_score = max(scores.values())
    if (max_score - min_score) == 0:  # to avoid division by zero if all scores are the same,
        return {doc_id: 0 for doc_id in scores}  # return 0 (opted for this rather than 1 becuase if everything is relevant, nothing it)
    
    # Apply normalisation and return
    return {doc_id: (score - min_score) / (max_score - min_score) for doc_id, score in scores.items()}