from data_structures import bow_document_collection

def term_specificity(collection: bow_document_collection, query: dict, evaluations: dict, theta_1: float | int, theta_2: float | int) -> dict:
    """
    DEPRECATED: Can only use evaluations from Task 5 onwards (cannot incorporate them to train specificity)

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

def calculate_weighted_f1_score(evaluations, model_results, threshold: float, beta: float = 1.0, top_k: int = None) -> float:
    """
    DEPRECATED: Not directly required in task, implemented out of interest. However, worth discussing Precision/Recall/F1 in report.
    
    Calculate the weighted F1 score across all topics for a given threshold and beta, optionally considering only the top_k results.
    """
    total_f1_score = 0
    count = 0

    for topic, relevancy in evaluations.items():
        predicted_scores = model_results.get(topic, {})

        # Sort and possibly limit the results to top_k if specified
        if top_k:
            top_items = sorted(predicted_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            filtered_scores = dict(top_items)  # Convert sorted list back to dict
        else:
            filtered_scores = {doc_id: score for doc_id, score in predicted_scores.items() if score > threshold}

        # Calculate true positives, false positives, and false negatives
        true_positives = sum(1 for doc_id in filtered_scores.keys() if relevancy.get(doc_id) == 1)
        false_positives = sum(1 for doc_id in filtered_scores.keys() if relevancy.get(doc_id) == 0)
        false_negatives = sum(1 for doc_id, is_relevant in relevancy.items() if is_relevant == 1 and doc_id not in filtered_scores)

        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

        # Calculate weighted F1 score
        if (beta**2 * precision + recall) != 0:
            f1_score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        else:
            f1_score = 0

        # Add to total F1 score and increment count
        total_f1_score += f1_score
        count += 1

    # Calculate average F1 score across all topics
    average_f1_score = total_f1_score / count if count > 0 else 0
    return average_f1_score

def f1_grid_search(evaluations, model_results, thresholds, top_ks: list = None):
    """
    DEPRECATED: Not directly required in task, implemented out of interest. However, worth discussing Precision/Recall/F1 in report.

    Example:
    low_threshold = 0.0000001
    step_size = 0.0001
    num_steps = 100
    thresholds = [low_threshold + i * step_size for i in range(num_steps)]
    top_ks = [10]
    f1_grid_search(evaluations, JM_LM_results, thresholds)
    
    Determine the best combination of threshold(s) and top_k results which yield the best F1 score.
    """

    best_score = -float('inf')  # Assuming higher score is better; adjust if needed
    best_params = {}

    # Loop over all combinations of threshold and top_k
    for threshold in thresholds:
        if top_ks:
            for top_k in top_ks:
                # Calculate average precision for the current combination of threshold and top_k
                average_f1  = calculate_weighted_f1_score(evaluations, model_results, threshold, top_k)
                
                # Check if the current score is better than what we've seen and update best score and parameters
                if average_f1  > best_score:
                    best_score = average_f1 
                    best_params = {'threshold': threshold}
        
        else:
            # Calculate average precision for the current combination of threshold
            average_f1  = calculate_weighted_f1_score(evaluations, model_results, threshold)
            
            # Check if the current score is better than what we've seen and update best score and parameters
            if average_f1  > best_score:
                best_score = average_f1
                best_params = {'threshold': threshold}

    return best_score, best_params