from data_structures import bow_document_collection

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