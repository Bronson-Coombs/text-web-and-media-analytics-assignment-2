import math

from data_structures import bow_document_collection
from ir_tools import score_normalisation

def BM25(collection: bow_document_collection, query: dict) -> dict:
    """
    BM25 ranking function for a collection of documents and a given query.
    Generates a score for a given documents term:frequency set.
    Incorporates term frequency (TF) and inverse document frequency (IDF) factors. 
    It accounts for term frequency saturation as well as document length bias.
    """
    
    # Type check to ensure coll is a bow_document_collection
    if not isinstance(collection, bow_document_collection):
        raise TypeError("collection: must be a bow_document_collection object.")
    
    # If no collection contains no documents, raise attribute error
    if len(collection.docs) == 0:
        raise AttributeError("collection: object contains no documents (bow_document objects).")
    
    # Type check to ensure query is a dict
    if not isinstance(query, dict):
        raise TypeError("query: must be a dict object.")
    
    # Setting parameters
    k_1 = 1.2  # Controls non-linear term frequency normalization (saturation)
    k_2 = 500  # Controls non-linear term frequency normalization for query terms
    b = 0.75  # Controls to what degree document length normalizes tf values

    N = len(collection.docs)  # total number of documents in the collection
    R = 0  # number of relevant documents for this query; predefined by task
    r_i = 0  # number of relevant documents containing query term i; predefined by task

    # Calculate the average document length across the entire collection
    total_corpus_length = sum(doc.doc_len for doc in collection.docs.values())
    mean_doc_len = total_corpus_length / N
    
    # Initialize dictionary with 0-score for each document
    doc_scores = {doc_id: 0 for doc_id in collection.docs}

    # Loop through each term in the query.
    for query_term, query_frequency in query.items():
        n_i = collection.term_doc_count.get(query_term, 0)  # the number of documents containing term i (0 if not present)

        # Calculate the inverse document frequency for the term
        idf_component = math.log10(((r_i + 0.5)/(R - r_i + 0.5)) / ((n_i - r_i + 0.5) / (N - n_i - R + r_i + 0.5)))

        # Component measures the rarity of the term across the entire collection; 
        # term appearing in fewer documents will have a higher IDF, making it more influential.
        # Formula ensures that no division by zero occurs by introducing additive smoothing of 0.5 to the numerator and denominator.

        for doc_id, doc in collection.docs.items():
            doc_len = doc.doc_len  # document length

            K = k_1 * ((1 - b) + b * doc_len / mean_doc_len)  # frequency normaliser
            
            document_term_frequency = doc.terms.get(query_term, 0)  # query term frequency within the document (0 if not present)
            
            # Calculate the term frequency normalization for the document term
            tf_component = ((k_1 + 1) * document_term_frequency) / (K + document_term_frequency)
            # This component adjusts the score based on the frequency of the term in the document.
            # The normalisation (denominator) prevents over-emphasis on terms that appear too frequently within a single document.
            # `k_1` controls the non-linear term frequency saturation, and `K` adjusts the weight based on document length.

            # Calculate the query term frequency normalization
            query_component = ((k_2 + 1) * query_frequency) / (k_2 + query_frequency)
            # Adjusts the score based on the query term's frequency.
            # Denominator prevents over-emphasis on query terms that appear frequently.
            
            score = idf_component * tf_component * query_component  # determine the score (can be non-negative, clamping used below to adjust)
            
            doc_scores[doc_id] += score  # update the document's score with the product of the IDF and TF components

    # Sort the documents by their score in descending order, normalise, and return
    doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))
    doc_scores = score_normalisation(doc_scores)
    return doc_scores

def JM_LM(collection: bow_document_collection, query: dict) -> dict:
    """
    Calculate the conditional probability of each document given a query using the Jelinek-Mercer smoothing Language Model.
    """
    
    # Type check(s)
    if not isinstance(collection, bow_document_collection):
        raise TypeError("collection: must be a bow_document_collection object.")
    if not isinstance(query, dict):
        raise TypeError("query: must be a dict object.")
    
    # Check if the collection contains any documents
    if len(collection.docs) == 0:
        raise AttributeError("bow_document_collection: object contains no documents (bow_document objects).")
       
    # Set lambda parameter for Jelinek-Mercer smoothing
    lambda_val = 0.4
    
    # Calculate the total length of the corpus by summing the lengths of all documents
    total_corpus_length = sum(doc.doc_len for doc in collection.docs.values())

    # Initialize dictionary with 0-score for each document
    doc_scores = {doc_id: 0 for doc_id in collection.docs}

    # Iterate through each term in the query
    for query_term in query:
        # Get the frequency of the query term in the entire collection
        term_collection_frequency = collection.term_doc_count.get(query_term, 0)
        
        # Iterate through each document in the collection
        for doc_id, doc in collection.docs.items():
            # Get the frequency of the query term in the current document
            term_document_frequency = doc.terms.get(query_term, 0)

            # Calculate the probability of the term occurring in the document
            p_doc = (term_document_frequency / doc.doc_len) if doc.doc_len > 0 else 0
            
            # Calculate the probability of the term occurring in the whole collection
            p_coll = (term_collection_frequency / total_corpus_length) if total_corpus_length > 0 else 0

            # Calculate the smoothed score for the term using Jelinek-Mercer smoothing
            score = (1 - lambda_val) * p_doc + lambda_val * p_coll

            # Multiply the score (add within the log space) to the cumulative product if it's greater than 0
            if score > 0:
                doc_scores[doc_id] += math.log(score)
                # Use of log adding the score avoids issues with scores that are initiated
                # with the multiplicative identity (1) but are not multiplied.

    # Sort the documents by their score in descending order and return the sorted dictionary (normalisation not required)
    doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))
    return doc_scores

