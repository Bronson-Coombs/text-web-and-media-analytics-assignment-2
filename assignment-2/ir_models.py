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
            saturation_component = ((k_1 + 1) * document_term_frequency) / (K + document_term_frequency)
            # This component adjusts the score based on the frequency of the term in the document.
            # The normalisation (denominator) prevents over-emphasis on terms that appear too frequently within a single document.
            # `k_1` controls the non-linear term frequency saturation, and `K` adjusts the weight based on document length.

            # Calculate the query term frequency normalization
            query_component = ((k_2 + 1) * query_frequency) / (k_2 + query_frequency)
            # Adjusts the score based on the query term's frequency.
            # Denominator prevents over-emphasis on query terms that appear frequently.
            
            score = idf_component * saturation_component * query_component  # determine the score
            
            doc_scores[doc_id] += score  # update the document's score with the product of the IDF and TF components

    # Normalise, sort, return
    doc_scores = score_normalisation(doc_scores)
    doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))
    return doc_scores

def JM_LM(collection: bow_document_collection, query: dict) -> dict:
    """
    Calculate the conditional probability of each document given a query using the Jelinek-Mercer smoothing Language Model.
    This implementation uses a multiplicative approach and ensures that documents with no relevant terms receive a score of 0.
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

    # Initialize dictionary with Multiplicate identity for each document
    doc_scores = {doc_id: 1 for doc_id in collection.docs}

    # To handle documents that do not have any terms from the query,
    # introduce a flag that tracks whether a document has been updated.
    updated_docs = {doc_id: False for doc_id in collection.docs}

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
                doc_scores[doc_id] *= score
                updated_docs[doc_id] = True

    # Adjust scores to 0 if no terms were relevent
    doc_scores = {doc_id: (score if updated_docs[doc_id] else 0) for doc_id, score in doc_scores.items()}

    # Sort the documents by their score in descending order and return the sorted dictionary (normalisation not required)
    doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))
    return doc_scores

def cosine_similarity(vector1: list, vector2: list) -> float:
    """
    This function calculates the cosine similarity for two vectors.

    Cosine similarity is a measure of similarity between two non-zero vectors
    of an inner product space that measures the cosine of the angle between them.
    The cosine similarity is particularly used in positive spaces where the
    outcome is neatly bounded in [0,1].
    """

    # Calculate the dot product of the two vectors; sum of the products of the corresponding elements of the vectors
    dot_product = sum(x * y for x, y in zip(vector1, vector2))

    # Compute the Euclidean norms (also called the magnitude or length); square root of the sum of the squares of the elements
    norm1 = math.sqrt(sum(x ** 2 for x in vector1))
    norm2 = math.sqrt(sum(x ** 2 for x in vector2))

    # Compute the cosine similarity; dot product of the vectors divided by the product of their norms.
    return dot_product / (norm1 * norm2) if (norm1 and norm2) != 0 else 0.0  # return similarity, handling 0 case for a Euclidean norm of 0

def vector_space_model(collection: bow_document_collection, query: dict) -> dict:
    """
    The Vector Space Model is an algorithm that calculates scores for each document
    based on a given query. The model incorporates certain dimensions that can be
    changed appropriately, such as similiarity measures, query and the equation for 
    calculating document weights. The following function uses the cosine similarity 
    and the tf-idf equation. It returns a dictionary that contains documents and 
    scores respectively.
    """

    # Type check to ensure coll is a bow_document_collection
    if not isinstance(collection, bow_document_collection):
        raise TypeError("collection: must be a bow_document_collection object.")
    
    # If collection contains no documents, raise attribute error
    if len(collection.docs) == 0:
        raise AttributeError("collection: object contains no documents (bow_document objects).")
    
    # Type check to ensure query is a dict
    if not isinstance(query, dict):
        raise TypeError("query: must be a dict object.")
    
    # Initializations
    document_similarity_scores = {}
    document_vectors = {}
    idf_component = {}

    collection_term_frequency = collection.term_frequency  # dict of the total number of times each term appears in collection
    N = len(collection.docs)  # total number of documents in the collection

    # Calculate idf for all terms
    for term, doc_freq in collection_term_frequency.items():
        idf_component[term] = math.log(N / doc_freq, 10)

    # Get all terms of the collection
    all_terms = set(collection_term_frequency.keys())

    # Represents a general n-dimensional vector, which will be used to construct
    # the query- and document-vector. Therefore, a normalization is not necessary, 
    # due to the fact that the algorithm doesn't compare documents and different queries against each other.
    vector = [0] * len(all_terms)
    query_vector = [0] * len(all_terms)

    # Convert the query parameter into a vector representation
    for index, term in enumerate(all_terms):
        query_vector[index] = query.get(term, 0)
    
    # Calculate weights for each document
    for doc_ID, doc in collection.docs.items():
        term_freq_dict = doc.get_term_list()

        # Loop through all terms
        for index, term in enumerate(all_terms):
            # Calculate the term weigths using tf-idf equation
            vector[index] = term_freq_dict.get(term, 0) * idf_component.get(term, 0)
        
        # Store weighted vector for every document
        document_vectors[doc_ID] = vector 

    # Calculate cosine similarity for query and documents
    for doc_ID, doc_vector in document_vectors.items():
        document_similarity_scores[doc_ID] = cosine_similarity(doc_vector, query_vector)

    # Return documents in descending order (based on values of the weights)
    return dict(sorted(document_similarity_scores.items(), key=lambda item: item[1], reverse=True))

def w5(collection: bow_document_collection, evaluations: dict, theta: int | float = 0) -> dict:
    """
    This function calculates the weight-5 score for the whole collection based 
    on a dictionary that contains relevant and irrelevant documents. Theta is 
    a parameter which can be changed due to receive reasonable results.
    This function returns a dictionary with features and their calculated 
    weights.
    """
    
    # Type check(s)
    if not isinstance(collection, bow_document_collection):
        raise TypeError("collection: must be a bow_document_collection object.")
    if not isinstance(evaluations, dict):
        raise TypeError("evaluations: value must be a dict.")
    if not isinstance(theta, (int, float)):
        raise TypeError("theta: value must be an int or float.")
    
    # Collection check
    if len(collection.docs) == 0:
        raise AttributeError("collection: object contains no documents (bow_document objects).")

    # Initialisations
    term_relevance_count = {}
    term_total_count = {}
    mean_weight5 = 0  # average

    N = len(collection.docs)  # total number of documents in the collection
    R = sum(1 for relevancy in evaluations.values() if relevancy == 1)  # total number of relevant documents

    # Count relevant and total occurrences of each term
    for doc_id, doc in collection.docs.items():
        for term in doc.terms.keys():
            if evaluations[doc_id] == 1:
                term_relevance_count[term] = term_relevance_count.get(term, 0) + 1  # initialise term with 1 if not present, otherwise add 1
            term_total_count[term] = term_total_count.get(term, 0) + 1  # initialise term with 1 if not present, otherwise add 1

    # Calculate the weight-5 score for each term
    weight5_scores = {}
    for term, relevance_count in term_relevance_count.items():
        n_tk = term_total_count[term]
        score_numerator = ((relevance_count + 0.5) / (R - relevance_count + 0.5))
        score_denominator = ((n_tk - relevance_count + 0.5) / (N - n_tk - R + relevance_count + 0.5))
        weight5_scores[term] = score_numerator / score_denominator

    # Calculate the mean of all weight-5 scores if there are relevant documents
    if R > 0:
        mean_weight5 = sum(weight5_scores.values()) / len(weight5_scores)

    # Select features based on the mean weight5-score and theta
    selected_features = {term: score for term, score in weight5_scores.items() if score > mean_weight5 + theta}
    return selected_features

def My_PRM(weighting_function, collection: bow_document_collection, query: dict, threshold: int, theta: int) -> dict:
    """
    This function represents a Pseudo-Relevance-Model (PRM) to rank documents 
    based on pseudo feedback. It accepts a weighting function as parameter such
    as the BM25 or Vector Space Model algorithm to calculate weights for 
    each document in a given collection with respect to a given query. The 
    parameters threshold and theta are used to fine-tune the algorithm. The PRM returns
    a sorted dictionary.
    """
    
    # Ensure that the weighting function is callable
    if not callable(weighting_function):
        raise TypeError("weighting_function must be a callable function.")
    
    # Initializations
    relevant_documents = {}
    doc_scores = {doc_id: 0 for doc_id in collection.docs}

    # 1) Based on the given query, PRM calculates bm25-score/VSM-score for each document
    weighting_result = weighting_function(collection, query)

    # 2) Based on a defined threshold (e.g. value=1.0), the algorithm marks relevant documents
    # (positive label 1) that are greater than the defined threshold, otherwise as irrelevant (negative label 0)
    for doc, score in weighting_result.items():
        relevant_documents[doc] = 1 if score > threshold else 0

    # 3) Calculates w5-score to identify a set of features
    w5_results = w5(collection=collection, evaluations=relevant_documents, theta=theta)

    # 4) Calculate document scores based on the identified features
    for doc_ID, doc in collection.docs.items():
        for term, frequency in doc.terms.items():
            doc_scores[doc_ID] += frequency * w5_results.get(term, 0)
        
    # Normalise, sort, return
    doc_scores = score_normalisation(doc_scores)
    doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))
    return doc_scores