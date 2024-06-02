import numpy as np
import pandas as pd

from scipy import stats

def calculate_precision(evaluations, model_results, threshold: float, top_k: int = None) -> pd.DataFrame:
    """
    Calculate the precision for each topic in a collection, optionally only considering the top_k results.
    If top_k is specified, precision is calculated based on the top_k highest scored documents.
    Average Precision across all topics is appended as a final row.
    """

    precisions = []

    for topic, relevancy in evaluations.items():
        predicted_scores = model_results.get(topic, {})

        # Sort and possibly limit the results to top_k if specified
        if top_k:
            top_items = sorted(predicted_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            filtered_scores = dict(top_items)
        else:
            filtered_scores = {doc_id: score for doc_id, score in predicted_scores.items() if score > threshold}

        # Calculate the number of predicted relevant documents that meet the threshold
        retrieved_docs = len(filtered_scores)

        # Calculate the number of correctly retrieved documents (true positives)
        true_positives = sum(1 for doc_id in filtered_scores.keys() if relevancy.get(doc_id) == 1)

        # Calculate precision
        precision = true_positives / retrieved_docs if retrieved_docs > 0 else 0

        # Append results to list for DataFrame conversion
        precisions.append({'topic': topic, 'precision': precision})

    # Create DataFrame
    precision_df = pd.DataFrame(precisions)

    # Calculate MAP (Average) and append as a new row
    map_score = precision_df['precision'].mean()
    average_row = pd.DataFrame([{'topic': 'MAP' if not top_k else 'Average', 'precision': map_score}])
    precision_df = pd.concat([precision_df, average_row], ignore_index=True)

    return precision_df

def calculate_dcg(evaluations, model_results, threshold, p):
    """
    Calculate the Discounted Cumulative Gain (DCG) at rank p (effectively top k) for each topic in a collection.
    DCG is calculated using a logarithmic discount factor to decrease the weight of relevance for documents retrieved later in the list.
    """

    dcgs = []

    for topic, relevancy in evaluations.items():
        predicted_scores = model_results.get(topic, {})

        # Sort predicted scores and limit results to top p
        top_p_scores = sorted(predicted_scores.items(), key=lambda x: x[1], reverse=True)[:p]

        # Calculate DCG using the logarithmic discount
        dcg = 0
        for rank, (doc_id, score) in enumerate(top_p_scores, start=1):
            relevance = 1 if score > threshold and relevancy.get(doc_id, 0) == 1 else 0
            if rank == 1:
                dcg += relevance  # no discount for the first item
            else:
                dcg += relevance / np.log2(rank)  # discounting starts from the second item

        # Append results to list for DataFrame conversion
        dcgs.append({'topic': topic, 'DCG': dcg})

    # Create DataFrame
    dcg_df = pd.DataFrame(dcgs)

    # Calculate Average DCG and append as a new row
    average_dcg = dcg_df['DCG'].mean()
    average_row = pd.DataFrame([{'topic': 'Average DCG', 'DCG': average_dcg}])
    dcg_df = pd.concat([dcg_df, average_row], ignore_index=True)

    return dcg_df

def compare_models(df, column_names, metric):
    """
    This function performs two-tailed t-tests to compare the performance of each model based on an evaluation metric.
    """

    # Extract the names of the columns with the relevant data
    prm_col = column_names[0]
    bm25_col = column_names[1]
    jmlm_col = column_names[2]


    # Get the arrays, excluding the average score
    prm_scores = df[prm_col][0:50]
    bm25_scores = df[bm25_col][0:50]
    jmlm_scores = df[jmlm_col][0:50]

    # PRM vs BM25
    t_stat_prm_vs_bm25, p_value_prm_vs_bm25 = stats.ttest_rel(prm_scores, bm25_scores, alternative = 'two-sided')       
    # PRM vs JM_LM
    t_stat_prm_vs_jmlm, p_value_prm_vs_jmlm = stats.ttest_rel(prm_scores, jmlm_scores, alternative = 'two-sided')
    # BM25 vs JM_LM
    t_stat_bm25_vs_jmlm, p_value_bm25_vs_jmlm = stats.ttest_rel(bm25_scores, jmlm_scores, alternative = 'two-sided')

    # Create names for the columns, including the metric name
    t_name = 't-statistic_' + metric
    p_name = 'p-value_' + metric

    # Store results in dataframe
    results = pd.DataFrame({

        'model_comparison': ['PRM vs BM25', 'PRM vs JM_LM', 'BM25 vs JM_LM'],
        t_name: [t_stat_prm_vs_bm25, t_stat_prm_vs_jmlm, t_stat_bm25_vs_jmlm],
        p_name: [p_value_prm_vs_bm25, p_value_prm_vs_jmlm, p_value_bm25_vs_jmlm]
    })

    return results