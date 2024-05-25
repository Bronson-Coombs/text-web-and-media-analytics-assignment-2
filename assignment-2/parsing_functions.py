import csv
import os
import pandas as pd
import regex as re
import string
import nltk

from data_structures import bow_document, bow_document_collection

def parse_stop_words(stop_word_path: str) -> list:
    """Parse defined list of stop words (assumes txt file with words delimited with ',')."""

    # Type check to ensure stop_word_path is a str
    if not isinstance(stop_word_path, str):
        raise TypeError("stop_word_path: value must be a str.")
    
    # NOTE: need attribute check the path exists

    # Open file in read mode
    with open(stop_word_path, 'r') as file:
        stop_words = file.read()  # read text in given file into stop_words

    # We know what the format is ahead of time, so not a lot of processing needed;
    # i.e., assumes we don't need to make something more robust and that we're using the same txt
    stop_words = stop_words.lower().split(",")  # tokenize stop_words; delimited with ','
    stop_words = list(set(stop_words))  # reduce stop_words to uniques
    
    return stop_words  # return stop_words as a list object

def tokenization(words: str) -> list:
    """Tokenize input text by removing line breaks, numbers, punctuation, normalizing whitespace, stripping leading/trailing spaces, and splitting into lowercased words."""

    # Type check to ensure words is a str
    if not isinstance(words, str):
        raise TypeError("words: value must be a str.")

    words = words.replace("\n", "")  # don't want line breaks to contribute
    words = re.sub(r'\d', '', words)  # not interested in numbers for this particular task, remove
    words = re.sub(f'[{re.escape(string.punctuation)}]', ' ', words)  # not interested in punctuation, remove
    words = re.sub(r'\s+|\t+|\v+|\n+|\r+|\f+', ' ', words).strip()  # standardise the whitespaces, remove leading/trailing whitespace
    words = words.lower()  # standardise words as lower
    words = words.split()  # tokenize, deftault split on space

    # Filter out small words; can be important in some queries, usually in combinations, opting not to handle for simplicity.
    # For example, with no discrete management of apostrophes (indicating contractions or posession) aside from replacement 
    # of punctuation with a single space, we will get the following: "Amelia's" → ["Amelia", "s"] → ["Amelia"].
    # Unless they are actual words (e.g., "I" versus "s" or "t"), they won't be removed in stopping process.
    words = [word for word in words if len(word) >= 3]

    return words  # return list object of string words

def parse_xml(stop_words: list, xml_path: str) -> bow_document:
    """Parse a single XML file, process text, and return an bow_document object with term frequencies."""
    
    # Type check to ensure stop_words is a list of str
    if not isinstance(stop_words, list) or not all(isinstance(word, str) for word in stop_words):
        raise TypeError("stop_words: must be a list of strings.")
    
    # Type check to ensure xml_path is a str
    if not isinstance(xml_path, str):
        raise TypeError("xml_path: value must be a str.")
    
    # Check if provided xml_path is a valid xml file, raise AttributeError if it is not
    if not ((os.path.isfile(xml_path)) and (xml_path.lower().endswith(".xml"))):
        raise AttributeError(f"""xml_path: '{xml_path}' is not a valid xml file.""")
        # NOTE: check is included here for targeting single xml (wheras parse_rcv1v2() executes this check in loop)

    # DOCUMENT PARSING - recognition of the content and structure of text documents
    # Open file in read mode
    with open(xml_path, 'r') as file:
        xml = file.read()  # read xml in given file

    text = re.search(r'<text>\s*((?:<p>.*?</p>\s*)+)</text>', xml, re.DOTALL)  # find all text within the <text> tag

    # If no text found, raise attribute error; else return match group 1
    if not text:
        raise AttributeError(fr"""xml_path: '{xml_path}' did not contain any text, see text tag (expect match at '<text>\s*((?:<p>.*?</p>\s*)+)</text>' with re.DOTALL).""") 
    else:
        text = text.group(1)

    # Replace HTML entities with their corresponding characters
    html_entities = {"&lt;": "<", "&gt;": ">", "&amp;": "&", "&quot;": "\"", "&apos;": "'", "&nbsp;": " " }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)
    
    text = re.sub(r'<.*?>', ' ', text).strip()  # remove any XML tags (p tags in our case)
    
    # TOKENIZING - forming words from sequence of characters; critically, generating a list of tokens
    words = tokenization(text)
    
    # POSTING - a collection of arbitrary data (including a pointer)
    item_id = re.search(r'<newsitem itemid="(\d+)"', xml)  # POINTER - a unique identifier of a document (item_id attribute from newsitem element in this case)

    if not item_id:
        # If no item_id found, raise attribute error
        raise AttributeError(f"""xml_path: '{xml_path}' did not contain pointer, see item_id attribute in newsitem tag (expect match at '<newsitem itemid="(\\d+)"').""") 
    else:
        item_id = item_id.group(1)  # otherwise, take group 1 of regex (just the \d+ match component)
        
    document = bow_document(item_id)  # initialise bow_document object with the pointer (item_id)

    # STOPPING - removing stop (function) words from the text being analysed; have little meaning on their own
    words = [word for word in words if word not in stop_words]
    
    # STEMMING - reducing words to their word stem, base or root form (remove morphological variations)
    stemmer = nltk.stem.PorterStemmer()  # Porter Stemmer: efficient for information retrieval and text processing tasks – can often create non-words in favour of faster speeds
    words = [stemmer.stem(word) for word in words] 
    
    # Iterate over each stemmed word
    for stemmed_word in words:
        document.add_term(stemmed_word)  # use method add_term to update the bow_document object (our arbitrary data)          

    return document  # return the bow_document object

def parse_collection(stop_words: list, input_path: str) -> bow_document_collection:
    """Parse XML documents in a directory, filter stop words, and return a collection of bow_document objects."""
    
    # Type check to ensure stop_words is a list of str
    if not isinstance(stop_words, list) or not all(isinstance(word, str) for word in stop_words):
        raise TypeError("stop_words: must be a list of strings.")
    
    # Type check to ensure input_path is a str
    if not isinstance(input_path, str):
        raise TypeError("input_path: value must be a str.")
    
    # NOTE: need to do attribute check to see if input_path exists

    collection = bow_document_collection()  # initialise bow_document_collection object (collection of bow_document objects)
    
    # Iterate through files in directory
    for xml_file in os.listdir(input_path):
        xml_path = os.path.join(input_path, xml_file)  # build path to files
        if ((os.path.isfile(xml_path)) and (xml_path.lower().endswith(".xml"))):
            doc = parse_xml(stop_words, xml_path)  # parse xml with xml_parser function
            collection.add_doc(doc)  # use method add_doc to update the bow_document_collection object

    # If no xmls parsed (i.e., collection length is 0), raise attribute error
    if len(collection.docs) == 0:
        raise AttributeError(f"""input_path: '{input_path}' did not contain any valid xml files.""")

    return collection  # return the bow_document_collection object

def parse_query(query: str, stop_words: list) -> dict:
    """Tokenize an input query, remove stop words, and return a dictionary of remaining word frequencies."""

    # Type check to ensure stop_words is a list of str
    if not isinstance(stop_words, list) or not all(isinstance(word, str) for word in stop_words):
        raise TypeError("stop_words: must be a list of strings.")
    
    # Type check to ensure query is a str
    if not isinstance(query, str):
        raise TypeError("query: value must be a string.")
    
    # TOKENIZING - forming words from sequence of characters; critically, generating a list of tokens
    words = tokenization(query)
    
    # STOPPING - removing stop (function) words from the text being analysed; have little meaning on their own
    words = [word for word in words if word not in stop_words]
    
    # STEMMING - reducing words to their word stem, base or root form (remove morphological variations)
    stemmer = nltk.stem.PorterStemmer()  # Porter Stemmer: efficient for information retrieval and text processing tasks – though can often create non-words in favour of faster speeds
    words = [stemmer.stem(word) for word in words]
    
    # Constrcut term:frequency dictionary by counting instances of each word (more efficient than for loop + if/else)
    query_term_frequency = {stemmed_word: words.count(stemmed_word) for stemmed_word in set(words)}

    return query_term_frequency  # return the dictionary containing word frequencies

def parse_query_set(query_set: str) -> pd.DataFrame:
    
    # Type check to ensure the query_set is a string
    if not isinstance(query_set, str):
        raise TypeError("query_set: value must be a string.")
    
    # Check to see if the query_set exists
    if not os.path.exists(query_set):
        raise AttributeError("query_set: file does not exist.")
        
    with open(query_set, 'r') as file:
        data = file.read()
    
    # Define regex pattern to split queries
    query_pattern = re.compile(r'<Query>(.*?)</Query>', re.DOTALL)
    queries = query_pattern.findall(data)
    
    # Initialize lists for storing parsed data
    nums, titles, descriptions, narratives = [], [], [], []
    
    # Define regex patterns to extract individual fields
    num_pattern = re.compile(r'<num>\s*Number:\s*R(\w+)', re.MULTILINE)
    title_pattern = re.compile(r'<title>([\w\s,.-]*)', re.MULTILINE)
    desc_pattern = re.compile(r'<desc>\s*Description:\s*(.*?)\n\n', re.DOTALL)
    narr_pattern = re.compile(r'<narr>\s*Narrative:\s*(.*?)\n\n', re.DOTALL)
    
    for query in queries:
        # Extract data using regex patterns
        num_match = num_pattern.search(query)
        title_match = title_pattern.search(query)
        desc_match = desc_pattern.search(query)
        narr_match = narr_pattern.search(query)
        
        nums.append(num_match.group(1) if num_match else pd.NA)
        titles.append(title_match.group(1).strip() if title_match else pd.NA)
        descriptions.append(desc_match.group(1).strip() if desc_match else pd.NA)
        narratives.append(narr_match.group(1).strip() if narr_match else pd.NA)

    # Create a pandas DataFrame
    query_frame = pd.DataFrame({
        'number': nums,
        'title': titles,
        'description': descriptions,
        'narrative': narratives
    })

    return query_frame

def parse_evaluations(evaluation_path: str) -> dict:
    # Type check to ensure the evaluation_path is a string
    if not isinstance(evaluation_path, str):
        raise TypeError("evaluation_path: value must be a string.")
    
    # Check to see if the evaluation_path exists
    if not os.path.exists(evaluation_path):
        raise AttributeError("evaluation_path: directory does not exist.")

    # Load document relevance from evaluation files
    evaluations = {}
    for filename in os.listdir(evaluation_path):
        if filename.startswith("Dataset") and filename.endswith(".txt"):
            filepath = os.path.join(evaluation_path, filename) # create full filepath
            data_key = re.sub(r'\D', '', filename)  # 'Dataset101.txt' → '101'

            evaluations[data_key] = {}  # initialise sub-dictionary

            with open(filepath, 'r') as file:
                for line in file:
                    parts = line.strip().split()  # remove whitespace and split the line
                    doc_id = parts[1]  # get the doc_id
                    relevance = int(parts[2])  # get relvency Boolean
                    
                    evaluations[data_key][doc_id] = relevance

    # Check if relevency judgements were found
    if not evaluations:
        raise FileNotFoundError(r"evaluation_path: no r'Dataset\d+.txt' files were found in the directory.")

    # return evaluation data
    return evaluations