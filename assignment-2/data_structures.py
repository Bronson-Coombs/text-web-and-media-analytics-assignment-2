class bow_document:
    def __init__(self, item_id: str):
        # Type check to ensure object is initialised correctly
        if not isinstance(item_id, str):
            raise TypeError("item_id: value must be a string.")
            # Technically could work with str or int indexing (for key in collection),
            # using *only* str ensures no double-up of pointers
            # (e.g. item_id '1' vs item_id 1)

        self.doc_id = item_id  # assigning doc_id from 'item_id'
        self.terms = {}  # dictionary for terms and their frequencies
        self._doc_len = 0  # document length, private attribute

    def add_term(self, term: str):
        """Add a term to the document or update its frequency if it already exists."""
        
        # Type check(s)
        if not isinstance(term, str):
            raise TypeError("term: value must be a string.")
        
        self.doc_len += 1  # extend doc_len

        if term in self.terms:
            self.terms[term] += 1  # add frequency if the term exists
        else:
            self.terms[term] = 1  # if it doesn't exist, add it (setting frequency to 1)
        
    def get_doc_id(self) -> str:
        """Return the document ID."""
        return self.doc_id
    
    def get_term_list(self, sorted_by_freq: bool = None) -> dict:
        """
        Return a list of terms occurring in the document, optionally sorted by their frequency.
        If sorted_by_freq is True, the terms are returned sorted by their frequency in descending order.
        If sorted_by_freq is False or None (default), the terms are returned in arbitrary order.
        """

        # Type check(s)
        if not isinstance(sorted_by_freq, (bool, type(None))):
            raise TypeError("sorted_by_freq: must be a boolean or None.")

        if sorted_by_freq:
            # If sorted_by_freq is True
            sorted_terms = sorted(self.terms.items(), key=lambda word: word[1], reverse=sorted_by_freq)  # generate a sorted list of terms by frequency
            return {term: freq for term, freq in sorted_terms}  # return key:value pairs based on sorted terms
        else:
            # If sorted_by_freq is False or None, return the terms as is (i.e., unsorted and as they are added in)
            return self.terms
        
    def get_bag_of_words(self, sorted_by_freq: bool = None) -> str:
        """Return full bag-of-words representation for bow_document object, including; doc_id, term_count, doc_len, and terms."""
        
        # Type check(s)
        if not isinstance(sorted_by_freq, (bool, type(None))):
            raise TypeError("sorted_by_freq: must be a boolean or None.")

        # Defining formatted string for bag-of-word representation
        bag_of_words = f"""doc_id='{self.doc_id}',term_count={len(self.get_term_list())},doc_len={self.doc_len},terms={self.get_term_list(sorted_by_freq)}"""

        return bag_of_words  # return BOW representation; this kind of data can be stored and "unpacked" easily
    
    @property  # accessor (get) method for doc_len
    def doc_len(self) -> int:
        """The doc_len property getter method."""
        return self._doc_len

    @doc_len.setter  # mutator (setter) method for doc_len
    def doc_len(self, value: int):
        """The doc_len property setter method."""
        
        # Type check(s)
        if not isinstance(value, int):
            raise TypeError("doc_len: must be an int.")
        
        # Logical check on doc_len
        if value < 0:
            raise ValueError("doc_len: must not be negative.")
        
        self._doc_len = value

class bow_document_collection:
    def __init__(self):
        self.docs = {}  # initialise dictionary to hold collection (dict) of doc_id:bow_document

        self.term_doc_count = {}  # initialise dictionary to track the number of documents each term appears in
        self.term_frequency = {}  # initialise dictionary to track the total number of times each term appears

    # Method to add a doc (bow_document object)
    def add_doc(self, doc: bow_document):
        """Add bow_document object to the collection, using doc_id as the key, and update the inverted index."""

        # Type check(s)
        if not isinstance(doc, bow_document):
            raise TypeError("doc: must be an instance of bow_document.")
        
        # Add to the docs dict; key as doc_id and value as bow_document object (doc_id:bow_document)
        self.docs[doc.get_doc_id()] = doc

        # Update term document count and total frequency for each term
        for term in doc.terms:
            if term in self.term_doc_count:
                self.term_doc_count[term] += 1  # add 1 if the term exists in the corpus dictionary
            else:
                self.term_doc_count[term] = 1  # if it does not exist in the corpus dictionary, initialise by setting to 1
            
            if term in self.term_frequency:
                self.term_frequency[term] += 1  # add 1 if the term exists in the corpus dictionary
            else:
                self.term_frequency[term] = 1  # if it does not exist in the corpus dictionary, initialise by setting to 1
    
    def get_collection_ids(self) -> str:
        """Return list of document IDs present in the collection."""

        # Type check to ensure doc_id is a string
        if not len(self.docs) > 0:
            raise AttributeError("bow_document_collection object is empty, no IDs to return.")  # Corrected to match the check
        
        doc_ids_str = "'" + "', '".join(self.docs.keys()) + "'"  # create a string that lists doc_ids

        collection_ids = f"bow_document_collection(doc_ids: {doc_ids_str})"  # format the return variable

        return collection_ids