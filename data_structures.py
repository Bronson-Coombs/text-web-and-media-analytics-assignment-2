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