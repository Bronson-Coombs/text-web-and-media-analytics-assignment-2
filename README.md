# **Text, Web, & Media Analytics Assignment 2**

## System Requirements and Setup

### Python Version
- It is recommended to use **Python 3.12.2** for compatibility with the provided scripts.
- Check your current Python version by running `python --version` in your console.
- If you need to install Python 3.12.2, download it from the [official Python website](https://www.python.org/downloads/release/python-3122/) or manage versions via [pyenv](https://github.com/pyenv/pyenv).

### Package Installation
Navigate to the project's directory in your terminal and execute the following command to install necessary dependencies:
`pip install -r requirements.txt`

This command installs all required packages listed in `requirements.txt`, including:
- `nltk`: Used for processing human language data.
- `numpy`: Supports large, multi-dimensional arrays and matrices.
- `pandas`: Provides data manipulation and analysis tools.
- `regex`: Offers advanced regular expression matching functionalities.
- `scipy`: Used for scientific and technical computing.
- `ipykernel`: Provides the kernel for Jupyter.
- `pyarrow`: Interface to Apache Arrow data structures.

## Project Structure

### Codebase
The project's codebase includes the following files organized for easy navigation and usage:
- `Text, Web, & Media Analytics Assignment 2.ipynb`: Main Jupyter Notebook for the project.
- `data_structures.py`, `ir_evaluations.py`, `ir_models.py`, `ir_tools.py`, `parsing_functions.py`: Python scripts that refactor functions and classes used within the notebook.

### Required Directories and Files

#### Data Collection
- **The50Queries.txt**: Contains all 50 queries organized with XML tags. Specific queries are marked by `<Query>`, `<num>`, `<title>`, `<desc>`, and `<narr>` tags.
- **Data_Collection**: Includes sub-directories (e.g., Data_C101, Data_C102, ... up to Data_C150) containing `.xml` news files. Each file should have a `<newsitem>` tag with an `itemid` attribute and a `<text>` tag.

#### Evaluation Benchmark
- Located in `EvaluationBenchmark`, this directory houses `.txt` files (e.g., Dataset101.txt, Dataset150.txt) containing relevance judgments for each query. These text files are formatted with three columns: query key, document IDs, and a binary relevance indicator (0 for not relevant, 1 for relevant).

#### Additional Files
- **common-english-words.txt**: Contains comma-delimited common English stop words used in stopping processes.

## Execution Instructions

To run the system:
1. Open the `Text, Web, & Media Analytics Assignment 2.ipynb` notebook in a Jupyter environment (supported by most IDEs including Visual Studio Code).
2. Execute the cells sequentially from top to bottom using the 'Run' button or the Shift+Enter shortcut.

Ensure that all required data files and directories are correctly set up as described above to prevent execution errors.

## Output Files

Running the notebook generates a new directory `RankingOutputs` containing `.dat` files for each query and model, systematically named (e.g., `BM25_R101Ranking.dat`). These files catalog the document relevance scores and are stored in the `RankoutOutputs` directory of the current working directory.

## Development Environment

The scripts and notebook were developed using **Visual Studio Code**.