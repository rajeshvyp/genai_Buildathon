
# ============================================================
# Program 2
# BUILD A PERSISTENT FAISS VECTOR DATABASE
# Project: AI-Powered RAG Chatbot for Tamil Nadu Agriculture
# ============================================================
#
# What this program does:
# 1. Reads extracted text files from data/processed.
# 2. Extracts document content and source metadata.
# 3. Splits long documents into smaller chunks.
# 4. Sends chunks to OpenAI to create embeddings.
# 5. Stores the embeddings in a local FAISS index.
# 6. Saves the index so the chatbot can reuse it later.
#
# IMPORTANT:
# Run this program for the initial index build.
# Do not rebuild the index every time you ask a question.
# ============================================================


# -------------------- IMPORTS -------------------------------

import argparse
# argparse lets us accept command-line options.
# We use --rebuild when we intentionally want to replace
# an existing vector database.

import os
# os lets us read environment variables, including OPENAI_API_KEY.

import re
# re = regular expressions.
# We use it to find source information inside text headers.

from pathlib import Path
# Path helps us work with folders and filenames safely
# on Windows without manually joining path strings.

from dotenv import load_dotenv
# Loads environment variables from our project's .env file.
# This allows Python to access OPENAI_API_KEY.

from langchain_core.documents import Document
# Document is LangChain's standard container for:
# - page_content: the actual text
# - metadata: source filename, URL, title, and other details

from langchain_text_splitters import RecursiveCharacterTextSplitter
# Splits long documents into smaller, overlapping chunks.
# Smaller chunks are easier to search for relevant information.

from langchain_openai import OpenAIEmbeddings
# Connects to OpenAI's embedding API.
# It converts each text chunk into a list of numbers (a vector).

from langchain_community.vectorstores import FAISS
# FAISS is the vector database/search library.
# LangChain provides this wrapper to save, load, and search
# vectors together with their document chunks.


# -------------------- PROJECT PATHS -------------------------

# Find the project root automatically.
# This file is in scrapers/, so its parent folder is the project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder containing the text files created by process_documents.py.
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"

# Folder where the persistent vector database will be saved.
VECTORSTORE_FOLDER = PROJECT_ROOT / "data" / "vectorstore"

# Location of the existing .env file.
ENV_FILE = PROJECT_ROOT / ".env"

# FAISS saves its numerical index in this file.
INDEX_FILE = VECTORSTORE_FOLDER / "index.faiss"

# LangChain saves the document store and mapping in this file.
METADATA_FILE = VECTORSTORE_FOLDER / "index.pkl"


# -------------------- EMBEDDING SETTINGS --------------------

# Model used to convert text into numerical vectors.
# This model supports multilingual text, including many languages.
# We are currently indexing the documents with usable extracted text.
EMBEDDING_MODEL = "text-embedding-3-small"

# Skip files whose extracted content is too short to be useful.
MIN_DOCUMENT_CHARACTERS = 100

# Maximum target size of each chunk, measured in characters.
CHUNK_SIZE = 1000

# Number of characters that can overlap between neighboring chunks.
# Overlap helps preserve context at chunk boundaries.
CHUNK_OVERLAP = 150


# -------------------- DOCUMENT LOADING ----------------------

def load_source_document(path):
    """
    Read one extracted text file and return a LangChain Document.

    The Document contains:
    - page_content: the text we want to search
    - metadata: information identifying its source
    """

    # Read the file as UTF-8.
    # utf-8-sig also handles files that begin with a UTF-8 BOM.
    # errors="replace" prevents unusual characters from crashing
    # the entire program.
    raw_text = path.read_text(
        encoding="utf-8-sig",
        errors="replace"
    ).strip()

    # Empty files cannot provide useful search results.
    if not raw_text:
        return None

    # Our processing script adds this marker before the actual text.
    # Split the file into its header and extracted content.
    marker = "--- Extracted text ---"

    if marker in raw_text:
        header, content = raw_text.split(marker, 1)
    else:
        # Some text files may not have this marker.
        # In that case, treat the whole file as content.
        header = ""
        content = raw_text

    content = content.strip()

    # Ignore files with almost no usable text.
    if len(content) < MIN_DOCUMENT_CHARACTERS:
        print(f"SKIP short/empty text: {path.name}")
        return None

    # Normalize whitespace and lowercase the text for checking.
    normalized = re.sub(r"\s+", " ", content).strip().lower()

    # Examples of text that is usually not useful for answering
    # agriculture-scheme questions.
    low_value_phrases = [
        "your browser does not support the audio tag",
        "accessibility options",
        "page not found",
        "403 forbidden",
        "404 not found",
    ]

    # If every listed phrase is present, the page is likely
    # not useful. This is a basic filter, not a perfect classifier.
    if all(phrase in normalized for phrase in low_value_phrases):
        print(f"SKIP low-value page: {path.name}")
        return None

    # Extract the original source filename from the header.
    source_file_match = re.search(
        r"^Source file:\s*(.+)$",
        header,
        flags=re.MULTILINE
    )

    # Extract the source type, such as PDF, MHT, or TXT.
    source_type_match = re.search(
        r"^Source type:\s*(.+)$",
        header,
        flags=re.MULTILINE
    )

    # Some collected articles contain a source URL in their text.
    source_url_match = re.search(
        r"^Source URL:\s*(https?://\S+)",
        content,
        flags=re.MULTILINE
    )

    # Look for an original page title when one is available.
    source_title_match = re.search(
        r"^Source title:\s*(.+)$",
        content,
        flags=re.MULTILINE
    )

    # If metadata is missing, use sensible fallback values.
    source_file = (
        source_file_match.group(1).strip()
        if source_file_match
        else path.name
    )

    source_type = (
        source_type_match.group(1).strip()
        if source_type_match
        else path.suffix.lstrip(".").upper()
    )

    source_url = (
        source_url_match.group(1).strip()
        if source_url_match
        else ""
    )

    source_title = (
        source_title_match.group(1).strip()
        if source_title_match
        else path.stem
    )

    # Return the content and metadata together.
    # Metadata will help the chatbot identify and cite its sources.
    return Document(
        page_content=content,
        metadata={
            "source_file": source_file,
            "source_type": source_type,
            "source_title": source_title,
            "source_url": source_url,
            "processed_file": str(
                path.relative_to(PROJECT_ROOT)
            ),
        },
    )


# -------------------- MAIN PROGRAM --------------------------

def main():

    # Set up command-line arguments.
    parser = argparse.ArgumentParser(
        description=(
            "Build a persistent FAISS index from processed documents."
        )
    )

    # By default, an existing index is protected from accidental
    # replacement. The --rebuild option explicitly requests a rebuild.
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Replace an existing index and regenerate embeddings."
        ),
    )

    args = parser.parse_args()

    # Load OPENAI_API_KEY from the project's .env file.
    load_dotenv(ENV_FILE)

    # Never print the API key itself.
    # Only check whether it exists.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            f"OPENAI_API_KEY was not found. Check: {ENV_FILE}"
        )

    # Make sure the extracted text folder exists.
    if not PROCESSED_FOLDER.exists():
        raise FileNotFoundError(
            f"Processed folder not found: {PROCESSED_FOLDER}"
        )

    # Check whether both expected saved index files already exist.
    index_exists = (
        INDEX_FILE.exists()
        and METADATA_FILE.exists()
    )

    # If an index already exists and the user did not request
    # a rebuild, stop here. This avoids unnecessary API calls.
    if index_exists and not args.rebuild:
        print("A persistent FAISS index already exists.")
        print(f"Index folder: {VECTORSTORE_FOLDER}")
        print("Nothing was rebuilt; no embeddings were regenerated.")
        print(
            "Use --rebuild only when you intentionally want to rebuild."
        )
        return

    # Only reach this point with an existing index if --rebuild
    # was explicitly supplied.
    if index_exists and args.rebuild:
        print("Rebuild requested: the existing index will be replaced.")

    # Find all processed .txt files, including files in subfolders.
    text_files = sorted(
        PROCESSED_FOLDER.rglob("*.txt")
    )

    if not text_files:
        raise RuntimeError(
            f"No extracted .txt files found in {PROCESSED_FOLDER}"
        )

    print("=" * 60)
    print("BUILDING PERSISTENT FAISS VECTOR STORE")
    print("=" * 60)
    print(f"Processed text files found: {len(text_files)}")

    # Load usable documents into memory.
    documents = []

    for path in text_files:
        try:
            document = load_source_document(path)

            # A None result means the file was empty, too short,
            # or filtered out.
            if document is not None:
                documents.append(document)
                print(
                    f"READY: {path.name} "
                    f"({len(document.page_content):,} chars)"
                )

        except Exception as error:
            # One unreadable file should not stop all the others.
            print(f"SKIP unreadable file {path.name}: {error}")

    if not documents:
        raise RuntimeError(
            "No usable documents were found to index."
        )

    # -------------------- CHUNKING --------------------------

    # Split long documents into overlapping chunks.
    # The splitter prefers paragraph and line boundaries where possible.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Each chunk inherits its original document metadata.
    # start_index records where the chunk starts in its source text.
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise RuntimeError("No chunks were produced.")

    print(f"\nDocuments selected: {len(documents)}")
    print(f"Chunks to embed: {len(chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("Sending document chunks to OpenAI for embeddings...")

    # -------------------- EMBEDDINGS ------------------------

    # Create the embedding client.
    # It uses OPENAI_API_KEY from the environment.
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    # Create a vector for each chunk and build the FAISS index.
    # This is the main step that sends text to OpenAI and may
    # incur API usage charges.
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # -------------------- PERSISTENT STORAGE ----------------

    # Ensure the output directory exists.
    VECTORSTORE_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the FAISS index and its associated document data.
    # These files remain on your computer after the program exits.
    vectorstore.save_local(
        str(VECTORSTORE_FOLDER)
    )

    # -------------------- FINAL REPORT ----------------------

    print("\n" + "=" * 60)
    print("VECTOR STORE SAVED")
    print("=" * 60)
    print(f"Source documents indexed: {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Saved folder: {VECTORSTORE_FOLDER}")
    print(f"FAISS index exists: {INDEX_FILE.exists()}")
    print(f"Metadata file exists: {METADATA_FILE.exists()}")
    print("\nThe saved index can now be reused without re-embedding.")
    print("Keep the index files together and do not delete them.")


# This condition means main() runs only when this file is
# executed directly, not when another Python file imports it.
if __name__ == "__main__":
    main()
