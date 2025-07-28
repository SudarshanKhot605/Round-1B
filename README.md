# PDF Section Extractor and Summarizer

A robust solution for **extracting**, **ranking**, and **summarizing** relevant sections from collections of PDF documents. This tool helps users quickly identify key information based on a **persona** and **task**, leveraging advanced **Natural Language Processing (NLP)** techniques.

---

## Features

* **Intelligent Heading Extraction**
  Uses [`PyMuPDF`](https://pymupdf.readthedocs.io/en/latest/) for text extraction, along with custom `PDFLineExtractor` and `classify_headings` modules to identify hierarchical headings (H1–H4).

* **Semantic Relevance Ranking**
  Applies **Sentence-BERT** embeddings (`all-mpnet-base-v2`) to semantically score headings against a given `task` and `persona`.

* **Abstractive Summarization**
  Generates concise summaries using a fine-tuned **BART** model (`distilbart-cnn-6-6`).

* **Configurable Parameters**
  Tune number of sections extracted, heading depth, scoring thresholds, and more.

* **Comprehensive Logging**
  Detailed logs are recorded to aid with debugging and performance evaluation.

---

## How It Works

### Workflow Breakdown

1. **Input Setup**
   Each collection directory contains a `challenge1b_input.json` file with:

   * `persona` (e.g., `"financial analyst"`)
   * `job_to_be_done` (e.g., `"analyze quarterly earnings"`)
   * List of PDFs to process

2. **Heading Identification**
   Extracts lines of text with formatting information (font size, bold, etc.) to detect structured headings.

3. **Relevance Scoring**
   Embeds each heading using Sentence-BERT and scores them based on similarity to the combined `task + persona`.

4. **Top Section Selection**
   Chooses the top **N** relevant headings (default: 5) across all documents.

5. **Content Summarization**
   Extracts text from the corresponding pages and summarizes using `distilbart-cnn-6-6`.

6. **Output Generation**
   Saves results into `challenge1b_output.json` for each collection.

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SudarshanKhot605/Round-1B.git
cd Round-1B
```

### 2. Prerequisites

* Docker: [Get Started with Docker](https://www.docker.com/get-started)

### 3. Project Structure

```
.
├── Dockerfile
├── requirements.txt
├── text_extraction.py
├── structure_analysis.py
├── main.py                  # Main entry point of the application
├── Challenge_1b/
│   ├── Collection_1/
│   │   ├── PDFs/
│   │   │   └── document1.pdf
│   │   └── challenge1b_input.json
│   ├── Collection_2/
│   │   ├── PDFs/
│   │   │   └── document2.pdf
│   │   └── challenge1b_input.json
└── ...
```

> **Note**: The main script must be named `main.py` as expected by the `Dockerfile`.

---

## Docker Usage

### Build the Docker Image

```bash
docker build -t pdf-processor .
```

This builds the image and installs dependencies, including NLP models.

### Run the Container

```bash
docker run --rm -v "$(pwd)/Challenge_1b:/app/Challenge_1b" pdf-processor
```

#### Explanation of Flags:

* `--rm`: Removes the container after execution
* `-v "$(pwd)/Challenge_1b:/app/Challenge_1b"`: Mounts local directory for input/output file sharing

After execution, output files like `challenge1b_output.json` will appear inside their respective `Collection_X` folders.

---

## Output Example

Each `challenge1b_output.json` file includes:

```json
{
  "collection_id": "Collection_1",
  "top_sections": [
    {
      "document": "document1.pdf",
      "heading": "Quarterly Earnings Overview",
      "page": 3,
      "score": 0.82,
      "summary": "The company reported a 12% increase in net income..."
    }
  ]
}
```

---

## Technologies Used

* PyMuPDF – PDF parsing
* Sentence-BERT – Semantic similarity
* Transformers (HuggingFace) – Text summarization
* Docker – Environment encapsulation

---

## Contact

For questions, bug reports, or feature requests, please use the [GitHub Issues](https://github.com/SudarshanKhot605/Round-1B/issues) page.

---

Let me know if you want a downloadable `README.md` file or additional documentation (e.g., API reference, developer guide).
