# PDF Section Extractor and Summarizer

This project provides a robust solution for extracting, ranking, and summarizing relevant sections from collections of PDF documents. It's designed to help users quickly identify key information based on a defined persona and specific task, leveraging advanced natural language processing techniques.

-----

## Features

  \* **Intelligent Heading Extraction**: Utilizes `PyMuPDF` for text extraction, coupled with custom `PDFLineExtractor` and `classify_headings` modules, to accurately identify and structure hierarchical headings (H1-H4) within PDFs.
  \* **Semantic Relevance Ranking**: Employs **Sentence-BERT embeddings** (`all-mpnet-base-v2`) to semantically score extracted headings against a given `task` and `persona`, ensuring the most pertinent sections are prioritized.
  \* **Abstractive Summarization**: Generates concise, high-quality summaries of the content on pages corresponding to the top-ranked sections. This is powered by a fine-tuned **BART model** (`distilbart-cnn-6-6`), capable of condensing large amounts of text.
  \* **Configurable Parameters**: Easily adjust settings like the number of top sections to extract and various heading validation rules to tailor the extraction process to your specific needs.
  \* **Comprehensive Logging**: Detailed logs are generated for each run, providing insights into the extraction, ranking, and summarization process, aiding in debugging and performance monitoring.

-----

## How It Works

The script operates by processing PDF document collections, each typically housed in its own directory. Here's a breakdown of the workflow:

1.  **Input Definition**: Each collection directory must contain a `challenge1b_input.json` file. This file specifies the `persona` (e.g., "financial analyst") and the `job_to_be_done` (`task`, e.g., "analyze quarterly earnings"). It also lists the PDF documents to be processed within that collection.
2.  **Heading Identification**: For each PDF, the system extracts text lines with their formatting (font size, bolding, etc.). 
3.  **Relevance Scoring**: The extracted headings are semantically encoded using Sentence-BERT. A similarity score is then calculated between each heading's embedding and the embedding of the `task` query, effectively ranking how relevant each heading is to the user's goal.
4.  **Top Section Selection**: After scoring all headings across all documents in a collection, the script selects the `NUM_TOP_SECTIONS` (default: 5) with the highest relevance scores.
5.  **Content Summarization**: For each of the selected top sections, the corresponding page's full text is extracted. This text is then fed into the `distilbart-cnn-6-6` model, which generates a concise abstractive summary, capturing the essence of the section's content.
6.  **Output Generation**: Finally, a `challenge1b_output.json` file is created in the collection's directory. This file contains metadata about the processed documents, a ranked list of the most important sections, and the generated summaries for each of these sections.

-----

## Setup and Installation

### Clone and Navigate to the Repository

To get started, clone the repository to your local machine and navigate into the project directory:

```bash
git clone https://github.com/SudarshanKhot605/Round-1B.git
cd Round-1B
```

### Prerequisites

  \* **Docker**: Ensure Docker is installed on your system. You can download it from [docker.com](https://www.google.com/search?q=https://www.docker.com/get-started).

### Project Structure

Organize your project directory like this:

```
.
├── Dockerfile
├── requirements.txt
├── text_extraction.py
├── structure_analysis.py
├── main.py  # This is the primary script (your pdf_section_extractor.py, renamed)
├── Challenge_1b/
│   ├── Collection_1/
│   │   ├── PDFs/
│   │   │   └── document1.pdf
│   │   ├── challenge1b_input.json
│   ├── Collection_2/
│   │   ├── PDFs/
│   │   │   └── document2.pdf
│   │   ├── challenge1b_input.json
└── ...
```

**Important**: Make sure your main script is named `main.py` as indicated in the `CMD` instruction of the `Dockerfile`. 

### Building the Docker Image

Navigate to your project's root directory in the terminal and build the Docker image:

```bash
docker build -t pdf-processor .
```

This command tags the image as `pdf-processor`. This process might take a few minutes as it downloads base images and installs all required dependencies, including the large language models.

### Running the Docker Container

Once the image is built, you can run a container. To ensure the output JSON files are saved to your host machine, it's highly recommended to **mount a volume**:

```bash
docker run --rm -v "$(pwd)/Challenge_1b:/app/Challenge_1b" pdf-processor
```

  \* `--rm`: This flag automatically removes the container after it finishes execution, keeping your system clean.
  \* `-v "$(pwd)/Challenge_1b:/app/Challenge_1b"`: This crucial part **mounts your local `Challenge_1b` directory** into the container's `/app/Challenge_1b` path. This means any `challenge1b_output.json` files generated by the script inside the container will directly appear in the corresponding `Collection_X` folders on your host machine.

The script's progress and output will be displayed in your terminal. Upon completion, the container will exit, and you'll find the generated JSON files in your local `Challenge_1b` directory.
