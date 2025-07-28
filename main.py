import os
import json
import fitz  # PyMuPDF
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime
from tqdm import tqdm
from math import comb
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import torch
from typing import List, Tuple
from text_extraction import PDFLineExtractor  # Import your extractor
from structure_analysis import classify_headings  # Import your heading classifier

# --- Configuration Constants ---
ROOT_DIR = "Challenge_1b"
NUM_TOP_SECTIONS = 5  # Final number of top sections per collection
MODEL_PATH = "models/all-mpnet-base-v2"
MODEL_DIR = "models/distilbart-cnn-6-6"
LOG_DIR = "logs"
# Heading extraction rules
CONFIG = {
    "HEADING_MIN_LEN": 5,
    "HEADING_MAX_LEN": 100,
    "HEADING_MIN_WORDS": 2,
    "HEADING_MAX_WORDS": 10,
    "HEADING_INLINE_CONTENT_WORD_THRESHOLD": 3
}

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)
# Create a unique log file per run, timestamped
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"process_{timestamp}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Load the local sentence-transformers model
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
tokenizer = BartTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
llm = BartForConditionalGeneration.from_pretrained(MODEL_DIR, local_files_only=True)


# Title-case helper
STOPWORDS = {
    'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at',
    'to', 'from', 'by', 'in', 'of', 'with'
}
FORBIDDEN_END_CHARS = '.,:;[(-=+/'

def _is_title_case(text: str, stopwords: set) -> bool:
    words = text.split()
    if not words or not words[0][0].isupper():
        return False
    for w in words[1:]:
        if w.lower() in stopwords:
            continue
        if not (w[0].isupper() and w[1:].islower()):
            return False
    return True


# def extract_headings_from_pdf(pdf_path: str) -> List[Tuple[str, int]]:
#     headings = []
#     try:
#         doc = fitz.open(pdf_path)
#     except Exception as e:
#         logging.error(f"Failed to open PDF {pdf_path}: {e}")
#         return []

#     for page_num, page in enumerate(doc, start=1):
#         for block in page.get_text("dict").get("blocks", []):
#             if "lines" not in block:
#                 continue
#             for line in block["lines"]:
#                 spans = line.get("spans", [])
#                 if not spans:
#                     continue
#                 text = " ".join(span["text"].strip() for span in spans).strip()
#                 if not text:
#                     continue
#                 word_count = len(text.split())
#                 if not (
#                     CONFIG["HEADING_MIN_LEN"] < len(text) < CONFIG["HEADING_MAX_LEN"] and
#                     CONFIG["HEADING_MIN_WORDS"] <= word_count <= CONFIG["HEADING_MAX_WORDS"] and
#                     text[0].isupper()
#                 ):
#                     continue
#                 if text.endswith(tuple(FORBIDDEN_END_CHARS)):
#                     continue
#                 if not _is_title_case(text, STOPWORDS):
#                     continue
#                 if ":" in text and len(text.split(':', 1)[1].split()) > CONFIG["HEADING_INLINE_CONTENT_WORD_THRESHOLD"]:
#                     continue
#                 avg_size = sum(span["size"] for span in spans) / len(spans)
#                 is_bold = any("bold" in span["font"].lower() for span in spans)
#                 if avg_size > 10 and (is_bold or avg_size > 13):
#                     headings.append((text, page_num))
#     logging.info(f"Extracted {len(headings)} headings from {os.path.basename(pdf_path)}")
#     return headings

def extract_headings_from_pdf(pdf_path: str) -> List[Tuple[str, int]]:
    """
    Extracts hierarchical outline (H1–H4) from a PDF and returns it in the same format
    as extract_headings_from_pdf().
    Uses advanced text extraction and classification methods.
    
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        List[Tuple[str, int]]: List of tuples where each tuple contains (heading_text, page_number)
    """
    headings = []
    try:
        # Step 1: Extract text lines with formatting using PDFLineExtractor
        extractor = PDFLineExtractor(pdf_path)
        extractor.extract_text_lines()
        
        # Step 2: Get formatted JSON data (in memory, not saved)
        json_data = extractor.get_pdf_lines(include_metadata=True)
        
        # Step 3: Process with heading classifier
        result = classify_headings(json_data)
        
        # Step 4: Convert outline format to List[Tuple[str, int]]
        outline = result.get("outline", [])
        for heading in outline:
            text = heading.get("text", "").strip()
            page = heading.get("page", 1)  # Default to page 1 if not specified
            if text:  # Only add non-empty headings
                headings.append((text, page))
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []
    
    return headings

# simple
def rank_sections_with_embeddings(sections: List[Tuple[str, int]], persona: str, task: str) -> List[Tuple[str, int, float]]:
    if not sections:
        logging.warning(f"No headings to rank for task: {task}")
        return []
    query = f"{task}"
    q_emb = model.encode(query, convert_to_tensor=True)
    texts, pages = zip(*sections)
    emb = model.encode(list(texts), convert_to_tensor=True)
    scores = torch.sum(q_emb * emb, dim=1)
    scored = []
    for idx, (title, page) in enumerate(sections):
        score = float(scores[idx])
        scored.append((title, page, score))
        logging.info(f"Relevance - '{title}' (Page {page}): {score:.4f}")
    return scored


# def is_heading(text: str, threshold: float = 0.5) -> bool:
#     """
#     Returns True if the classifier assigns higher score to "Heading" than "Sentence".
#     You can raise threshold if you want to be more conservative.
#     """
#     out = classifier(text, candidate_labels=["Heading", "Sentence"])
#     return out["scores"][0] >= threshold and out["labels"][0] == "Heading"


def compute_relevance_score_pascal(heading: str, task: str) -> float:
    """
    Compute relevance of heading to task by:
      1. Splitting task into words (n words).
      2. For k from 1 to n, form all consecutive k-grams of task.
      3. Compute cosine similarity of each k-gram with heading embedding.
      4. Average similarities for each k (avg_k).
      5. Weight each avg_k by Pascal's binomial coefficient C(n, k).
      6. Return weighted average: sum_k [C(n,k) * avg_k] / sum_k C(n,k).
    """
    words = task.split()
    n = len(words)
    if n == 0:
        return 0.0
    # Precompute embeddings
    heading_emb = model.encode(heading, convert_to_tensor=True)
    total_weight = 0.0
    weighted_sum = 0.0
    for k in range(1, n+1):
        # sliding k-grams
        grams = [" ".join(words[i:i+k]) for i in range(n-k+1)]
        gram_embs = model.encode(grams, convert_to_tensor=True)
        sims = util.cos_sim(heading_emb, gram_embs)[0]
        avg_k = float(sims.mean()) if sims.numel() > 0 else 0.0
        weight = comb(n, k)
        weighted_sum += weight * avg_k
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def extract_page_text(pdf_path: str, page_number: int, task: str = None, persona: str = None) -> str:
    """
    Extracts text from the specified page of a PDF and generates an abstractive summary
    using the distilBART model (sshleifer/distilbart-cnn-6-6).

    Args:
        pdf_path (str): Path to the PDF file.
        page_number (int): 1-based page number to extract.
        task (str, optional): Ignored by this model (for compatibility).
        persona (str, optional): Ignored by this model (for compatibility).

    Returns:
        str: The generated summary text.
    """
    # Extract raw text from the PDF page
    with fitz.open(pdf_path) as doc:
        raw_text = doc[page_number - 1].get_text().strip()

    # Tokenize and summarize
    inputs = tokenizer(
        raw_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024  # distilBART supports up to 1024 tokens
    )

    # Generate summary
    summary_ids = llm.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=300,
        min_length=40,
        early_stopping=True
    )

    # Decode to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def process_collection(collection_path: str):
    logging.info(f"Processing collection: {collection_path}")
    input_path = os.path.join(collection_path, "challenge1b_input.json")
    data = json.load(open(input_path))
    persona = data["persona"]["role"]
    task = data["job_to_be_done"]["task"]

    all_scored = []
    for doc in tqdm(data["documents"], desc="Scoring headings"):
        pdf_path = os.path.join(collection_path, "PDFs", doc["filename"])
        sections = extract_headings_from_pdf(pdf_path)
        scored = rank_sections_with_embeddings(sections, persona, task)
        for title, page, score in scored:
            all_scored.append({
                "document": doc["filename"],
                "title": title,
                "page": page,
                "score": score
            })
    all_scored.sort(key=lambda x: -x["score"])
    top_sections = all_scored[:NUM_TOP_SECTIONS]

    for sec in top_sections:
        sec["refined_text"] = extract_page_text(
            os.path.join(collection_path, "PDFs", sec["document"]),
            sec["page"],
            task,
            persona,
        )

    output = {
        "metadata": {
            "input_documents": [d["filename"] for d in data["documents"]],
            "persona": persona,
            "job_to_be_done": task
        },
        "extracted_sections": [
            {"document": sec["document"], "section_title": sec["title"], "importance_rank": idx+1, "page_number": sec["page"]}
            for idx, sec in enumerate(top_sections)
        ],
        "subsection_analysis": [
            {"document": sec["document"], "refined_text": sec["refined_text"], "page_number": sec["page"]}
            for sec in top_sections
        ]
    }

    out_path = os.path.join(collection_path, "challenge1b_output.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Wrote final JSON for {collection_path}")


def main():
    for coll in os.listdir(ROOT_DIR):
        path = os.path.join(ROOT_DIR, coll)
        if os.path.isdir(path) and coll.startswith("Collection"):
            process_collection(path)

if __name__ == "__main__":
    main()
