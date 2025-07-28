import fitz  # PyMuPDF
import json
from typing import List, Dict, Any
from datetime import datetime


class PDFLineExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.pdf_lines = []

    def extract_text_lines(self) -> List[Dict[str, Any]]:
        print("Extracting text lines with formatting from PDF...")
        all_lines = []

        for page_num, page in enumerate(self.doc, start=1):
            print(f"Processing page {page_num}/{len(self.doc)}")

            try:
                blocks = page.get_text("dict")["blocks"]
                page_lines = []

                for block in blocks:
                    if "lines" not in block:
                        continue

                    for line in block["lines"]:
                        spans = line.get("spans", [])
                        if not spans:
                            continue

                        full_text = "".join([span.get("text", "") for span in spans]).strip()
                        if not full_text:
                            continue

                        first_span = spans[0]
                        bbox = self.get_line_bbox(spans)

                        # Guard: Only calculate if bbox is valid
                        space_above = 0
                        space_below = 0
                        x0, y0, x1, y1 = 0, 0, 0, 0
                        
                        if bbox:
                            x0, y0, x1, y1 = bbox
                            prev_line = page_lines[-1] if page_lines else None
                            if prev_line and "bbox" in prev_line:
                                space_above = round(y0 - prev_line["bbox"][3], 2)
                                space_below = round(prev_line["bbox"][1] - y1, 2)

                        line_data = {
                            "text": full_text,
                            "font_size": round(first_span.get("size", 0.0), 2),
                            "font": first_span.get("font", "Unknown"),
                            "is_bold": "Bold" in first_span.get("font", ""),
                            "is_italic": "Italic" in first_span.get("font", ""),
                            "is_underlined": first_span.get("flags", 0) & 4 != 0,
                            "is_center": self.is_centered(line, page.rect),
                            "bbox": bbox,
                            "x0": round(x0, 2),
                            "y0": round(y0, 2),
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "space_above": space_above,
                            "space_below": space_below,
                            "page": page_num
                        }

                        page_lines.append(line_data)

                all_lines.extend(page_lines)

            except Exception as e:
                print(f"Error extracting text from page: {e}")

        self.pdf_lines = all_lines
        print(f"Extracted {len(all_lines)} text lines")
        return all_lines

    def get_line_bbox(self, spans: List[Dict[str, Any]]):
        try:
            x0 = min([span["bbox"][0] for span in spans if "bbox" in span])
            y0 = min([span["bbox"][1] for span in spans if "bbox" in span])
            x1 = max([span["bbox"][2] for span in spans if "bbox" in span])
            y1 = max([span["bbox"][3] for span in spans if "bbox" in span])
            return (x0, y0, x1, y1)
        except Exception:
            return None

    def is_centered(self, line: Dict[str, Any], rect: fitz.Rect):
        try:
            spans = line.get("spans", [])
            if not spans:
                return False
            bbox = spans[0].get("bbox", None)
            if not bbox:
                return False
            x0, x1 = bbox[0], bbox[2]
            center_of_line = (x0 + x1) / 2
            page_center = (rect.x0 + rect.x1) / 2
            return abs(center_of_line - page_center) < 50  # pixels
        except:
            return False

    def get_pdf_lines(self, include_metadata=True) -> List[Dict[str, Any]]:
        lines = []
        for l in self.pdf_lines:
            line = {"text": l["text"], "page": l["page"]}
            if include_metadata:
                line.update({
                    "font_size": l.get("font_size", 0),
                    "font": l.get("font", "Unknown"),
                    "is_bold": l.get("is_bold", False),
                    "is_italic": l.get("is_italic", False),
                    "is_underlined": l.get("is_underlined", False),
                    "is_center": l.get("is_center", False),
                    "x0": l.get("x0", 0),
                    "y0": l.get("y0", 0),
                    "x1": l.get("x1", 0),
                    "y1": l.get("y1", 0),
                    "space_above": l.get("space_above", 0),
                    "space_below": l.get("space_below", 0),
                })
            lines.append(line)
        return lines

    def save_lines_to_file(self, output_path: str, include_metadata: bool = True):
        lines = self.get_pdf_lines(include_metadata)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(lines, f, indent=4)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    pdf_path = r"pdfs\Dinner Ideas - Mains_1.pdf" # Use path.join for cross-platform compatibility
    extractor = PDFLineExtractor(pdf_path)
    extractor.extract_text_lines()
    extractor.save_lines_to_file("Dinner Ideas - Mains_1-new.json")
