import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import traceback
import re
import enchant
import string

# Create English dictionary
d = enchant.Dict("en_US")

def classify_string(text: str) -> bool:
    """
    Returns True if text is:
     - something starting with a digit (e.g. '1. Preamble')
     - an all‑caps acronym of length>=2 (e.g. 'RFP')
     - a valid English word (e.g. 'Proposal', 'plan')
     - contains at least one valid English word (NEW: more lenient)
     
    NEW RULES:
     - Must be at least 3 characters long
     - Must NOT be only numbers and special characters (no letters)
     - If not starting with a number, first word must start with capital letter
     
    Otherwise returns False.
    """
    text = text.strip()
    if not text:
        return False

    # NEW RULE 1: Minimum length check - must be at least 3 characters
    if len(text) < 3:
        logger.debug(f"❌ Rejected (too short): '{text}' (length: {len(text)})")
        return False

    # Check for consecutive special characters that shouldn't be in headings
    consecutive_patterns = ['--', '..', '==', '**', '^^', '<<', '>>', '//', '\\\\', '~~']
    if any(pattern in text for pattern in consecutive_patterns):
        return False
    
    # NEW RULE 2: Reject if text contains only numbers and special characters (no letters)
    if not any(c.isalpha() for c in text):
        logger.debug(f"❌ Rejected (no letters): '{text}'")
        return False
    
    # NEW RULE 3: Check capitalization rule for non-numeric starts
    if not text[0].isdigit():
        # Extract first word
        words = text.split()
        if words:
            first_word = words[0].strip(string.punctuation)  # Remove leading/trailing punctuation
            if first_word and first_word[0].islower():
                logger.debug(f"❌ Rejected (first word not capitalized): '{text}' (first word: '{first_word}')")
                return False
    
    # 1) Numeric start - always allowed if it passes the new rules above
    if text[0].isdigit():
        return True

    # 2) Trim off surrounding punctuation for the next checks
    cleaned = text.strip(string.punctuation)

    # 3) Acronym: all uppercase letters, length>=2 (but we already checked min length above)
    if cleaned.isupper() and len(cleaned) >= 2:
        return True

    # 4) Check if entire text is a valid English word
    if d.check(cleaned):
        return True

    # 5) Check if text contains at least one valid English word
    words = cleaned.split()
    for word in words:
        word_clean = word.strip(string.punctuation)
        if len(word_clean) >= 2 and d.check(word_clean):
            return True
    
    # 6) Allow common title/heading patterns even if not in dictionary
    # This helps with proper nouns, technical terms, etc.
    if len(cleaned) >= 3:
        # Check if it has reasonable letter patterns (not just numbers/symbols)
        alpha_count = sum(1 for c in cleaned if c.isalpha())
        if alpha_count >= len(cleaned) * 0.5:  # At least 50% letters
            return True

    return False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeaderFooterDetector:
    def __init__(self, page_height=792, page_width=612):  # Standard letter size
        self.page_height = page_height
        self.page_width = page_width
        self.header_threshold = 0.12  # Top 12%
        self.footer_threshold = 0.88  # Bottom 12%
        self.min_repetition = 2  # Minimum pages to consider repetition
        
        
    def detect_headers_footers(self, all_pages_data):
        """
        Main detection method combining multiple strategies
        """
        results = {}
        
        for page_num, elements in all_pages_data.items():
            # Strategy 1: Position-based detection
            pos_headers, pos_footers, pos_content = self._detect_by_position(elements)
            
            # Strategy 2: Style-based detection
            style_headers, style_footers, style_content = self._detect_by_style(elements)
            
            # Combine results (intersection for higher confidence)
            final_headers = self._combine_detections(pos_headers, style_headers)
            final_footers = self._combine_detections(pos_footers, style_footers)
            
            # Remove headers/footers from main content
            header_indices = {elem.get('original_index') for elem in final_headers}
            footer_indices = {elem.get('original_index') for elem in final_footers}
            
            main_content = [
                elem for elem in elements 
                if elem.get('original_index') not in header_indices 
                and elem.get('original_index') not in footer_indices
            ]
            
            results[page_num] = {
                'headers': final_headers,
                'footers': final_footers,
                'content': main_content
            }
        
        # Strategy 3: Cross-page repetition analysis
        self._refine_by_repetition(results)
        
        return results
    
    def _detect_by_position(self, elements):
        """Position-based detection"""
        headers = []
        footers = []
        content = []
        
        header_y_limit = self.page_height * self.header_threshold
        footer_y_limit = self.page_height * self.footer_threshold
        
        for element in elements:
            y_pos = element.get('y', self.page_height / 2)
            
            if y_pos <= header_y_limit:
                headers.append(element)
            elif y_pos >= footer_y_limit:
                footers.append(element)
            else:
                content.append(element)
        
        return headers, footers, content
    
    def _detect_by_style(self, elements):
        """Style-based detection"""
        if not elements:
            return [], [], []
        
        # Calculate main font size
        font_sizes = [elem.get('font_size', 12) for elem in elements]
        main_font_size = max(set(font_sizes), key=font_sizes.count)
        
        headers = []
        footers = []
        content = []
        
        for element in elements:
            font_size = element.get('font_size', 12)
            text = element.get('text', '').strip()
            y_pos = element.get('y', self.page_height / 2)
            
            # Style indicators
            is_small_font = font_size < main_font_size * 0.85
            is_page_number = self._is_page_number(text)
            is_short = len(text) < 60
            is_italic = element.get('is_italic', False)
            
            # Header/footer likelihood score
            hf_score = 0
            if is_small_font: hf_score += 2
            if is_page_number: hf_score += 3
            if is_short: hf_score += 1
            if is_italic: hf_score += 1
            
            # Position-based classification with style weighting
            if y_pos <= self.page_height * 0.15:  # Top area
                if hf_score >= 2:
                    headers.append(element)
                else:
                    content.append(element)
            elif y_pos >= self.page_height * 0.85:  # Bottom area
                if hf_score >= 2:
                    footers.append(element)
                else:
                    content.append(element)
            else:
                content.append(element)
        
        return headers, footers, content
    
    def _is_page_number(self, text):
        """Check if text looks like a page number"""
        # Simple page number patterns
        patterns = [
            r'^\d+$',  # Just a number
            r'^page\s+\d+$',  # "page 1"
            r'^p\.\s*\d+$',  # "p. 1"
            r'^\d+\s*/\s*\d+$',  # "1 / 10"
            r'^-\s*\d+\s*-$',  # "- 1 -"
        ]
        
        text_lower = text.lower().strip()
        return any(re.match(pattern, text_lower) for pattern in patterns)
    
    def _combine_detections(self, detection1, detection2):
        """Combine two detection results"""
        # Use intersection for higher confidence
        indices1 = {elem.get('original_index') for elem in detection1}
        indices2 = {elem.get('original_index') for elem in detection2}
        
        common_indices = indices1.intersection(indices2)
        
        # Return elements from first detection that are in common
        return [elem for elem in detection1 if elem.get('original_index') in common_indices]
    
    def _refine_by_repetition(self, results):
        """Refine detection using cross-page repetition analysis"""
        # Collect potential headers/footers across pages
        header_patterns = defaultdict(list)
        footer_patterns = defaultdict(list)
        
        for page_num, page_data in results.items():
            for header in page_data['headers']:
                pattern = self._create_pattern(header)
                header_patterns[pattern].append((page_num, header))
            
            for footer in page_data['footers']:
                pattern = self._create_pattern(footer)
                footer_patterns[pattern].append((page_num, footer))
        
        # Find truly repeated patterns
        repeated_headers = {
            pattern: pages for pattern, pages in header_patterns.items()
            if len(pages) >= self.min_repetition
        }
        
        repeated_footers = {
            pattern: pages for pattern, pages in footer_patterns.items()
            if len(pages) >= self.min_repetition
        }
        
        # Update results with refined detection
        for page_num, page_data in results.items():
            # Mark repeated elements as confirmed headers/footers
            for pattern, pages in repeated_headers.items():
                for p_num, element in pages:
                    if p_num == page_num:
                        element['is_repeated_header'] = True
            
            for pattern, pages in repeated_footers.items():
                for p_num, element in pages:
                    if p_num == page_num:
                        element['is_repeated_footer'] = True
    
    def _create_pattern(self, element):
        """Create a pattern for repetition detection"""
        text = element.get('text', '').strip()
        y_norm = round(element.get('y', 0) / self.page_height, 2)
        x_norm = round(element.get('x', 0) / self.page_width, 2)
        
        # For page numbers, use position only
        if self._is_page_number(text):
            return ('PAGE_NUMBER', y_norm, x_norm)
        
        # For other text, use text + position
        return (text, y_norm, x_norm)

@dataclass
class TextElement:
    """Represents a processed text element with validated attributes"""
    text: str = ""
    page: int = 1
    font_size: float = 12.0
    font: str = "Arial"
    is_bold: bool = False
    is_italic: bool = False
    is_underlined: bool = False
    is_center: bool = False
    space_above: float = 0.0
    space_below: float = 0.0
    original_index: int = 0
    x: float = 0.0  # Added x coordinate for header/footer detection
    y: float = 0.0  # Added y coordinate for header/footer detection
    
    def __post_init__(self):
        """Validate and clean the text element data"""
        try:
            # Clean and validate text
            self.text = str(self.text).strip() if self.text is not None else ""
            
            # Validate numeric fields
            self.page = max(1, int(self.page)) if self.page is not None else 1
            self.font_size = float(self.font_size) if self.font_size is not None else 12.0
            self.space_above = float(self.space_above) if self.space_above is not None else 0.0
            self.space_below = float(self.space_below) if self.space_below is not None else 0.0
            self.x = float(self.x) if self.x is not None else 0.0
            self.y = float(self.y) if self.y is not None else 0.0

            if hasattr(self, 'x0') and self.x == 0.0:
                self.x = float(getattr(self, 'x0', 0.0))
            if hasattr(self, 'y0') and self.y == 0.0:
                self.y = float(getattr(self, 'y0', 0.0))
            
            # Validate string fields
            self.font = str(self.font) if self.font is not None else "Arial"
            
            # Validate boolean fields
            self.is_bold = bool(self.is_bold) if self.is_bold is not None else False
            self.is_italic = bool(self.is_italic) if self.is_italic is not None else False
            self.is_underlined = bool(self.is_underlined) if self.is_underlined is not None else False
            self.is_center = bool(self.is_center) if self.is_center is not None else False
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Error validating text element: {e}. Using default values.")
            self._set_defaults()
    
    def _set_defaults(self):
        """Set default values for invalid data"""
        self.text = ""
        self.page = 1
        self.font_size = 12.0
        self.font = "Arial"
        self.is_bold = False
        self.is_italic = False
        self.is_underlined = False
        self.is_center = False
        self.space_above = 0.0
        self.space_below = 0.0
        self.x = 0.0
        self.y = 0.0
    
    def to_dict(self):
        """Convert TextElement to dictionary for header/footer detection"""
        return {
            'text': self.text,
            'page': self.page,
            'font_size': self.font_size,
            'font': self.font,
            'is_bold': self.is_bold,
            'is_italic': self.is_italic,
            'is_underlined': self.is_underlined,
            'is_center': self.is_center,
            'space_above': self.space_above,
            'space_below': self.space_below,
            'original_index': self.original_index,
            'x': self.x,
            'y': self.y
        }

@dataclass
class HeadingGroup:
    """Represents a group of headings with similar formatting"""
    font_size: float
    is_bold: bool
    is_italic: bool
    is_center: bool
    font: str
    elements: List[TextElement] = field(default_factory=list)
    level: Optional[str] = None
    is_underlined: bool = False
    
    def get_signature(self) -> Tuple:
        """Get unique signature for grouping"""
        return (self.font_size, self.is_bold, self.is_italic, self.is_center, self.font)
    
    def add_element(self, element: TextElement):
        """Add element to the group"""
        self.elements.append(element)
    
    def get_priority_score(self) -> float:
        """
        ENHANCED: Calculate priority score with better differentiation including word count bonus
        """
        # Base score from font size (primary factor)
        score = self.font_size * 100
        
        # Style bonuses for better differentiation
        if self.is_bold:
            score += 11  # Increased from 4
        if self.is_center:
            score += 5  # Increased from 4
        if self.is_italic:
            score += 11   # Increased from 2
        if hasattr(self, 'is_underlined') and self.is_underlined:
            score += 11
        
        
        # Spacing-based bonus (headers often have more space)
        if hasattr(self, 'space_above') and self.space_above > 10:
            score += 5
        if hasattr(self, 'space_below') and self.space_below > 10:
            score += 3
        
        # NEW: Word count bonus - fewer words get higher bonus
        word_count_bonus = self._calculate_word_count_bonus()
        score += word_count_bonus
        
        return score

    def _calculate_word_count_bonus(self) -> float:
        """
        Calculate bonus score based on word count - fewer words = higher bonus
        No bonus if more than 8 words
        """
        if not self.elements:
            return 0.0
        
        total_bonus = 0.0
        element_count = 0
        
        for element in self.elements:
            # Count words in this element
            words = element.text.strip().split()
            word_count = len([word for word in words if word.strip()])  # Count non-empty words
            
            # Calculate bonus for this element
            if word_count <= 8:
                if word_count == 1:
                    element_bonus = 20.0  # Very high bonus for single words
                elif word_count <= 2:
                    element_bonus = 15.0  # High bonus for 2 words
                elif word_count <= 3:
                    element_bonus = 12.0  # Good bonus for 3 words
                elif word_count <= 4:
                    element_bonus = 10.0   # Medium bonus for 4 words
                elif word_count <= 5:
                    element_bonus = 8.0   # Small bonus for 5 words
                elif word_count <= 6:
                    element_bonus = 5.0   # Smaller bonus for 6 words
                elif word_count <= 7:
                    element_bonus = 3.0   # Very small bonus for 7 words
                else:  # word_count == 8
                    element_bonus = 1.0   # Minimal bonus for 8 words
            else:
                element_bonus = 0.0  # No bonus for more than 8 words
            
            total_bonus += element_bonus
            element_count += 1
        
        # Average the bonus across all elements in the group
        average_bonus = total_bonus / element_count if element_count > 0 else 0.0
        
        # Log the word count analysis
        if self.elements and hasattr(self.elements[0], 'text'):
            sample_text = self.elements[0].text[:30] + "..." if len(self.elements[0].text) > 30 else self.elements[0].text
            sample_words = len(self.elements[0].text.strip().split())
            logger.debug(f"📝 Word count bonus: '{sample_text}' ({sample_words} words) = +{average_bonus:.1f}")
        
        return average_bonus

class HeadingClassifier:
    """Main class for classifying headings with robust error handling and header/footer detection"""
    
    def __init__(self):
        self.elements: List[TextElement] = []
        self.groups: List[HeadingGroup] = []
        self.title: Optional[str] = None
        self.title_elements: List[TextElement] = []  # Track elements used for title
        self.outline: List[Dict[str, Any]] = []
        self.font_size_threshold = 30  # Threshold for filtering common text
        self.header_footer_detector = HeaderFooterDetector()
        self.excluded_indices: set = set()  # Track indices excluded due to headers/footers
        self.max_text_length_for_lowest = 50  # Max length for lowest level headings
        
 
    def _validate_input(self, data: Any) -> bool:
        """Validate input data structure"""
        try:
            if not isinstance(data, list):
                logger.error("Input data must be a list")
                return False
            
            if len(data) == 0:
                logger.warning("Input data is empty")
                return False
            
            # Check if at least some elements are dictionaries
            dict_count = sum(1 for item in data if isinstance(item, dict))
            if dict_count == 0:
                logger.error("No valid dictionary elements found in input")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating input: {e}")
            return False
    
    def _parse_text_elements(self, data: List[Dict[str, Any]]):
        """Parse and validate text elements"""
        self.elements = []
        
        for i, item in enumerate(data):
            try:
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dictionary item at index {i}")
                    continue
                
                # Create TextElement with validation
                element = TextElement(
                    text=item.get('text', ''),
                    page=item.get('page', 1),
                    font_size=item.get('font_size', 12.0),
                    font=item.get('font', 'Arial'),
                    is_bold=item.get('is_bold', False),
                    is_italic=item.get('is_italic', False),
                    is_underlined=item.get('is_underlined', False),
                    is_center=item.get('is_center', False),
                    space_above=item.get('space_above', 0.0),
                    space_below=item.get('space_below', 0.0),
                    x=item.get('x', item.get('x0', 0.0)),  # Try x first, then x0
                    y=item.get('y', item.get('y0', 0.0)),  # Try y first, then y0
                    original_index=i
                )
                
                # Skip empty text elements
                if element.text:
                    self.elements.append(element)
                else:
                    logger.warning(f"Skipping empty text element at index {i}")
                    
            except Exception as e:
                logger.warning(f"Error parsing element at index {i}: {e}")
                continue
        
        logger.info(f"Successfully parsed {len(self.elements)} text elements")
    
    def _detect_and_remove_headers_footers(self):
        """Detect and remove headers/footers from elements"""
        try:
            logger.info("Starting header/footer detection...")
            
            # Group elements by page
            pages_data = defaultdict(list)
            for element in self.elements:
                pages_data[element.page].append(element.to_dict())
            
            # Run header/footer detection
            hf_results = self.header_footer_detector.detect_headers_footers(pages_data)
            
            # Collect indices of headers and footers to exclude
            header_footer_indices = set()
            total_headers = 0
            total_footers = 0
            
            for page_num, page_data in hf_results.items():
                for header in page_data['headers']:
                    header_footer_indices.add(header['original_index'])
                    total_headers += 1
                
                for footer in page_data['footers']:
                    header_footer_indices.add(footer['original_index'])
                    total_footers += 1
            
            # Store excluded indices for reporting
            self.excluded_indices = header_footer_indices
            
            # Filter out header/footer elements
            original_count = len(self.elements)
            self.elements = [
                element for element in self.elements 
                if element.original_index not in header_footer_indices
            ]
            
            logger.info(f"Header/Footer detection complete:")
            logger.info(f"  - Found {total_headers} header elements")
            logger.info(f"  - Found {total_footers} footer elements")
            logger.info(f"  - Filtered from {original_count} to {len(self.elements)} elements")
            
        except Exception as e:
            logger.error(f"Error in header/footer detection: {e}")
            logger.error(traceback.format_exc())
            # Continue without header/footer filtering

    def _filter_by_font_size(self):
        """Smart font size filtering - keep elements that could be headings"""
        if not self.elements:
            return
        
        font_size_counts = Counter(element.font_size for element in self.elements)
        total_elements = len(self.elements)
        
        logger.info(f"📊 Font size distribution: {dict(font_size_counts)}")
        
        # Strategy 1: If we have very few font sizes, keep all
        if len(font_size_counts) <= 3:
            logger.info(f"✅ Only {len(font_size_counts)} font sizes found - keeping all elements")
            return
        
        # Strategy 2: Smart filtering based on percentage and heading likelihood
        excluded_sizes = set()
        kept_sizes = set()
        
        for font_size, count in font_size_counts.items():
            percentage = (count / total_elements) * 100
            
            # Exclude if it's more than 50% of all elements (likely body text)
            if percentage > 50:
                excluded_sizes.add(font_size)
                logger.info(f"🚫 Excluding font size {font_size} ({count} elements, {percentage:.1f}% - likely body text)")
            else:
                kept_sizes.add(font_size)
                logger.info(f"✅ Keeping font size {font_size} ({count} elements, {percentage:.1f}%)")
        
        # NEW: Apply classify_string filter early to reduce elements before grouping
        pre_filter_count = len(self.elements)
        self.elements = [
            element for element in self.elements 
            if classify_string(element.text)
        ]
        post_filter_count = len(self.elements)
        
        if post_filter_count != pre_filter_count:
            logger.info(f"📝 Early classify_string filtering: {pre_filter_count} → {post_filter_count} elements")
        
        # Safety check: Always keep at least the largest font sizes
        if not kept_sizes:
            # Emergency fallback - keep the 3 largest font sizes
            largest_sizes = sorted(font_size_counts.keys(), reverse=True)[:3]
            kept_sizes = set(largest_sizes)
            excluded_sizes = set(font_size_counts.keys()) - kept_sizes
            logger.warning(f"⚠️ Emergency fallback - keeping largest font sizes: {largest_sizes}")
        
        # Filter elements - only exclude the problematic sizes
        original_count = len(self.elements)
        self.elements = [
            element for element in self.elements 
            if element.font_size not in excluded_sizes
        ]
        
        logger.info(f"📝 Font size filtering: {original_count} → {len(self.elements)} elements")
        
        # Final safety check
        if len(self.elements) == 0:
            logger.error("🚨 All elements filtered out - restoring all elements as emergency fallback")
            # Restore all elements by reloading from original data
            # This is a safety net - should rarely happen
            font_size_counts = Counter(element.font_size for element in self.elements)
            logger.info("🔄 Restoring all elements to prevent empty processing")
            # We need to reload elements, so let's disable filtering entirely
            self.font_size_threshold = float('inf')  # Disable threshold

    def _create_groups(self):
        """Group elements by formatting attributes - FIXED to handle spatial proximity"""
        try:
            group_map = defaultdict(list)
            
            for element in self.elements:
                # Create base signature without is_center
                base_signature = (
                    round(element.font_size, 1),  # Round to handle minor differences
                    element.is_bold,
                    element.is_italic,
                    element.font.lower().replace('-', '').replace(' ', '')  # Normalize font
                )
                
                # Look for existing groups with same base formatting on same page
                merged_to_existing = False
                
                for signature, existing_elements in group_map.items():
                    if (len(signature) >= 4 and signature[:4] == base_signature and 
                        existing_elements and existing_elements[0].page == element.page):
                        
                        # Check if spatially close to any element in this group
                        for existing_elem in existing_elements:
                            y_diff = abs(element.y - existing_elem.y)
                            # Dynamic threshold based on font size - larger fonts get more space
                            dynamic_threshold = max(3, element.font_size * 0.2)  # Minimum 8, scales with font
                            if y_diff <= dynamic_threshold:
                                group_map[signature].append(element)
                                merged_to_existing = True
                                logger.info(f"Merged element '{element.text[:30]}...' with existing group (y_diff: {y_diff:.1f}, threshold: {dynamic_threshold:.1f})")
                                break
                        
                        if merged_to_existing:
                            break
                
                # If not merged, create new group
                if not merged_to_existing:
                    full_signature = base_signature + (element.is_center,)
                    group_map[full_signature].append(element)
            
            # Create HeadingGroup objects
            self.groups = []
            for signature, elements in group_map.items():
                if len(signature) >= 4:  # Ensure we have all required fields
                    font_size = signature[0]
                    is_bold = signature[1]
                    is_italic = signature[2]
                    font = signature[3] if isinstance(signature[3], str) else "Arial"
                    is_center = signature[4] if len(signature) > 4 else False
                    is_group_underlined = any(getattr(elem, 'is_underlined', False) for elem in elements)
                    
                    group = HeadingGroup(
                        font_size=font_size,
                        is_bold=is_bold,
                        is_italic=is_italic,
                        is_center=is_center,
                        font=font,
                        is_underlined=is_group_underlined, 
                        elements=elements
                    )
                    self.groups.append(group)
            
            logger.info(f"Created {len(self.groups)} heading groups with spatial proximity grouping")
            
        except Exception as e:
            logger.error(f"Error creating groups: {e}")
            # Fallback to simple grouping
            self.groups = [HeadingGroup(
                font_size=12.0,
                is_bold=False,
                is_italic=False,
                is_center=False,
                font="Arial",
                elements=self.elements
            )]

    def _combine_consecutive_similar_elements(self):
        """Combine consecutive elements with identical formatting that should be one heading"""
        try:
            if not self.groups:
                return
            
            logger.info("🔄 Combining consecutive similar elements...")
            
            # For each group, check if there are consecutive elements that should be combined
            for group in self.groups:
                if len(group.elements) <= 1:
                    continue
                
                # Sort elements by page and original index
                group.elements.sort(key=lambda e: (e.page, e.original_index))
                
                # Group consecutive elements that are spatially close
                combined_elements = []
                current_group = [group.elements[0]]
                
                for i in range(1, len(group.elements)):
                    prev_elem = group.elements[i-1]
                    curr_elem = group.elements[i]
                    
                    # Check if elements are consecutive and spatially close
                    # Much stricter consecutiveness check
                    is_consecutive = (curr_elem.original_index - prev_elem.original_index <= 1)  # Only truly consecutive  # Allow some gap
                    is_same_page = (curr_elem.page == prev_elem.page)
                    # Dynamic spatial threshold based on font size
                    # Much stricter spatial threshold
                    spatial_threshold = max(2, curr_elem.font_size * 0.15)
                    is_spatially_close = (abs(curr_elem.y - prev_elem.y) <= spatial_threshold)
                    
                    if is_consecutive and is_same_page and is_spatially_close:
                        # Add to current group
                        current_group.append(curr_elem)
                    else:
                        # Finalize current group and start new one
                        if len(current_group) > 1:
                            # Combine the text of elements in current_group
                            combined_elem = self._combine_elements_into_one(current_group)
                            combined_elements.append(combined_elem)
                        else:
                            combined_elements.extend(current_group)
                        
                        current_group = [curr_elem]
                
                # Handle the last group
                if len(current_group) > 1:
                    combined_elem = self._combine_elements_into_one(current_group)
                    combined_elements.append(combined_elem)
                else:
                    combined_elements.extend(current_group)
                
                # Update the group's elements
                if len(combined_elements) < len(group.elements):
                    logger.info(f"✅ Combined {len(group.elements)} elements into {len(combined_elements)} for group with font_size={group.font_size}")
                    group.elements = combined_elements
        
        except Exception as e:
            logger.error(f"Error combining consecutive elements: {e}")

    def _combine_elements_into_one(self, elements: List['TextElement']) -> 'TextElement':
        """Combine multiple elements into a single element"""
        if not elements:
            return None
        
        if len(elements) == 1:
            return elements[0]
        
        # Sort by original index
        elements.sort(key=lambda e: e.original_index)
        
        # Use the first element as base
        base_elem = elements[0]
        
        # Combine text using spatial reconstruction
        combined_text = self._reconstruct_title_text(elements)
        
        # Create new combined element
        combined_element = TextElement(
            text=combined_text,
            page=base_elem.page,
            font_size=base_elem.font_size,
            font=base_elem.font,
            is_bold=base_elem.is_bold,
            is_italic=base_elem.is_italic,
            is_underlined=base_elem.is_underlined,
            is_center=base_elem.is_center,
            space_above=base_elem.space_above,
            space_below=elements[-1].space_below,  # Use last element's space_below
            x=base_elem.x,
            y=base_elem.y,
            original_index=base_elem.original_index  # Keep first element's index
        )
        
        return combined_element

    def _is_valid_title_text(self, text: str) -> bool:
        """
        Check if text is valid for a title using industry-standard criteria
        """
        try:
            if not text or not text.strip():
                return False
            
            text = text.strip()
            
            # Length criteria
            if len(text) < 3 or len(text) > 200:
                return False
            
            # Character composition analysis
            total_chars = len(text)
            alpha_chars = sum(1 for c in text if c.isalpha())
            digit_chars = sum(1 for c in text if c.isdigit())
            space_chars = sum(1 for c in text if c.isspace())
            punct_chars = sum(1 for c in text if c in '.,!?;:()-[]{}"\'-')
            
            # Must have reasonable proportion of alphabetic characters
            if alpha_chars < total_chars * 0.4:  # At least 40% letters
                return False
            
            # Too many digits suggests it's not a title
            if digit_chars > total_chars * 0.5:  # More than 50% digits
                return False
            
            # Pattern-based exclusions
            patterns_to_exclude = [
                r'^[0-9\s\-\.\/]+$',  # Only numbers, spaces, dashes, dots, slashes
                r'^[A-Z]{3,}\s*[0-9]+$',  # Pattern like "ABC 123" or "FORM123"
                r'^\d+[\.\-\s]*\d*$',  # Pure numeric patterns
                r'^[^\w\s]{3,}$',  # Only special characters
                # r'.*\b(form|doc|file|page|section|appendix|exhibit|schedule)\s*\d+\b.*',  # Document references
                r'.*\b(rev|version|ver|v)\s*[\d\.]+\b.*',  # Version numbers
                r'.*\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b.*',  # Dates
                r'.*\b[A-Z]{2,}\-\d+\b.*',  # Code patterns like "ABC-123"
                r'^(table|figure|chart|graph|image|photo)\s+\d+.*',  # Figure/table references
                r'.*-{2,}.*',  # Change to this - excludes any text containing 2+ consecutive dashes
            ]
            
            text_lower = text.lower()
            for pattern in patterns_to_exclude:
                if re.match(pattern, text_lower, re.IGNORECASE):
                    return False
            
            # Word-based analysis
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            
            if not words:  # No valid words found
                return False
            
            # Check for minimum meaningful words
            if len(words) == 1 and len(words[0]) < 4:  # Single short word
                return False
            
            # Dictionary word check (basic English common words)
            common_english_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
                'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
                'about', 'above', 'after', 'again', 'against', 'all', 'any', 'as', 'because', 'before',
                'below', 'between', 'both', 'but', 'during', 'each', 'few', 'from', 'further',
                'if', 'into', 'more', 'most', 'no', 'not', 'only', 'other', 'over', 'same', 'some',
                'such', 'than', 'through', 'under', 'until', 'up', 'very', 'while', 'within', 'without'
            }
            
            # Count recognizable words (either common words or words with reasonable letter patterns)
            recognizable_words = 0
            for word in words:
                word_lower = word.lower()
                # Accept if it's a common word or follows reasonable patterns
                if (word_lower in common_english_words or 
                    len(word) >= 3 or  # Short words might be valid
                    self._has_reasonable_letter_pattern(word)):
                    recognizable_words += 1
            
            # At least 60% of words should be recognizable
            if len(words) > 1 and recognizable_words / len(words) < 0.6:
                return False
            
            # Content quality checks
            # if self._contains_excessive_repetition(text):
            #     return False
            
            if self._looks_like_code_or_technical_id(text):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating title text '{text}': {e}")
            return True  # Be permissive on error


    def _is_potential_title_fragment(self, text: str) -> bool:
        """Check if text could be part of a title (more lenient than full validation)"""
        if not text or not text.strip():
            return False
        
        text = text.strip()
        
        # Very basic checks for fragments
        if len(text) < 1:
            return False
        
        # Allow short fragments that might be part of larger text
        if len(text) <= 3:
            return text.isalnum() or text in [':', '-', '(', ')', '.', ',']
        
        # For longer fragments, check for reasonable content
        alpha_count = sum(1 for c in text if c.isalpha())
        total_chars = len(text)
        
        # At least 30% letters (more lenient than title validation)
        return alpha_count >= total_chars * 0.3

    def _find_similar_title_elements(self, reference_group: 'HeadingGroup', target_page: int) -> List['TextElement']:
        """Find all elements on the target page with similar formatting to the reference group"""
        similar_elements = []
        
        # Get reference characteristics
        ref_font_size = reference_group.font_size
        ref_font = reference_group.font.lower()
        ref_is_bold = reference_group.is_bold
        ref_is_italic = reference_group.is_italic
        
        # Collect elements from all groups that match the criteria
        for group in self.groups:
            # Check if group has similar characteristics
            font_size_diff = abs(group.font_size - ref_font_size)
            font_match = group.font.lower() == ref_font or self._fonts_are_similar(group.font, reference_group.font)
            
            if (font_size_diff <= 1.0 and  # Font size within 1 point
                font_match and             # Same or similar font
                group.is_bold == ref_is_bold and
                group.is_italic == ref_is_italic):
                
                # Add elements from this page
                for element in group.elements:
                    if element.page == target_page:
                        similar_elements.append(element)
        
        # Sort by original index (document order)
        similar_elements.sort(key=lambda e: e.original_index)
        
        return similar_elements

    def _remove_overlapping_elements(self, elements: List['TextElement']) -> List['TextElement']:
        """Remove overlapping/duplicate elements based on spatial coordinates and text content"""
        if len(elements) <= 1:
            return elements
        
        # Sort by original index to maintain document order
        sorted_elements = sorted(elements, key=lambda e: e.original_index)
        filtered_elements = []
        
        for current_elem in sorted_elements:
            should_keep = True
            current_text = current_elem.text.strip()
            
            for existing_elem in filtered_elements:
                existing_text = existing_elem.text.strip()
                
                # Check for spatial overlap (elements at nearly same position)
                # Stricter overlap detection
                x_overlap = abs(current_elem.x - existing_elem.x) < 3
                y_overlap = abs(current_elem.y - existing_elem.y) < 2
                
                # Check for text containment
                text_contained = (current_text in existing_text or 
                                existing_text in current_text)
                
                # If elements overlap spatially and have similar/contained text, skip the shorter one
                if x_overlap and y_overlap and text_contained:
                    if len(current_text) <= len(existing_text):
                        should_keep = False
                        break
                    else:
                        # Current text is longer, remove the existing shorter one
                        filtered_elements.remove(existing_elem)
            
            if should_keep:
                filtered_elements.append(current_elem)
        
        return filtered_elements

    def _fonts_are_similar(self, font1: str, font2: str) -> bool:
        """Check if two fonts are similar enough to be considered the same"""
        f1 = font1.lower().replace('-', '').replace(' ', '')
        f2 = font2.lower().replace('-', '').replace(' ', '')
        
        # Direct match
        if f1 == f2:
            return True
        
        # Check for common variations
        font_families = {
            'arial': ['arial', 'arialblack', 'arialbold'],
            'times': ['times', 'timesnewroman', 'timesbold'],
            'helvetica': ['helvetica', 'arial'],
            'calibri': ['calibri', 'calibribold']
        }
        
        for family, variants in font_families.items():
            if f1 in variants and f2 in variants:
                return True
        
        return False

    def _reconstruct_title_text(self, elements: List['TextElement']) -> str:
        """Reconstruct complete title text from fragmented elements"""
        if not elements:
            return ""
        
        elements = self._remove_overlapping_elements(elements)
        
        # Strategy 1: Try to reconstruct by overlapping text analysis
        reconstructed = self._reconstruct_by_overlap_analysis(elements)
        if reconstructed:
            return reconstructed
        
        # Strategy 2: Try positional reconstruction
        reconstructed = self._reconstruct_by_position(elements)
        if reconstructed:
            return reconstructed
        
        # Strategy 3: Simple concatenation with deduplication
        return self._simple_concatenation_with_dedup(elements)

    def _reconstruct_by_overlap_analysis(self, elements: List['TextElement']) -> str:
        """Reconstruct text by analyzing overlapping fragments"""
        if len(elements) <= 1:
            return elements[0].text.strip() if elements else ""
        
        # Sort elements by their original index
        sorted_elements = sorted(elements, key=lambda e: e.original_index)
        
        # Try to find the longest coherent sequence
        text_fragments = [elem.text.strip() for elem in sorted_elements if elem.text.strip()]
        
        if not text_fragments:
            return ""
        
        # Look for patterns in the fragments
        reconstructed = self._merge_overlapping_fragments(text_fragments)
        
        return reconstructed

    def _merge_overlapping_fragments(self, fragments: List[str]) -> str:
        """Merge overlapping text fragments into coherent text - IMPROVED VERSION"""
        if not fragments:
            return ""
        
        if len(fragments) == 1:
            return fragments[0]
        
        # Remove exact duplicates first
        unique_fragments = []
        for frag in fragments:
            if frag not in unique_fragments:
                unique_fragments.append(frag)
        
        if len(unique_fragments) == 1:
            return unique_fragments[0]
        
        # Sort fragments by length (longest first) to prioritize complete text
        unique_fragments.sort(key=len, reverse=True)
        
        # Start with the longest fragment
        result = unique_fragments[0]
        
        for i in range(1, len(unique_fragments)):
            current_fragment = unique_fragments[i]
            
            # Skip if current fragment is completely contained in result
            if current_fragment.lower() in result.lower():
                continue
            
            # Try to find overlap with the result
            merged = self._merge_two_fragments(result, current_fragment)
            if merged != result and len(merged) > len(result):  # Only accept if merge makes text longer
                result = merged
            elif not any(word.lower() in result.lower() for word in current_fragment.split() if len(word) > 2):
                # If no significant word overlap, append as continuation
                result = result + " " + current_fragment
        
        return result.strip()

    def _merge_two_fragments(self, text1: str, text2: str) -> str:
        """Try to merge two potentially overlapping fragments"""
        # Check for suffix-prefix overlap
        max_overlap = min(len(text1), len(text2)) // 2
        
        for overlap_len in range(max_overlap, 0, -1):
            if text1[-overlap_len:].lower() == text2[:overlap_len].lower():
                # Found overlap
                merged = text1 + text2[overlap_len:]
                return merged
        
        # Check for prefix-suffix overlap (reverse)
        for overlap_len in range(max_overlap, 0, -1):
            if text2[-overlap_len:].lower() == text1[:overlap_len].lower():
                # Found overlap
                merged = text2 + text1[overlap_len:]
                return merged
        
        # Check if one is contained in the other
        if text1.lower() in text2.lower():
            return text2
        elif text2.lower() in text1.lower():
            return text1
        
        return text1  # No merge possible

    def _is_continuation(self, text1: str, text2: str) -> bool:
        """Check if text2 is a logical continuation of text1"""
        # Simple heuristics for continuation
        if not text1 or not text2:
            return False
        
        # If text1 ends with incomplete word and text2 starts completing it
        if text1[-1].isalpha() and text2[0].isalpha():
            return True
        
        # If text1 ends with punctuation that suggests continuation
        if text1[-1] in ':-,':
            return True
        
        return False

    def _reconstruct_by_position(self, elements: List['TextElement']) -> str:
        """Reconstruct text based on spatial positioning (if x,y coordinates available)"""
        if not all(hasattr(elem, 'x') and hasattr(elem, 'y') for elem in elements):
            return ""
        
        # Group elements by approximate y-position (same line)
        lines = defaultdict(list)
        
        for elem in elements:
            # Round y-position to group elements on the same line
            line_key = round(elem.y / 10) * 10  # Group within 5 units
            lines[line_key].append(elem)
        
        # Sort lines by y-position (top to bottom)
        sorted_lines = sorted(lines.items(), key=lambda x: x[0])
        
        # For each line, sort elements by x-position (left to right)
        line_texts = []
        for y_pos, line_elements in sorted_lines:
            line_elements.sort(key=lambda e: e.x)
            line_text = " ".join(elem.text.strip() for elem in line_elements if elem.text.strip())
            if line_text:
                line_texts.append(line_text)
        
        # Combine lines
        return " ".join(line_texts)

    def _simple_concatenation_with_dedup(self, elements: List['TextElement']) -> str:
        """Simple concatenation with basic deduplication"""
        if not elements:
            return ""
        
        # Get unique text pieces in document order
        seen_texts = set()
        unique_texts = []
        
        for elem in sorted(elements, key=lambda e: e.original_index):
            text = elem.text.strip()
            if text and text not in seen_texts:
                unique_texts.append(text)
                seen_texts.add(text)
        
        return " ".join(unique_texts)

    def _simple_title_combination(self, elements: List['TextElement']) -> str:
        """Fallback: simple combination of all element texts"""
        if not elements:
            return ""
        
        # Sort by original index and combine
        sorted_elements = sorted(elements, key=lambda e: e.original_index)
        texts = [elem.text.strip() for elem in sorted_elements if elem.text.strip()]
        
        # Remove obvious duplicates
        unique_texts = []
        for text in texts:
            if not unique_texts or text != unique_texts[-1]:
                unique_texts.append(text)
        
        combined = " ".join(unique_texts)
        
        # Basic cleanup
        combined = re.sub(r'\s+', ' ', combined)  # Multiple spaces to single
        combined = combined.strip()
        
        return combined

    def _is_similar_title_formatting(self, group1: HeadingGroup, group2: HeadingGroup) -> bool:
        """
        Check if two groups have similar formatting that could indicate they're part of the same title
        """
        try:
            # Must have same bold and italic characteristics
            if group1.is_bold != group2.is_bold or group1.is_italic != group2.is_italic:
                return False
            
            # Must have same centering
            if group1.is_center != group2.is_center:
                return False
            
            # Font sizes should be very close (within 2 points)
            font_size_diff = abs(group1.font_size - group2.font_size)
            if font_size_diff > 2.0:
                return False
            
            # Font should be the same or similar
            if group1.font.lower() != group2.font.lower():
                # Allow some common font variations
                font1_normalized = self._normalize_font_name(group1.font)
                font2_normalized = self._normalize_font_name(group2.font)
                if font1_normalized != font2_normalized:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error comparing group formatting: {e}")
            return False

    def _normalize_font_name(self, font_name: str) -> str:
        """Normalize font names for comparison"""
        font_lower = font_name.lower().strip()
        
        # Common font family mappings
        font_families = {
            'arial': 'arial',
            'helvetica': 'arial',  # Similar sans-serif
            'times': 'times',
            'times new roman': 'times',
            'calibri': 'calibri',
            'georgia': 'georgia',
        }
        
        for family_key, family_name in font_families.items():
            if family_key in font_lower:
                return family_name
        
        return font_lower

    def _combine_title_parts(self, title_parts: List[str]) -> str:
        """
        Intelligently combine title parts with appropriate spacing
        """
        if not title_parts:
            return ""
        
        if len(title_parts) == 1:
            return title_parts[0]
        
        combined = ""
        
        for i, part in enumerate(title_parts):
            if i == 0:
                combined = part
            else:
                # Determine appropriate spacing
                prev_part = title_parts[i-1]
                current_part = part
                
                # Check if we need spacing
                spacing = self._determine_title_spacing(prev_part, current_part)
                combined += spacing + current_part
        
        return combined.strip()

    def _determine_title_spacing(self, prev_part: str, current_part: str) -> str:
        """
        Determine appropriate spacing between title parts
        """
        # If previous part ends with punctuation, use space
        if prev_part and prev_part[-1] in '.!?:;':
            return " "
        
        # If current part starts with punctuation, no space
        if current_part and current_part[0] in '.,!?:;':
            return ""
        
        # If parts look like they should be on separate lines (common in titles)
        if (len(prev_part) > 20 and len(current_part) > 5) or \
        any(keyword in current_part.lower() for keyword in ['subtitle', 'volume', 'part', 'chapter']):
            return " "
        
        # Default: single space
        return " "
    def _has_reasonable_letter_pattern(self, word: str) -> bool:
        """Check if word has reasonable vowel-consonant patterns"""
        if len(word) < 2:
            return True
        
        vowels = set('aeiouAEIOU')
        has_vowel = any(c in vowels for c in word)
        has_consonant = any(c.isalpha() and c not in vowels for c in word)
        
        # Should have both vowels and consonants for longer words
        if len(word) >= 4:
            return has_vowel and has_consonant
        
        return True

    # def _contains_excessive_repetition(self, text: str) -> bool:
    #     """Check for excessive character or pattern repetition"""
    #     # Check for repeated characters
    #     for i in range(len(text) - 2):
    #         if text[i] == text[i+1] == text[i+2]:  # 3 consecutive identical chars
    #             return True
        
    #     # Check for repeated short patterns
    #     words = text.split()
    #     if len(words) > 1:
    #         word_counts = Counter(words)
    #         for word, count in word_counts.items():
    #             if len(word) <= 3 and count > 2:  # Short word repeated > 2 times
    #                 return True
        
    #     return False

    def _looks_like_code_or_technical_id(self, text: str) -> bool:
        """Check if text looks like code, IDs, or technical references"""
        patterns = [
            r'.*[a-zA-Z]+\d+[a-zA-Z]+\d+.*',  # Mixed letters and numbers pattern
            r'.*\b[A-Z]{2,}_[A-Z0-9_]+\b.*',  # Underscore separated caps
            r'.*\b[a-z]+[A-Z][a-z]*[A-Z].*',  # camelCase patterns
            r'.*\b\w*[0-9]{3,}\w*\b.*',  # Words with 3+ consecutive digits
            r'.*\b[A-F0-9]{8,}\b.*',  # Hex-like patterns
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        return False
    

    def _log_final_hierarchy(self):
        """Log the final hierarchy structure for debugging"""
        try:
            logger.info("=== Final Hierarchy Structure ===")
            
            # Group by level for clear display
            level_groups = defaultdict(list)
            for group in self.groups:
                level_groups[group.level].append(group)
            
            # Sort levels (TITLE first, then H1, H2, etc.)
            level_order = ['TITLE'] + [f'H{i}' for i in range(1, 10)]
            
            for level in level_order:
                if level in level_groups:
                    groups = level_groups[level]
                    for group in groups:
                        sample_text = ""
                        if group.elements:
                            first_elem = min(group.elements, key=lambda e: e.original_index)
                            sample_text = first_elem.text[:50] + "..." if len(first_elem.text) > 50 else first_elem.text
                        
                        logger.info(f"{level}: font_size={group.font_size}, bold={group.is_bold}, "
                                f"center={group.is_center}, elements={len(group.elements)}")
                        logger.info(f"    Sample: '{sample_text}'")
            
            logger.info("=== End Hierarchy Structure ===")
            
        except Exception as e:
            logger.error(f"Error logging hierarchy: {e}")

    def _determine_hierarchy(self):
        """Initial hierarchy determination - will be reassigned after title identification"""
        try:
            if not self.groups:
                return
            
            # Sort groups by priority score (descending - highest priority first)
            sorted_groups = sorted(self.groups, key=lambda g: g.get_priority_score(), reverse=True)
            
            # Assign TEMPORARY levels (these will be reassigned after title identification)
            level_names = ['TEMP1', 'TEMP2', 'TEMP3', 'TEMP4', 'TEMP5']
            
            for i, group in enumerate(sorted_groups):
                if i < len(level_names):
                    group.level = level_names[i]
                    logger.info(f"Assigned temporary {level_names[i]} to group with priority score: {group.get_priority_score():.2f}")
                else:
                    group.level = f'TEMP{i+1}'
            
            # Update the groups list with sorted order
            self.groups = sorted_groups
            
            logger.info(f"Assigned temporary hierarchy levels to {len(self.groups)} groups")
            
        except Exception as e:
            logger.error(f"Error determining hierarchy: {e}")
            for i, group in enumerate(self.groups):
                group.level = f'TEMP{i+1}'


    def process_input(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Main processing function with comprehensive error handling"""
        try:
            logger.info(f"Starting processing of {len(data) if data else 0} text elements")
            
            # Step 1: Input validation and sanitization
            if not self._validate_input(data):
                return self._create_error_output("Invalid input data")
            
            # Step 2: Parse and validate text elements
            self._parse_text_elements(data)
            
            if not self.elements:
                return self._create_error_output("No valid text elements found")
            
            # Step 3: Header/Footer detection and removal
            self._detect_and_remove_headers_footers()
            
            if not self.elements:
                return self._create_error_output("No content elements found after header/footer removal")
            
            # NEW Step 3.5: Apply classify_string filter EARLY to minimize brackets
            self._apply_early_classify_string_filter()
            
            if not self.elements:
                return self._create_error_output("No valid heading text found after classify_string filtering")
            
            # Step 4: Font size filtering
            self._filter_by_font_size()
            
            if not self.elements:
                return self._create_error_output("No heading-level text found after filtering")
            
            # Step 5: Group elements by formatting
            self._create_groups()

            # Step 5.5: Combine consecutive similar elements
            self._combine_consecutive_similar_elements()
            
            # Step 6: Determine TEMPORARY hierarchy (will be reassigned)
            self._determine_hierarchy()
            
            # Step 7: Identify title (this will mark one group as 'TITLE')
            self._identify_title()

            # Step 8: CRITICAL - Remove duplicate headings BEFORE hierarchy reassignment
            self._remove_duplicate_headings()

            # Step 9: CRITICAL FIX - Reassign hierarchy starting from H1 after title identification
            self._reassign_hierarchy_after_title()

            # Step 10: Create outline
            self._create_outline()
            
            # Step 11: Generate output
            return self._generate_output()
            
        except Exception as e:
            logger.error(f"Unexpected error in process_input: {e}")
            logger.error(traceback.format_exc())
            return self._create_error_output(f"Processing failed: {str(e)}")
    
    def _identify_title(self):
        """Identify title based on HIGHEST scoring group that passes validation"""
        try:
            if not self.groups:
                self.title = ""
                self.title_elements = []
                return
            
            # Sort groups by score in descending order (highest first)
            sorted_groups = sorted(self.groups, key=lambda g: g.get_priority_score(), reverse=True)
            
            logger.info(f"🔍 Checking {len(sorted_groups)} groups for title in priority order:")
            
            # Check each group starting from highest score
            title_found = False
            for i, group in enumerate(sorted_groups):
                score = group.get_priority_score()
                
                # Get all elements from this group
                group_elements = group.elements
                if not group_elements:
                    logger.info(f"  Rank {i+1}: Score {score:.1f} - ❌ No elements")
                    continue
                    
                # Check if this group appears on a single page (individual group check)
                pages_in_group = set(elem.page for elem in group_elements)
                
                # Reconstruct text from this group
                reconstructed_title = self._reconstruct_title_text(group_elements)
                
                # Log what we're checking
                sample_text = reconstructed_title[:50] + "..." if len(reconstructed_title) > 50 else reconstructed_title
                logger.info(f"  Rank {i+1}: Score {score:.1f} - '{sample_text}' (pages: {sorted(pages_in_group)})")
                
                # Title validation criteria
                if not reconstructed_title:
                    logger.info(f"    ❌ Empty text after reconstruction")
                    continue
                    
                if not self._is_valid_title_text(reconstructed_title):
                    logger.info(f"    ❌ Failed title text validation")
                    continue
                
                # Prefer titles that appear early in document (first few pages)
                earliest_page = min(pages_in_group)
                if earliest_page > 3:  # More strict - first 3 pages only
                    logger.info(f"    ❌ Appears too late (page {earliest_page})")
                    continue
                
                # Prefer shorter titles (reasonable length)
                if len(reconstructed_title) > 150:  # More strict length limit
                    logger.info(f"    ❌ Too long ({len(reconstructed_title)} chars)")
                    continue
                    
                # Prefer titles that appear on single page (but not required)
                single_page_bonus = len(pages_in_group) == 1
                logger.info(f"    📍 Single page: {single_page_bonus}")
                
                # SUCCESS - This is our title
                self.title = reconstructed_title
                self.title_elements = group_elements
                group.level = 'TITLE'
                
                logger.info(f"✅ Title found: '{self.title}' (score: {score:.1f}, page: {earliest_page})")
                title_found = True
                break
            
            if not title_found:
                self.title = ""
                self.title_elements = []
                logger.info("❌ No valid title found - all groups failed validation")
                        
        except Exception as e:
            logger.error(f"Error identifying title: {e}")
            logger.error(traceback.format_exc())
            self.title = ""
            self.title_elements = []

    def _apply_early_classify_string_filter(self):
        """
        Apply classify_string filtering early to remove non-heading text before bracket creation.
        This significantly reduces the number of elements and creates fewer, more meaningful brackets.
        """
        try:
            if not self.elements:
                return
            
            original_count = len(self.elements)
            logger.info(f"🔍 Applying early classify_string filter to {original_count} elements...")
            
            # Filter elements using classify_string
            filtered_elements = []
            rejected_count = 0
            
            for element in self.elements:
                if classify_string(element.text):
                    filtered_elements.append(element)
                    logger.debug(f"✅ Kept: '{element.text[:50]}...'")
                else:
                    rejected_count += 1
                    logger.debug(f"❌ Rejected: '{element.text[:50]}...'")
            
            # Update elements list
            self.elements = filtered_elements
            
            logger.info(f"📊 Early classify_string filtering results:")
            logger.info(f"  - Original elements: {original_count}")
            logger.info(f"  - Kept elements: {len(self.elements)}")
            logger.info(f"  - Rejected elements: {rejected_count}")
            logger.info(f"  - Reduction: {((rejected_count / original_count) * 100):.1f}%")
            
            # Log some examples of what was kept
            if self.elements:
                logger.info("📝 Sample of kept elements:")
                for i, elem in enumerate(self.elements[:5]):  # Show first 5
                    logger.info(f"  {i+1}. '{elem.text[:60]}...' (font: {elem.font_size})")
            
        except Exception as e:
            logger.error(f"Error in early classify_string filtering: {e}")
            logger.error(traceback.format_exc())

    def _reassign_hierarchy_after_title(self):
        """Smart hierarchy assignment with 10-point score brackets and 30% exclusion rule"""
        try:
            logger.info("=== STARTING BRACKET-BASED HIERARCHY ASSIGNMENT ===")
            
            # Separate title groups from non-title groups
            title_groups = [g for g in self.groups if g.level == 'TITLE']
            non_title_groups = [g for g in self.groups if g.level != 'TITLE']
            
            if not non_title_groups:
                logger.warning("⚠️  No non-title groups found")
                return
            
            # Get all scores and create brackets
            all_scores = []
            score_to_groups = defaultdict(list)
            for group in non_title_groups:
                score = group.get_priority_score()
                all_scores.append(score)
                score_to_groups[score].append(group)
            
            # Sort scores in descending order
            all_scores = sorted(set(all_scores), reverse=True)
            logger.info(f"📊 Found {len(all_scores)} unique scores: {all_scores}")
            
            # Create score brackets with 10-point ranges
            brackets = self._create_score_brackets(all_scores)
            logger.info(f"🎯 Created {len(brackets)} score brackets:")
            for i, bracket in enumerate(brackets):
                logger.info(f"  Bracket {i+1}: {bracket['range'][0]:.1f} - {bracket['range'][1]:.1f} ({len(bracket['scores'])} scores)")
            
            # NEW: Apply absolute exclusion rule (exclude brackets with >40 entries)
            brackets = self._apply_absolute_exclusion_rule(brackets, score_to_groups, max_entries_threshold=40)
            
            # Apply hierarchy rules based on number of brackets (after exclusion)
            num_brackets = len(brackets)
            original_bracket_count = len(self._create_score_brackets(all_scores))  # Get original count before exclusion

            logger.info(f"📊 Brackets after absolute exclusion: {num_brackets} (original: {original_bracket_count})")
            if num_brackets == 0:
                # No brackets left after exclusion
                logger.info("🚫 No brackets remaining after 65% exclusion - excluding all")
                for group in non_title_groups:
                    group.level = 'EXCLUDED'
                    
            elif num_brackets == 1:
                # Special handling: if we started with more brackets but absolute rule left us with 1,
                if original_bracket_count == 1:
                    logger.info("🚫 Only 1 bracket found originally - excluding all as likely body text")
                    for group in non_title_groups:
                        group.level = 'EXCLUDED'
                else:
                    logger.info("✅ 1 bracket remaining after >40 exclusion - treating as H1")
                    self._assign_bracket_to_level(brackets[0], score_to_groups, 'H1')
                    
            elif num_brackets == 2:
                # Two brackets: H1 and conditional H2
                self._assign_bracket_to_level(brackets[0], score_to_groups, 'H1')
                
                # Check if second bracket should be included
                should_include = self._should_include_bracket(brackets[1], score_to_groups)
                if should_include:
                    self._assign_bracket_to_level(brackets[1], score_to_groups, 'H2')
                    logger.info("✅ Including second bracket as H2")
                else:
                    self._assign_bracket_to_level(brackets[1], score_to_groups, 'EXCLUDED')
                    logger.info("🚫 Excluding second bracket (text too long)")
                    
            elif num_brackets == 3:
                # Three brackets: H1, H2, and conditional H3
                self._assign_bracket_to_level(brackets[0], score_to_groups, 'H1')
                self._assign_bracket_to_level(brackets[1], score_to_groups, 'H2')
                
                # Check if third bracket should be included
                should_include = self._should_include_bracket(brackets[2], score_to_groups)
                if should_include:
                    self._assign_bracket_to_level(brackets[2], score_to_groups, 'H3')
                    logger.info("✅ Including third bracket as H3")
                else:
                    self._assign_bracket_to_level(brackets[2], score_to_groups, 'EXCLUDED')
                    logger.info("🚫 Excluding third bracket (text too long)")
                    
            else:  # num_brackets >= 4
                # Four or more brackets: Use top 3, exclude rest
                logger.info(f"🎯 {num_brackets} brackets found - using only top 3, excluding rest")
                
                # Assign top 3 brackets
                self._assign_bracket_to_level(brackets[0], score_to_groups, 'H1')
                self._assign_bracket_to_level(brackets[1], score_to_groups, 'H2')
                
                # Check if third bracket should be included
                should_include = self._should_include_bracket(brackets[2], score_to_groups)
                if should_include:
                    self._assign_bracket_to_level(brackets[2], score_to_groups, 'H3')
                    logger.info("✅ Including third bracket as H3")
                else:
                    self._assign_bracket_to_level(brackets[2], score_to_groups, 'EXCLUDED')
                    logger.info("🚫 Excluding third bracket (text too long)")
                
                # Exclude all remaining brackets
                for i in range(3, num_brackets):
                    self._assign_bracket_to_level(brackets[i], score_to_groups, 'EXCLUDED')
                    logger.info(f"🚫 Excluding bracket {i+1} (beyond top 3)")
            
            # Update groups list - only include non-excluded groups
            valid_groups = title_groups + [g for g in non_title_groups if g.level != 'EXCLUDED']
            excluded_count = len([g for g in non_title_groups if g.level == 'EXCLUDED'])
            self.groups = valid_groups
            
            # Log final assignments
            logger.info("=== 🎯 FINAL BRACKET-BASED HIERARCHY ASSIGNMENTS ===")
            level_counts = defaultdict(int)
            for group in self.groups:
                level_counts[group.level] += 1
                if group.elements:
                    sample = group.elements[0].text[:40] + "..." if len(group.elements[0].text) > 40 else group.elements[0].text
                    logger.info(f"✅ {group.level}: '{sample}' (score: {group.get_priority_score():.1f})")
            
            logger.info(f"📊 Level distribution: {dict(level_counts)}")
            logger.info(f"🚫 Excluded {excluded_count} groups using bracket system")
            
        except Exception as e:
            logger.error(f"🚨 Error in bracket-based hierarchy assignment: {e}")
            logger.error(traceback.format_exc())

    def _apply_absolute_exclusion_rule(self, brackets, score_to_groups, max_entries_threshold=40):
        """
        Apply absolute exclusion rule: exclude any bracket that contains more than max_entries_threshold entries.
        
        Args:
            brackets: List of bracket dictionaries
            score_to_groups: Dictionary mapping scores to groups
            max_entries_threshold: Maximum number of entries allowed in a bracket (default: 40)
            
        Returns:
            Updated brackets list with exclusion applied
        """
        try:
            if not brackets:
                logger.info("📊 No brackets to process")
                return brackets
            
            # Calculate entries per bracket (after duplicate removal)
            bracket_entry_counts = []
            total_entries = 0

            for bracket in brackets:
                bracket_entries = 0
                for score in bracket['scores']:
                    for group in score_to_groups[score]:
                        bracket_entries += len(group.elements)
                bracket_entry_counts.append(bracket_entries)
                total_entries += bracket_entries

            logger.info(f"📊 Bracket entry counts (post-duplicate removal): {bracket_entry_counts}")
            
            logger.info(f"📊 Entry analysis (threshold: {max_entries_threshold}):")
            logger.info(f"  - Total entries across all brackets: {total_entries}")
            logger.info(f"  - Entries per bracket: {bracket_entry_counts}")
            
            # Filter brackets based on absolute threshold
            filtered_brackets = []
            excluded_brackets = []
            
            for i, bracket in enumerate(brackets):
                bracket_entries = bracket_entry_counts[i]
                
                if bracket_entries > max_entries_threshold:
                    logger.warning(f"🚫 Bracket {i+1}: {bracket_entries} entries (>{max_entries_threshold}) - EXCLUDING")
                    excluded_brackets.append((i, bracket))
                    
                    # Mark groups in this bracket as excluded
                    for score in bracket['scores']:
                        for group in score_to_groups[score]:
                            group.level = 'EXCLUDED'
                            logger.debug(f"🚫 Excluded group due to absolute rule: score {score:.1f}")
                else:
                    logger.info(f"✅ Bracket {i+1}: {bracket_entries} entries (<={max_entries_threshold}) - KEEPING")
                    filtered_brackets.append(bracket)
            
            # Log results
            if excluded_brackets:
                excluded_indices = [idx for idx, _ in excluded_brackets]
                logger.info(f"📊 Absolute exclusion rule applied:")
                logger.info(f"  - Kept brackets: {len(filtered_brackets)} (indices: {[i for i in range(len(brackets)) if i not in excluded_indices]})")
                logger.info(f"  - Excluded brackets: {len(excluded_brackets)} (indices: {excluded_indices})")
                logger.info(f"  - Threshold: {max_entries_threshold} entries")
            else:
                logger.info(f"✅ All brackets kept - none exceeded {max_entries_threshold} entries")
            
            return filtered_brackets
            
        except Exception as e:
            logger.error(f"🚨 Error applying absolute exclusion rule: {e}")
            logger.error(traceback.format_exc())
            return brackets  # Return original brackets on error
    
    def _create_score_brackets(self, sorted_scores):
        """Create score brackets with 10-point ranges"""
        if not sorted_scores:
            return []
        
        brackets = []
        i = 0
        
        while i < len(sorted_scores):
            # Start a new bracket with the current highest available score
            bracket_start = sorted_scores[i]
            bracket_end = bracket_start - 15
            
            # Find all scores that fall within this bracket
            bracket_scores = []
            while i < len(sorted_scores) and sorted_scores[i] >= bracket_end:
                bracket_scores.append(sorted_scores[i])
                i += 1
            
            brackets.append({
                'range': (bracket_start, bracket_end),
                'scores': bracket_scores
            })
            
            logger.info(f"🎯 Created bracket: {bracket_start:.1f} to {bracket_end:.1f} with {len(bracket_scores)} scores")
        
        return brackets

    def _assign_bracket_to_level(self, bracket, score_to_groups, level):
        """Assign all groups in a bracket to a specific level"""
        for score in bracket['scores']:
            for group in score_to_groups[score]:
                group.level = level
                logger.info(f"📌 Assigned {level} to group with score {score:.1f}")

    def _should_include_bracket(self, bracket, score_to_groups):
        """Check if a bracket should be included based on text length"""
        # Get all groups in this bracket
        bracket_groups = []
        for score in bracket['scores']:
            bracket_groups.extend(score_to_groups[score])
        
        # Apply the same text length logic as before
        return self._should_include_lowest_score(bracket_groups, self.max_text_length_for_lowest)

    def _should_include_lowest_score(self, lowest_score_groups, max_text_length=50):
        """
        Check if lowest score groups should be included based on text length
        
        Args:
            lowest_score_groups: List of groups with the lowest score
            max_text_length: Maximum allowed text length for inclusion (default: 50)
        
        Returns:
            bool: True if should include, False if should exclude
        """
        try:
            # Check all elements in lowest score groups
            for group in lowest_score_groups:
                for element in group.elements:
                    text_length = len(element.text.strip())
                    
                    # If ANY element has text length <= threshold, include the whole score group
                    if text_length <= max_text_length:
                        logger.info(f"📏 Found short text in lowest score: '{element.text.strip()}' (length: {text_length})")
                        return True
                    else:
                        logger.info(f"📏 Long text in lowest score: '{element.text.strip()[:30]}...' (length: {text_length})")
            
            # If no short text found, exclude
            logger.info(f"📏 All texts in lowest score exceed {max_text_length} characters - excluding")
            return False
            
        except Exception as e:
            logger.error(f"Error checking lowest score text length: {e}")
            return False  # Default to excluding on error
    
    def _get_priority_score_for_group(self, group):
        """Calculate priority score for a group consistently"""
        return group.get_priority_score()

    def _remove_duplicate_headings(self):
        """Remove only the specific text elements that appear more than 5 times"""
        try:
            logger.info("🔍 Starting duplicate text detection and removal...")
            
            if not self.groups:
                return
            
            # Step 1: Count all heading texts across all groups
            text_count = defaultdict(list)  # text -> list of (group, element) pairs
            
            for group in self.groups:
                if group.level not in ['TITLE', 'EXCLUDED']:  # Only check actual headings
                    for element in group.elements:
                        # Normalize text for comparison
                        normalized_text = ' '.join(element.text.strip().lower().split())
                        
                        if normalized_text and len(normalized_text) > 2:  # Skip very short texts
                            text_count[normalized_text].append((group, element))
            
            # Step 2: Identify texts that appear more than 5 times
            texts_to_remove = set()
            for normalized_text, group_element_pairs in text_count.items():
                if len(group_element_pairs) > 5:
                    texts_to_remove.add(normalized_text)
                    logger.info(f"🚫 Text appears {len(group_element_pairs)} times (>5): '{normalized_text[:50]}...'")
            
            if not texts_to_remove:
                logger.info("✅ No texts appearing more than 5 times found")
                return
            
            # Step 3: Remove specific elements (not entire groups)
            elements_removed = 0
            groups_modified = 0
            empty_groups = []
            
            for group in self.groups:
                if group.level not in ['TITLE', 'EXCLUDED']:
                    original_count = len(group.elements)
                    
                    # Filter out elements with repeating text
                    new_elements = []
                    for element in group.elements:
                        normalized_text = ' '.join(element.text.strip().lower().split())
                        
                        if normalized_text not in texts_to_remove:
                            new_elements.append(element)
                        else:
                            elements_removed += 1
                            logger.info(f"🗑️  Removed element: '{element.text.strip()[:50]}...' from {group.level}")
                    
                    # Update group with filtered elements
                    group.elements = new_elements
                    
                    if len(group.elements) != original_count:
                        groups_modified += 1
                    
                    # Track groups that became empty
                    if len(group.elements) == 0:
                        empty_groups.append(group)
            
            # Step 4: Remove groups that became completely empty
            if empty_groups:
                original_group_count = len(self.groups)
                self.groups = [group for group in self.groups if group not in empty_groups]
                logger.info(f"🗑️  Removed {len(empty_groups)} groups that became empty after text removal")
                logger.info(f"📊 Groups: {original_group_count} → {len(self.groups)}")
            
            logger.info(f"✅ Duplicate removal complete:")
            logger.info(f"  - Removed {elements_removed} text elements appearing >5 times")
            logger.info(f"  - Modified {groups_modified} groups")
            logger.info(f"  - Removed {len(empty_groups)} empty groups")
            
            # Step 5: Log remaining groups for verification
            remaining_elements = sum(len(group.elements) for group in self.groups if group.level not in ['TITLE', 'EXCLUDED'])
            logger.info(f"📊 Remaining heading elements: {remaining_elements}")
            
        except Exception as e:
            logger.error(f"🚨 Error removing duplicate texts: {e}")
            logger.error(traceback.format_exc())

    def _create_outline(self):
        """Create the outline from grouped elements with hierarchy order correction"""
        try:
            self.outline = []
            
            # Create a set of original indices for title elements for fast lookup
            title_indices = set(element.original_index for element in self.title_elements)
            
            # Define allowed heading levels (H1-H3 as specified)
            allowed_levels = {'H1', 'H2', 'H3', 'H4', 'H5', 'H6'}
            
            logger.info("📝 Creating outline - Current group levels:")
            h1_groups_found = 0
            for group in self.groups:
                if group.level == 'H1':
                    h1_groups_found += 1
                sample_text = ""
                if group.elements:
                    first_elem = min(group.elements, key=lambda e: e.original_index)
                    sample_text = first_elem.text[:30] + "..." if len(first_elem.text) > 30 else first_elem.text
                logger.info(f"  {group.level}: '{sample_text}' (elements: {len(group.elements)})")
            
            if h1_groups_found == 0:
                logger.error("🚨 CRITICAL: No H1 groups found when creating outline!")
            else:
                logger.info(f"✅ Found {h1_groups_found} H1 groups")
            
            # Collect all elements with their levels
            all_elements = []
            
            for group in self.groups:
                # Only include groups with allowed levels
                if group.level in allowed_levels:
                    logger.info(f"✅ Including {group.level} group with {len(group.elements)} elements")
                    for element in group.elements:
                        all_elements.append((element, group.level))
            
            # Sort by page and original index to maintain document order
            all_elements.sort(key=lambda x: (x[0].page, x[0].original_index))
            
            logger.info(f"🔍 Processing {len(all_elements)} potential outline elements")
            
            # Create outline entries
            h1_entries_added = 0
            
            for element, level in all_elements:
                # Skip if this element was used in title construction
                if element.original_index in title_indices:
                    logger.debug(f"⏭️  Skipping title element: '{element.text[:30]}...'")
                    continue
                
                # Skip empty elements
                element_text = element.text.strip()
                if not element_text:
                    continue

                # # Apply classify_string filter
                # if classify_string(element_text):
                self.outline.append({
                    "level": level,
                    "text": element_text,
                    "page": element.page,
                    # "original_level": level  # Store original level for reference
                })
                
                if level == 'H1':
                    h1_entries_added += 1
                    logger.info(f"✅ Added H1: '{element_text[:50]}...' (page {element.page})")
                else:
                    logger.info(f"✅ Added {level}: '{element_text[:50]}...' (page {element.page})")
            
            # CRITICAL VERIFICATION
            if h1_entries_added == 0:
                logger.error("🚨 CRITICAL: No H1 entries added to outline!")
                logger.error("This suggests the hierarchy assignment is still failing.")
            else:
                logger.info(f"✅ SUCCESS: {h1_entries_added} H1 entries added to outline")
            
            # NEW: Apply comprehensive title and hierarchy order correction
            self._correct_title_and_hierarchy_order()
            
            logger.info(f"📊 Final outline contains {len(self.outline)} entries")
            
            # Log summary by level (after correction)
            level_counts = {'H1': 0, 'H2': 0, 'H3': 0, 'H4': 0}
            for entry in self.outline:
                if entry['level'] in level_counts:
                    level_counts[entry['level']] += 1
            
            # Optional: Apply additional quality filtering
            self._filter_outline_by_quality()
            for level, count in level_counts.items():
                logger.info(f"  {level}: {count} entries")
            
        except Exception as e:
            logger.error(f"Error creating outline: {e}")
            logger.error(traceback.format_exc())
            self.outline = []

    def _validate_heading_quality(self, text: str) -> bool:
        """
        Additional validation for heading quality beyond classify_string.
        More strict rules for what constitutes a good heading.
        """
        try:
            text = text.strip()
            if not text:
                return False
            
            # Rule 1: Must contain at least one letter
            if not any(c.isalpha() for c in text):
                logger.debug(f"❌ Heading rejected (no letters): '{text}'")
                return False
            
            # Rule 2: If doesn't start with number, first word must be capitalized
            if not text[0].isdigit():
                words = text.split()
                if words:
                    first_word = words[0].strip(string.punctuation)
                    if first_word and first_word[0].islower():
                        logger.debug(f"❌ Heading rejected (not capitalized): '{text}'")
                        return False
            
            # Rule 3: Reject if mostly numbers and special characters
            total_chars = len(text)
            alpha_chars = sum(1 for c in text if c.isalpha())
            digit_chars = sum(1 for c in text if c.isdigit())
            
            # At least 30% should be letters, unless it starts with a number
            if not text[0].isdigit() and alpha_chars < total_chars * 0.3:
                logger.debug(f"❌ Heading rejected (too few letters): '{text}' ({alpha_chars}/{total_chars})")
                return False
            
            # Rule 4: Reject obvious non-headings
            suspicious_patterns = [
                r'^[0-9\s\-\.\/\(\)]+$',  # Only numbers, spaces, and basic punctuation
                r'^[^\w\s]+$',  # Only special characters
                r'^\s*[\-\=\*\#]{3,}\s*$',  # Lines of dashes, equals, asterisks, etc.
                r'^\s*[\.]{3,}\s*$',  # Multiple dots
            ]
            
            for pattern in suspicious_patterns:
                if re.match(pattern, text):
                    logger.debug(f"❌ Heading rejected (suspicious pattern): '{text}'")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating heading quality for '{text}': {e}")
            return True  # Be permissive on error

    def _filter_outline_by_quality(self):
        """
        Apply additional quality filtering to the final outline.
        Call this after _create_outline() if you want extra strict filtering.
        """
        try:
            if not self.outline:
                return
            
            original_count = len(self.outline)
            filtered_outline = []
            
            for entry in self.outline:
                if self._validate_heading_quality(entry['text']):
                    filtered_outline.append(entry)
                else:
                    logger.info(f"🚫 Filtered out low-quality heading: '{entry['text'][:50]}...'")
            
            self.outline = filtered_outline
            
            if len(self.outline) != original_count:
                logger.info(f"📊 Quality filtering: {original_count} → {len(self.outline)} outline entries")
            
        except Exception as e:
            logger.error(f"Error in quality filtering: {e}")
            logger.error(traceback.format_exc())
    def _correct_hierarchy_order(self):
        """
        Correct hierarchy order to ensure proper sequence: H1 -> H2 -> H3 etc.
        If sequence is violated (e.g., H2, H2, H1), reassign levels (H1, H1, H2)
        """
        try:
            if not self.outline:
                return
            
            logger.info("🔧 Starting hierarchy order correction...")
            
            # Step 1: Get unique levels in document order
            unique_levels = []
            seen_levels = set()
            
            for entry in self.outline:
                level = entry['level']
                if level not in seen_levels:
                    unique_levels.append(level)
                    seen_levels.add(level)
            
            logger.info(f"📊 Current level sequence: {unique_levels}")
            
            # Step 2: Define proper hierarchy order
            proper_order = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
            
            # Step 3: Check if reordering is needed
            needs_correction = False
            for i in range(len(unique_levels)):
                expected_position = proper_order.index(unique_levels[i]) if unique_levels[i] in proper_order else i
                if expected_position != i:
                    needs_correction = True
                    break
            
            if not needs_correction:
                logger.info("✅ Hierarchy order is already correct")
                return
            
            # Step 4: Create level mapping for correction
            level_mapping = {}
            available_levels = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
            
            for i, current_level in enumerate(unique_levels):
                if i < len(available_levels):
                    new_level = available_levels[i]
                    level_mapping[current_level] = new_level
                    logger.info(f"🔄 Mapping {current_level} → {new_level}")
                else:
                    # Fallback for too many levels
                    level_mapping[current_level] = f"H{i+1}"
                    logger.warning(f"⚠️  Mapping {current_level} → H{i+1} (beyond standard levels)")
            
            # Step 5: Apply the mapping
            corrections_made = 0
            for entry in self.outline:
                original_level = entry['level']
                new_level = level_mapping.get(original_level, original_level)
                
                if new_level != original_level:
                    entry['level'] = new_level
                    corrections_made += 1
                    logger.debug(f"🔄 Corrected: '{entry['text'][:30]}...' from {original_level} to {new_level}")
            
            # Step 6: Log final sequence
            final_sequence = []
            seen_final = set()
            for entry in self.outline:
                level = entry['level']
                if level not in seen_final:
                    final_sequence.append(level)
                    seen_final.add(level)
            
            logger.info(f"✅ Hierarchy correction complete:")
            logger.info(f"  - Before: {unique_levels}")
            logger.info(f"  - After:  {final_sequence}")
            logger.info(f"  - Corrections made: {corrections_made}")
            
            # Step 7: Verify the correction worked
            is_valid = self._validate_hierarchy_sequence(final_sequence)
            if is_valid:
                logger.info("✅ Hierarchy sequence is now valid")
            else:
                logger.warning("⚠️  Hierarchy sequence may still have issues")
            
        except Exception as e:
            logger.error(f"🚨 Error in hierarchy order correction: {e}")
            logger.error(traceback.format_exc())

    def _correct_title_and_hierarchy_order(self):
        """
        Comprehensive correction of title and hierarchy order.
        If title appears after any heading in document order, title becomes empty and gets reassigned.
        All subsequent levels shift accordingly.
        """
        try:
            if not self.outline:
                return
            
            logger.info("🔧 Starting comprehensive title and hierarchy order correction...")
            
            # Step 1: Check title position relative to ALL headings (not just in outline)
            title_needs_correction = False
            title_position_info = None
            
            if self.title and self.title_elements:
                # Get the position of title elements in the original document
                title_positions = [elem.original_index for elem in self.title_elements]
                min_title_position = min(title_positions) if title_positions else float('inf')
                
                logger.info(f"📍 Title '{self.title[:50]}...' elements at positions: {title_positions}")
                
                # Check all outline entries (headings) positions
                heading_before_title = []
                
                for entry in self.outline:
                    # Find the original elements that correspond to this outline entry
                    for group in self.groups:
                        if group.level == entry['level']:
                            for element in group.elements:
                                if (element.text.strip() == entry['text'].strip() and 
                                    element.page == entry['page']):
                                    
                                    if element.original_index < min_title_position:
                                        heading_before_title.append({
                                            'text': entry['text'],
                                            'level': entry['level'],
                                            'position': element.original_index,
                                            'page': element.page
                                        })
                                        logger.info(f"📍 Heading '{entry['text'][:50]}...' at position {element.original_index} appears BEFORE title at {min_title_position}")
                
                # If ANY heading appears before title in document order, correction is needed
                if heading_before_title:
                    title_needs_correction = True
                    logger.warning(f"⚠️  Found {len(heading_before_title)} headings appearing before title in document order!")
                    for heading in heading_before_title:
                        logger.warning(f"    - {heading['level']}: '{heading['text'][:50]}...' (pos: {heading['position']})")
            
            # Step 2: Apply title correction if needed
            if title_needs_correction:
                logger.info("🔄 Applying title order correction - clearing title...")
                
                # Clear the title
                original_title = self.title
                self.title = ""
                logger.info(f"🗑️  Cleared title: '{original_title}'")
                
                # Add the former title as the first heading (H1)
                if self.title_elements:
                    # Create a new outline entry for the former title
                    title_element = self.title_elements[0]  # Use first title element
                    new_title_entry = {
                        "level": "H1",  # Former title becomes H1
                        "text": original_title,
                        "page": title_element.page,
                        # "original_level": "TITLE"
                    }
                    
                    # Insert at the beginning of outline
                    self.outline.insert(0, new_title_entry)
                    logger.info(f"📌 Added former title as H1 at beginning: '{original_title[:50]}...'")
            
            # Step 3: Get unique levels in document order (after title correction)
            unique_levels = []
            seen_levels = set()
            
            for entry in self.outline:
                level = entry['level']
                if level not in seen_levels and level != 'TITLE':  # Skip TITLE level
                    unique_levels.append(level)
                    seen_levels.add(level)
            
            logger.info(f"📊 Current level sequence (after title correction): {unique_levels}")
            
            # Step 4: Check if hierarchy correction is needed
            proper_order = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
            needs_hierarchy_correction = False
            
            for i in range(len(unique_levels)):
                if i < len(proper_order):
                    expected_level = proper_order[i]
                    if unique_levels[i] != expected_level:
                        needs_hierarchy_correction = True
                        break
            
            # Step 5: Apply hierarchy correction if needed
            corrections_made = 0
            if needs_hierarchy_correction or title_needs_correction:
                logger.info("🔄 Applying hierarchy level corrections...")
                
                # Create level mapping
                level_mapping = {}
                available_levels = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
                
                for i, current_level in enumerate(unique_levels):
                    if i < len(available_levels):
                        new_level = available_levels[i]
                        level_mapping[current_level] = new_level
                        logger.info(f"🔄 Mapping {current_level} → {new_level}")
                    else:
                        level_mapping[current_level] = f"H{i+1}"
                        logger.warning(f"⚠️  Mapping {current_level} → H{i+1} (beyond standard levels)")
                
                # Apply the mapping to all outline entries
                for entry in self.outline:
                    if entry['level'] in level_mapping:
                        original_level = entry['level']
                        new_level = level_mapping[original_level]
                        
                        if new_level != original_level:
                            entry['level'] = new_level
                            corrections_made += 1
                            logger.debug(f"🔄 Corrected: '{entry['text'][:30]}...' from {original_level} to {new_level}")
            
            # Step 6: Log final results
            final_sequence = []
            seen_final = set()
            for entry in self.outline:
                level = entry['level']
                if level not in seen_final and level != 'TITLE':
                    final_sequence.append(level)
                    seen_final.add(level)
            
            if title_needs_correction or corrections_made > 0:
                logger.info(f"✅ Title and hierarchy correction complete:")
                logger.info(f"  - Title cleared: {title_needs_correction}")
                logger.info(f"  - Final title: '{self.title}' {'(CLEARED)' if not self.title else ''}")
                logger.info(f"  - Hierarchy corrections: {corrections_made}")
                logger.info(f"  - Final sequence: {final_sequence}")
                logger.info(f"  - Total outline entries: {len(self.outline)}")
            else:
                logger.info("✅ No title or hierarchy corrections needed")
            
            # Step 7: Verify the correction worked
            is_valid = self._validate_hierarchy_sequence(final_sequence)
            if is_valid:
                logger.info("✅ Final sequence is valid")
            else:
                logger.warning("⚠️  Final sequence may still have issues")
            
        except Exception as e:
            logger.error(f"🚨 Error in title and hierarchy correction: {e}")
            logger.error(traceback.format_exc())
    def _find_title_in_outline(self):
        """Helper method to find if title text appears in the outline"""
        if not self.title or not self.outline:
            return None
        
        title_text_normalized = self.title.lower().strip()
        
        for i, entry in enumerate(self.outline):
            entry_text_normalized = entry['text'].lower().strip()
            
            # Check for exact match or significant overlap
            if (entry_text_normalized == title_text_normalized or 
                (len(title_text_normalized) > 20 and title_text_normalized in entry_text_normalized) or
                (len(entry_text_normalized) > 20 and entry_text_normalized in title_text_normalized)):
                return i
        
        return None

    def _validate_hierarchy_sequence(self, sequence):
        """Validate that hierarchy sequence is logically correct"""
        try:
            if not sequence:
                return True
            
            proper_order = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
            
            # Check if sequence follows proper order
            last_index = -1
            for level in sequence:
                if level in proper_order:
                    current_index = proper_order.index(level)
                    if current_index <= last_index:
                        return False  # Out of order
                    last_index = current_index
                else:
                    logger.warning(f"⚠️  Unknown level in sequence: {level}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating hierarchy sequence: {e}")
            return False
    def _generate_output(self) -> Dict[str, Any]:
        """Generate the final output dictionary with duplicate verification"""
        try:
            # Final duplicate check on outline
            self._verify_no_duplicates_in_outline()
            
            output = {
                "title": self.title if self.title else "",
                "outline": self.outline,
                "processing_info": {
                    "headers_footers_excluded": len(self.excluded_indices),
                    "title_elements_excluded": len(self.title_elements),
                    "total_outline_entries": len(self.outline),
                    "heading_levels_included": "H1-H3 only"
                }
            }
            
            logger.info("✅ Successfully generated output")
            return output
            
        except Exception as e:
            logger.error(f"Error generating output: {e}")
            return self._create_error_output(f"Failed to generate output: {str(e)}")

    def _verify_no_duplicates_in_outline(self):
        """Final verification - log any remaining duplicates but don't remove them"""
        try:
            seen_texts = defaultdict(int)
            
            for entry in self.outline:
                normalized_text = entry["text"].strip().lower()
                seen_texts[normalized_text] += 1
            
            duplicates = {text: count for text, count in seen_texts.items() if count > 1}
            
            if duplicates:
                logger.info(f"📊 Final outline contains {len(duplicates)} texts with multiple entries:")
                for text, count in duplicates.items():
                    logger.info(f"  - '{text[:50]}...' appears {count} times")
            else:
                logger.info("✅ Final verification: All outline entries are unique")
                
        except Exception as e:
            logger.error(f"Error in duplicate verification: {e}")

    def _create_error_output(self, error_message: str) -> Dict[str, Any]:
        """Create error output format"""
        logger.error(f"Creating error output: {error_message}")
        return {
            "title": "",
            "outline": [],
            "error": error_message
        }

def classify_headings(input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function to classify headings from input data with header/footer detection
    
    Args:
        input_data: List of dictionaries containing text elements
        
    Returns:
        Dictionary with title and outline in specified format (H1-H3 only)
    """
    try:
        classifier = HeadingClassifier()
        result = classifier.process_input(input_data)
        return result
        
    except Exception as e:
        logger.error(f"Fatal error in classify_headings: {e}")
        logger.error(traceback.format_exc())
        return {
            "title": "",
            "outline": [],
            "error": f"Fatal error: {str(e)}"
        }

def save_output(result: Dict[str, Any], filename: str = "output-learn-acrobat-2-exp.json"):
    """
    Save the result to a JSON file
    
    Args:
        result: The classification result
        filename: Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        logger.info(f"Output saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving output: {e}")

# Example usage
if __name__ == "__main__":
    import os
    if os.path.exists("data/TOPJUMP-PARTY-INVITATION-20161003-V01.json"):
        with open("data/TOPJUMP-PARTY-INVITATION-20161003-V01.json", "r", encoding="utf-8") as f:
            sample_data = json.load(f)
        
        result = classify_headings(sample_data)
        print(json.dumps(result, indent=2))
        save_output(result)
    else:
        print("Test data file not found. Run best-support.py for integrated workflow.")
