"""
Utilities for extracting technical terms and relationships from text.
Optimized for technical documentation analysis.
"""

import re
import logging
import json
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from collections import defaultdict
from enum import Enum, auto

__all__ = [
    "extract_technical_terms",
    "extract_technical_terms_regex",
    "extract_technical_terms_spacy",
    "extract_concept_relationships",
    "extract_document_relationships",
    "extract_hierarchy_relationships",
    "find_best_term_match",
    "RelationType",
    "COMMON_TECHNICAL_CATEGORIES"  # Export this for use in other modules
]

logger = logging.getLogger(__name__)

# Try to load spaCy model for better technical term extraction
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    logger.warning("spaCy model not available, falling back to regex patterns for term extraction")

# Common technical terminology categories that can appear in technical documents
# Enhanced with additional domains from utils.py
COMMON_TECHNICAL_CATEGORIES = {
    # Programming and Development
    "programming": ["function", "method", "class", "object", "variable", "parameter", "argument",
                   "api", "interface", "library", "framework", "runtime", "compiler", "interpreter"],

    # Data and Databases
    "data": ["database", "query", "schema", "table", "record", "field", "index", "key", "join",
            "sql", "nosql", "orm", "etl", "data model", "data structure"],

    # Web Technologies
    "web": ["http", "https", "rest", "soap", "api", "endpoint", "request", "response",
           "frontend", "backend", "client", "server", "html", "css", "javascript"],

    # Infrastructure and DevOps
    "infrastructure": ["server", "cloud", "container", "kubernetes", "docker", "vm",
                      "ci/cd", "pipeline", "deployment", "infrastructure", "network"],

    # AI and Machine Learning
    "ai": ["algorithm", "model", "neural network", "training", "inference", "classification",
          "regression", "clustering", "deep learning", "machine learning", "dataset"],

    # Building Automation Systems
    "building_automation": ["hvac", "temperature", "sensor", "controller", "thermostat", "building",
                           "zone", "setpoint", "automation", "bms", "bas", "actuator", "relay"],

    # Niagara Framework Specific
    "niagara": ["niagara", "tridium", "jace", "fox", "workbench", "ax", "n4", "station",
                "bajascript", "baja", "hierarchy", "nav", "wiresheet", "px", "ord"],

    # Station Components
    "station": ["station", "device", "driver", "network", "point", "history", "alarm", "schedule",
               "trend", "config", "component", "service", "extension", "module"],

    # Hierarchy and Navigation
    "hierarchy": ["hierarchy", "folder", "ordFolder", "navTree", "navigation", "slot", "tree",
                 "structure", "organization", "tags", "tagging", "spaces"]
}

# Domain-specific terms used by document processing
# This is maintained for backward compatibility if other code expects it
DOMAIN_SPECIFIC_TERMS = COMMON_TECHNICAL_CATEGORIES

# Common technical document content sections
CONTENT_SECTION_PATTERNS = [
    r"(?i)^#+\s*(introduction|overview|background|context)",
    r"(?i)^#+\s*(architecture|design|structure|system|framework)",
    r"(?i)^#+\s*(implementation|development|code|solution)",
    r"(?i)^#+\s*(api|endpoint|method|function|parameter)",
    r"(?i)^#+\s*(database|storage|data model|schema)",
    r"(?i)^#+\s*(usage|examples|tutorial|how to|guide)",
    r"(?i)^#+\s*(installation|setup|configuration|deployment)",
    r"(?i)^#+\s*(limitations|constraints|challenges|issues)",
    r"(?i)^#+\s*(security|authentication|authorization)",
    r"(?i)^#+\s*(testing|validation|verification)",
    r"(?i)^#+\s*(performance|optimization|scalability|efficiency)",
    r"(?i)^#+\s*(future work|roadmap|todo|planned features)"
]

class RelationType(str, Enum):
    """Relationship types between technical concepts."""
    PART_OF = "part_of"           # Component is part of a larger system
    USES = "uses"                 # One component uses/depends on another
    IMPLEMENTS = "implements"     # Component implements an interface/abstract concept
    EXTENDS = "extends"           # Component extends/inherits from another
    RELATES_TO = "relates_to"     # General relationship between components
    CONFIGURES = "configures"     # One component configures another
    PRECEDES = "precedes"         # Sequential/process relationship
    ALTERNATIVE = "alternative"   # Component is an alternative to another
    INCOMPATIBLE = "incompatible" # Components cannot be used together
    UNKNOWN = "unknown"           # Relationship type couldn't be determined

def extract_technical_terms(text: str) -> List[str]:
    """
    Extract technical terms from text with domain awareness.
    Uses multiple methods and combines results for better coverage.

    Args:
        text: Text to extract terms from

    Returns:
        List of filtered technical terms
    """
    if not text:
        return []

    # First try domain-specific extraction
    terms = set()

    # Use regex-based extraction (works without dependencies)
    regex_terms = extract_technical_terms_regex(text)
    terms.update(regex_terms)

    # Use spaCy if available
    if SPACY_AVAILABLE:
        spacy_terms = extract_technical_terms_spacy(text, nlp)
        terms.update(spacy_terms)

    # Filter duplicates and normalize
    filtered_terms = []
    seen = set()

    for term in terms:
        normalized = term.lower().strip()
        if normalized and len(normalized) > 2 and normalized not in seen:
            filtered_terms.append(normalized)
            seen.add(normalized)

    return filtered_terms

def extract_technical_terms_regex(text: str) -> List[str]:
    """
    Extract technical terms using regex patterns optimized for technical docs.

    Args:
        text: Text to extract terms from

    Returns:
        List of filtered technical terms
    """
    if not text:
        return []

    # Common technical terms patterns
    patterns = [
        # Camel case (e.g., useState, DataFrame)
        r'(?<![A-Z])[A-Z][a-z]+(?:[A-Z][a-z]+)+',

        # Snake case variables and functions (e.g., process_data, extract_features)
        r'\b[a-z]+(?:_[a-z]+){1,}\b',

        # Acronyms and initialisms (e.g., API, HTTP, JWT)
        r'\b[A-Z]{2,}\b',

        # Library and package names (e.g., tensorflow, scikit-learn)
        r'\b[a-z]+(?:-[a-z]+)+\b',

        # Function calls and method references
        r'\b[a-zA-Z_][a-zA-Z0-9_]*\(\)',

        # Class names and types (PascalCase)
        r'\b[A-Z][a-zA-Z0-9]*\b'
    ]

    # Extract terms using patterns
    terms = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        terms.update(matches)

    # Extract from common categories
    for category, category_terms in COMMON_TECHNICAL_CATEGORIES.items():
        for term in category_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text.lower()):
                terms.add(term)

    # Clean up and filter terms
    stop_words = {"the", "and", "this", "that", "with", "from", "have", "has", "had", "not", "are", "were"}
    filtered_terms = []

    for term in terms:
        # Remove trailing parentheses from function patterns
        clean_term = term.replace('()', '')

        # Skip short terms and stop words
        if len(clean_term) < 3 or clean_term.lower() in stop_words:
            continue

        filtered_terms.append(clean_term)

    return filtered_terms

def extract_technical_terms_spacy(text: str, nlp) -> List[str]:
    """
    Extract technical terms using spaCy NER and pattern matching.

    Args:
        text: Text to extract terms from
        nlp: spaCy nlp model

    Returns:
        List of filtered technical terms
    """
    if not text or not nlp:
        return []

    # Process the text with spaCy
    doc = nlp(text)

    # Extract terms based on POS patterns and named entities
    terms = set()

    # Add named entities
    for ent in doc.ents:
        if ent.label_ in {"ORG", "PRODUCT", "WORK_OF_ART", "EVENT"}:
            terms.add(ent.text)

    # Add noun chunks (potential technical terms)
    for chunk in doc.noun_chunks:
        # Extract compound nouns and technical adjective-noun pairs
        if chunk.root.dep_ in {"nsubj", "dobj", "pobj"}:
            # Check if it contains technical-looking words
            if any(token.is_alpha and len(token.text) > 2 and not token.is_stop for token in chunk):
                terms.add(chunk.text)

    # Extract terms based on specific POS patterns
    for token in doc:
        # Get compound terms
        if token.dep_ == "compound" and token.head.pos_ in {"NOUN", "PROPN"}:
            compound_term = token.text + " " + token.head.text
            terms.add(compound_term)

        # Get proper nouns that might be technical terms
        if token.pos_ == "PROPN" and not token.is_stop:
            terms.add(token.text)

    # Filter and clean
    filtered_terms = []
    for term in terms:
        clean_term = re.sub(r'\s+', ' ', term).strip()
        if clean_term and len(clean_term) > 2:
            filtered_terms.append(clean_term.lower())

    return filtered_terms

def find_best_term_match(text: str, terms: Set[str]) -> Optional[str]:
    """
    Find the best matching technical term for a given text.
    Performs fuzzy matching to handle variations in technical terms.

    Args:
        text: The text to match against terms
        terms: Set of known technical terms

    Returns:
        Best matching term or None if no match
    """
    if not text or not terms:
        return None

    # Exact match
    text_lower = text.lower()
    for term in terms:
        if term.lower() == text_lower:
            return term

    # Contained match (text is contained in a term)
    for term in terms:
        if text_lower in term.lower():
            return term

    # Contained match (term is contained in text)
    matching_terms = []
    for term in terms:
        if term.lower() in text_lower:
            matching_terms.append((term, len(term)))

    # Return the longest matching term if found
    if matching_terms:
        return sorted(matching_terms, key=lambda x: x[1], reverse=True)[0][0]

    # No good match found
    return None

def extract_concept_relationships(
    text: str,
    known_concepts: Optional[Set[str]] = None,
    min_confidence: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Extract relationships between concepts in text with configurable confidence.

    Args:
        text: Text to analyze
        known_concepts: Optional set of known concepts to look for
        min_confidence: Minimum confidence threshold for relationships

    Returns:
        List of concept relationships
    """
    # Extract technical terms if known_concepts not provided
    if not known_concepts:
        technical_terms = extract_technical_terms(text)
    else:
        technical_terms = list(known_concepts)

    # Use document relationship extraction
    return extract_document_relationships(text, technical_terms, min_confidence)

def extract_document_relationships(
    text: str,
    technical_terms: List[str],
    min_confidence: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Extract relationships between technical terms.

    Args:
        text: Document text to analyze
        technical_terms: List of identified technical terms in the document
        min_confidence: Minimum confidence threshold for relationships

    Returns:
        List of concept relationships
    """
    if not text or not technical_terms:
        return []

    # Relationship patterns and their confidence scores
    relationship_patterns = [
        # "X is a Y" pattern (strong hierarchical)
        (r'(?i)\b(%s)\s+is\s+(?:a|an)\s+(%s)\b', 0.9),

        # "X is part of Y" pattern (strong compositional)
        (r'(?i)\b(%s)\s+is\s+(?:part|component)\s+of\s+(%s)\b', 0.9),

        # "X contains Y" pattern (compositional)
        (r'(?i)\b(%s)\s+contains\s+(%s)\b', 0.8),

        # "X consists of Y" pattern (compositional)
        (r'(?i)\b(%s)\s+consists\s+of\s+(%s)\b', 0.8),

        # "X has Y" pattern (moderate compositional)
        (r'(?i)\b(%s)\s+has\s+(?:a|an)?\s*(%s)\b', 0.7),

        # "Y in X" pattern (moderate associative)
        (r'(?i)\b(%s)\s+in\s+(?:the|a|an)?\s*(%s)\b', 0.6),

        # "Y of X" pattern (weak associative)
        (r'(?i)\b(%s)\s+of\s+(?:the|a|an)?\s*(%s)\b', 0.5),

        # "X and Y" pattern (very weak associative)
        (r'(?i)\b(%s)\s+and\s+(%s)\b', 0.3)
    ]

    relationships = []

    # Escape special regex characters in technical terms
    escaped_terms = [re.escape(term) for term in technical_terms]

    # Generate combinations of terms to search for relationships
    for i, term1 in enumerate(technical_terms):
        term1_escaped = escaped_terms[i]

        for j, term2 in enumerate(technical_terms):
            if i == j:
                continue

            term2_escaped = escaped_terms[j]

            # Check for relationships using patterns
            for pattern_template, confidence in relationship_patterns:
                # Create pattern with the current terms
                pattern = pattern_template % (term1_escaped, term2_escaped)
                matches = re.findall(pattern, text)

                if matches:
                    # Extract context surrounding the match
                    for match in matches:
                        # Find match position
                        match_text = f"{term1} {match} {term2}"
                        match_pos = text.lower().find(match_text.lower())

                        if match_pos >= 0:
                            # Extract context (50 chars before and after)
                            start = max(0, match_pos - 50)
                            end = min(len(text), match_pos + len(match_text) + 50)
                            context = text[start:end]

                            # Add relationship if confidence is high enough
                            if confidence >= min_confidence:
                                relationship = {
                                    "source": term1,
                                    "target": term2,
                                    "confidence": confidence,
                                    "context": context.strip(),
                                    "relationship_type": _infer_relationship_type(pattern_template)
                                }
                                relationships.append(relationship)

    return relationships

def _infer_relationship_type(pattern_template: str) -> str:
    """Infer relationship type from pattern template."""
    if "is a" in pattern_template:
        return "is_a"
    elif "part of" in pattern_template or "component of" in pattern_template:
        return "part_of"
    elif "contains" in pattern_template or "consists of" in pattern_template:
        return "contains"
    elif "has" in pattern_template:
        return "has"
    elif "in" in pattern_template:
        return "in"
    elif "of" in pattern_template:
        return "of"
    elif "and" in pattern_template:
        return "associated_with"
    else:
        return "related_to"

def extract_hierarchy_relationships(
    text: str,
    technical_terms: List[str],
    min_confidence: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Extract hierarchical relationships between technical terms.

    Args:
        text: Document text to analyze
        technical_terms: List of identified technical terms in the document
        min_confidence: Minimum confidence threshold for relationships

    Returns:
        List of hierarchical relationships
    """
    hierarchy_patterns = [
        # Parent-child patterns
        (r"{child}.*is\s+a\s+(?:type\s+of|kind\s+of|subclass\s+of).*{parent}", 0.85, RelationType.EXTENDS),
        (r"{child}.*(?:inherits|extends|derives).*from.*{parent}", 0.9, RelationType.EXTENDS),
        (r"{child}.*(?:is\s+part\s+of|belongs\s+to|included\s+in).*{parent}", 0.85, RelationType.PART_OF),
        (r"{parent}.*contains.*{child}", 0.8, RelationType.PART_OF),
        (r"{parent}.*consists\s+of.*{child}", 0.85, RelationType.PART_OF),
        # Dependency patterns
        (r"{dependent}.*(?:depends\s+on|requires|needs).*{dependency}", 0.8, RelationType.USES),
        (r"{dependent}.*(?:calls|invokes|utilizes).*{dependency}", 0.75, RelationType.USES),
        # Implementation patterns
        (r"{implementer}.*implements.*{interface}", 0.9, RelationType.IMPLEMENTS),
        (r"{implementer}.*provides\s+(?:an\s+)?implementation\s+(?:of|for).*{interface}", 0.85, RelationType.IMPLEMENTS),
    ]

    hierarchical_relationships = []
    terms_set = set(technical_terms)

    for term1 in technical_terms:
        for term2 in technical_terms:
            if term1 == term2:
                continue

            for pattern_template, confidence, rel_type in hierarchy_patterns:
                # Try both directions for each term pair
                pattern1 = pattern_template.replace("{parent}", re.escape(term1)).replace("{child}", re.escape(term2))
                pattern1 = pattern_template.replace("{interface}", re.escape(term1)).replace("{implementer}", re.escape(term2))
                pattern1 = pattern_template.replace("{dependency}", re.escape(term1)).replace("{dependent}", re.escape(term2))

                pattern2 = pattern_template.replace("{parent}", re.escape(term2)).replace("{child}", re.escape(term1))
                pattern2 = pattern_template.replace("{interface}", re.escape(term2)).replace("{implementer}", re.escape(term1))
                pattern2 = pattern_template.replace("{dependency}", re.escape(term2)).replace("{dependent}", re.escape(term1))

                # Check first direction
                if re.search(pattern1, text, re.IGNORECASE):
                    hierarchical_relationships.append({
                        "source": term2,
                        "target": term1,
                        "type": rel_type,
                        "confidence": confidence
                    })

                # Check second direction
                if re.search(pattern2, text, re.IGNORECASE):
                    hierarchical_relationships.append({
                        "source": term1,
                        "target": term2,
                        "type": rel_type,
                        "confidence": confidence
                    })

    # Filter by confidence threshold
    return [rel for rel in hierarchical_relationships if rel["confidence"] >= min_confidence]

def extract_procedures_and_parameters(text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract procedures and parameters from text content.

    Args:
        text: Document text to analyze

    Returns:
        Tuple of (procedures, parameters) lists
    """
    procedures = []
    parameters = []

    # Extract procedures first
    procedure_blocks = _extract_procedure_blocks(text)

    # Process each procedure block to extract steps and parameters
    for proc_index, proc_block in enumerate(procedure_blocks):
        # Basic procedure metadata
        procedure = {
            "procedure_id": f"procedure_{proc_index}",
            "title": proc_block.get("title", f"Procedure {proc_index + 1}"),
            "content": proc_block.get("content", ""),
            "page": proc_block.get("page", 0),
            "steps": [],
            "parameters": []
        }

        # Extract steps for this procedure
        steps = _extract_steps(proc_block.get("content", ""))
        procedure["steps"] = steps

        # Extract parameters from the procedure
        proc_parameters = _extract_parameters_from_text(proc_block.get("content", ""))
        procedure["parameters"] = [param["name"] for param in proc_parameters]

        # Add the procedure to our list
        procedures.append(procedure)

        # Add extracted parameters to our parameter list
        for param in proc_parameters:
            # Enrich with procedure context
            param["procedure_id"] = procedure["procedure_id"]
            param["procedure_title"] = procedure["title"]
            param["page"] = proc_block.get("page", 0)
            parameters.append(param)

    # Extract standalone parameters (not in procedures)
    standalone_params = _extract_parameters_from_text(text)

    # Filter out parameters already found in procedures
    proc_param_names = {param["name"] for param in parameters}
    standalone_params = [param for param in standalone_params if param["name"] not in proc_param_names]

    # Add standalone parameters to our list
    parameters.extend(standalone_params)

    return procedures, parameters

def _extract_procedure_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Extract procedure blocks from text based on headers and content patterns.

    Args:
        text: Document text

    Returns:
        List of procedure blocks with title, content, and page info
    """
    procedure_blocks = []

    # Pattern to identify procedure headers
    procedure_header_patterns = [
        r"(?:^|\n)(?:#+\s+)(.*?procedure.*?)(?:\n)",
        r"(?:^|\n)(?:#+\s+)(.*?step.*?by.*?step.*?)(?:\n)",
        r"(?:^|\n)(?:#+\s+)(.*?instructions.*?)(?:\n)",
        r"(?:^|\n)(?:#+\s+)(.*?how\s+to.*?)(?:\n)",
        r"(?:^|\n)(?:#+\s+)(.*?guide.*?)(?:\n)"
    ]

    # Look for procedure headers and extract the following content
    for pattern in procedure_header_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            header = match.group(1).strip()
            start_pos = match.end()

            # Find the end of this section (next header or end of text)
            next_header_match = re.search(r"\n#+\s+", text[start_pos:])
            if next_header_match:
                end_pos = start_pos + next_header_match.start()
            else:
                end_pos = len(text)

            # Extract the content
            content = text[start_pos:end_pos].strip()

            # Estimate page number (simple approach - could be improved)
            page = _estimate_page_from_position(text, start_pos)

            # Add this procedure block
            procedure_blocks.append({
                "title": header,
                "content": content,
                "page": page
            })

    # If no procedure headers found, look for numbered steps sections
    if not procedure_blocks:
        # Pattern for sequences of numbered steps
        step_sequences = _find_step_sequences(text)

        for i, (start_pos, end_pos, steps_text) in enumerate(step_sequences):
            # Try to find a title before the steps
            title_match = re.search(r"(?:^|\n)([^\n]+)(?:\n+)", text[:start_pos][-200:])
            title = title_match.group(1).strip() if title_match else f"Procedure {i+1}"

            # Estimate page
            page = _estimate_page_from_position(text, start_pos)

            procedure_blocks.append({
                "title": title,
                "content": steps_text,
                "page": page
            })

    return procedure_blocks

def _find_step_sequences(text: str) -> List[Tuple[int, int, str]]:
    """
    Find sequences of numbered steps in text.

    Args:
        text: Document text

    Returns:
        List of tuples with (start_position, end_position, steps_text)
    """
    # Patterns for numbered steps
    step_patterns = [
        r"(?:^|\n)(?:\s*)((?:\d+\.\s+.+(?:\n|$)){3,})",     # 1. Step one\n 2. Step two\n
        r"(?:^|\n)(?:\s*)((?:Step\s+\d+\.\s+.+(?:\n|$)){3,})",  # Step 1. Do this\n Step 2. Do that\n
        r"(?:^|\n)(?:\s*)((?:\d+\)\s+.+(?:\n|$)){3,})"      # 1) Step one\n 2) Step two\n
    ]

    sequences = []

    for pattern in step_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE)

        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            steps_text = match.group(1)
            sequences.append((start_pos, end_pos, steps_text))

    # Sort by position in the document
    sequences.sort(key=lambda x: x[0])

    return sequences

def _extract_steps(text: str) -> List[Dict[str, Any]]:
    """
    Extract individual steps from procedure text.

    Args:
        text: Procedure text

    Returns:
        List of step dictionaries
    """
    steps = []

    # Patterns for step detection
    step_patterns = [
        r"(?:^|\n)(?:\s*)(\d+)\.\s+(.+)(?:\n|$)",     # 1. Step one
        r"(?:^|\n)(?:\s*)Step\s+(\d+)\.\s+(.+)(?:\n|$)",  # Step 1. Do this
        r"(?:^|\n)(?:\s*)(\d+)\)\s+(.+)(?:\n|$)",     # 1) Step one
        r"(?:^|\n)(?:\s*)Step\s+(\d+)(?:\s*)?:(?:\s*)(.+)(?:\n|$)"  # Step 1: Do this
    ]

    # Try each pattern
    for pattern in step_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE)

        for match in matches:
            step_num = match.group(1)
            step_text = match.group(2).strip()

            # Skip empty steps
            if not step_text:
                continue

            # Find warnings or cautions
            warnings = []
            warning_match = re.search(r"(?:warning|caution|note|important):?\s*([^\n]+)", step_text, re.IGNORECASE)
            if warning_match:
                warnings.append(warning_match.group(1).strip())

            # Extract parameters from step
            parameters = _extract_parameters_from_text(step_text)

            steps.append({
                "step_number": int(step_num),
                "content": step_text,
                "warnings": warnings,
                "parameters": [p["name"] for p in parameters]
            })

    # Sort steps by number
    steps.sort(key=lambda x: x["step_number"])

    return steps

def _extract_parameters_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract parameters and their values from text.

    Args:
        text: Text to analyze

    Returns:
        List of parameter dictionaries
    """
    parameters = []
    seen_names = set()

    # Patterns for parameter extraction
    param_patterns = [
        # Parameter: Value format
        r"(?:^|\n)(?:\s*)([A-Za-z][A-Za-z0-9_\s-]+?)(?:\s*):(?:\s*)(.+?)(?:\n|$)",

        # Parameter = Value format
        r"(?:^|\n)(?:\s*)([A-Za-z][A-Za-z0-9_\s-]+?)(?:\s*)=(?:\s*)(.+?)(?:\n|$)",

        # Set Parameter to Value format
        r"(?:Set|Configure)\s+([A-Za-z][A-Za-z0-9_\s-]+?)\s+to\s+(.+?)(?:\.|,|\n|$)",

        # Technical parameters in parentheses
        r"([A-Za-z][A-Za-z0-9_\s-]+?)\s*\(([^)]+)\)"
    ]

    for pattern in param_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE)

        for match in matches:
            name = match.group(1).strip()
            value = match.group(2).strip()

            # Skip too short parameter names or common words
            if len(name) < 3 or name.lower() in {"the", "and", "set", "this", "that", "with", "from", "note"}:
                continue

            # Skip if already found
            if name.lower() in seen_names:
                continue

            # Skip if name is too long (likely not a parameter)
            if len(name) > 50:
                continue

            # Try to determine parameter type
            param_type = _determine_parameter_type(value)

            parameters.append({
                "name": name,
                "value": value,
                "type": param_type,
                "description": _generate_parameter_description(name, value, param_type)
            })

            seen_names.add(name.lower())

    return parameters

def _determine_parameter_type(value: str) -> str:
    """
    Determine parameter type from its value.

    Args:
        value: Parameter value

    Returns:
        Parameter type string
    """
    # Check for numeric types
    if re.match(r"^\d+$", value):
        return "integer"
    elif re.match(r"^\d+\.\d+$", value):
        return "float"

    # Check for boolean
    if value.lower() in {"true", "false", "yes", "no", "on", "off"}:
        return "boolean"

    # Check for units
    if re.search(r"\d+\s*(?:px|em|cm|mm|m|kg|s|ms|hz|db|[°%])", value, re.IGNORECASE):
        return "measurement"

    # Default to string
    return "string"

def _generate_parameter_description(name: str, value: str, param_type: str) -> str:
    """
    Generate a standardized description for a parameter.

    Args:
        name: Parameter name
        value: Parameter value
        param_type: Parameter type

    Returns:
        Parameter description
    """
    # Basic description template
    description = f"{name} with value {value}"

    # Add type-specific details
    if param_type == "integer":
        description += f" (integer value)"
    elif param_type == "float":
        description += f" (decimal value)"
    elif param_type == "boolean":
        description += f" (boolean setting)"
    elif param_type == "measurement":
        match = re.search(r"(\d+\.?\d*)\s*([a-zA-Z°%]+)", value)
        if match:
            unit = match.group(2)
            description += f" (measurement in {unit})"

    return description

def _estimate_page_from_position(text: str, position: int) -> int:
    """
    Estimate page number from position in text.
    Looks for page markers near the position.

    Args:
        text: Full document text
        position: Character position in text

    Returns:
        Estimated page number
    """
    # Look for page number indicators near the position
    window_start = max(0, position - 1000)
    window_end = min(len(text), position + 1000)
    window = text[window_start:window_end]

    # Try different page number patterns
    page_patterns = [
        r"(?:Page|PAGE)[\s]*(\d+)",
        r"(?:Pg\.|pg\.)[\s]*(\d+)",
        r"-\s*(\d+)\s*-",  # Center page numbers like "- 42 -"
        r"\[Page\s*(\d+)\]",  # [Page 42] format
        r"\(Page\s*(\d+)\)",  # (Page 42) format
        r"(?:Page|PAGE)[\s]*(\d+)[\s]*(?:of|OF)[\s]*\d+"  # "Page X of Y" format
    ]

    for pattern in page_patterns:
        match = re.search(pattern, window)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue

    # If no page number found, estimate based on position in document
    # Assume average page is ~3000 characters
    estimated_page = position // 3000 + 1
    return estimated_page
