"""
Document generation, summarization and manipulation utilities.
"""

import os
import re
import logging
import hashlib
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Union, Callable
from datetime import datetime
from collections import defaultdict

# These imports reference the new modular layout
from .extraction import extract_technical_terms, ALL_DOMAIN_TERMS, DOMAIN_SPECIFIC_TERMS

__all__ = [
    "generate_document_summary",
    "create_directory_if_not_exists",
    "generate_file_hash"
]

logger = logging.getLogger(__name__)


def create_directory_if_not_exists(directory: Union[str, Path]) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        directory: Directory path
    """
    if isinstance(directory, str):
        directory = Path(directory)

    directory.mkdir(parents=True, exist_ok=True)


def generate_file_hash(file_path: Union[str, Path]) -> str:
    """
    Generate hash for file content.
    Useful for caching and detecting changes.

    Args:
        file_path: Path to file

    Returns:
        Hash of file content
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        return ""

    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


async def generate_document_summary(
    text: str,
    technical_terms: List[str],
    relationships: List[Union[Dict[str, Any], Any]],  # Handles object or dict
    openai_client = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive document summary with enhanced domain awareness.
    Optimized for technical documentation including Tridium Niagara framework.

    Args:
        text: Document text (markdown content)
        technical_terms: Extracted technical terms
        relationships: Concept relationships (dict or ConceptRelationship objects)
        openai_client: Optional OpenAI client for LLM assistance

    Returns:
        Dictionary with summary information
    """
    import re
    from datetime import datetime
    from collections import defaultdict

    if not text or not technical_terms:
        return {"error": "Insufficient document content for summary generation"}

    # 1. Get term frequencies for importance weighting
    term_frequencies = {}
    for term in technical_terms:
        # Count case-insensitive occurrences
        count = len(re.findall(rf'\b{re.escape(term)}\b', text, re.IGNORECASE))
        term_frequencies[term] = count

    # 2. Identify primary terms (most frequent and central in relationship graph)
    # Calculate centrality in the relationship graph
    centrality = defaultdict(int)
    for rel in relationships:
        # Check if relationship is a dictionary or an object
        if isinstance(rel, dict):
            # Dictionary style
            centrality[rel["source"]] += 1
            centrality[rel["target"]] += 1
        else:
            # Object style (ConceptRelationship)
            centrality[rel.source] += 1
            centrality[rel.target] += 1

    # Identify domain-specific terms in the document
    domain_term_boost = {}
    for term in technical_terms:
        term_lower = term.lower()

        # Check if term is in our domain-specific terms list
        for category, domain_terms in DOMAIN_SPECIFIC_TERMS.items():
            for domain_term in domain_terms:
                if domain_term.lower() in term_lower or term_lower in domain_term.lower():
                    # Different weight for different categories
                    if category in ["niagara", "station", "hierarchy"]:
                        domain_term_boost[term] = 0.4  # Core platform concepts
                    elif category in ["trend", "interval", "visualization"]:
                        domain_term_boost[term] = 0.5  # Visualization concepts
                    elif category in ["component", "node", "programming"]:
                        domain_term_boost[term] = 0.3  # Development concepts
                    else:
                        domain_term_boost[term] = 0.2  # Other domain concepts
                    break

    # Combine frequency, centrality and domain knowledge for importance score
    term_importance = {}
    for term in technical_terms:
        # Base frequency score (logarithmic scaling to prevent domination by frequency)
        freq_score = math.log(1 + term_frequencies.get(term, 0)) / max(1, math.log(1 + max(term_frequencies.values())))

        # Centrality score - how connected the term is
        central_score = centrality.get(term, 0) / max(centrality.values(), default=1) if centrality else 0

        # Domain-specific boost
        domain_boost = domain_term_boost.get(term, 0.0)

        # Combined score with weights
        term_importance[term] = (freq_score * 0.4) + (central_score * 0.3) + domain_boost

    # Get top terms by importance
    primary_terms = sorted(term_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    primary_concepts = [term for term, _ in primary_terms]

    # 3. Identify main topics using section headers
    section_headers = re.findall(r'(?:^|\n)#{1,5}\s+(.+?)(?=\n|$)', text, re.MULTILINE)
    if not section_headers:
        # Try alternative header patterns if markdown headers not found
        section_headers = re.findall(r'(?:^|\n)([A-Z][A-Za-z\s]{2,50})(?:\n|$)', text, re.MULTILINE)

    # 4. Identify key relationship types
    relationship_types = defaultdict(int)
    for rel in relationships:
        if isinstance(rel, dict):
            relationship_types[rel["type"]] += 1
        else:
            # Get type value for ConceptRelationship object
            rel_type = rel.type
            if hasattr(rel_type, 'value'):  # Handle enum types
                rel_type = rel_type.value
            relationship_types[rel_type] += 1

    top_relationship_types = sorted(relationship_types.items(), key=lambda x: x[1], reverse=True)[:5]

    # 5. Identify document type based on key concepts
    document_type = "Technical Document"  # Default

    # Count terms in each domain category
    category_counts = defaultdict(int)
    for term in technical_terms:
        term_lower = term.lower()
        for category, domain_terms in DOMAIN_SPECIFIC_TERMS.items():
            for domain_term in domain_terms:
                if domain_term.lower() in term_lower or term_lower in domain_term.lower():
                    category_counts[category] += 1
                    break

    # Determine primary categories
    if category_counts:
        top_category = max(category_counts.items(), key=lambda x: x[1])[0]

        # Map category to document type
        doc_type_map = {
            "niagara": "Niagara Framework Overview",
            "station": "Station Configuration Guide",
            "hierarchy": "Hierarchy Navigation Documentation",
            "component": "Component Reference Guide",
            "hvac": "HVAC Control Documentation",
            "control": "Control System Design",
            "alarm": "Alarm Management Guide",
            "trend": "Trend Visualization Documentation",
            "interval": "Time Series Configuration Guide",
            "visualization": "Data Visualization Guide",
            "module": "Module Development Guide",
            "node": "Function Block Reference",
            "programming": "Programming Guide",
            "security": "Security Configuration Guide",
            "database": "Database Integration Guide",
            "network": "Network Protocol Guide"
        }

        document_type = doc_type_map.get(top_category, "Technical Document")

    # 6. Generate standard insights about the document
    # Find document title from first significant header or make a sensible title
    title = section_headers[0] if section_headers else document_type

    # Generate key insights about the document
    insights = []

    # Add insight about primary topics
    if primary_concepts:
        primary_terms_text = ", ".join(primary_concepts[:5])
        insights.append(f"Primary concepts: {primary_terms_text}")

    # Add insight about main sections
    if section_headers:
        section_text = ", ".join(section_headers[:5])
        insights.append(f"Main sections: {section_text}")

    # Add insight about relationship types
    if top_relationship_types:
        rel_text = ", ".join([f"{rel_type} ({count})" for rel_type, count in top_relationship_types[:3]])
        insights.append(f"Key relationship types: {rel_text}")

    # Add domain-specific insight based on document type
    if "trend" in document_type.lower() or "visualization" in document_type.lower():
        insights.append("Contains information about visualizing time-series data and configuring trends")

    if "hierarchy" in document_type.lower() or "navigation" in document_type.lower():
        insights.append("Describes the hierarchical structure for organizing building automation components")

    if "node" in document_type.lower() or "function" in document_type.lower():
        insights.append("Documents function blocks and data transformation capabilities")

    # 7. Enhance with LLM if available
    enhanced_title = title
    enhanced_insights = insights.copy()
    description = ""

    if openai_client:
        try:
            # Sample a few paragraphs from the document
            text_samples = []
            paragraphs = text.split('\n\n')

            # Get some paragraphs from beginning, middle, and end
            if len(paragraphs) > 10:
                text_samples = paragraphs[:2]  # Beginning
                text_samples.extend(paragraphs[len(paragraphs)//2:len(paragraphs)//2+2])  # Middle
                text_samples.extend(paragraphs[-2:])  # End
            else:
                text_samples = paragraphs

            sample_text = '\n\n'.join(text_samples)

            # Add domain context to the prompt
            domain_context = """
            This document appears to be technical documentation related to the Tridium Niagara Framework,
            which is a software platform for building automation and IoT systems. Key areas may include:
            - Building automation and controls
            - Hierarchical navigation structure
            - Data visualization and trend analysis
            - Time-series data management
            - Programming and component configuration
            """

            # Prompt for LLM summary generation
            prompt = f"""
            You are analyzing a technical document with the following information:

            {domain_context}

            Key terms: {', '.join(primary_concepts[:15])}
            Main sections: {', '.join(section_headers[:10])}
            Document type: {document_type}

            Sample content:
            {sample_text[:2000]}

            Based on this information, please provide:
            1. A descriptive document title (NOT just "Technical Document")
            2. 3-5 key insights about what this document contains
            3. A concise one-paragraph document description (150-200 words)

            Format your response as JSON with keys: "title", "key_insights" (array), "description" (string)
            """

            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            # Parse the response
            import json
            try:
                result = json.loads(response.choices[0].message.content)

                if "title" in result and result["title"] and result["title"] != "Technical Document":
                    enhanced_title = result["title"]

                if "key_insights" in result and result["key_insights"]:
                    # Combine statistical insights with LLM-generated insights
                    enhanced_insights = result["key_insights"]

                    # Make sure we include primary concepts
                    has_concepts = any("concept" in insight.lower() for insight in enhanced_insights)
                    if not has_concepts and primary_concepts:
                        enhanced_insights.append(f"Primary concepts: {', '.join(primary_concepts[:5])}")

                if "description" in result and result["description"]:
                    description = result["description"]
                    # Add description as an insight if it's not already part of insights
                    if description and not any(description in insight for insight in enhanced_insights):
                        enhanced_insights.insert(0, description)

            except json.JSONDecodeError:
                # Fallback to standard insights if JSON parsing fails
                pass

        except Exception as e:
            import logging
            logging.warning(f"Error using LLM for summary enhancement: {str(e)}")
            # Fallback to standard insights

    # 8. Build the complete summary
    summary = {
        "title": enhanced_title,
        "primary_concepts": primary_concepts,
        "primary_concept_scores": {term: score for term, score in primary_terms},
        "section_structure": section_headers[:10] if len(section_headers) > 10 else section_headers,
        "relationship_types": dict(top_relationship_types),
        "document_type": document_type,
        "total_technical_terms": len(technical_terms),
        "total_relationships": len(relationships),
        "key_insights": enhanced_insights,
        "description": description,  # Add description field
        "domain_context": "Tridium Niagara Framework", # Add explicit domain context
        "generated_at": datetime.utcnow().isoformat()
    }

    return summary
