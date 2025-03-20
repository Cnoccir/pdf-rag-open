"""
Response generator node for LangGraph-based PDF RAG system.
Generates comprehensive responses using retrieved content and knowledge synthesis.
Supports both single-document and research modes.
"""

import logging
import os
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from openai import OpenAI

from app.chat.langgraph.state import GraphState, GenerationState

logger = logging.getLogger(__name__)

# Standard response generation prompt template
RESPONSE_GENERATION_PROMPT = """
You are an AI assistant specialized in answering queries about technical documents.
Generate a comprehensive response to the query based on the retrieved content.

Query: {query}

Retrieved Content:
{content}

Knowledge Synthesis:
{synthesis}

Your task:
1. Answer the query directly and accurately based on the retrieved content
2. Include relevant technical details and explanations
3. Cite specific sources for key information using [1], [2], etc. notation
4. Present a coherent and well-structured response
5. Acknowledge any limitations or gaps in the information

Use the knowledge synthesis to guide your response, but focus on addressing the query directly.
Make citations specific and relevant - cite the exact source where a piece of information comes from.
"""

# Response generation prompt template for procedures
PROCEDURE_PROMPT = """
You are an AI assistant specialized in explaining technical procedures and steps.
Generate a comprehensive response to the query based on the retrieved procedures.

Query: {query}

Retrieved Procedures:
{procedures}

Parameters and Settings:
{parameters}

Your task:
1. Present a clear, step-by-step explanation of the procedure
2. Format steps in a numbered list for clarity
3. Explain the purpose of each step when possible
4. Highlight important warnings, cautions, or notes
5. Explain any parameters or settings that are relevant to the procedure
6. Cite specific sources for key information using [1], [2], etc. notation

If multiple procedures are available, choose the most relevant one or synthesize the information.
Make sure to present the steps in a logical order and explain any technical terms.
"""

# Response generation prompt template for research mode
RESEARCH_MODE_PROMPT = """
You are an AI assistant specializing in cross-document research and analysis.
Generate a comprehensive response combining insights from multiple documents.

Query: {query}

Retrieved Content:
{content}

Cross-Document Insights:
{insights}

Document Information:
{document_info}

Your task:
1. Answer the query by synthesizing information across all documents
2. Highlight similarities, differences, and complementary information between documents
3. Cite specific sources using [1], [2], etc. notation, mentioning document names
4. Organize your response in a clear, structured manner
5. Identify any gaps or contradictions between the documents

Make sure to explicitly mention when different documents provide different perspectives on the same topic.
For each key point, indicate which document(s) the information comes from.
If documents contradict each other, present both perspectives fairly.
"""

def generate_response(state: GraphState) -> GraphState:
    """
    Generate the final response based on retrieved content and knowledge synthesis.
    Enhanced to handle multi-level chunks and procedure-specific results.
    Supports both single-document and research modes.

    Args:
        state: Current graph state containing retrieval results and knowledge synthesis

    Returns:
        Updated graph state with generated response
    """
    # Validate required states
    if not state.retrieval_state or not state.query_state:
        logger.warning("Missing retrieval or query state for response generation")
        error_msg = "Missing required information to provide a response"

        state.generation_state = GenerationState(
            response="I'm sorry, but I don't have enough information to provide a response.",
            citations=[],
            metadata={"error": error_msg}
        )
        return state

    # Check if we have elements to work with
    has_elements = state.retrieval_state.elements and len(state.retrieval_state.elements) > 0

    if not has_elements:
        logger.warning("No elements retrieved, generating fallback response")

        # Check for specific database errors
        error_msg = ""
        if state.retrieval_state.metadata and "error" in state.retrieval_state.metadata:
            error_msg = state.retrieval_state.metadata["error"]

        # Generate appropriate fallback response
        if "MongoDB" in error_msg or "Qdrant" in error_msg:
            fallback_msg = ("I'm having trouble connecting to the document database. "
                           "This might be due to a temporary issue. "
                           "Please try again in a few moments.")
        else:
            # Generic fallback for when retrieval simply found nothing relevant
            fallback_msg = ("I couldn't find specific information in the document to answer your query. "
                           "Could you please rephrase your question or provide more details about what you're looking for?")

        state.generation_state = GenerationState(
            response=fallback_msg,
            citations=[],
            metadata={"fallback": True, "reason": "no_results"}
        )
        return state

    logger.info(f"Generating response for query: {state.query_state.query[:50]}...")

    try:
        # Determine if we're in research mode
        is_research_mode = False
        if state.query_state.pdf_ids and len(state.query_state.pdf_ids) > 1:
            is_research_mode = True
            logger.info("Using research mode for response generation")
        elif state.retrieval_state.metadata and state.retrieval_state.metadata.get("research_mode"):
            is_research_mode = True
            logger.info("Research mode detected in retrieval metadata")

        # Determine if this is a procedure-focused query
        is_procedure_query = state.query_state.procedure_focused

        # Check if we have procedure results
        has_procedures = (hasattr(state.retrieval_state, 'procedures_retrieved') and
                         state.retrieval_state.procedures_retrieved and
                         len(state.retrieval_state.procedures_retrieved) > 0)

        # Prioritize procedure handling if applicable
        if is_procedure_query or has_procedures:
            logger.info("Using procedure-focused response generation")
            return generate_procedure_response(state)

        # Format retrieved content for the prompt
        content_str, citation_map = format_retrieved_content(state.retrieval_state.elements)

        # Format knowledge synthesis if available
        synthesis_str = ""
        if state.research_state and hasattr(state.research_state, 'research_context'):
            synthesis_str = format_knowledge_synthesis(state.research_state.research_context)

        # Format research insights if available
        insights_str = ""
        if is_research_mode and state.research_state:
            insights_str = format_research_insights(state.research_state)

        # Initialize OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Choose appropriate prompt based on mode
        if is_research_mode:
            # Get document information for research mode
            document_info = format_document_information(state)

            prompt = RESEARCH_MODE_PROMPT.format(
                query=state.query_state.query,
                content=content_str,
                insights=insights_str or "No specific cross-document insights available.",
                document_info=document_info
            )
        else:
            prompt = RESPONSE_GENERATION_PROMPT.format(
                query=state.query_state.query,
                content=content_str,
                synthesis=synthesis_str or "No knowledge synthesis available."
            )

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for improved response quality
            messages=[
                {"role": "system", "content": "You are an expert assistant for technical documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        # Get the response text
        response_text = response.choices[0].message.content

        # Extract citations from the response
        citations = extract_citations(response_text, citation_map)

        # Create generation state
        state.generation_state = GenerationState(
            response=response_text,
            citations=citations,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4",
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "research_mode": is_research_mode,
                "document_count": len(state.query_state.pdf_ids) if state.query_state.pdf_ids else 1,
                "chunk_levels_used": state.retrieval_state.chunk_levels_used if hasattr(state.retrieval_state, 'chunk_levels_used') else [],
                "embedding_types_used": state.retrieval_state.embedding_types_used if hasattr(state.retrieval_state, 'embedding_types_used') else []
            },
            token_usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )

        logger.info(
            f"Response generation complete with {len(citations)} citations, "
            f"{response.usage.total_tokens} total tokens"
        )

        return state

    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}", exc_info=True)

        # Generate error response
        error_response = "I'm sorry, but I encountered an error while generating a response. Please try again or rephrase your query."

        # Create error state
        state.generation_state = GenerationState(
            response=error_response,
            citations=[],
            metadata={"error": str(e)}
        )

        return state

def generate_procedure_response(state: GraphState) -> GraphState:
    """
    Generate a procedure-focused response based on the retrieved procedures.

    Args:
        state: Current graph state containing retrieval results

    Returns:
        Updated graph state with generated procedure response
    """
    try:
        # Get procedure results and parameter results
        procedures = state.retrieval_state.procedures_retrieved if hasattr(state.retrieval_state, 'procedures_retrieved') else []
        parameters = state.retrieval_state.parameters_retrieved if hasattr(state.retrieval_state, 'parameters_retrieved') else []

        # Format procedures for the prompt
        procedures_str = ""
        citation_map = {}

        for i, proc in enumerate(procedures):
            # Create citation ID
            citation_id = f"[{i+1}]"

            # Get metadata
            metadata = proc.get("metadata", {})

            # Add to citation map
            citation_map[citation_id] = {
                "id": proc.get("id", "") or proc.get("element_id", f"proc_{i}") or proc.get("procedure_id", f"proc_{i}"),
                "type": "procedure",
                "pdf_id": proc.get("pdf_id", ""),
                "page": proc.get("page", 0),
                "document_title": metadata.get("document_title", f"Document {proc.get('pdf_id', '')}")
            }

            # Format procedure with citation marker
            procedures_str += f"\n--- Procedure {citation_id} ---\n"
            procedures_str += proc.get("content", "")

            # Add steps if available
            if "steps" in proc and proc["steps"]:
                procedures_str += "\n\nSteps:\n"
                for step in proc["steps"]:
                    step_num = step.get("step_number", 0)
                    step_content = step.get("content", "")
                    procedures_str += f"{step_num}. {step_content}\n"

                    # Add warnings if available
                    if "warnings" in step and step["warnings"]:
                        procedures_str += "Warning: " + ", ".join(step["warnings"]) + "\n"

        # Format parameters for the prompt
        parameters_str = ""
        for i, param in enumerate(parameters):
            param_name = param.get("name", "")
            param_value = param.get("value", "")
            param_desc = param.get("description", "")
            param_type = param.get("type", "")

            parameters_str += f"\n- Parameter: {param_name}"
            if param_value:
                parameters_str += f" = {param_value}"
            if param_type:
                parameters_str += f" (Type: {param_type})"
            if param_desc and param_desc != f"{param_name} with value {param_value}":
                parameters_str += f"\n  Description: {param_desc}"

        if not parameters_str:
            parameters_str = "No specific parameter information available."

        # Initialize OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Create the procedure prompt
        prompt = PROCEDURE_PROMPT.format(
            query=state.query_state.query,
            procedures=procedures_str,
            parameters=parameters_str
        )

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",  # Use the most capable model
            messages=[
                {"role": "system", "content": "You are an expert assistant for technical procedures."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        # Get the response text
        response_text = response.choices[0].message.content

        # Extract steps if possible
        procedure_steps = extract_procedure_steps(response_text)

        # Extract citations from the response
        citations = extract_citations(response_text, citation_map)

        # Create generation state
        state.generation_state = GenerationState(
            response=response_text,
            citations=citations,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4",
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "is_procedure": True,
                "procedure_count": len(procedures)
            },
            token_usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            procedure_steps=procedure_steps
        )

        logger.info(
            f"Procedure response generation complete with {len(citations)} citations, "
            f"{response.usage.total_tokens} total tokens"
        )

        return state

    except Exception as e:
        logger.error(f"Procedure response generation failed: {str(e)}", exc_info=True)

        # Generate error response
        error_response = "I'm sorry, but I encountered an error while generating a procedure response. Please try again or rephrase your query."

        # Create error state
        state.generation_state = GenerationState(
            response=error_response,
            citations=[],
            metadata={"error": str(e)}
        )

        return state

def format_retrieved_content(elements: List[Dict[str, Any]]) -> tuple:
    """
    Format retrieved content for the prompt with enhanced chunk information.

    Args:
        elements: Retrieved elements

    Returns:
        Tuple of (formatted content string, citation map)
    """
    content_str = ""
    citation_map = {}

    # Sort elements by score if available
    sorted_elements = sorted(
        elements,
        key=lambda x: x.get("metadata", {}).get("score", 0) if isinstance(x.get("metadata"), dict) else 0,
        reverse=True
    )

    # Limit to most relevant elements
    elements_to_use = sorted_elements[:12]

    for i, element in enumerate(elements_to_use):
        # Create citation ID
        citation_id = f"[{i+1}]"

        # Get metadata safely
        metadata = element.get("metadata", {})

        # Get content type and additional info
        content_type = element.get("content_type", "text")
        chunk_level = metadata.get("chunk_level", "")
        embedding_type = metadata.get("embedding_type", "")

        # Get document info
        pdf_id = element.get("pdf_id", "")
        page = metadata.get("page", 0) or element.get("page", 0)
        document_title = metadata.get("document_title", f"Document {pdf_id}")

        # Add citation info
        citation_map[citation_id] = {
            "id": element.get("id", "") or element.get("element_id", f"element_{i}"),
            "type": content_type,
            "pdf_id": pdf_id,
            "page": page,
            "document_title": document_title,
            "chunk_level": chunk_level,
            "embedding_type": embedding_type
        }

        # Format section info if available
        section_info = ""
        if "section" in metadata and metadata["section"]:
            section_info = f", Section: {metadata['section']}"
        elif "section_headers" in metadata and metadata["section_headers"]:
            if isinstance(metadata["section_headers"], list) and metadata["section_headers"]:
                section_info = f", Section: {' > '.join(metadata['section_headers'])}"

        # Format the content with citation marker and enhanced information
        content_str += f"\n--- Content {citation_id} (Type: {content_type}{section_info}, "
        content_str += f"Document: {document_title}, Page: {page}) ---\n"

        # Add chunk level and embedding type if available
        if chunk_level or embedding_type:
            content_str += f"[Level: {chunk_level}, Embedding: {embedding_type}]\n"

        # Add the actual content
        content = element.get("content", "")
        if len(content) > 1000:
            content = content[:1000] + "..."
        content_str += content

    return content_str, citation_map

def format_document_information(state: GraphState) -> str:
    """
    Format document information for research mode.

    Args:
        state: Graph state with document information

    Returns:
        Formatted document information string
    """
    info_str = "Documents included in this analysis:\n"

    # Try to get document titles from research state
    if state.research_state and state.research_state.metadata and "document_meta" in state.research_state.metadata:
        document_meta = state.research_state.metadata["document_meta"]

        # Handle different metadata formats
        if isinstance(document_meta, dict):
            for pdf_id, info in document_meta.items():
                if isinstance(info, dict) and "title" in info:
                    info_str += f"- Document ID: {pdf_id}, Title: {info['title']}\n"
                elif isinstance(info, str):
                    info_str += f"- Document ID: {pdf_id}, Title: {info}\n"
                else:
                    info_str += f"- Document ID: {pdf_id}\n"

    # Fallback to PDF IDs if no titles available
    elif state.query_state and state.query_state.pdf_ids:
        for pdf_id in state.query_state.pdf_ids:
            info_str += f"- Document ID: {pdf_id}\n"

    # Fallback to retrieval state metadata
    elif state.retrieval_state and state.retrieval_state.metadata and "cross_document_info" in state.retrieval_state.metadata:
        cross_doc_info = state.retrieval_state.metadata["cross_document_info"]

        if "document_titles" in cross_doc_info:
            doc_titles = cross_doc_info["document_titles"]
            for pdf_id, title in doc_titles.items():
                info_str += f"- Document ID: {pdf_id}, Title: {title}\n"

    return info_str

def format_knowledge_synthesis(research_context: Any) -> str:
    """
    Format knowledge synthesis for the prompt.

    Args:
        research_context: Research context with synthesis information

    Returns:
        Formatted synthesis string
    """
    synthesis_str = ""

    if hasattr(research_context, "summary") and research_context.summary:
        synthesis_str += f"Summary: {research_context.summary}\n\n"

    if hasattr(research_context, "facts") and research_context.facts:
        synthesis_str += "Key Facts:\n"
        for fact in research_context.facts[:5]:  # Limit to most important facts
            synthesis_str += f"- {fact}\n"
        synthesis_str += "\n"

    if hasattr(research_context, "insights") and research_context.insights:
        synthesis_str += "Insights:\n"
        for insight in research_context.insights[:3]:  # Limit to key insights
            synthesis_str += f"- {insight}\n"

    if not synthesis_str:
        # Try direct dictionary access if attributes not available
        if hasattr(research_context, "__getitem__"):
            try:
                if "summary" in research_context:
                    synthesis_str += f"Summary: {research_context['summary']}\n\n"

                if "facts" in research_context and research_context["facts"]:
                    synthesis_str += "Key Facts:\n"
                    for fact in research_context["facts"][:5]:
                        synthesis_str += f"- {fact}\n"
                    synthesis_str += "\n"

                if "insights" in research_context and research_context["insights"]:
                    synthesis_str += "Insights:\n"
                    for insight in research_context["insights"][:3]:
                        synthesis_str += f"- {insight}\n"
            except:
                pass

    return synthesis_str

def format_research_insights(research_state: Any) -> str:
    """
    Format research insights for cross-document analysis.

    Args:
        research_state: Research state with cross-document insights

    Returns:
        Formatted insights string
    """
    insights_str = ""

    if hasattr(research_state, "insights") and research_state.insights:
        insights_str += "Cross-Document Insights:\n"
        for insight in research_state.insights:
            insights_str += f"- {insight}\n"
        insights_str += "\n"

    if hasattr(research_state, "cross_references") and research_state.cross_references:
        insights_str += "Common Concepts Across Documents:\n"
        for i, concept in enumerate(research_state.cross_references[:5]):
            if isinstance(concept, dict) and "name" in concept:
                doc_count = concept.get("document_count", 0)
                docs = concept.get("documents", [])
                insights_str += f"- {concept['name']} (appears in {doc_count} documents: {', '.join(docs[:3])})\n"
            else:
                insights_str += f"- {concept}\n"

    # Try to access metadata if no attributes found
    if not insights_str and hasattr(research_state, "metadata"):
        meta = research_state.metadata
        if "cross_document_info" in meta and "shared_concepts" in meta["cross_document_info"]:
            shared_concepts = meta["cross_document_info"]["shared_concepts"]
            insights_str += "Common Concepts Across Documents:\n"
            for concept in shared_concepts[:5]:
                if isinstance(concept, dict) and "name" in concept:
                    insights_str += f"- {concept['name']}\n"
                else:
                    insights_str += f"- {concept}\n"

    return insights_str

def extract_citations(response_text: str, citation_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract citations from the response text.

    Args:
        response_text: Generated response text
        citation_map: Mapping of citation markers to document information

    Returns:
        List of citation objects
    """
    citations = []

    # Find all citation markers [1], [2], etc.
    citation_pattern = r'\[(\d+)\]'
    matches = re.findall(citation_pattern, response_text)

    # Process each citation
    for match in matches:
        citation_id = f"[{match}]"

        if citation_id in citation_map:
            citation_data = citation_map[citation_id].copy()

            # Check if already added (avoid duplicates)
            if not any(c.get("id") == citation_data.get("id") for c in citations):
                citations.append(citation_data)

    return citations

def extract_procedure_steps(response_text: str) -> List[Dict[str, Any]]:
    """
    Extract procedure steps from the response text.

    Args:
        response_text: Generated response text

    Returns:
        List of procedure step objects
    """
    steps = []

    # Try to find numbered steps (common formats)
    step_patterns = [
        r'(\d+)\.\s+(.*?)(?=\n\d+\.|\n\n|$)',  # 1. Step description
        r'Step\s+(\d+)[:.\s]+\s*(.*?)(?=Step\s+\d+|$)',  # Step 1: Step description
        r'(\d+)\)\s+(.*?)(?=\n\d+\)|\n\n|$)'   # 1) Step description
    ]

    # Try each pattern until we find steps
    for pattern in step_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            for step_num, step_text in matches:
                # Clean up step text
                step_text = step_text.strip()

                # Add to steps list
                steps.append({
                    "step_number": int(step_num),
                    "content": step_text
                })

            # If we found steps, sort by step number and return
            steps.sort(key=lambda x: x["step_number"])
            return steps

    # If no structured steps found, try to find paragraphs that might be steps
    if not steps:
        paragraphs = re.split(r'\n\n+', response_text)
        for i, para in enumerate(paragraphs):
            if para.strip() and len(para) > 20:  # Reasonable length for a step
                steps.append({
                    "step_number": i + 1,
                    "content": para.strip()
                })

    return steps
