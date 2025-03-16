"""
Response generator node for LangGraph-based PDF RAG system.
This node generates the final response to the user query based on retrieved content.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from openai import OpenAI
from datetime import datetime

from app.chat.langgraph.state import RetrievalState, GraphState, GenerationState, ResearchState
from app.chat.types import ContentElement

logger = logging.getLogger(__name__)

# Response generation prompt template
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
3. Cite specific sources for key information
4. Present a coherent and well-structured response
5. Acknowledge any limitations or gaps in the information

Use the knowledge synthesis to guide your response, but focus on addressing the query directly.
"""

def generate_response(state: GraphState) -> GraphState:
    """
    Generate the final response based on retrieved content and knowledge synthesis.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state with generated response
    """
    if (not state.retrieval_state or not state.query_state or 
        (hasattr(state, 'research_state') and not state.research_state)):
        logger.error("Retrieval state, query state, and research state are required for response generation")
        raise ValueError("Required states are missing")
    
    if not state.retrieval_state.elements:
        logger.warning("No elements retrieved, generating fallback response")
        # Generate fallback response
        state.generation_state = GenerationState(
            retrieval_state=state.retrieval_state,
            response="I couldn't find specific information to answer your query. Could you please rephrase or provide more details?",
            metadata={"fallback": True}
        )
        return state
    
    logger.info(f"Generating response for query: {state.query_state.query}")
    
    try:
        # Format retrieved content for the prompt
        content_str = ""
        citation_map = {}
        
        for i, element in enumerate(state.retrieval_state.elements[:15]):  # Limit to avoid token limits
            content_type = element.type.value if hasattr(element.type, 'value') else str(element.type)
            citation_id = f"[{i+1}]"
            citation_map[citation_id] = {
                "id": element.id,
                "type": content_type,
                "pdf_id": element.pdf_id,
                "page": element.page if hasattr(element, "page") else None
            }
            
            content_str += f"\n--- Content {citation_id} (Type: {content_type}) ---\n"
            content_str += element.content[:1500] + ("..." if len(element.content) > 1500 else "")
        
        # Format knowledge synthesis if available
        synthesis_str = ""
        if state.research_state and state.research_state.research_context:
            context = state.research_state.research_context
            synthesis_str += f"Summary: {context.summary}\n\n"
            
            if context.facts:
                synthesis_str += "Key Facts:\n"
                for fact in context.facts:
                    synthesis_str += f"- {fact}\n"
                synthesis_str += "\n"
                
            if context.insights:
                synthesis_str += "Insights:\n"
                for insight in context.insights:
                    synthesis_str += f"- {insight}\n"
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Prepare the prompt
        prompt = RESPONSE_GENERATION_PROMPT.format(
            query=state.query_state.query,
            content=content_str,
            synthesis=synthesis_str
        )
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are an expert assistant for technical documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        # Get the response text
        response_text = response.choices[0].message.content
        
        # Extract citations from the response
        citations = []
        for citation_id in citation_map:
            if citation_id in response_text:
                citations.append(citation_map[citation_id])
        
        # Create generation state
        state.generation_state = GenerationState(
            retrieval_state=state.retrieval_state,
            response=response_text,
            citations=citations,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "model": "gpt-4-0125-preview",
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            },
            token_usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
        
        logger.info(
            f"Response generation complete with {len(citations)} citations "
            f"and {response.usage.total_tokens} total tokens"
        )
        
        return state
        
    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}", exc_info=True)
        # Generate error response
        state.generation_state = GenerationState(
            retrieval_state=state.retrieval_state,
            response="I'm sorry, but I encountered an error while generating a response. Please try again or rephrase your query.",
            metadata={"error": str(e)}
        )
        return state
