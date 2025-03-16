import os
import argparse
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import Dict, Any
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorInspector:
    def __init__(self):
        load_dotenv()
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    async def inspect_vectors(self, pdf_id: str):
        """Inspect vector metadata and embeddings by type for debugging."""
        try:
            logger.info(f"[INSPECT] Analyzing vectors for PDF {pdf_id}")

            # Query all vectors for the PDF
            response = await asyncio.to_thread(
                self.index.query,
                vector=[0] * 1536,  # Dummy vector for metadata-only query
                filter={"pdf_id": pdf_id},
                top_k=1000,
                include_metadata=True,
                include_values=True
            )

            # Group vectors by content type
            vectors_by_type = {
                'text': [],
                'table': [],
                'image': [],
                'header': []
            }

            for match in response.matches:
                content_type = match.metadata.get('type', 'text')
                vectors_by_type[content_type].append({
                    'id': match.id,
                    'page': match.metadata.get('page'),
                    'metadata': match.metadata,
                    'embedding': match.values,
                    'score': match.score
                })

            # Log analysis for each content type
            logger.info("\n=== Vector Analysis ===")
            for content_type, vectors in vectors_by_type.items():
                logger.info(f"\n{content_type.upper()} Vectors: {len(vectors)}")
                if vectors:
                    sample = vectors[0]
                    logger.info(f"Sample Metadata Structure for {content_type.upper()}:")
                    logger.info(json.dumps(sample['metadata'], indent=2))
                    logger.info(f"Sample Embedding (First 5 Values): {sample['embedding'][:5]}")

                    # Check for image_base64 in image vectors
                    if content_type == 'image':
                        image_base64_present = 'image_base64' in sample['metadata']
                        logger.info(f"Image Base64 Present: {image_base64_present}")
                        if image_base64_present:
                            logger.info(f"Image Base64 Size: {len(sample['metadata']['image_base64'])} bytes")

            return vectors_by_type

        except Exception as e:
            logger.error(f"[INSPECT] Error inspecting vectors: {str(e)}")
            raise

    async def remove_all_vectors(self, pdf_id: str):
        """Removes all vectors associated with a given PDF ID."""
        try:
            logger.info(f"[REMOVE] Removing all vectors for PDF {pdf_id}")

            # Fetch all vector IDs for the PDF
            query_response = await asyncio.to_thread(
                self.index.query,
                vector=[0] * 1536,  # Dummy vector for metadata-only query
                filter={"pdf_id": pdf_id},
                top_k=10000, # Adjust top_k as needed. Pinecone limits to 10k in one call. If you have more, you would need to implement paging
                include_metadata=False,
                include_values=False
            )
            ids_to_delete = [match.id for match in query_response.matches]

            if ids_to_delete:
                # Delete the vectors
                await asyncio.to_thread(self.index.delete, ids=ids_to_delete)
                logger.info(f"[REMOVE] Successfully removed {len(ids_to_delete)} vectors.")
            else:
                logger.info(f"[REMOVE] No vectors found for PDF {pdf_id}.")

        except Exception as e:
            logger.error(f"[REMOVE] Error removing vectors: {str(e)}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect or remove vectors for a given PDF ID.")
    parser.add_argument("--pdf_id", type=str, required=True, help="The PDF ID to process.")
    parser.add_argument("--remove", action="store_true", help="Remove all vectors for the PDF ID.")
    args = parser.parse_args()

    inspector = VectorInspector()

    if args.remove:
        asyncio.run(inspector.remove_all_vectors(args.pdf_id))
    else:
        asyncio.run(inspector.inspect_vectors(args.pdf_id))
