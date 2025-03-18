#!/usr/bin/env python3
"""
Database initialization script for Neo4j.
Run this script once to set up all required indices and constraints for the application.
"""

import os
import sys
import asyncio
import logging
from typing import List, Dict, Any, Optional
from neo4j import AsyncGraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

async def initialize_database() -> bool:
    """Initialize the Neo4j database with all required schema elements."""

    logger.info(f"Connecting to Neo4j at {NEO4J_URI}")
    driver = AsyncGraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    try:
        # Test connection
        logger.info("Testing database connection...")
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS test")
            record = await result.single()
            if not record or record.get("test") != 1:
                logger.error("Database connection test failed!")
                return False

            logger.info("Database connection successful")

            # 1. Create constraints
            logger.info("Creating constraints...")
            try:
                # Document ID uniqueness constraint
                await session.run(
                    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS " +
                    "FOR (d:Document) REQUIRE d.id IS UNIQUE"
                )

                # Content element ID uniqueness constraint
                await session.run(
                    "CREATE CONSTRAINT content_element_id_unique IF NOT EXISTS " +
                    "FOR (c:ContentElement) REQUIRE c.id IS UNIQUE"
                )

                logger.info("Constraints created successfully")
            except Exception as e:
                logger.warning(f"Error creating constraints (may already exist): {str(e)}")

            # 2. Create basic indices
            logger.info("Creating basic indices...")
            try:
                # PDF ID index for documents
                await session.run(
                    "CREATE INDEX document_pdf_id IF NOT EXISTS " +
                    "FOR (d:Document) ON (d.pdf_id)"
                )

                # PDF ID index for content elements
                await session.run(
                    "CREATE INDEX content_pdf_id IF NOT EXISTS " +
                    "FOR (c:ContentElement) ON (c.pdf_id)"
                )

                # Content type index
                await session.run(
                    "CREATE INDEX content_type_index IF NOT EXISTS " +
                    "FOR (c:ContentElement) ON (c.content_type)"
                )

                logger.info("Basic indices created successfully")
            except Exception as e:
                logger.warning(f"Error creating basic indices: {str(e)}")

            # 3. Create vector index
            logger.info(f"Creating vector index with {EMBEDDING_DIMENSIONS} dimensions...")
            try:
                # Try the standard Neo4j approach
                try:
                    await session.run(
                        "CALL db.index.vector.createNodeIndex(" +
                        "'content_vector', 'ContentElement', 'embedding', $dimensions, 'cosine')",
                        {"dimensions": EMBEDDING_DIMENSIONS}
                    )
                    logger.info("Vector index 'content_vector' created successfully")
                except Exception as ve:
                    if "equivalent index already exists" in str(ve):
                        logger.info("Vector index 'content_vector' already exists")
                    else:
                        logger.warning(f"Vector index creation failed: {str(ve)}")
                        logger.warning("Attempting alternative approach...")

                        # Try Neo4j 4.4+ syntax if the first one failed
                        try:
                            await session.run(
                                "CALL db.index.vector.createNodeIndex(" +
                                "'content_vector', 'ContentElement', 'embedding', $dimensions, 'cosine')",
                                {"dimensions": EMBEDDING_DIMENSIONS}
                            )
                            logger.info("Vector index 'content_vector' created with alternative approach")
                        except Exception as ve2:
                            logger.error(f"Alternative vector index creation also failed: {str(ve2)}")
            except Exception as e:
                logger.error(f"Error creating vector index: {str(e)}")

            # 4. Check existing vector indexes
            logger.info("Checking existing vector indexes...")
            try:
                # Get Neo4j version
                version_result = await session.run("CALL dbms.components() YIELD name, versions RETURN versions[0] as version")
                version_record = await version_result.single()
                neo4j_version = version_record.get("version", "unknown") if version_record else "unknown"
                logger.info(f"Neo4j version: {neo4j_version}")

                # Different approach based on Neo4j version
                if neo4j_version.startswith("5."):
                    # Neo4j 5.x uses SHOW INDEXES
                    show_indexes_result = await session.run("SHOW INDEXES")
                    # Use values() because fetch() might need arguments in some driver versions
                    records = await show_indexes_result.values()

                    vector_indexes = []
                    for record in records:
                        record_str = str(record)
                        if "content_vector" in record_str:
                            vector_indexes.append(record_str)

                    if vector_indexes:
                        logger.info(f"Found vector indexes: {len(vector_indexes)}")
                        for idx in vector_indexes:
                            logger.info(f"Vector index: {idx}")
                    else:
                        logger.warning("No vector indexes found")

                else:
                    # Neo4j 4.x uses a different approach
                    logger.info("Using Neo4j 4.x index listing approach")
                    list_indexes_result = await session.run("CALL db.indexes()")
                    records = await list_indexes_result.values()

                    vector_indexes = []
                    for record in records:
                        record_str = str(record)
                        if "content_vector" in record_str:
                            vector_indexes.append(record_str)

                    if vector_indexes:
                        logger.info(f"Found vector indexes: {len(vector_indexes)}")
                    else:
                        logger.warning("No vector indexes found")
            except Exception as e:
                logger.error(f"Error checking vector indexes: {str(e)}")

            # 5. Create simple content index as fallback for keyword search
            logger.info("Creating content index as fallback for keyword search...")
            try:
                await session.run(
                    "CREATE INDEX content_text_index IF NOT EXISTS " +
                    "FOR (c:ContentElement) ON (c.content)"
                )
                logger.info("Created content index successfully")
            except Exception as e:
                logger.error(f"Error creating content index: {str(e)}")

            logger.info("Database initialization complete")
            return True

    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}", exc_info=True)
        return False
    finally:
        await driver.close()

if __name__ == "__main__":
    try:
        success = asyncio.run(initialize_database())
        if success:
            logger.info("✅ Database initialization successful")
            sys.exit(0)
        else:
            logger.error("❌ Database initialization failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Initialization script error: {str(e)}", exc_info=True)
        sys.exit(1)
