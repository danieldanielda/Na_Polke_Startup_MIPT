"""
Example usage of RAG Consultant Agent

This example shows how to use the RAG Consultant Agent to query the RAG system
with product information obtained from the barcode parser agent.
"""

import asyncio
from src.api.v1.schemas import RAGQueryRequest
from rag_crew import RAGConsultantCrew

async def example_rag_query():
    """
    Example: Query RAG system with product information from barcode parser
    """
    # Product information from barcode parser agent
    product_info = """
    Product Name: L'Oreal Paris Revitalift Anti-Wrinkle Cream
    Brand: L'Oreal Paris
    Category: Skincare / Anti-Aging
    """
    
    # Create RAG query request
    request = RAGQueryRequest(
        product_info=product_info,
        collection_id="default",  # or your specific collection ID
        system_prompt="system_common_prompt"  # or your custom prompt
    )
    
    # Execute RAG query
    inputs = {
        "product_info": request.product_info,
        "collection_id": request.collection_id,
        "system_prompt": request.system_prompt
    }
    
    result = RAGConsultantCrew().crew().kickoff(inputs=inputs)
    
    print("RAG Query Result:")
    print(result)
    
    return result

if __name__ == "__main__":
    asyncio.run(example_rag_query())
