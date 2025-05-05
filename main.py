import uvicorn
from fastapi import FastAPI, HTTPException, status

import config
from database.mongodb import initialize_mongodb

# Import models and services
from models.pydantic_models import ErrorResponse, MessageInput
from services.bedrock_service import generate_embedding
from services.conversation_service import (
    add_conversation_message,
    generate_conversation_summary,
    get_conversation_context,
    search_memory,
)
from services.memory_service import find_similar_memories
from utils import error_utils

# Initialize FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description=config.APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Initialize MongoDB on startup
initialize_mongodb()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/conversation/")
async def add_message(message: MessageInput):
    """Add a message to the conversation history"""
    try:
        return await add_conversation_message(message)
    except Exception as error:
        error_response = error_utils.handle_exception(error)
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(**error_response),
        )


@app.get("/retrieve_memory/")
async def retrieve_memory(user_id: str, text: str):
    """
    Retrieve memory items, context, summary, and similar memory nodes in a single request
    """
    try:
        # Generate embedding for the query text
        vector_query = generate_embedding(text)

        # Search for relevant memory items
        memory_items = await search_memory(user_id, text)

        # Get similar memory nodes from the memory tree
        similar_memories = await find_similar_memories(user_id, vector_query)

        if memory_items["documents"] == "No documents found":
            return {
                "related_conversation": "No conversation found",
                "conversation_summary": "No summary found",
                "similar_memories": (
                    similar_memories
                    if similar_memories
                    else "No similar memories found"
                ),
            }

        # Extract conversation ID from the first memory item
        object_id = memory_items["documents"][0]["_id"]

        # Retrieve conversation context around the matching memory item
        context = await get_conversation_context(object_id)

        # Generate a detailed summary for the conversation
        summary = await generate_conversation_summary(context["documents"])

        memories = [
            {
                "content": memory["content"],
                "summary": memory["summary"],
                "similarity": memory["similarity"],
                "importance": memory["effective_importance"],
            }
            for memory in similar_memories
        ]

        result = {
            "related_conversation": context["documents"],
            "conversation_summary": summary["summary"],
            "similar_memories": memories if memories else "No similar memories found",
        }

        return result
    except Exception as error:
        error_response = error_utils.handle_exception(error)
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(**error_response),
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        reload=config.DEBUG,
    )
