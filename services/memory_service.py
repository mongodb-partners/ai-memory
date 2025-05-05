import datetime
from bson.objectid import ObjectId
import pymongo
from config import MAX_DEPTH, SIMILARITY_THRESHOLD, REINFORCEMENT_FACTOR, DECAY_FACTOR
from database.mongodb import memory_nodes
from services.bedrock_service import generate_embedding, send_to_bedrock
from utils.helpers import cosine_similarity
from typing import List, Dict
from config import MEMORY_NODES_VECTOR_SEARCH_INDEX_NAME
from utils.logger import logger

async def find_similar_memories(
    user_id: str, embedding: List[float], top_n: int = 3
) -> List[Dict]:
    """
    Find most similar memory nodes from the memory tree using vector search

    Args:
        user_id: User ID to filter by
        embedding: Query embedding vector
        top_n: Number of similar memories to return

    Returns:
        List of similar memory nodes with similarity scores
    """
    try:
        response = memory_nodes.aggregate(
            [
                {
                    "$vectorSearch": {
                        "index": MEMORY_NODES_VECTOR_SEARCH_INDEX_NAME,
                        "path": "embeddings",
                        "queryVector": embedding,
                        "numCandidates": 100,
                        "limit": top_n,
                        "filter": {"user_id": user_id},
                    }
                },
                {"$addFields": {"similarity": {"$meta": "vectorSearchScore"}}},
                {
                    "$project": {
                        "_id": 1,
                        "content": 1,
                        "summary": 1,
                        "importance": 1,
                        "effective_importance": {
                            "$multiply": [
                                "$importance",
                                {"$add": [1, {"$ln": {"$add": ["$access_count", 1]}}]},
                            ]
                        },
                        "similarity": 1,
                        "access_count": 1,
                        "timestamp": 1,
                        "embeddings": 1,
                    }
                },
            ]
        )

        results = []
        for doc in response:
            doc_id = str(doc.pop("_id"))
            doc["id"] = doc_id
            results.append(doc)

        return results
    except Exception as e:
        logger.error(f"Error finding similar memory nodes: {str(e)}")
        raise


async def update_importance(user_id, embedding):
    """Update importance of memories based on similarity to new content"""
    cursor = memory_nodes.find({"user_id": user_id})
    for doc in cursor:
        doc_id = doc["_id"]
        memory_embedding = doc["embeddings"]
        similarity = cosine_similarity(embedding, memory_embedding)
        if similarity > SIMILARITY_THRESHOLD:
            # Reinforce similar memories
            new_importance = doc["importance"] * REINFORCEMENT_FACTOR
            new_access_count = doc["access_count"] + 1
        else:
            # Decay less relevant memories
            new_importance = doc["importance"] * DECAY_FACTOR
            new_access_count = doc["access_count"]
        # Update in database
        memory_nodes.update_one(
            {"_id": doc_id},
            {"$set": {"importance": new_importance, "access_count": new_access_count}},
        )


async def prune_memories(user_id):
    """Prune less important memories exceeding the maximum depth"""
    count = memory_nodes.count_documents({"user_id": user_id})
    if count > MAX_DEPTH:
        # Find low importance memories to delete
        cursor = (
            memory_nodes.find({"user_id": user_id})
            .sort([("importance", pymongo.ASCENDING)])
            .limit(count - MAX_DEPTH)
        )
        # Delete them
        for doc in cursor:
            memory_nodes.delete_one({"_id": doc["_id"]})


async def remember_content(request):
    """Store a new memory for the user, integrating with existing memories"""
    try:
        # Input validation
        if not request.content.strip():
            return {"message": "Cannot remember empty content"}
        # Generate embedding for the content
        embeddings = generate_embedding(request.content)
        # Check for similar existing memories before creating a new one
        similar_memories = await find_similar_memories(request.user_id, embeddings)
        # If we already have very similar memories, reinforce them instead
        for memory in similar_memories:
            if memory["similarity"] > 0.85:  # High similarity threshold
                # Update existing memory instead of creating a new one
                memory_nodes.update_one(
                    {"_id": ObjectId(memory["id"])},
                    {
                        "$set": {
                            "importance": memory["importance"] * REINFORCEMENT_FACTOR,
                            "access_count": memory["access_count"] + 1,
                            "last_accessed": datetime.datetime.now(
                                datetime.timezone.utc
                            ),
                        }
                    },
                )
                return {
                    "message": "Reinforced existing memory",
                    "memory_id": memory["id"],
                }
        # For new memories, assess importance
        importance_assessment_prompt = (
            "On a scale of 1-10, rate the importance of remembering this information long-term. "
            "Consider factors like: uniqueness of information, actionability, personal significance, "
            "and whether it contains key facts or decisions. Respond with just a number.\n\n"
            f"Text to evaluate: {request.content}"
        )
        importance_rating_text = await send_to_bedrock(importance_assessment_prompt)
        # Extract numeric rating (handle potential non-numeric responses)
        try:
            importance_rating = float(
                "".join(c for c in importance_rating_text if c.isdigit() or c == ".")
            )
            # Normalize to 0-1 range
            importance_score = min(max(importance_rating / 10, 0.1), 1.0)
        except ValueError:
            # Default if we can't parse the rating
            importance_score = 0.5
        # Generate a concise summary
        summary_prompt = (
            "Create a one-sentence summary of the key information in this text. Be specific and concise:\n\n"
            + request.content
        )
        summary = await send_to_bedrock(summary_prompt)
        # Create new memory node
        new_memory = {
            "user_id": request.user_id,
            "content": request.content,
            "summary": summary,
            "importance": importance_score,
            "access_count": 0,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "last_accessed": datetime.datetime.now(datetime.timezone.utc),
            "embeddings": embeddings,
        }
        # Save to database
        result = memory_nodes.insert_one(new_memory)
        memory_id = str(result.inserted_id)
        # Find similar memories for potential merging
        similar_memories = await find_similar_memories(request.user_id, embeddings)
        # Merge with similar memories if they exceed threshold but aren't identical
        for memory in similar_memories:
            if memory["id"] != memory_id and 0.7 < memory["similarity"] < 0.85:
                # Combine content using AI
                combined_content_prompt = (
                    "These two texts contain related information. Combine them into a single cohesive text "
                    "that preserves all important details from both without redundancy:\n\n"
                    f"TEXT 1: {new_memory['content']}\n\n"
                    f"TEXT 2: {memory['content']}"
                )
                combined_content = await send_to_bedrock(
                    f"{combined_content_prompt}\n\nCombine these texts effectively."
                )
                # Update metrics
                updated_importance = (
                    max(new_memory["importance"], memory["importance"]) * 1.1
                )
                updated_access_count = (
                    new_memory["access_count"] + memory["access_count"]
                )
                # Average embeddings
                updated_embeddings = [
                    (a + b) / 2 for a, b in zip(embeddings, memory["embeddings"])
                ]
                # Generate new summary
                summary_prompt = (
                    "Create a one-sentence summary capturing the key information:\n\n"
                    + combined_content
                )
                summary = await send_to_bedrock(
                    f"{summary_prompt}\n\nCreate a concise summary."
                )
                # Update the memory
                memory_nodes.update_one(
                    {"_id": ObjectId(memory_id)},
                    {
                        "$set": {
                            "content": combined_content,
                            "summary": summary,
                            "importance": updated_importance,
                            "access_count": updated_access_count,
                            "embeddings": updated_embeddings,
                        }
                    },
                )
                # Delete the merged memory
                memory_nodes.delete_one({"_id": ObjectId(memory["id"])})
                break
        # Update importance of other memories based on relationship to this memory
        await update_importance(request.user_id, embeddings)
        # Prune excessive memories if needed
        await prune_memories(request.user_id)
        logger.info(f"Memory created for user {request.user_id}: {summary[:50]}...")
        return {
            "message": f"Remembered: {new_memory['summary']}",
            "memory_id": memory_id,
            "importance": importance_score,
        }
    except Exception as error:
        logger.error(f"Error remembering content: {error}")
        raise
