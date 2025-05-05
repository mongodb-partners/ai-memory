import json
import pymongo
from bson.objectid import ObjectId
from bson import json_util
from database.mongodb import conversations
from database.models import Message
from services.bedrock_service import generate_embedding, send_to_bedrock
from models.pydantic_models import RememberRequest
from services.memory_service import remember_content
from utils.logger import logger
import config

def hybrid_search(query, vector_query, user_id, weight=0.5, top_n=10):
    """
    Perform a hybrid search operation on MongoDB by combining full-text and vector (semantic) search results.
    """
    pipeline = [
        {
            "$search": {
                "index":config.CONVERSATIONS_FULLTEXT_SEARCH_INDEX_NAME,
                "text": {"query": query, "path": "text"},
            }
        },
        {"$match": {"user_id": user_id}},
        {"$addFields": {"fts_score": {"$meta": "searchScore"}}},
        {"$setWindowFields": {"output": {"maxScore": {"$max": "$fts_score"}}}},
        {
            "$addFields": {
                "normalized_fts_score": {"$divide": ["$fts_score", "$maxScore"]}
            }
        },
        {
            "$project": {
                "text": 1,
                "type": 1,
                "timestamp": 1,
                "conversation_id": 1,
                "normalized_fts_score": 1,
            }
        },
        {
            "$unionWith": {
                "coll": "conversations",
                "pipeline": [
                    {
                        "$vectorSearch": {
                            "index": config.CONVERSATIONS_VECTOR_SEARCH_INDEX_NAME,
                            "queryVector": vector_query,
                            "path": "embeddings",
                            "numCandidates": 200,
                            "limit": top_n,
                            "filter": {"user_id": user_id},
                        }
                    },
                    {"$addFields": {"vs_score": {"$meta": "vectorSearchScore"}}},
                    {
                        "$setWindowFields": {
                            "output": {"maxScore": {"$max": "$vs_score"}}
                        }
                    },
                    {
                        "$addFields": {
                            "normalized_vs_score": {
                                "$divide": ["$vs_score", "$maxScore"]
                            }
                        }
                    },
                    {
                        "$project": {
                            "text": 1,
                            "type": 1,
                            "timestamp": 1,
                            "conversation_id": 1,
                            "normalized_vs_score": 1,
                        }
                    },
                ],
            }
        },
        {
            "$group": {
                "_id": "$_id",  # Group by document ID
                "fts_score": {"$max": "$normalized_fts_score"},
                "vs_score": {"$max": "$normalized_vs_score"},
                "text_field": {"$first": "$text"},
                "type_field": {"$first": "$type"},
                "timestamp_field": {"$first": "$timestamp"},
                "conversation_id_field": {"$first": "$conversation_id"},
            }
        },
        {
            "$addFields": {
                "hybrid_score": {
                    "$add": [
                        {"$multiply": [weight, {"$ifNull": ["$vs_score", 0]}]},
                        {"$multiply": [1 - weight, {"$ifNull": ["$fts_score", 0]}]},
                    ]
                }
            }
        },
        {"$sort": {"hybrid_score": -1}},  # Sort by combined hybrid score descending
        {"$limit": top_n},  # Limit final output
        {
            "$project": {
                "_id": 1,
                "fts_score": 1,
                "vs_score": 1,
                "score": "$hybrid_score",
                "text": "$text_field",
                "type": "$type_field",
                "timestamp": "$timestamp_field",
                "conversation_id": "$conversation_id_field",
            }
        },
    ]
    # Execute the aggregation pipeline and return the results
    try:
        results = list(conversations.aggregate(pipeline))
        return results
    except Exception as e:
        logger.error(f"Error in hybrid_search: {e}")
        raise

async def add_conversation_message(message_input):
    """Add a message to the conversation history"""
    try:
        new_message = Message(message_input)
        conversations.insert_one(new_message.to_dict())
        # For significant human messages, create a memory node
        if message_input.type == "human" and len(message_input.text) > 30:
            try:
                memory_content = (
                    f"From conversation {message_input.conversation_id}: {message_input.text}"
                )
                logger.info(f"Creating memory for user {message_input.user_id}: {memory_content}")
                await remember_content(
                    RememberRequest(user_id=message_input.user_id, content=memory_content)
                )
            except Exception as memory_error:
                logger.error(f"Error creating memory: {str(memory_error)}")
                raise
        return {"message": "Message added successfully"}
    except Exception as error:
        logger.error(str(error))
        raise

async def search_memory(user_id, query):
    """
    Searches memory items by user_id and a textual query using hybrid search.
    """
    try:
        # Generate embedding for the query text
        vector_query = generate_embedding(query)
        # Perform hybrid search over the stored messages
        documents = hybrid_search(query, vector_query, user_id, weight=0.8, top_n=5)
        # Filter results by minimum hybrid score threshold
        relevant_results = [doc for doc in documents if doc["score"] >= 0.70]
        if not relevant_results:
            return {"documents": "No documents found"}
        else:
            return {"documents": [serialize_document(doc) for doc in relevant_results]}
    except Exception as error:
        logger.error(str(error))
        raise

async def get_conversation_context(_id):
    """
    Fetches conversation records with context surrounding a specific message
    """
    try:
        # Fetch the conversation record for the given object ID
        conversation_record = conversations.find_one(
            {"_id": ObjectId(_id)},
            projection={
                "_id": 0,
                "embeddings": 0,
            },
        )
        if not conversation_record:
            return {"documents": "No documents found"}
        # Extract metadata
        user_id = conversation_record["user_id"]
        conversation_id = conversation_record["conversation_id"]
        timestamp = conversation_record["timestamp"]
        message_type = conversation_record["type"]
        if message_type == "ai":
            # Get more preceding context for AI messages
            prev_limit = 4
            next_limit = 2
        else:
            # Balance for human messages
            prev_limit = 3
            next_limit = 3
        # Get messages before target
        prev_cursor = (
            conversations.find(
                {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "timestamp": {"$lte": timestamp},
                },
                projection={
                    "_id": 0,
                    "embeddings": 0,
                },
            )
            .sort("timestamp", pymongo.DESCENDING)
            .limit(prev_limit)
        )
        context = list(prev_cursor)
        # Get messages after target
        next_cursor = (
            conversations.find(
                {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "timestamp": {"$gt": timestamp},
                },
                projection={
                    "_id": 0,
                    "embeddings": 0,
                },
            )
            .sort("timestamp", pymongo.ASCENDING)
            .limit(next_limit)
        )
        context_after = list(next_cursor)
        # Combine and sort all messages by timestamp
        conversation_with_context = sorted(
            context + context_after,
            key=lambda x: x["timestamp"],
        )
        return {"documents": conversation_with_context}
    except Exception as error:
        logger.error(str(error))
        raise

async def generate_conversation_summary(documents):
    """
    Generates a detailed and structured summary for a conversation provided in JSON format.
    """
    try:
        # Construct a prompt with detailed instructions and conversation history
        prompt = (
            f"You are an advanced AI assistant skilled in analyzing and summarizing conversation histories while preserving all essential details.\n"
            f"Given the following conversation data in JSON format, generate a detailed and structured summary that captures all key points, topics discussed, decisions made, and relevant insights.\n\n"
            f"Ensure your summary follows these guidelines:\n"
            f"- **Maintain Clarity & Accuracy:** Include all significant details, technical discussions, and conclusions.\n"
            f"- **Preserve Context & Meaning:** Avoid omitting important points that could alter the conversation's intent.\n"
            f"- **Organized Structure:** Present the summary in a logical flow or chronological order.\n"
            f"- **Key Highlights:** Explicitly state major questions asked, AI responses, decisions made, and follow-up discussions.\n"
            f"- **Avoid Redundancy:** Summarize effectively without unnecessary repetition.\n\n"
            f"### Output Format:\n"
            f"- **Topic:** Briefly describe the conversation's purpose.\n"
            f"- **Key Discussion Points:** Outline the main topics covered.\n"
            f"- **Decisions & Takeaways:** Highlight key conclusions or next steps.\n"
            f"- **Unresolved Questions (if any):** Mention pending queries or areas needing further clarification.\n\n"
            f"Provide a **clear, structured, and comprehensive** summary ensuring no critical detail is overlooked.\n\n"
            f"Input JSON: {json.dumps(documents, default=json_util.default)}"
        )
        # Send prompt to Bedrock and wait for summary response
        summary = await send_to_bedrock(prompt)
        return {"summary": summary}
    except Exception as error:
        logger.error(str(error))
        raise

def serialize_document(doc):
    """Helper function to serialize MongoDB documents."""
    doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
    return doc