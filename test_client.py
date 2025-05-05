import asyncio
import httpx
import json
import uuid
from datetime import datetime, timezone
from pprint import pprint

# Configuration
BASE_URL = "http://localhost:8182"
TIMEOUT = 600.0  # Seconds

# Test user and conversation IDs
USER_ID = f"test_user_{uuid.uuid4().hex[:8]}"
CONVERSATION_ID = f"test_conversation_{uuid.uuid4().hex[:8]}"

async def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{BASE_URL}/health")
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
        return True

async def add_test_conversation():
    """Add a test conversation with multiple messages"""
    print("\n=== Adding Test Conversation ===")
    
    conversation_messages = [
        {
            "user_id": USER_ID,
            "conversation_id": CONVERSATION_ID,
            "type": "human",
            "text": "Hello! I'd like to discuss my preferences for a new project.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "user_id": USER_ID,
            "conversation_id": CONVERSATION_ID,
            "type": "ai",
            "text": "I'd be happy to discuss your project preferences. What type of project are you working on?",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "user_id": USER_ID,
            "conversation_id": CONVERSATION_ID,
            "type": "human",
            "text": "I'm building a data analytics platform. I prefer Python for backend and React for frontend. Also, I'd like to use MongoDB Atlas for storage with vector search capabilities.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "user_id": USER_ID,
            "conversation_id": CONVERSATION_ID,
            "type": "ai",
            "text": "Great choices! Python works well for data analytics backends, and React offers a flexible frontend. MongoDB Atlas with vector search is excellent for handling both structured data and vector embeddings. Would you like some architecture recommendations?",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "user_id": USER_ID,
            "conversation_id": CONVERSATION_ID,
            "type": "human", 
            "text": "Yes, please. I'd also like to deploy this on AWS and integrate with their machine learning services. My email is test@example.com if you need to send me any documentation.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "user_id": USER_ID,
            "conversation_id": CONVERSATION_ID,
            "type": "ai",
            "text": "Ok. Thanks for the input. I'll send you the documentation to your email. For AWS deployment, I recommend using EC2 for compute resources and S3 for storage. You can also leverage AWS Lambda for serverless functions. Would you like to discuss any specific AWS services?",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
    ]
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        all_responses = []
        
        for i, message in enumerate(conversation_messages):
            print(f"Adding message {i+1}/{len(conversation_messages)}...")
            response = await client.post(
                f"{BASE_URL}/conversation/",
                json=message
            )
            
            if response.status_code != 200:
                print(f"Error adding message: {response.status_code}")
                print(f"Response: {response.text}")
                return False
            
            all_responses.append(response.json())
            
            # # Small delay to ensure proper ordering
            # await asyncio.sleep(0.5)
        
        print("All messages added successfully!")
        return all_responses

async def test_retrieve_memory(query_text):
    """Test memory retrieval with the given query"""
    print(f"\n=== Testing Memory Retrieval for '{query_text}' ===")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(
            f"{BASE_URL}/retrieve_memory/",
            params={"user_id": USER_ID, "text": query_text}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Pretty print summary
            print("\n--- Conversation Summary ---")
            print(result.get("conversation_summary", "No summary available"))
            
            # Print similar memories
            print("\n--- Similar Memories ---")
            similar_memories = result.get("similar_memories", [])
            if isinstance(similar_memories, list):
                for i, memory in enumerate(similar_memories[:3]):  # Show at most 3
                    print(f"\nMemory {i+1}:")
                    print(f"Content: {memory.get('content', 'N/A')[:100]}...")  # Truncate long content
                    print(f"Summary: {memory.get('summary', 'N/A')}")
                    print(f"Importance: {memory.get('importance', 'N/A')}")
                    print(f"Similarity: {memory.get('similarity', 'N/A')}")
            else:
                print(similar_memories)
            
            return result
        else:
            print(f"Error retrieving memory: {response.text}")
            return None

async def run_comprehensive_tests():
    """Run a comprehensive test suite"""
    # Test health check
    health_ok = await test_health_check()
    if not health_ok:
        print("Health check failed, aborting tests")
        return
    
    # Add test conversation
    await add_test_conversation()
    
    # Allow time for embedding generation and memory processing
    print("\nWaiting for memory processing to complete...")
    await asyncio.sleep(3)
    
    # Test various memory retrieval queries
    test_queries = [
        "project preferences",
        "tech stack",
        "email contact",
        "MongoDB Atlas",
        "AWS deployment"
    ]
    
    for query in test_queries:
        await test_retrieve_memory(query)
        # Small delay between queries
        await asyncio.sleep(1)
    
    print("\n=== All Tests Completed ===")

async def test_conversation_evolution():
    """Test how the memory system evolves with repeated similar information"""
    print("\n=== Testing Memory Evolution ===")
    
    # Define a series of related messages about the same topic
    evolution_messages = [
        {
            "user_id": USER_ID,
            "conversation_id": f"{CONVERSATION_ID}_evolution",
            "type": "human",
            "text": "I prefer to be contacted via email at test@example.com",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "user_id": USER_ID,
            "conversation_id": f"{CONVERSATION_ID}_evolution",
            "type": "ai",
            "text": "I've noted your preference for email communication at test@example.com",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    ]
    
    # Add the initial messages
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for message in evolution_messages:
            await client.post(f"{BASE_URL}/conversation/", json=message)
    
    # Check memory state
    print("\nInitial memory state:")
    await test_retrieve_memory("contact preferences")
    await asyncio.sleep(2)
    
    # Add reinforcement messages
    reinforcement_message = {
        "user_id": USER_ID,
        "conversation_id": f"{CONVERSATION_ID}_evolution",
        "type": "human",
        "text": "Just to confirm, my email is test@example.com and I prefer email over phone calls",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        await client.post(f"{BASE_URL}/conversation/", json=reinforcement_message)
    
    #  Wait for processing
    await asyncio.sleep(2)
    
    # Check memory state after reinforcement
    print("\nMemory state after reinforcement:")
    await test_retrieve_memory("contact preferences")

if __name__ == "__main__":
    # Create and run the async event loop
    loop = asyncio.get_event_loop()
    
    # Choose which test set to run
    loop.run_until_complete(run_comprehensive_tests())
    #loop.run_until_complete(test_conversation_evolution())