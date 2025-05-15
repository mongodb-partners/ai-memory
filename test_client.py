import asyncio
import httpx
import uuid
from datetime import datetime, timezone, timedelta

# Configuration
BASE_URL = "http://localhost:8182"
TIMEOUT = 600.0  # Seconds

# Test user and conversation IDs
USER_ID = f"test_user_{uuid.uuid4().hex[:8]}"

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

async def add_conversation(conversation_id, messages):
    """Add a conversation with the provided messages"""
    print(f"\n=== Adding Conversation: {conversation_id} ===")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        all_responses = []
        
        for i, message in enumerate(messages):
            # Add user ID and conversation ID to each message
            message["user_id"] = USER_ID
            message["conversation_id"] = conversation_id
            
            # Add timestamp if not provided
            if "timestamp" not in message:
                # Simulate conversations happening at different times
                message["timestamp"] = (datetime.now(timezone.utc) - 
                                       timedelta(hours=10-i/2)).isoformat()
            
            print(f"Adding message {i+1}/{len(messages)}...")
            response = await client.post(
                f"{BASE_URL}/conversation/",
                json=message
            )
            
            if response.status_code != 200:
                print(f"Error adding message: {response.status_code}")
                print(f"Response: {response.text}")
                return False
            
            all_responses.append(response.json())
            
            # Small delay to ensure proper ordering
            await asyncio.sleep(0.1)
        
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
                    print(f"Content: {memory.get('content', 'N/A')[:300]}...")  # Truncate long content
                    print(f"Summary: {memory.get('summary', 'N/A')}")
                    print(f"Importance: {memory.get('importance', 'N/A')}")
                    print(f"Similarity: {memory.get('similarity', 'N/A')}")
            else:
                print(similar_memories)
            
            return result
        else:
            print(f"Error retrieving memory: {response.text}")
            return None

async def run_comprehensive_memory_evolution_test():
    """Run a comprehensive test showcasing memory evolution across different topics"""
    # Test health check
    health_ok = await test_health_check()
    if not health_ok:
        print("Health check failed, aborting tests")
        return
    
    # Define conversations on various topics
    conversations = {
        # 1. Technology preferences
        "tech_preferences": [
            {"type": "human", "text": "Hello! I'm setting up a new project and want to discuss my tech preferences."},
            {"type": "ai", "text": "Great! I'd be happy to discuss your technology preferences. What kind of project are you working on?"},
            {"type": "human", "text": "I'm building a data analytics platform. I prefer Python for backend and React for frontend."},
            {"type": "ai", "text": "Excellent choices! Python is great for data processing and analytics, while React gives you a robust frontend framework. Any database preferences?"},
            {"type": "human", "text": "Yes, I like MongoDB for document storage and PostgreSQL for relational data."},
            {"type": "ai", "text": "Got it. MongoDB for document storage and PostgreSQL for relational data. That's a solid combination for flexibility and performance."}
        ],
        
        # 2. Personal information
        "personal_info": [
            {"type": "human", "text": "Just so you know, my name is Alex Chen and I work as a data scientist."},
            {"type": "ai", "text": "Thanks for sharing that information, Alex. It's nice to meet you. How long have you been working as a data scientist?"},
            {"type": "human", "text": "About 5 years now. I specialize in NLP and computer vision projects."},
            {"type": "ai", "text": "That's impressive, Alex! 5 years in NLP and computer vision is substantial experience. Are you currently working on any interesting projects?"},
            {"type": "human", "text": "Yes, I'm developing a sentiment analysis tool for customer feedback at my company, DataInsight Corp."},
            {"type": "ai", "text": "That sounds fascinating! Sentiment analysis for customer feedback can provide valuable insights for DataInsight Corp. I'd be interested to hear more about your approach."}
        ],
        
        # 3. Travel preferences
        "travel_preferences": [
            {"type": "human", "text": "I'm planning a vacation and wanted to mention my travel preferences."},
            {"type": "ai", "text": "That sounds exciting! I'd be happy to hear about your travel preferences to better assist you in the future."},
            {"type": "human", "text": "I prefer beach destinations over mountains, and I always travel with my family of four."},
            {"type": "ai", "text": "I see, you prefer beach destinations when traveling with your family of four. Are there any particular beach destinations you've enjoyed in the past?"},
            {"type": "human", "text": "Yes, we loved Hawaii and the Mediterranean coast. We usually travel during summer break in July."},
            {"type": "ai", "text": "Hawaii and the Mediterranean coast are wonderful choices! And I've noted that you typically travel with your family during the July summer break."}
        ],
        
        # 4. Food preferences
        "food_preferences": [
            {"type": "human", "text": "I should tell you about my dietary preferences for future reference."},
            {"type": "ai", "text": "Thank you for sharing! Knowing your dietary preferences will help me provide more relevant suggestions."},
            {"type": "human", "text": "I'm vegetarian, but I do eat dairy products. I'm also allergic to nuts."},
            {"type": "ai", "text": "I'll remember that you're vegetarian who consumes dairy products and have a nut allergy. This is important information, especially for any food-related discussions."},
            {"type": "human", "text": "Exactly. And I particularly enjoy Italian and Indian cuisines."},
            {"type": "ai", "text": "Great! I've noted that you enjoy Italian and Indian cuisines, which offer many excellent vegetarian options. I'll keep these preferences in mind for future recommendations."}
        ],
        
        # 5. Contact information (with evolution)
        "contact_info": [
            {"type": "human", "text": "You can contact me at alex.chen@example.com if needed."},
            {"type": "ai", "text": "Thank you for sharing your email address, alex.chen@example.com. I'll make note of this contact information."},
            {"type": "human", "text": "Actually, I have a new work email now: alex.chen@datainsight.com"},
            {"type": "ai", "text": "I've updated your contact information. Your current email is alex.chen@datainsight.com. I'll use this email for any future references."},
            {"type": "human", "text": "Perfect. And my work phone is 555-123-4567 if there's ever an urgent matter."},
            {"type": "ai", "text": "I've added your work phone number, 555-123-4567, to your contact information along with your email alex.chen@datainsight.com."}
        ],
        
        # 6. Project deadlines (with evolution)
        "project_deadlines": [
            {"type": "human", "text": "I need to finish this analytics dashboard by June 15th."},
            {"type": "ai", "text": "I've noted your deadline of June 15th for the analytics dashboard project. That's about a month from now."},
            {"type": "human", "text": "The client just moved the deadline up to June 1st! I need to adjust my schedule."},
            {"type": "ai", "text": "I understand the urgency. Your analytics dashboard deadline has been moved from June 15th to June 1st, which is two weeks earlier than planned."},
            {"type": "human", "text": "Actually, we negotiated and settled on June 8th as the final deadline."},
            {"type": "ai", "text": "Thank you for the update. I've noted that the final agreed deadline for the analytics dashboard is now June 8th, which is a compromise between the original June 15th date and the client's requested June 1st date."}
        ],
        
        # 7. Learning interests
        "learning_interests": [
            {"type": "human", "text": "I'm interested in learning more about deep learning and transformer models."},
            {"type": "ai", "text": "That's a fascinating area! Deep learning and transformer models have revolutionized NLP and many other fields. Would you like me to focus on any particular aspects?"},
            {"type": "human", "text": "I'm particularly interested in BERT and its applications in text classification."},
            {"type": "ai", "text": "BERT is indeed powerful for text classification tasks. I'll keep in mind your interest in BERT and its applications for text classification for future discussions."},
            {"type": "human", "text": "I've also started exploring GPT models and prompt engineering recently."},
            {"type": "ai", "text": "That's great! GPT models and prompt engineering are cutting-edge areas. I've noted your expanding interests from BERT text classification to now including GPT models and prompt engineering."}
        ],
        
        # 8. Home automation setup (with evolution)
        "home_automation": [
            {"type": "human", "text": "I have a smart home system with Philips Hue lights and a Nest thermostat."},
            {"type": "ai", "text": "That's a nice smart home setup with Philips Hue lights and a Nest thermostat. Are you using any central hub to control them?"},
            {"type": "human", "text": "Yes, I use Samsung SmartThings as my hub and Google Assistant for voice control."},
            {"type": "ai", "text": "Great! You have a comprehensive setup with Philips Hue lights and a Nest thermostat controlled through Samsung SmartThings and Google Assistant voice commands."},
            {"type": "human", "text": "I just upgraded and added Amazon Echo devices throughout the house for better coverage."},
            {"type": "ai", "text": "I see you've expanded your smart home system. You've added Amazon Echo devices throughout your house for better coverage, while maintaining your existing setup of Philips Hue lights, Nest thermostat, and Samsung SmartThings hub."}
        ],
        
        # 9. Fitness goals
        "fitness_goals": [
            {"type": "human", "text": "I'm trying to improve my fitness by running 5km three times a week."},
            {"type": "ai", "text": "That's an excellent fitness goal! Running 5km three times weekly is a great way to build cardiovascular endurance. How long have you been following this routine?"},
            {"type": "human", "text": "I started about a month ago. My goal is to participate in a half marathon in November."},
            {"type": "ai", "text": "That's impressive progress! You've been running 5km three times weekly for a month, building up to a half marathon in November. That gives you a clear training progression path."},
            {"type": "human", "text": "Yes, and I'm incorporating strength training twice a week to improve overall performance."},
            {"type": "ai", "text": "Excellent approach! Your fitness regimen now includes running 5km three times weekly plus strength training twice weekly, all working toward your half marathon goal in November. The strength training will definitely complement your running performance."}
        ],
        
        # 10. Reading list
        "reading_list": [
            {"type": "human", "text": "I'm currently reading 'Thinking Fast and Slow' by Daniel Kahneman."},
            {"type": "ai", "text": "That's an excellent choice! 'Thinking Fast and Slow' explores cognitive biases and decision-making processes. How are you finding it so far?"},
            {"type": "human", "text": "It's fascinating. I'm also planning to read 'Atomic Habits' by James Clear next."},
            {"type": "ai", "text": "Both great books! 'Thinking Fast and Slow' gives insights into how we think, while 'Atomic Habits' offers practical strategies for behavior change. They complement each other nicely."},
            {"type": "human", "text": "I've added 'Deep Work' by Cal Newport to my list as well, focusing on productivity books this year."},
            {"type": "ai", "text": "I see you have a thoughtful reading progression: starting with cognitive science ('Thinking Fast and Slow'), moving to habit formation ('Atomic Habits'), and adding focused productivity ('Deep Work'). There's a clear theme of self-improvement and cognitive enhancement in your reading list this year."}
        ]
    }
    
    # Add all conversations with small delays between them
    for topic, messages in conversations.items():
        conversation_id = f"test_conversation_{topic}_{uuid.uuid4().hex[:5]}"
        await add_conversation(conversation_id, messages)
        # Allow time for processing
        await asyncio.sleep(2)
    
    # Test memory retrieval with various queries to demonstrate evolution
    test_queries = [
        # General professional context
        "Tell me about Alex's professional background",
        # Technology preferences that evolved
        "What programming languages and databases does the user prefer?",
        # Travel preferences evolution
        "What are the user's vacation preferences?",
        # Food preferences that became more specific
        "What food does the user like and are there any dietary restrictions?",
        # Contact information that changed
        "What is the user's email address?",
        # Project deadline that shifted multiple times
        "When is the analytics dashboard due?",
        # Learning interests that expanded
        "What AI topics is the user interested in learning about?",
        # Home automation setup that evolved
        "Describe the user's smart home setup",
        # Fitness goals that built upon each other
        "What are the user's fitness goals and routine?",
        # Reading interests that show a pattern
        "What books is the user interested in?"
    ]
    
    # Run memory retrieval tests
    for query in test_queries:
        await test_retrieve_memory(query)
        await asyncio.sleep(1)
    
    print("\n=== Memory Evolution Test Completed ===")

if __name__ == "__main__":
    # Create and run the async event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_comprehensive_memory_evolution_test())