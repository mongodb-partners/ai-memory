# AI-Memory-Service

## Table of Contents
1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Core Components](#3-core-components)
4. [Memory Cognitive Architecture](#4-memory-cognitive-architecture)
5. [Installation & Deployment](#5-installation--deployment)
6. [Configuration](#6-configuration)
7. [API Reference](#7-api-reference)
8. [Memory Operations](#8-memory-operations)
9. [Search Capabilities](#9-search-capabilities)
10. [Security & Monitoring](#10-security--monitoring)
11. [Development Guide](#11-development-guide)
12. [Advanced Features](#12-advanced-features)

## 1. Overview

AI Memory Service is an advanced cognitive memory system designed to transform AI memory from simple storage into a sophisticated architecture that evaluates, reinforces, and connects knowledge like a human brain. By combining MongoDB Atlas's vector search capabilities with AWS Bedrock's AI services, the system builds memory networks that prioritize important information, strengthen connections through repetition, and recall relevant context precisely when needed.

Key features include:

- Importance-weighted memory storage and retrieval
- Dynamic reinforcement and decay of memories
- Semantic merging of related memories
- Hybrid search combining vector and full-text capabilities
- Contextual conversation retrieval with AI-powered summarization
- Automatic importance assessment of new information
- Memory pruning based on cognitive principles

## 2. System Architecture

```mermaid
graph TD
    Client(Client Application) --> FastAPI[FastAPI Service]
    FastAPI --> MemSvc[Memory Service]
    FastAPI --> ConvSvc[Conversation Service]
    FastAPI --> BedSvc[Bedrock Service]
    
    MemSvc --> Atlas[(MongoDB Atlas)]
    ConvSvc --> Atlas
    MemSvc & ConvSvc --> BedSvc
    
    BedSvc --> EmbedModel[Embedding Model]
    BedSvc --> LLMModel[LLM Model]
    
    Atlas --> ConvColl[Conversations Collection]
    Atlas --> MemColl[Memory Nodes Collection]
    
    ConvColl --> ConvVecIdx[Vector Search Index]
    ConvColl --> ConvTxtIdx[Fulltext Search Index]
    MemColl --> MemVecIdx[Vector Search Index]
```

The architecture follows a service-oriented design where each component has a specific responsibility in the cognitive memory pipeline, leveraging MongoDB Atlas for advanced vector storage and AWS Bedrock for AI reasoning capabilities.

## 3. Core Components

1. **FastAPI Service Layer**
   - Purpose: Handles HTTP requests and orchestrates memory operations
   - Technologies: Python 3.10+, FastAPI 0.115+
   - Key endpoints: Add conversation messages, retrieve memories, search memories

2. **MongoDB Atlas Integration**
   - Purpose: Provides persistent storage with vector search capabilities
   - Collections:
     - Conversations: Stores raw conversation history with embeddings
     - Memory Nodes: Stores processed memory nodes with importance ratings
   - Indexes:
     - Vector search indexes for semantic retrieval
     - Full-text search indexes for keyword retrieval
     - Importance indexes for memory prioritization

3. **AWS Bedrock Service**
   - Purpose: Delivers AI capabilities for embedding and reasoning
   - Models:
     - Embedding Model: Generates vector representations (Titan)
     - LLM Model: Performs reasoning tasks (Claude)
   - Operations:
     - Embedding generation for semantic search
     - Importance assessment of new information
     - Memory summarization and merging
     - Conversation context summarization

4. **Memory Service**
   - Purpose: Manages the cognitive memory operations
   - Features:
     - Memory creation with importance assessment
     - Memory reinforcement and decay
     - Related memory merging
     - Memory pruning based on importance

5. **Conversation Service**
   - Purpose: Handles conversation storage and retrieval
   - Features:
     - Conversation history storage
     - Context retrieval around specific points
     - Hybrid search across conversations
     - AI-powered conversation summarization

## 4. Memory Cognitive Architecture

The system implements a cognitive architecture for memory that mimics human memory processes:

```mermaid
flowchart TD
    Start([New Content]) --> Embed[Generate Embeddings]
    Embed --> CheckSim{Similar Memory Exists?}
    
    CheckSim -->|Is >0.85 Similar| Reinforce[Reinforce Existing Memory]
    CheckSim -->|Not Similar| Assess[Assess Importance with LLM]
    
    Assess --> CreateSumm[Generate Summary]
    CreateSumm --> CreateNode[Create Memory Node]
    CreateNode --> CheckMerge{Mergeable Memories?}
    
    CheckMerge -->|Similar 0.7-0.85| Merge[Merge Related Memories]
    CheckMerge -->|Not Similar| UpdateImportance[Update Other Memories]
    
    Merge --> UpdateImportance
    Reinforce --> UpdateImportance
    
    UpdateImportance --> CheckPrune{Memory Count > Max?}
    CheckPrune -->|Yes| Prune[Prune Least Important]
    CheckPrune -->|No| Complete([Complete])
```

Key cognitive processes:
- **Importance Assessment**: Using AI to evaluate memory significance
- **Memory Reinforcement**: Strengthening memories through repetition
- **Memory Decay**: Gradually reducing importance of unused memories
- **Memory Merging**: Combining related information for coherent knowledge
- **Memory Pruning**: Removing less important memories when capacity is reached

## 5. Installation & Deployment

### Prerequisites
- Python 3.10+
- MongoDB Atlas account with vector search capability
- AWS account with Bedrock access
- Docker (optional)

### Local Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mongodb-partners/ai-memory.git
   cd ai-memory
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (see Configuration section)

4. Run the application:
   ```bash
   python main.py
   ```

### Docker Deployment
1. Build the Docker image:
   ```bash
   docker build -t ai-memory .
   ```

2. Run the container:
   ```bash
   docker run -p 8182:8182 --env-file .env ai-memory
   ```

## 6. Configuration

### Environment Variables
Configure the application using environment variables or a `.env` file:

```
# MongoDB Atlas Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DB_NAME=ai_memory

# AWS Configuration
AWS_REGION=us-east-1
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1
LLM_MODEL_ID=us.anthropic.claude-3-7-sonnet-20250219-v1:0

# Memory System Parameters
MAX_DEPTH=5
SIMILARITY_THRESHOLD=0.7
DECAY_FACTOR=0.99
REINFORCEMENT_FACTOR=1.1

# Service Configuration
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8182
DEBUG=False
```

### Memory Parameters

The cognitive behavior of the memory system can be tuned through these parameters:

- **MAX_DEPTH**: Maximum number of memories per user (default: 5)
- **SIMILARITY_THRESHOLD**: Threshold for memory reinforcement (default: 0.7)
- **DECAY_FACTOR**: Rate at which memories fade (default: 0.99)
- **REINFORCEMENT_FACTOR**: Strength of memory reinforcement (default: 1.1)

## 7. API Reference

### Endpoints

- **POST /conversation/**
  - Purpose: Add a message to the conversation history
  - Request Body: MessageInput model
  - Response: Confirmation message
  - Example:
    ```json
    {
      "user_id": "user123",
      "conversation_id": "conv456",
      "type": "human",
      "text": "I prefer to be contacted via email at john@example.com",
      "timestamp": "2023-06-10T14:30:00Z"
    }
    ```

- **GET /retrieve_memory/**
  - Purpose: Retrieve memory items, context, and similar memory nodes
  - Query Parameters: user_id, text
  - Response: Related conversation, conversation summary, and similar memories
  - Example URL: `/retrieve_memory/?user_id=user123&text=contact preference`

- **GET /health**
  - Purpose: Health check endpoint
  - Response: Status information

### Models

Key data models:

- **MessageInput**: Represents a conversation message
- **MemoryNode**: Represents a memory node with importance scoring
- **SearchRequest**: Request for memory search
- **ErrorResponse**: Standardized error response

## 8. Memory Operations

### Memory Creation

New memories are created from significant human messages:
1. Message is converted to embeddings
2. Similar memories are checked
3. If no similar memory exists, importance is assessed
4. A summary is generated
5. The memory node is created with metadata

### Memory Retrieval

```mermaid
flowchart TD
    Query[User Query] --> Embed[Generate Query Embedding]
    Embed --> ParallelOps[Parallel Operations]
    ParallelOps --> HybridSearch[Hybrid Search]
    ParallelOps --> MemorySearch[Memory Nodes Search]
    HybridSearch --> Combine[Combine Results]
    MemorySearch --> CalcImp[Calculate Effective Importance]
    Combine & CalcImp --> Response[Build Response]
```

Memories are retrieved through a sophisticated process:
1. Query embeddings are generated
2. Hybrid search combines vector and text search
3. Memory nodes are searched directly
4. Context is retrieved around matching points
5. Summaries are generated for conversations
6. Results are combined with importance weighing

### Memory Updating

Memories evolve through:
1. Reinforcement when similar content appears
2. Decay when not accessed
3. Merging when related information is found
4. Pruning when capacity is exceeded

## 9. Search Capabilities

### Hybrid Search Mechanism

```mermaid
flowchart TD
    Start([Search Query]) --> Split[Split Processing]
    Split --> Text[Text Query]
    Split --> Vector[Vector Query]
    Text --> TextSearch["$search Aggregation"]
    Vector --> VectorSearch["$vectorSearch Aggregation"]
    TextSearch & VectorSearch --> Union["$unionWith Operation"]
    Union --> Group["$group by document ID"]
    Group --> Weighted["Weighted Score Combination"]
    Weighted --> Sort["Sort by hybrid_score"]
```

The system combines multiple search methodologies:
1. Vector search for semantic understanding
2. Full-text search for keyword precision
3. Score normalization across methodologies
4. Weighted combination of results
5. Context retrieval and summarization

### Vector Search Configuration

MongoDB Atlas vector search is configured for optimal performance:
- Vector dimension: 1536 (Titan embeddings)
- Similarity metric: Cosine similarity
- Query filter: User ID filtering
- numCandidates: 200 (tunable parameter)

## 10. Security & Monitoring

### Security Considerations
- Use HTTPS for all API communications
- Implement authentication for API access
- Regularly rotate AWS and MongoDB credentials
- Apply least privilege principle for service users
- Consider encryption for sensitive memory content

### Monitoring & Logging
- The system uses the `logger.py` for structured logging
- Key metrics to monitor:
  - Memory creation rate and distribution
  - Average memory importance scores
  - Prune frequency and volume
  - Query latency for memory retrieval
  - AWS Bedrock API usage and costs
- Log levels can be configured based on operational needs

## 11. Development Guide

### Adding New Features
1. Extend the appropriate service module
2. Update models if necessary
3. Add new API endpoints in main.py
4. Update MongoDB indexes if needed
5. Document changes and update tests

### Testing
- Test memory creation with various importance levels
- Verify memory reinforcement and decay behavior
- Benchmark hybrid search performance
- Test with different memory parameters

### Best Practices
- Use type hints and descriptive variable names
- Document all functions with docstrings
- Use the logger for all significant operations
- Handle exceptions appropriately
- Consider backward compatibility for API changes

## 12. Advanced Features

### Memory Hierarchy
The system can be extended to support hierarchical memory structures with parent-child relationships between memories.

### Enhanced Importance Assessment
The importance evaluation can be made more sophisticated by considering:
- User feedback on memory relevance
- Time-based relevance decay
- Domain-specific importance metrics
- User behavior patterns

### Multi-modal Memory
Future versions can incorporate:
- Image embeddings and memory
- Voice pattern memory
- Document and structured data memory
- Cross-modal associative memory

### Federation and Privacy
Advanced implementations can support:
- Federated memory across services
- Privacy-preserving memory operations
- Selective forgetting capabilities
- Memory encryption for sensitive data

This documentation provides a comprehensive overview of the AI-Memory-Service's cognitive architecture. For specific implementation details, refer to the comments and docstrings within the codebase.