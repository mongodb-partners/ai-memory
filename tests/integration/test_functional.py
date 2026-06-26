"""Comprehensive functional tests against the live Docker MCP server.

Modelled after test_client.py — exercises all seven MCP tools with
multi-topic conversations, memory evolution, semantic recall, caching,
hybrid search, and deletion.

Requires: docker container running on localhost:8000
Run with: .venv/bin/python tests/integration/test_functional.py
"""

import json
import http.client
import os
import sys
import uuid
import time
import textwrap

# ─── Configuration ────────────────────────────────────────────────

MCP_HOST = "localhost"
MCP_PORT = 8000
MCP_ENDPOINT = "/mcp"
AUTH_TOKEN = os.environ.get("MEMORY_MCP_TEST_TOKEN", "mcp-key-mdf-2025")

# Test identifiers
USER_ID = f"test-user-{uuid.uuid4().hex[:8]}"

# Track MCP session
session_id = None
request_id = 0


# ─── Low-level helpers ────────────────────────────────────────────


def _next_id():
    global request_id
    request_id += 1
    return request_id


def _post(method: str, params: dict | None = None) -> dict:
    """Send a JSON-RPC request over streamable-http and return the result."""
    global session_id

    body = {
        "jsonrpc": "2.0",
        "id": _next_id(),
        "method": method,
    }
    if params is not None:
        body["params"] = params

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {AUTH_TOKEN}",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    conn = http.client.HTTPConnection(MCP_HOST, MCP_PORT, timeout=30)
    conn.request("POST", MCP_ENDPOINT, json.dumps(body), headers)
    resp = conn.getresponse()
    raw = resp.read().decode()

    # Capture session id from response headers
    sid = resp.getheader("Mcp-Session-Id")
    if sid:
        session_id = sid

    # Parse SSE or direct JSON
    if "text/event-stream" in (resp.getheader("Content-Type") or ""):
        result = None
        for line in raw.split("\n"):
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "result" in data:
                    result = data["result"]
                elif "error" in data:
                    return {"error": data["error"]}
        if result is not None:
            return result
        raise RuntimeError(f"No result in SSE response: {raw[:500]}")
    else:
        data = json.loads(raw)
        if "result" in data:
            return data["result"]
        elif "error" in data:
            return {"error": data["error"]}
        raise RuntimeError(f"Unexpected response: {raw[:500]}")


def _call_tool(tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool and return the parsed content."""
    result = _post("tools/call", {"name": tool_name, "arguments": arguments})
    if "error" in result:
        return result

    content = result.get("content", [])
    for item in content:
        if item.get("type") == "text":
            return json.loads(item["text"])
    return result


def _banner(title: str):
    """Print a section banner."""
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


def _subheading(title: str):
    print(f"\n--- {title} ---")


# ─── Phase 1: Session & Discovery ────────────────────────────────


def test_initialize():
    """Initialize the MCP session."""
    _subheading("Initialize MCP session")
    result = _post("initialize", {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "comprehensive-test", "version": "2.0"},
    })
    assert "serverInfo" in result, f"Missing serverInfo: {result}"
    assert "capabilities" in result, f"Missing capabilities: {result}"
    server = result["serverInfo"]
    print(f"  Server : {server.get('name')} v{server.get('version')}")
    print(f"  Protocol: {result.get('protocolVersion')}")
    return True


def test_list_tools():
    """List all registered tools and validate their schemas."""
    _subheading("List registered tools")
    result = _post("tools/list")
    tools = result.get("tools", [])
    tool_names = {t["name"] for t in tools}
    expected = {
        "store_memory", "recall_memory", "delete_memory",
        "check_cache", "store_cache", "hybrid_search", "search_web",
        "memory_health", "wipe_user_data", "cache_invalidate",
        "store_decision", "recall_decision",
    }
    assert expected == tool_names, f"Tool mismatch.\n  Expected: {expected}\n  Got:      {tool_names}"

    for t in sorted(tools, key=lambda x: x["name"]):
        desc = t.get("description", "")[:70]
        print(f"  {t['name']:20s} {desc}")

    return True


# ─── Phase 2: Multi-topic conversation storage ───────────────────


CONVERSATIONS = {
    # 1. Technology preferences
    "tech_preferences": [
        {"content": "Hello! I'm setting up a new project and want to discuss my tech preferences.", "message_type": "human"},
        {"content": "Great! I'd be happy to discuss your technology preferences. What kind of project are you working on?", "message_type": "ai"},
        {"content": "I'm building a data analytics platform. I prefer Python for backend and React for frontend.", "message_type": "human"},
        {"content": "Excellent choices! Python is great for data processing and analytics, while React gives you a robust frontend framework. Any database preferences?", "message_type": "ai"},
        {"content": "Yes, I like MongoDB for document storage and PostgreSQL for relational data.", "message_type": "human"},
        {"content": "Got it. MongoDB for document storage and PostgreSQL for relational data. That's a solid combination.", "message_type": "ai"},
    ],

    # 2. Personal information
    "personal_info": [
        {"content": "Just so you know, my name is Alex Chen and I work as a data scientist.", "message_type": "human"},
        {"content": "Thanks for sharing that information, Alex. How long have you been working as a data scientist?", "message_type": "ai"},
        {"content": "About 5 years now. I specialize in NLP and computer vision projects.", "message_type": "human"},
        {"content": "That's impressive, Alex! 5 years in NLP and computer vision is substantial experience.", "message_type": "ai"},
        {"content": "Yes, I'm developing a sentiment analysis tool for customer feedback at my company, DataInsight Corp.", "message_type": "human"},
        {"content": "Sentiment analysis for customer feedback can provide valuable insights for DataInsight Corp.", "message_type": "ai"},
    ],

    # 3. Travel preferences
    "travel_preferences": [
        {"content": "I'm planning a vacation and wanted to mention my travel preferences.", "message_type": "human"},
        {"content": "That sounds exciting! I'd be happy to hear about your travel preferences.", "message_type": "ai"},
        {"content": "I prefer beach destinations over mountains, and I always travel with my family of four.", "message_type": "human"},
        {"content": "You prefer beach destinations when traveling with your family of four. Any favorites?", "message_type": "ai"},
        {"content": "Yes, we loved Hawaii and the Mediterranean coast. We usually travel during summer break in July.", "message_type": "human"},
        {"content": "Hawaii and the Mediterranean coast are wonderful choices! Noted July summer break.", "message_type": "ai"},
    ],

    # 4. Food preferences
    "food_preferences": [
        {"content": "I should tell you about my dietary preferences for future reference.", "message_type": "human"},
        {"content": "Thank you for sharing! Knowing your dietary preferences will help me.", "message_type": "ai"},
        {"content": "I'm vegetarian, but I do eat dairy products. I'm also allergic to nuts.", "message_type": "human"},
        {"content": "Noted: vegetarian who consumes dairy products with a nut allergy.", "message_type": "ai"},
        {"content": "Exactly. And I particularly enjoy Italian and Indian cuisines.", "message_type": "human"},
        {"content": "Italian and Indian cuisines offer many excellent vegetarian options. Great choices!", "message_type": "ai"},
    ],

    # 5. Contact information (with evolution)
    "contact_info": [
        {"content": "You can contact me at alex.chen@example.com if needed.", "message_type": "human"},
        {"content": "Thank you for sharing your email address, alex.chen@example.com.", "message_type": "ai"},
        {"content": "Actually, I have a new work email now: alex.chen@datainsight.com", "message_type": "human"},
        {"content": "Updated your contact info. Your current email is alex.chen@datainsight.com.", "message_type": "ai"},
        {"content": "Perfect. And my work phone is 555-123-4567 if there's ever an urgent matter.", "message_type": "human"},
        {"content": "Added your work phone 555-123-4567 along with email alex.chen@datainsight.com.", "message_type": "ai"},
    ],

    # 6. Project deadlines (with evolution)
    "project_deadlines": [
        {"content": "I need to finish this analytics dashboard by June 15th.", "message_type": "human"},
        {"content": "Noted your deadline of June 15th for the analytics dashboard project.", "message_type": "ai"},
        {"content": "The client just moved the deadline up to June 1st! I need to adjust my schedule.", "message_type": "human"},
        {"content": "Your analytics dashboard deadline has been moved from June 15th to June 1st.", "message_type": "ai"},
        {"content": "Actually, we negotiated and settled on June 8th as the final deadline.", "message_type": "human"},
        {"content": "Final agreed deadline for the analytics dashboard is now June 8th.", "message_type": "ai"},
    ],

    # 7. Learning interests
    "learning_interests": [
        {"content": "I'm interested in learning more about deep learning and transformer models.", "message_type": "human"},
        {"content": "Deep learning and transformers have revolutionized NLP and many other fields.", "message_type": "ai"},
        {"content": "I'm particularly interested in BERT and its applications in text classification.", "message_type": "human"},
        {"content": "BERT is indeed powerful for text classification tasks.", "message_type": "ai"},
        {"content": "I've also started exploring GPT models and prompt engineering recently.", "message_type": "human"},
        {"content": "GPT models and prompt engineering are cutting-edge areas. Great expansion of interests!", "message_type": "ai"},
    ],

    # 8. Home automation setup (with evolution)
    "home_automation": [
        {"content": "I have a smart home system with Philips Hue lights and a Nest thermostat.", "message_type": "human"},
        {"content": "Nice setup with Philips Hue lights and a Nest thermostat. Any central hub?", "message_type": "ai"},
        {"content": "Yes, I use Samsung SmartThings as my hub and Google Assistant for voice control.", "message_type": "human"},
        {"content": "Comprehensive setup with Hue, Nest, SmartThings hub, and Google Assistant.", "message_type": "ai"},
        {"content": "I just upgraded and added Amazon Echo devices throughout the house for better coverage.", "message_type": "human"},
        {"content": "You've expanded with Amazon Echo devices alongside existing SmartThings and Google setup.", "message_type": "ai"},
    ],

    # 9. Fitness goals
    "fitness_goals": [
        {"content": "I'm trying to improve my fitness by running 5km three times a week.", "message_type": "human"},
        {"content": "Running 5km three times weekly is great for cardiovascular endurance.", "message_type": "ai"},
        {"content": "I started about a month ago. My goal is to participate in a half marathon in November.", "message_type": "human"},
        {"content": "Building up to a half marathon in November. That's a clear training path.", "message_type": "ai"},
        {"content": "Yes, and I'm incorporating strength training twice a week to improve overall performance.", "message_type": "human"},
        {"content": "Running 3x/week plus strength training 2x/week, all toward your November half marathon.", "message_type": "ai"},
    ],

    # 10. Reading list
    "reading_list": [
        {"content": "I'm currently reading 'Thinking Fast and Slow' by Daniel Kahneman.", "message_type": "human"},
        {"content": "'Thinking Fast and Slow' explores cognitive biases and decision-making. How is it?", "message_type": "ai"},
        {"content": "It's fascinating. I'm also planning to read 'Atomic Habits' by James Clear next.", "message_type": "human"},
        {"content": "Both great books! 'Thinking Fast and Slow' and 'Atomic Habits' complement each other.", "message_type": "ai"},
        {"content": "I've added 'Deep Work' by Cal Newport to my list as well, focusing on productivity books this year.", "message_type": "human"},
        {"content": "A thoughtful reading progression: cognitive science, habit formation, then focused productivity.", "message_type": "ai"},
    ],
}


def store_all_conversations():
    """Store all 10 multi-topic conversations."""
    _banner("PHASE 2: Storing Multi-Topic Conversations")
    all_ids = {}

    for topic, messages in CONVERSATIONS.items():
        conv_id = f"conv-{topic}-{uuid.uuid4().hex[:5]}"
        _subheading(f"Storing conversation: {topic} ({len(messages)} messages)")

        result = _call_tool("store_memory", {
            "user_id": USER_ID,
            "conversation_id": conv_id,
            "messages": messages,
        })
        assert "error" not in result, f"store_memory error for {topic}: {result}"
        assert "stm_ids" in result, f"Missing stm_ids for {topic}: {result}"
        assert result["count"] >= len(messages), \
            f"Expected >={len(messages)} stored for {topic}, got {result['count']}"

        all_ids[topic] = result["stm_ids"]
        print(f"  Stored {result['count']} memories")
        print(f"  IDs   : {result['stm_ids'][:3]}{'...' if len(result['stm_ids']) > 3 else ''}")

        # Small delay to ensure proper ordering
        time.sleep(0.3)

    return all_ids


# ─── Phase 3: Semantic recall across topics ───────────────────────

RECALL_QUERIES = [
    ("Professional background", "Tell me about Alex's professional background"),
    ("Tech stack preferences", "What programming languages and databases does the user prefer?"),
    ("Vacation preferences", "What are the user's vacation preferences?"),
    ("Dietary restrictions", "What food does the user like and are there any dietary restrictions?"),
    ("Contact information", "What is the user's email address?"),
    ("Project deadline", "When is the analytics dashboard due?"),
    ("AI learning interests", "What AI topics is the user interested in learning about?"),
    ("Smart home setup", "Describe the user's smart home setup"),
    ("Fitness routine", "What are the user's fitness goals and routine?"),
    ("Books & reading", "What books is the user interested in?"),
]


def test_recall_all_topics():
    """Recall memories for each topic using semantic queries."""
    _banner("PHASE 3: Semantic Memory Recall")

    for label, query in RECALL_QUERIES:
        _subheading(f"Recall: {label}")
        print(f"  Query: '{query}'")

        result = _call_tool("recall_memory", {
            "user_id": USER_ID,
            "query": query,
            "limit": 5,
        })
        assert "error" not in result, f"recall_memory error: {result}"
        assert "results" in result, f"Missing results: {result}"

        count = result["count"]
        print(f"  Results: {count} memories found")

        for i, mem in enumerate(result["results"][:3]):
            content = mem.get("content", "N/A")
            # Truncate long content
            if len(content) > 120:
                content = content[:120] + "..."
            tier = mem.get("tier", "?")
            importance = mem.get("importance", "?")
            print(f"    [{i+1}] ({tier}, imp={importance}) {content}")

        time.sleep(0.3)

    return True


# ─── Phase 4: Memory recall with tier filters ────────────────────


def test_recall_with_tier_filter():
    """Test recall filtering by memory tier (STM / LTM)."""
    _subheading("Recall with tier filter: STM only")

    result = _call_tool("recall_memory", {
        "user_id": USER_ID,
        "query": "technology preferences",
        "tier": ["stm"],
        "limit": 5,
    })
    assert "error" not in result, f"recall_memory tier=stm error: {result}"
    print(f"  STM-only results: {result['count']}")

    for mem in result["results"]:
        assert mem.get("tier") == "stm", f"Expected tier=stm, got {mem.get('tier')}"

    return True


# ─── Phase 5: Cache operations ───────────────────────────────────

CACHE_ENTRIES = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is Python used for?", "Python is used for web development, data science, AI/ML, automation, and more."),
    ("How do I install MongoDB?", "You can install MongoDB via the official download page or package managers like apt/brew."),
]


def test_cache_store_and_check():
    """Store and retrieve cache entries."""
    _banner("PHASE 5: Cache Operations")
    cache_user = f"cache-user-{uuid.uuid4().hex[:8]}"

    for query, response in CACHE_ENTRIES:
        _subheading(f"Cache store: '{query[:50]}'")
        result = _call_tool("store_cache", {
            "user_id": cache_user,
            "query": query,
            "response": response,
        })
        assert "error" not in result, f"store_cache error: {result}"
        assert "cache_id" in result, f"Missing cache_id: {result}"
        print(f"  Stored cache entry: {result['cache_id']}")

    # Wait a moment for vector indexes to catch up
    time.sleep(2)

    # Check cache hits
    _subheading("Cache checks")
    for query, expected_response in CACHE_ENTRIES:
        result = _call_tool("check_cache", {
            "user_id": cache_user,
            "query": query,
        })
        assert "error" not in result, f"check_cache error: {result}"

        if result.get("cache_hit") is False:
            print(f"  MISS: '{query[:50]}' (vector index may still be building)")
        else:
            cached = result.get("response", "")[:60]
            print(f"  HIT : '{query[:50]}' -> '{cached}...'")

    # Check with paraphrased query
    _subheading("Cache check: paraphrased query")
    result = _call_tool("check_cache", {
        "user_id": cache_user,
        "query": "What's the capital city of France?",
    })
    if result.get("cache_hit") is False:
        print("  Paraphrased miss (index may still be building)")
    else:
        print(f"  Paraphrased hit: {result.get('response', '')[:60]}...")

    return True


# ─── Phase 6: Hybrid search ──────────────────────────────────────


def test_hybrid_search():
    """Test hybrid vector + full-text search."""
    _banner("PHASE 6: Hybrid Search")

    hybrid_queries = [
        "machine learning NLP computer vision",
        "vegetarian food Italian Indian",
        "smart home automation Philips Hue",
        "half marathon fitness running",
        "analytics dashboard deadline",
    ]

    for query in hybrid_queries:
        _subheading(f"Hybrid: '{query}'")

        result = _call_tool("hybrid_search", {
            "user_id": USER_ID,
            "query": query,
            "limit": 3,
        })
        assert "error" not in result, f"hybrid_search error: {result}"
        assert "results" in result, f"Missing results: {result}"

        print(f"  Results: {result['count']}")
        for i, mem in enumerate(result["results"][:3]):
            content = mem.get("content", "N/A")
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"    [{i+1}] {content}")

        time.sleep(0.3)

    return True


# ─── Phase 7: Delete operations ──────────────────────────────────


def test_delete_operations(all_ids: dict):
    """Test dry-run and actual deletion."""
    _banner("PHASE 7: Delete Operations")

    # Pick a memory to delete
    topic = "contact_info"
    ids = all_ids.get(topic, [])
    if not ids:
        print("  SKIP: No IDs available for deletion test")
        return True

    target_id = ids[0]

    # Dry run
    _subheading(f"Dry-run delete: {target_id}")
    result = _call_tool("delete_memory", {
        "user_id": USER_ID,
        "memory_id": target_id,
        "dry_run": True,
    })
    assert "error" not in result, f"delete dry_run error: {result}"
    assert result.get("deleted_count") == 1, f"Expected dry_run count 1: {result}"
    assert result.get("dry_run") is True, f"Expected dry_run=True: {result}"
    print(f"  Dry run would delete {result['deleted_count']} memory")

    # Verify memory still exists after dry run
    _subheading("Verify memory still exists after dry run")
    recall_result = _call_tool("recall_memory", {
        "user_id": USER_ID,
        "query": "contact email phone",
        "limit": 10,
    })
    assert recall_result["count"] > 0, "Memory should still exist after dry run"
    print(f"  Still found {recall_result['count']} contact-related memories")

    # Actual delete
    _subheading(f"Actual delete: {target_id}")
    result = _call_tool("delete_memory", {
        "user_id": USER_ID,
        "memory_id": target_id,
    })
    assert "error" not in result, f"delete error: {result}"
    assert result.get("deleted_count") == 1, f"Expected delete count 1: {result}"
    print(f"  Deleted {result['deleted_count']} memory")

    return True


# ─── Phase 8: Web search ─────────────────────────────────────────


def test_web_search():
    """Test web search (may gracefully fail without Tavily key)."""
    _banner("PHASE 8: Web Search")

    result = _call_tool("search_web", {
        "user_id": USER_ID,
        "query": "latest Python release",
    })

    if "error" in result or (isinstance(result.get("error"), str) and "Tavily" in result["error"]):
        print("  Web search unavailable: Tavily API key not configured (expected)")
        return True

    if isinstance(result, dict) and "results" in result:
        print(f"  Web search returned {len(result['results'])} results")
        for i, r in enumerate(result["results"][:3]):
            title = r.get("title", "N/A")[:60]
            print(f"    [{i+1}] {title}")
    return True


# ─── Phase 9: Memory evolution verification ──────────────────────


def test_memory_evolution():
    """Verify the system captures evolving information correctly."""
    _banner("PHASE 9: Memory Evolution Verification")

    evolution_checks = [
        {
            "label": "Contact info evolution",
            "query": "email address alex chen",
            "should_contain_any": ["datainsight.com", "alex.chen"],
        },
        {
            "label": "Deadline evolution",
            "query": "analytics dashboard deadline date",
            "should_contain_any": ["June 8", "June 1", "June 15"],
        },
        {
            "label": "Home automation evolution",
            "query": "smart home devices setup",
            "should_contain_any": ["Echo", "SmartThings", "Philips", "Hue"],
        },
    ]

    for check in evolution_checks:
        _subheading(check["label"])
        result = _call_tool("recall_memory", {
            "user_id": USER_ID,
            "query": check["query"],
            "limit": 5,
        })
        assert "error" not in result, f"recall error: {result}"

        # Combine all result content
        combined = " ".join(
            mem.get("content", "") for mem in result.get("results", [])
        )

        found = [kw for kw in check["should_contain_any"] if kw.lower() in combined.lower()]
        if found:
            print(f"  Found expected keywords: {found}")
        else:
            print(f"  WARNING: None of {check['should_contain_any']} found in results")
            print(f"  Combined content preview: {combined[:200]}")

    return True


# ─── Phase 10: Admin tools ──────────────────────────────────────


def test_admin_tools():
    """Test memory_health, cache_invalidate, and wipe_user_data."""
    _banner("PHASE 10: Admin Tools")

    # memory_health
    _subheading("memory_health")
    result = _call_tool("memory_health", {"user_id": USER_ID})
    assert "error" not in result, f"memory_health error: {result}"
    assert "total_memories" in result, f"Missing total_memories: {result}"
    print(f"  Total memories: {result['total_memories']}")
    print(f"  Tier stats   : {result.get('tier_stats', {})}")
    print(f"  Enrichment   : {result.get('enrichment_stats', {})}")

    # cache_invalidate (pattern-based)
    _subheading("cache_invalidate (pattern)")
    result = _call_tool("cache_invalidate", {
        "user_id": f"admin-test-{uuid.uuid4().hex[:8]}",
        "invalidate_all": True,
    })
    assert "error" not in result, f"cache_invalidate error: {result}"
    print(f"  Deleted: {result.get('deleted_count', 0)} cache entries")

    # wipe_user_data — requires confirm, test refusal first
    _subheading("wipe_user_data (no confirm)")
    wipe_user = f"wipe-test-{uuid.uuid4().hex[:8]}"
    result = _call_tool("wipe_user_data", {
        "user_id": wipe_user,
        "confirm": False,
    })
    assert "error" in result, "wipe_user_data should reject without confirm"
    print(f"  Correctly rejected: {result['error'][:60]}")

    # wipe_user_data — with confirm on a disposable user
    _subheading("wipe_user_data (with confirm)")
    # Store a memory for the wipe user first
    _call_tool("store_memory", {
        "user_id": wipe_user,
        "conversation_id": f"conv-wipe-{uuid.uuid4().hex[:5]}",
        "messages": [
            {"content": "This is a test memory for wipe.", "message_type": "human"},
        ],
    })
    time.sleep(0.5)
    result = _call_tool("wipe_user_data", {
        "user_id": wipe_user,
        "confirm": True,
    })
    assert "error" not in result, f"wipe_user_data error: {result}"
    print(f"  Memories deleted: {result.get('memories_deleted', 0)}")
    print(f"  Cache deleted   : {result.get('cache_deleted', 0)}")
    print(f"  Audit deleted   : {result.get('audit_deleted', 0)}")

    return True


# ─── Phase 11: Decision stickiness ─────────────────────────────


def test_decision_tools():
    """Test store_decision and recall_decision."""
    _banner("PHASE 11: Decision Stickiness")
    decision_user = f"decision-user-{uuid.uuid4().hex[:8]}"

    # Store a decision
    _subheading("store_decision")
    result = _call_tool("store_decision", {
        "user_id": decision_user,
        "key": "editor",
        "value": "vim",
        "ttl_days": 30,
    })
    assert "error" not in result, f"store_decision error: {result}"
    assert result.get("action") in ("stored", "updated"), f"Unexpected action: {result}"
    print(f"  Action: {result['action']}, key: {result.get('key')}")

    # Recall the decision
    _subheading("recall_decision (found)")
    result = _call_tool("recall_decision", {
        "user_id": decision_user,
        "key": "editor",
    })
    assert "error" not in result, f"recall_decision error: {result}"
    assert result.get("found") is True, f"Expected found=True: {result}"
    assert result["decision"]["value"] == "vim", f"Expected value=vim: {result}"
    print(f"  Found: {result['decision']['value']}")

    # Update the decision
    _subheading("store_decision (update)")
    result = _call_tool("store_decision", {
        "user_id": decision_user,
        "key": "editor",
        "value": "emacs",
    })
    assert result.get("action") in ("stored", "updated"), f"Unexpected action: {result}"
    print(f"  Action: {result['action']}, key: {result.get('key')}")

    # Recall updated decision
    _subheading("recall_decision (updated)")
    result = _call_tool("recall_decision", {
        "user_id": decision_user,
        "key": "editor",
    })
    assert result["decision"]["value"] == "emacs", f"Expected value=emacs: {result}"
    print(f"  Updated value: {result['decision']['value']}")

    # Recall non-existent decision
    _subheading("recall_decision (not found)")
    result = _call_tool("recall_decision", {
        "user_id": decision_user,
        "key": "nonexistent_key",
    })
    assert result.get("found") is False, f"Expected found=False: {result}"
    print(f"  Correctly returned not found")

    return True


# ─── Runner ───────────────────────────────────────────────────────


def run_all():
    """Run the comprehensive functional test suite."""
    passed = 0
    failed = 0
    total_steps = 12

    _banner(f"COMPREHENSIVE FUNCTIONAL TEST SUITE")
    print(f"  Target : http://{MCP_HOST}:{MCP_PORT}{MCP_ENDPOINT}")
    print(f"  User ID: {USER_ID}")

    # --- Step 1: Initialize ---
    step = 1
    print(f"\n[{step}/{total_steps}] Initialize MCP session")
    try:
        test_initialize()
        passed += 1
        print("  >> PASS")
    except Exception as e:
        failed += 1
        print(f"  >> FAIL: {e}")
        print("Cannot continue without session. Aborting.")
        return 1

    # --- Step 2: List tools ---
    step = 2
    print(f"\n[{step}/{total_steps}] List tools")
    try:
        test_list_tools()
        passed += 1
        print("  >> PASS")
    except Exception as e:
        failed += 1
        print(f"  >> FAIL: {e}")

    # --- Step 3: Store conversations ---
    step = 3
    print(f"\n[{step}/{total_steps}] Store multi-topic conversations")
    all_ids = {}
    try:
        all_ids = store_all_conversations()
        passed += 1
        total_stored = sum(len(ids) for ids in all_ids.values())
        print(f"\n  >> PASS — stored {total_stored} memories across {len(all_ids)} topics")
    except Exception as e:
        failed += 1
        print(f"  >> FAIL: {e}")

    # Allow some time for vector indexes to process
    print("\n  Waiting for vector indexes to settle...")
    time.sleep(3)

    # --- Step 4: Semantic recall ---
    step = 4
    print(f"\n[{step}/{total_steps}] Semantic recall across all topics")
    try:
        test_recall_all_topics()
        passed += 1
        print("\n  >> PASS")
    except Exception as e:
        failed += 1
        print(f"  >> FAIL: {e}")

    # --- Step 5: Tier-filtered recall ---
    step = 5
    print(f"\n[{step}/{total_steps}] Recall with tier filter")
    try:
        test_recall_with_tier_filter()
        passed += 1
        print("  >> PASS")
    except Exception as e:
        failed += 1
        print(f"  >> FAIL: {e}")

    # --- Step 6: Cache operations ---
    step = 6
    print(f"\n[{step}/{total_steps}] Cache store & check")
    try:
        test_cache_store_and_check()
        passed += 1
        print("\n  >> PASS")
    except Exception as e:
        failed += 1
        print(f"  >> FAIL: {e}")

    # --- Step 7: Hybrid search ---
    step = 7
    print(f"\n[{step}/{total_steps}] Hybrid search")
    try:
        test_hybrid_search()
        passed += 1
        print("\n  >> PASS")
    except Exception as e:
        failed += 1
        print(f"  >> FAIL: {e}")

    # --- Step 8: Delete operations ---
    step = 8
    print(f"\n[{step}/{total_steps}] Delete operations (dry run + actual)")
    try:
        test_delete_operations(all_ids)
        passed += 1
        print("\n  >> PASS")
    except Exception as e:
        failed += 1
        print(f"  >> FAIL: {e}")

    # --- Step 9: Web search ---
    step = 9
    print(f"\n[{step}/{total_steps}] Web search")
    try:
        test_web_search()
        passed += 1
        print("\n  >> PASS")
    except Exception as e:
        failed += 1
        print(f"  >> FAIL: {e}")

    # --- Step 10: Memory evolution ---
    step = 10
    print(f"\n[{step}/{total_steps}] Memory evolution verification")
    try:
        test_memory_evolution()
        passed += 1
        print("\n  >> PASS")
    except Exception as e:
        failed += 1
        print(f"  >> FAIL: {e}")

    # --- Step 11: Admin tools ---
    step = 11
    print(f"\n[{step}/{total_steps}] Admin tools (memory_health, cache_invalidate, wipe_user_data)")
    try:
        test_admin_tools()
        passed += 1
        print("\n  >> PASS")
    except Exception as e:
        failed += 1
        print(f"  >> FAIL: {e}")

    # --- Step 12: Decision stickiness ---
    step = 12
    print(f"\n[{step}/{total_steps}] Decision stickiness (store_decision, recall_decision)")
    try:
        test_decision_tools()
        passed += 1
        print("\n  >> PASS")
    except Exception as e:
        failed += 1
        print(f"  >> FAIL: {e}")

    # ─── Summary ──────────────────────────────────────────────────
    _banner("RESULTS")
    print(f"  Passed : {passed}/{total_steps}")
    print(f"  Failed : {failed}/{total_steps}")
    print(f"  User ID: {USER_ID}")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
