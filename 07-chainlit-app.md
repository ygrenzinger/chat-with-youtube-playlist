# Component: Chainlit Chat Application

## Purpose

Provide a user-friendly chat interface for interacting with the Flow Conference RAG system, with streaming responses, source citations, and video links.

## Technology Choice: Chainlit

| Requirement | Chainlit Capability |
|-------------|---------------------|
| Python native | ✅ Direct integration with RAG pipeline |
| Streaming responses | ✅ Built-in support |
| Source citations | ✅ Elements system |
| Easy deployment | ✅ Docker, Cloud options |
| Customizable UI | ✅ Theming, components |
| Session management | ✅ User sessions built-in |

## User Interface Design

### Main Chat Interface

```
┌─────────────────────────────────────────────────────────────────┐
│  🎬 Flow Conference Assistant                              [⚙️] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 👋 Welcome! I can help you explore Flow Conference talks. │ │
│  │                                                           │ │
│  │ Ask me about:                                             │ │
│  │ • RAG systems and embeddings                              │ │
│  │ • LLMs and prompt engineering                             │ │
│  │ • Vector databases                                        │ │
│  │ • Any topic from the conference                           │ │
│  │                                                           │ │
│  │ 💡 Try these:                                             │ │
│  │ [What is RAG?] [Best chunking strategies] [Compare DBs]   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 👤 What did speakers say about vector databases?          │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 🤖 Several speakers discussed vector databases at the     │ │
│  │    conference:                                            │ │
│  │                                                           │ │
│  │    **Performance Considerations**                         │ │
│  │    In "Scaling RAG Systems", the speaker explained that   │ │
│  │    vector DB choice depends heavily on your query         │ │
│  │    patterns and data volume [Source 1].                   │ │
│  │                                                           │ │
│  │    **Comparison of Options**                              │ │
│  │    Another talk compared Pinecone, Weaviate, and Chroma,  │ │
│  │    noting that sqlite-vec works well for smaller          │ │
│  │    deployments [Source 2].                                │ │
│  │                                                           │ │
│  │    📚 Sources                                             │ │
│  │    ├─ [1] Scaling RAG Systems (12:30)                     │ │
│  │    │      https://youtube.com/watch?v=abc&t=750           │ │
│  │    └─ [2] Vector DBs Compared (5:45)                      │ │
│  │           https://youtube.com/watch?v=xyz&t=345           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  💬 Type your question...                           [Send ➤]   │
└─────────────────────────────────────────────────────────────────┘
```

## Application Structure

### File Organization

```
src/
├── app.py                 # Main Chainlit application
├── components/
│   ├── welcome.py         # Welcome message and examples
│   ├── sources.py         # Source citation formatting
│   └── settings.py        # User settings panel
├── pipeline/
│   ├── __init__.py
│   ├── rag.py             # RAG pipeline integration
│   └── streaming.py       # Streaming helpers
└── utils/
    ├── formatting.py      # Response formatting
    └── validation.py      # Input validation

config/
├── chainlit.md            # Welcome screen markdown
└── .chainlit/
    └── config.toml        # Chainlit configuration
```

### Main Application (app.py)

```python
import chainlit as cl
from pipeline.rag import RAGPipeline
from components.welcome import get_welcome_message, get_example_actions
from components.sources import create_source_elements
from utils.formatting import format_response_with_sources
import logging

# Initialize pipeline (singleton)
pipeline = None

def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        from config import load_config
        config = load_config()
        pipeline = RAGPipeline(config)
    return pipeline


@cl.on_chat_start
async def start():
    """
    Initialize chat session.
    Called when a new user starts a conversation.
    """
    # Initialize session state
    cl.user_session.set("history", [])
    cl.user_session.set("settings", get_default_settings())
    
    # Send welcome message with example buttons
    welcome_msg = cl.Message(
        content=get_welcome_message(),
        actions=get_example_actions()
    )
    await welcome_msg.send()


@cl.on_message
async def main(message: cl.Message):
    """
    Handle user messages.
    Main conversation loop.
    """
    user_query = message.content.strip()
    
    # Validate input
    if not user_query:
        await cl.Message(content="Please enter a question.").send()
        return
    
    if len(user_query) > 1000:
        await cl.Message(content="Question is too long. Please keep it under 1000 characters.").send()
        return
    
    # Get pipeline and settings
    rag = get_pipeline()
    settings = cl.user_session.get("settings", {})
    
    # Show searching indicator
    async with cl.Step(name="🔍 Searching conference talks...", type="tool") as step:
        # Retrieve relevant chunks
        try:
            retrieval_result = await rag.retrieve(user_query)
            step.output = f"Found {len(retrieval_result['chunks'])} relevant sources"
        except Exception as e:
            logging.error(f"Retrieval error: {e}")
            await cl.Message(content="Sorry, I encountered an error searching the talks. Please try again.").send()
            return
    
    # Check if we found relevant content
    if not retrieval_result["chunks"]:
        await cl.Message(
            content=f"I couldn't find relevant content about \"{user_query}\" in the conference talks. Try rephrasing your question or asking about a different topic."
        ).send()
        return
    
    # Create response message for streaming
    response_msg = cl.Message(content="")
    await response_msg.send()
    
    # Stream the response
    full_response = ""
    try:
        async for token in rag.generate_streaming(user_query, retrieval_result["context"]):
            full_response += token
            await response_msg.stream_token(token)
    except Exception as e:
        logging.error(f"Generation error: {e}")
        response_msg.content = "Sorry, I encountered an error generating the response. Please try again."
        await response_msg.update()
        return
    
    # Add source elements
    if settings.get("show_sources", True):
        source_elements = create_source_elements(retrieval_result["chunks"])
        response_msg.elements = source_elements
        await response_msg.update()
    
    # Update conversation history
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("history", history[-20:])  # Keep last 20 turns


@cl.action_callback("example_query")
async def on_example_click(action: cl.Action):
    """
    Handle clicks on example query buttons.
    """
    # Create a message as if user typed it
    await cl.Message(content=action.value, author="user").send()
    
    # Process through main handler
    msg = cl.Message(content=action.value)
    await main(msg)


@cl.on_settings_update
async def handle_settings_update(settings: dict):
    """
    Handle user settings changes.
    """
    cl.user_session.set("settings", settings)
    await cl.Message(content="✓ Settings updated").send()


def get_default_settings() -> dict:
    return {
        "num_sources": 5,
        "show_sources": True,
        "include_timestamps": True,
        "language": "auto"
    }
```

### Welcome Component (components/welcome.py)

```python
import chainlit as cl

def get_welcome_message() -> str:
    return """# 🎬 Flow Conference Assistant

Welcome! I can help you explore insights from all the Flow Conference talks.

## What I can do:
- Answer questions about topics covered in the talks
- Summarize specific presentations
- Compare different speakers' perspectives
- Link you directly to relevant video moments

## Tips:
- Be specific in your questions for better results
- Ask follow-up questions to dive deeper
- Click on source links to watch the relevant video sections
"""

def get_example_actions() -> list:
    examples = [
        ("What is RAG?", "What is RAG and how does it work?"),
        ("Chunking strategies", "What are the best practices for chunking documents?"),
        ("Vector databases", "Compare different vector database options"),
        ("Embeddings", "How do embeddings capture semantic meaning?"),
    ]
    
    return [
        cl.Action(
            name="example_query",
            value=query,
            label=f"💡 {label}",
            description=f"Ask: {query}"
        )
        for label, query in examples
    ]
```

### Source Citation Component (components/sources.py)

```python
import chainlit as cl
from typing import List

def create_source_elements(chunks: List[dict]) -> List[cl.Element]:
    """
    Create Chainlit elements for source citations.
    """
    elements = []
    
    for i, chunk in enumerate(chunks):
        # Format timestamp
        start_ts = format_timestamp(chunk["start_time"])
        end_ts = format_timestamp(chunk["end_time"])
        
        # Create source text
        source_content = f"""### 📺 {chunk['video_title']}

**Timestamp:** {start_ts} - {end_ts}

**Section:** {chunk.get('section_type', 'N/A').replace('_', ' ').title()}

---

{chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}

---

🔗 [Watch on YouTube]({chunk['youtube_link']})
"""
        
        # Create text element
        element = cl.Text(
            name=f"Source {i + 1}",
            content=source_content,
            display="side"  # Show in sidebar when clicked
        )
        elements.append(element)
    
    return elements


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def create_source_summary(chunks: List[dict]) -> str:
    """
    Create a markdown summary of sources for inline display.
    """
    if not chunks:
        return ""
    
    lines = ["\n\n---\n📚 **Sources:**\n"]
    
    for i, chunk in enumerate(chunks):
        start_ts = format_timestamp(chunk["start_time"])
        lines.append(
            f"- [{i+1}] **{chunk['video_title']}** ({start_ts}) - "
            f"[Watch]({chunk['youtube_link']})"
        )
    
    return "\n".join(lines)
```

### Settings Panel (components/settings.py)

```python
import chainlit as cl

def get_settings_inputs() -> list:
    """
    Define user-configurable settings.
    """
    return [
        cl.input_widget.Slider(
            id="num_sources",
            label="Number of sources to retrieve",
            initial=5,
            min=1,
            max=10,
            step=1
        ),
        cl.input_widget.Switch(
            id="show_sources",
            label="Show source details",
            initial=True
        ),
        cl.input_widget.Switch(
            id="include_timestamps",
            label="Include video timestamps in response",
            initial=True
        ),
        cl.input_widget.Select(
            id="language",
            label="Response language",
            initial_value="auto",
            values=["auto", "French", "English"]
        )
    ]
```

### Streaming Helpers (pipeline/streaming.py)

```python
import chainlit as cl
from typing import AsyncIterator

async def stream_with_sources(
    token_stream: AsyncIterator[str],
    chunks: list,
    message: cl.Message
):
    """
    Stream tokens and add sources at the end.
    """
    full_response = ""
    
    async for token in token_stream:
        full_response += token
        await message.stream_token(token)
    
    # Add source summary at the end
    source_summary = create_source_summary(chunks)
    if source_summary:
        await message.stream_token(source_summary)
        full_response += source_summary
    
    return full_response
```

## Configuration Files

### chainlit.md (Welcome Screen)

```markdown
# 🎬 Flow Conference Assistant

Welcome to the Flow Conference knowledge base!

I have access to all the talks from Flow Conference and can help you:
- **Find information** about specific topics
- **Summarize** presentations
- **Compare** different speakers' perspectives
- **Link you directly** to relevant video moments

## Getting Started

Just type your question in the chat below. For example:
- "What is RAG and how does it work?"
- "What are best practices for document chunking?"
- "Compare vector database options discussed at the conference"

## Tips for Better Results

1. **Be specific** - "How does BGE-M3 handle multilingual content?" works better than "tell me about embeddings"
2. **Ask follow-ups** - Dive deeper into interesting topics
3. **Check the sources** - Click on source links to watch the original talks

---

*Powered by BGE-M3 embeddings, sqlite-vec, and Claude*
```

### .chainlit/config.toml

```toml
[project]
name = "Flow Conference RAG"
enable_hierarchical_messages = false

[UI]
name = "Flow Conference Assistant"
description = "Chat with Flow Conference talks"
default_collapse_content = true
default_expand_messages = false
hide_cot = false
show_readme_as_default = true

[UI.theme]
primary_color = "#3B82F6"  # Blue
background_color = "#FFFFFF"
font_family = "Inter, sans-serif"

[features]
streaming = true
speech_to_text.enabled = false
latex.enabled = false

[meta]
generated_by = "Flow Conference RAG System"
```

## Error Handling and Edge Cases

```python
# In app.py

@cl.on_message
async def main(message: cl.Message):
    """Handle user messages with comprehensive error handling."""
    
    try:
        # ... main logic ...
        pass
        
    except ConnectionError:
        await cl.Message(
            content="⚠️ Unable to connect to the search service. Please try again in a moment."
        ).send()
        
    except TimeoutError:
        await cl.Message(
            content="⏱️ The request took too long. Please try a simpler question."
        ).send()
        
    except Exception as e:
        logging.exception("Unexpected error in message handler")
        await cl.Message(
            content="😕 Something went wrong. Please try again or rephrase your question."
        ).send()


# Rate limiting (optional)
from datetime import datetime, timedelta
from collections import defaultdict

rate_limits = defaultdict(list)

def check_rate_limit(session_id: str, max_requests: int = 20, window_minutes: int = 1) -> bool:
    """Simple rate limiting per session."""
    now = datetime.now()
    window_start = now - timedelta(minutes=window_minutes)
    
    # Clean old entries
    rate_limits[session_id] = [
        ts for ts in rate_limits[session_id] 
        if ts > window_start
    ]
    
    if len(rate_limits[session_id]) >= max_requests:
        return False
    
    rate_limits[session_id].append(now)
    return True
```

## Conversation Memory

```python
# Handle follow-up questions using conversation history

@cl.on_message
async def main(message: cl.Message):
    user_query = message.content.strip()
    history = cl.user_session.get("history", [])
    
    # Detect follow-up questions
    if is_follow_up(user_query):
        # Enhance query with context from history
        enhanced_query = enhance_with_history(user_query, history)
        user_query = enhanced_query
    
    # ... rest of processing ...


def is_follow_up(query: str) -> bool:
    """Detect if query is a follow-up to previous conversation."""
    follow_up_indicators = [
        "what about", "how about", "and", "also",
        "more about", "tell me more", "elaborate",
        "why", "can you explain", "what do you mean",
        "et", "aussi", "pourquoi", "plus de détails"
    ]
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in follow_up_indicators)


def enhance_with_history(query: str, history: list) -> str:
    """Add context from recent history to query."""
    if not history:
        return query
    
    # Get last user message and assistant response
    recent = history[-2:] if len(history) >= 2 else history
    context = " ".join([m["content"] for m in recent])
    
    # Simple enhancement - prepend context
    # In production, use LLM for query rewriting
    return f"Given the previous discussion about: {context[:200]}... {query}"
```

## Deployment

### Requirements

```
# requirements.txt
chainlit>=1.0.0
anthropic>=0.20.0
numpy>=1.24.0
FlagEmbedding>=1.2.0
sqlite-vec>=0.1.0
langdetect>=1.0.9
pyyaml>=6.0
python-dotenv>=1.0.0
```

### Environment Variables

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
CHAINLIT_AUTH_SECRET=your-secret-key
DATABASE_PATH=data/flow_conference.db
LOG_LEVEL=INFO
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run Chainlit
CMD ["chainlit", "run", "src/app.py", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  flow-rag:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_PATH=/app/data/flow_conference.db
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run in development mode
chainlit run src/app.py --watch

# Run in production mode
chainlit run src/app.py --host 0.0.0.0 --port 8000
```

## Performance Optimization

### Caching

```python
from functools import lru_cache
import hashlib

# Cache embeddings for repeated queries
@lru_cache(maxsize=1000)
def cached_embed_query(query: str):
    """Cache query embeddings."""
    return embedder.embed_query(query)

# Cache frequent query results
query_cache = {}

def get_cached_results(query: str, ttl_seconds: int = 300):
    """Get cached results if available and fresh."""
    cache_key = hashlib.md5(query.encode()).hexdigest()
    
    if cache_key in query_cache:
        result, timestamp = query_cache[cache_key]
        if time.time() - timestamp < ttl_seconds:
            return result
    
    return None
```

### Connection Pooling

```python
# Initialize pipeline once, reuse across requests
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline(config)
    return pipeline
```

## Success Criteria

- [ ] Welcome message displays correctly
- [ ] Example queries work as expected
- [ ] Streaming responses display smoothly
- [ ] Source citations are clickable and accurate
- [ ] YouTube links open correct timestamp
- [ ] Settings persist within session
- [ ] Error messages are user-friendly
- [ ] Response time < 3 seconds for first token
- [ ] Works on mobile devices
- [ ] Handles concurrent users (10+)
