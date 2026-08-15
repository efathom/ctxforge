# ctxforge Examples

End-to-end examples demonstrating the ctxforge framework with real services.

## Quick Start (In-Memory Mode)

The simplest way to run the demo uses in-memory storage (no PostgreSQL required):

```bash
# 1. Install dependencies
pip install openai chromadb python-dotenv pyyaml

# 2. Set your OpenAI API key
export OPENAI_API_KEY=sk-your-key-here

# 3. Run the scripted demo
cd /path/to/ctxforge
python -m ctxforge.examples.run_demo

# Or run interactive mode
python -m ctxforge.examples.run_demo --mode interactive

# Show configuration summary
python -m ctxforge.examples.run_demo --show-config
```

## Configuration

The demo uses the framework's `ConfigLoader` to load settings from YAML files with environment variable overrides.

### Configuration File

The default configuration is in `engine_config.yaml`:

```yaml
# Engine identity
name: "context-engine-demo"
version: "1.0.0"

# LLM Configuration
llm:
  provider: openai
  model: ${OPENAI_MODEL:-gpt-4}      # Uses env var or default
  api_key: ${OPENAI_API_KEY}          # Required env var
  temperature: 0.7

# Storage Configuration
storage:
  session:
    backend: memory                   # memory, redis, postgres
  memory:
    store_backend: memory             # memory, redis, postgres, mysql
    vector:
      backend: chromadb               # memory, chromadb, pinecone
      extra_params:
        persist_directory: ./chroma_data

# And more...
```

### Environment Variable Overrides

Environment variables with `CTXFORGE_` prefix override YAML settings:

```bash
# Override LLM model
export CTXFORGE_LLM_MODEL=gpt-4o

# Override retrieval limit
export CTXFORGE_RETRIEVAL_DEFAULT_LIMIT=10

# Enable debug mode
export CTXFORGE_DEBUG=true
```

### Custom Configuration File

Use a custom config file:

```bash
python -m ctxforge.examples.run_demo --config my_config.yaml
```

### Configuration Priority

Settings are loaded in this order (later overrides earlier):
1. Default values in `EngineConfig`
2. Values from YAML file
3. Environment variables (`CTXFORGE_*` prefix)
4. Direct environment variables (`OPENAI_API_KEY`, etc.)

## Full Setup (with Docker Compose)

The fastest way to bring up all backing services (PostgreSQL, MySQL, Neo4j) is the
provided Docker Compose file at the repo root:

```bash
# From the repo root (ctxforge/)
docker compose up -d
```

This starts:
- PostgreSQL on `localhost:5432` (db/user/pass: `ctxforge`/`ctxforge`/`password`), schema from `examples/init.sql`
- MySQL on `localhost:3306` (db/user/pass: `ctxforge`/`ctxforge`/`password`)
- Neo4j on `bolt://localhost:7687` (user/pass: `neo4j`/`password`)

ChromaDB runs in-process (persist_directory), and the embedding server is expected at
`LOCAL_EMBEDDING_BASE_URL` (default `http://localhost:8080/v1`; run e.g. TEI on that port).

Stop the containers with `docker compose down` (data volumes persist; use `down -v` to wipe them).

## Full Setup (with Local PostgreSQL)

For production-like testing with PostgreSQL:

### 1. Install PostgreSQL

**macOS (Homebrew):**
```bash
brew install postgresql@15
brew services start postgresql@15
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**Windows:**
Download and install from https://www.postgresql.org/download/windows/

### 2. Create Database and User

```bash
# Connect to PostgreSQL as admin
psql postgres

# Create user and database
CREATE USER contextengine WITH PASSWORD 'password';
CREATE DATABASE contextengine OWNER contextengine;
GRANT ALL PRIVILEGES ON DATABASE contextengine TO contextengine;
\q
```

### 3. Initialize Schema

```bash
# Run the initialization script
psql -U contextengine -d contextengine -f ctxforge/examples/init.sql
```

Or manually in psql:
```bash
psql -U contextengine -d contextengine
\i ctxforge/examples/init.sql
```

### 4. Configure Environment

```bash
# Copy example config
cp ctxforge/examples/env.example .env

# Edit with your settings
nano .env
```

Example `.env`:
```
OPENAI_API_KEY=sk-your-key-here
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=contextengine
POSTGRES_USER=contextengine
POSTGRES_PASSWORD=password
```

### 5. Install Dependencies

```bash
pip install openai chromadb python-dotenv asyncpg
```

### 6. Run Demo

```bash
# Scripted demo with PostgreSQL
python -m ctxforge.examples.run_demo --postgres

# Interactive demo with PostgreSQL
python -m ctxforge.examples.run_demo --mode interactive --postgres
```

## Demo Modes

### Scripted Demo (Default)

Runs through a predefined conversation that demonstrates:
- Memory extraction (name, preferences, location)
- PII detection (email address)
- Semantic memory retrieval
- Context-aware responses

```bash
python -m ctxforge.examples.run_demo
```

### Interactive Demo

Chat freely with the AI assistant:

```bash
python -m ctxforge.examples.run_demo --mode interactive
```

Commands in interactive mode:
- `/memories` - Show all stored memories
- `/quit` - Exit the demo

## What the Demo Shows

### 1. Storage Backends

- **In-Memory**: Default, no setup required
- **PostgreSQL**: Production-ready with `--postgres` flag
- **ChromaDB**: Vector storage for semantic search

### 2. Middleware Pipeline

```
User Input → Rate Limit → PII Detection → Audit Log → LLM
```

- **Rate Limiting**: Token bucket (10 req/sec, burst 50)
- **PII Detection**: Detects emails, phones, SSNs
- **Audit Logging**: Records all interactions

### 3. Memory Extraction

The demo extracts and stores:
- Personal facts (name, location)
- Preferences (favorite things)
- Entities (dates, emails, etc.)

### 4. Semantic Retrieval

When you ask questions, the system:
1. Embeds your query using OpenAI
2. Searches ChromaDB for similar memories
3. Includes relevant memories in the context
4. Generates personalized responses

## Example Session

```
👤 User: Hi! My name is Alice and I'm a software engineer.

🛡️ Processing through middleware...
   ✅ No issues detected

🔍 Retrieving relevant memories...
   ℹ️ No relevant memories found

📝 Extracting memories...
   ✅ Extracted 2 potential memories
      💾 Stored: [SEMANTIC] User's name is Alice
      💾 Stored: [SEMANTIC] User is a software engineer

🤖 Assistant: Nice to meet you, Alice! How exciting that you're a 
software engineer. What kind of projects do you work on?

──────────────────────────────────────────────────────────

👤 User: What do you remember about me?

🔍 Retrieving relevant memories...
   ✅ Found 2 relevant memories
      • User's name is Alice...
      • User is a software engineer...

🤖 Assistant: I remember that your name is Alice and you're a 
software engineer! Is there anything specific you'd like to 
chat about regarding your work or interests?
```

## Files

| File | Description |
|------|-------------|
| `config.py` | Environment configuration loader |
| `run_demo.py` | Main demo script |
| `init.sql` | Database schema initialization |
| `env.example` | Example environment variables |

Note: OpenAI providers are in `ctxforge/llm/openai_provider.py`.

## Troubleshooting

### "OPENAI_API_KEY environment variable is required"

```bash
export OPENAI_API_KEY=sk-your-key-here
```

### "openai package is required"

```bash
pip install openai
```

### PostgreSQL connection refused

Check PostgreSQL is running:
```bash
# macOS
brew services list | grep postgresql

# Linux
sudo systemctl status postgresql
```

Check connection manually:
```bash
psql -U contextengine -d contextengine -h localhost
```

### PostgreSQL authentication failed

Update pg_hba.conf to allow password authentication:
```bash
# Find pg_hba.conf location
psql -U postgres -c "SHOW hba_file;"

# Edit and change 'peer' or 'ident' to 'md5' for local connections
# Then restart PostgreSQL
```

### ChromaDB errors

Clear the ChromaDB data and try again:
```bash
rm -rf ./chroma_data
```

## Customization

### Use Different Models

```bash
export OPENAI_MODEL=gpt-3.5-turbo
export OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

### Adjust Rate Limits

Edit `run_demo.py`:
```python
RateLimitMiddleware(
    limiter=TokenBucketLimiter(rate=100.0, capacity=200),
)
```

### Add Custom Extractors

```python
from ctxforge.extraction import LLMExtractor

# Add LLM-based extraction (uses more tokens)
extractor = HybridExtractor(
    extractors=[
        PatternExtractor(),
        EntityExtractor(),
        LLMExtractor(llm_provider=self.llm_provider),
    ]
)
```

## Cleanup

Stop PostgreSQL (if you want to):
```bash
# macOS
brew services stop postgresql@15

# Linux
sudo systemctl stop postgresql
```

Remove data:
```bash
# Drop database
psql postgres -c "DROP DATABASE contextengine;"

# Remove ChromaDB data
rm -rf ./chroma_data
```
