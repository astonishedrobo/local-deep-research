## Quickstart

### 1. Install `uv`
First install `uv` on the system first following the official [instruction](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Run the Agent
Copy the Github repo 
```bash
git clone https://github.com/astonishedrobo/local-deep-research.git
cd local-deep-research
```

```bash
# For local testing
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking

# For prototype deployment
UV_NO_SANDBOX=1 uvx --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking --tunnel --no-browser
```

### 3. Dir Structure

All the unstructured data (e.g CSVs) must reside under the data/ dir.

### 4. Environ Variables

Before running the program, set the required API environment variables.

```bash
# Create the .env file
touch .env

# Put the api keys in following format
OPENAI_API_KEY=<your_openai_api_key>
LANGSMITH_API_KEY=<your_langsmith_api_key>
LANGSMITH_TRACING=true
LANGSMITH_PROJECT="deep-agent"

GRAPHRAG_API_KEY=<your_openai_api_key>
GRAPHRAG_LLM_MODEL=gpt-4o-mini
GRAPHRAG_EMBEDDING_MODEL=text-embedding-3-small
```