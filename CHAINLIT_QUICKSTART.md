# 🚀 Chainlit Quick Start Guide

## Prerequisites
- Python 3.10, 3.11, or 3.12 (**Not 3.13** - dependency conflicts with graphrag)
- uv package manager (will be auto-installed if missing)

## Quick Start (3 steps)

### 1. Clone the repository
```bash
git clone <repository-url>
cd agentic_al
```

### 2. Set up API keys
The script will create a `.env` file template. You need to add your API keys:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional (for additional features)
ANTHROPIC_API_KEY=your-anthropic-key
TAVILY_API_KEY=your-tavily-key
```

### 3. Run the start script
```bash
./start_chainlit.sh
```

That's it! The script will:
- ✅ Install uv if needed
- ✅ Create `.env` template if missing
- ✅ Generate authentication secret automatically
- ✅ Sync all dependencies using uv (much faster than pip!)
- ✅ Start the Chainlit server on http://localhost:8000

## First Time Login

The application uses **auto-registration**. Just enter any email/password on the login page:
- New credentials → Automatically registers a new user
- Existing credentials → Logs you in

**Example:**
- Email: `user@example.com`
- Password: `mypassword123`

## Features

- 🔍 Multi-agent research orchestration
- 🛠️ Real-time tool call visualization
- 📊 Hierarchical agent interaction display
- 💬 Interactive chat interface
- 📝 Research note tracking
- 🔒 User authentication with chat history

## Customization

### Change Port
```bash
./start_chainlit.sh 3000  # Start on port 3000
```

### Manual Start
If you prefer to start manually:
```bash
# Make sure .env is configured
source .env

# Start Chainlit with uv
uv run chainlit run chainlit_app.py -w
```

## Troubleshooting

### "No solution found when resolving dependencies"
Make sure you're using Python 3.10-3.12 (not 3.13). Check with: `python --version`

### "ModuleNotFoundError"
Run: `uv sync`

### "OpenAI API key not set"
Edit `.env` and add your OpenAI API key

### Port already in use
Use a different port: `./start_chainlit.sh 8001`

## Known Issues

### ~~Database Schema Errors~~ (FIXED in latest version)

**Previous Issue:** Errors like `no such column: e.props` or `no such column: defaultOpen`

**Root Cause:** The `chainlit_app.py` had a custom `init_database()` function that created tables with an outdated schema, missing columns like `props` in elements and `defaultOpen` in steps.

**Fix Applied:** Removed the custom database initialization. Chainlit 2.9.0's `SQLAlchemyDataLayer` now automatically creates tables with the correct schema.

**If you still see schema errors:**
```bash
# Stop the server (Ctrl+C)
rm chainlit_data.db
./start_chainlit.sh
```

This will create a fresh database with the correct schema.

### "Author not found for thread_id" Error After Database Recreation

**Symptom:** Error `ValueError: Author not found for thread_id <some-id>` when accessing the app

**Cause:** Your browser has cached thread IDs from the old database in localStorage. After recreating the database, those threads no longer exist.

**Solution:** Clear your browser data for localhost:
1. **Chrome/Edge:** Open DevTools (F12) → Application tab → Storage → Clear site data
2. **Firefox:** Open DevTools (F12) → Storage tab → Local Storage → Delete all
3. **Or use Incognito/Private mode:** Open http://localhost:8000 in a new private window

Alternatively, clear localStorage programmatically:
```javascript
// Open browser console (F12) and run:
localStorage.clear();
location.reload();
```

## System Theme

## System Theme

The app automatically follows your operating system's theme (light/dark mode).

## Support

For issues or questions, please check the main README.md or create an issue on GitHub.
