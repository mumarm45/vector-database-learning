# Food Search

A RAG-powered food recommendation chatbot using ChromaDB vector database and Anthropic Claude LLM.

## Features

- **Semantic Food Search** - Find foods using natural language queries
- **AI-Powered Recommendations** - Claude provides intelligent, contextual responses
- **Comparison Mode** - Compare two different food preferences side-by-side
- **Rich Food Database** - Includes cuisine types, calories, ingredients, and health benefits

## Installation

```bash
# Install dependencies
pip install -e .

# Or using uv
uv sync
```

## Configuration

Create a `.env` file with your API keys:

```env
ANTHROPIC_API_KEY=your_api_key_here
LLM_MODEL=claude-3-haiku-20240307
```

## Usage

Run the chatbot:

```bash
food-search
```

### Example Queries

- "I want something spicy and healthy for dinner"
- "What Italian dishes do you recommend under 400 calories?"
- "I'm craving comfort food for a cold evening"
- "Suggest some protein-rich breakfast options"

### Commands

- `help` - Show detailed help menu
- `compare` - Compare recommendations for two different queries
- `quit` - Exit the chatbot

## Project Structure

```
food_search/
├── src/
│   ├── app.py                 # Main entry point
│   ├── FoodDataSet.json       # Food database
│   └── modules/
│       ├── create_chroma.py   # ChromaDB setup
│       ├── load_food_data.py  # Data loading utilities
│       ├── search_food.py     # Search functions
│       └── llm_food_rag_search.py  # RAG chatbot logic
└── pyproject.toml
```

## Dependencies

- ChromaDB 1.0.12
- Sentence Transformers 4.1.0
- Anthropic SDK
- Python 3.12+
