# ChromaDB Example

Basic examples demonstrating ChromaDB vector database fundamentals with sentence transformer embeddings.

## Features

- **Grocery Collection** - Simple food item similarity search
- **Employee Collection** - Search employees by skills, experience, and department
- **Books Collection** - Find books by themes, genres, and descriptions
- **Filtered Queries** - Combine semantic search with metadata filters

## Installation

```bash
# Using uv
uv sync

# Or using pip
pip install chromadb sentence-transformers numpy torch
```

## Usage

```bash
python main.py
```

## Examples

### Grocery Search
```python
collection = perform_operations()
perform_similarity_search(collection, "red fruit")
```

### Employee Search with Filters
```python
collection = employee_collection()
refine_filter(collection, "Python developer with web development experience")
# Filters: experience >= 5 years, location in SF/NY/Seattle
```

### Book Search with Filters
```python
collection = book_collection()
refine_book_filter(collection, "magical fantasy adventure")
# Filters: rating >= 4.0, genre in Fantasy/Sci-Fi
```

## Tech Stack

- ChromaDB (< 1.0.0)
- Sentence Transformers (all-MiniLM-L6-v2)
- Python 3.12+
