# Kerala Ayurveda RAG System - Code

## Overview

This directory contains the Python implementation of the RAG system for Kerala Ayurveda content. The system implements:

- Hybrid retrieval (BM25 + dense embeddings)
- Cursor-based fact-checking
- Citation extraction and validation
- Answer generation with grounding

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from rag_system import KeralaAyurvedaRAG

# Initialize system (point to directory with .md and .csv files)
rag = KeralaAyurvedaRAG(corpus_path="../")

# Answer a query
result = rag.answer_user_query("What are the benefits of Ashwagandha?")

print(result['answer'])
print(f"Confidence: {result['confidence_score']}")
for citation in result['citations']:
    print(f"  - {citation['doc_id']}#{citation['section']}")
```

### Running the Demo

```bash
python rag_system.py
```

This will run example queries and display results.

## File Structure

- `rag_system.py`: Main RAG implementation
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Integration with LLM

The current implementation includes a `_generate_mock_answer()` method for demonstration. To use with an actual LLM:

1. Replace `_generate_mock_answer()` with a call to your LLM API (OpenAI, Anthropic, etc.)
2. Set `use_llm=True` in `answer_user_query()`
3. Add API key configuration

Example integration:

```python
import openai

def _generate_answer(self, prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

## Notes

- The system requires the corpus files (`.md` and `.csv`) to be in the parent directory by default
- Embedding generation may take a few minutes on first run
- BM25 and dense retrieval can work independently if one fails to load


