# How Fact-Checking Works in the Kerala Ayurveda RAG System

## Overview

The fact-checking system uses **cursor-based retrieval** to verify every factual claim in the generated answer. This prevents hallucinations and ensures all information is grounded in the corpus.

---

## Step-by-Step Process

### Step 1: Generate Draft Answer
```
User Query: "What are the benefits of Ashwagandha?"
    ↓
Initial Hybrid Retrieval (top-5 chunks)
    ↓
LLM generates draft answer:
"Ashwagandha is traditionally used to support stress adaptation. 
Some people notice effects within a few weeks."
```

### Step 2: Extract Factual Sentences
The system splits the answer into individual sentences and filters out:
- Very short sentences (< 20 characters)
- Sentences that are just citations
- Questions or conversational phrases

**Example extracted sentences:**
1. "Ashwagandha is traditionally used to support stress adaptation."
2. "Some people notice effects within a few weeks."

### Step 3: Cursor-Based Scanning (For Each Sentence)

For **each factual sentence**, the system:

#### 3a. Extract Key Entities
- Product names: "Ashwagandha"
- Concepts: "stress adaptation", "effects"
- Time references: "few weeks"

#### 3b. Cursor Scan the Corpus
```
Sentence: "Ashwagandha is traditionally used to support stress adaptation."
    ↓
Search corpus for: "Ashwagandha" + "stress adaptation" + "traditionally used"
    ↓
Process chunks in batches of 10:
  Batch 1: chunks 1-10   → Check similarity
  Batch 2: chunks 11-20   → Check similarity
  Batch 3: chunks 21-30   → Check similarity
  ...
  (Stop early if 3+ supporting chunks found)
```

#### 3c. Calculate Semantic Similarity
For each chunk found, calculate how similar it is to the sentence:

```python
sentence = "Ashwagandha is traditionally used to support stress adaptation."
chunk_content = "In Ayurveda, Ashwagandha is traditionally used to support 
                 the body's ability to adapt to stress..."

similarity = cosine_similarity(
    embedding(sentence),
    embedding(chunk_content)
)
# Result: 0.91 (high similarity = well supported)
```

#### 3d. Apply Threshold
- **Normal claims:** similarity ≥ 0.6 → **SUPPORTED**
- **Safety claims:** similarity ≥ 0.75 → **SUPPORTED** (stricter)
- **Below threshold:** → **UNSUPPORTED** (flag for removal)

### Step 4: Early-Stop Heuristics

The cursor scan stops early if:
1. ✅ **3+ chunks support the claim** (high confidence, no need to scan more)
2. ❌ **50 chunks scanned without support** (likely unsupported)
3. 📄 **End of relevant document section** (no more relevant content)

This makes fact-checking efficient without scanning the entire corpus.

### Step 5: Collect Results

```python
{
  "supported_sentences": [
    {
      "sentence": "Ashwagandha is traditionally used to support stress adaptation.",
      "max_similarity": 0.91,
      "supporting_chunk": {
        "doc_id": "product_ashwagandha_tablets_internal.md",
        "section": "Traditional Positioning",
        "excerpt": "In Ayurveda, Ashwagandha is traditionally used to..."
      }
    }
  ],
  "unsupported_sentences": [
    {
      "sentence": "Clinical studies prove Ashwagandha works in 3 days.",
      "max_similarity": 0.35,  # Below 0.6 threshold
      "reason": "No evidence found in corpus"
    }
  ]
}
```

### Step 6: Final Answer Processing

**If unsupported claims found:**
- Option A: Remove the unsupported sentence
- Option B: Add note: "[Note: This claim could not be verified in the corpus and may require review.]"

**Final answer:**
```
"Ashwagandha is traditionally used to support stress adaptation 
[source:product_ashwagandha_tablets_internal.md#Traditional Positioning]. 
Some people notice effects within a few weeks 
[source:faq_general_ayurveda_patients.md#2. How long does it take to see results?]."
```

---

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User Query: "What are the benefits of Ashwagandha?"     │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Initial Retrieval (Hybrid: BM25 + Dense)                 │
│    → Get top-5 chunks                                        │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. LLM Generates Draft Answer                                │
│    "Ashwagandha supports stress. Effects in weeks."          │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Extract Factual Sentences                                 │
│    Sentence 1: "Ashwagandha supports stress"                │
│    Sentence 2: "Effects in weeks"                           │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. For Each Sentence: Cursor-Based Fact-Check                │
│                                                              │
│    Sentence 1: "Ashwagandha supports stress"                 │
│    ├─ Extract entities: ["Ashwagandha", "stress"]          │
│    ├─ Cursor scan corpus (batches of 10 chunks)            │
│    ├─ Calculate similarity with each chunk                  │
│    └─ Max similarity: 0.91 → ✅ SUPPORTED                   │
│                                                              │
│    Sentence 2: "Effects in weeks"                           │
│    ├─ Extract entities: ["effects", "weeks"]                 │
│    ├─ Cursor scan corpus                                    │
│    ├─ Calculate similarity                                   │
│    └─ Max similarity: 0.78 → ✅ SUPPORTED                    │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Final Answer with Citations                               │
│    "Ashwagandha supports stress [source:...]                │
│     Effects in weeks [source:...]"                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Code Implementation

### Key Function: `_fact_check_with_cursor()`

```python
def _fact_check_with_cursor(self, answer: str, initial_chunks: List[Dict]) -> Dict:
    """Use cursor-based retrieval to fact-check the answer."""
    
    # Step 1: Extract factual sentences
    sentences = self._extract_factual_sentences(answer)
    
    unsupported_claims = []
    supporting_evidence = []
    
    # Step 2: For each sentence, cursor scan
    for sentence in sentences:
        # Cursor scan: search corpus for this specific sentence
        cursor_chunks = self._cursor_scan(sentence, batch_size=10, max_chunks=50)
        
        # Step 3: Find best matching chunk
        max_similarity = 0.0
        best_chunk = None
        
        for scored_chunk in cursor_chunks:
            chunk = scored_chunk['chunk']
            # Calculate semantic similarity
            similarity = self._semantic_similarity(sentence, chunk.content)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_chunk = chunk
        
        # Step 4: Apply threshold
        if max_similarity < 0.6:  # Threshold for normal claims
            unsupported_claims.append({
                'sentence': sentence,
                'max_similarity': max_similarity
            })
        else:
            supporting_evidence.append({
                'sentence': sentence,
                'max_similarity': max_similarity,
                'supporting_chunk': best_chunk
            })
    
    return {
        'unsupported_claims': unsupported_claims,
        'supporting_evidence': supporting_evidence
    }
```

### Semantic Similarity Calculation

```python
def _semantic_similarity(self, text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts."""
    
    # Convert both texts to embeddings
    emb1 = self.embedding_model.encode([text1])[0]
    emb2 = self.embedding_model.encode([text2])[0]
    
    # Calculate cosine similarity (0.0 to 1.0)
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    
    return float(similarity)
    # Returns: 0.0 (completely different) to 1.0 (identical meaning)
```

---

## Example: Fact-Checking in Action

### Input Sentence
```
"Ashwagandha is traditionally used to support stress adaptation."
```

### Cursor Scan Results

**Chunk 1:**
- Source: `product_ashwagandha_tablets_internal.md#Traditional Positioning`
- Content: "In Ayurveda, Ashwagandha is traditionally used to support the body's ability to adapt to stress"
- Similarity: **0.91** ✅

**Chunk 2:**
- Source: `products_catalog.csv#KA-P002`
- Content: "Ashwagandha Stress Balance Tablets, Stress & Sleep, Stress resilience"
- Similarity: **0.65** ✅

**Chunk 3:**
- Source: `faq_general_ayurveda_patients.md#3. Can Ayurveda help with stress?`
- Content: "Ayurveda approaches stress through herbs like Ashwagandha"
- Similarity: **0.72** ✅

**Result:** Max similarity = 0.91 → **SUPPORTED** (well above 0.6 threshold)

---

## Why This Approach Works

1. **Prevents Hallucination:** Every claim is verified against the corpus
2. **Efficient:** Early-stop heuristics avoid scanning entire corpus
3. **Semantic Understanding:** Uses embeddings, not just keyword matching
4. **Safety-First:** Stricter thresholds (0.75) for safety-critical claims
5. **Transparent:** Returns unsupported claims for review

---

## Thresholds Explained

| Claim Type | Threshold | Rationale |
|------------|-----------|-----------|
| **Normal claims** | 0.6 | General information, product benefits |
| **Safety claims** | 0.75 | Contraindications, drug interactions, pregnancy |
| **Unsupported** | < 0.6 | Flag for removal or review |

**Why 0.6?**
- Below 0.6: Likely hallucination or misinterpretation
- 0.6-0.75: Probably supported, but verify
- Above 0.75: Strongly supported, high confidence

---

## Integration in Agentic Workflow

The fact-checking happens in **Step 4** of the 5-step workflow:

1. Query Understanding
2. Hybrid Retrieval
3. Answer Generation
4. **Cursor-Based Fact-Checking** ← You are here
5. Response Formatting

See `partB_agent_workflow_and_eval.md` for the complete workflow.

---

## Summary

**Fact-checking = Cursor-based semantic verification**

1. Extract sentences from answer
2. For each sentence: scan corpus with cursor (batches of 10)
3. Calculate semantic similarity with each chunk
4. If max similarity < threshold → flag as unsupported
5. Remove or mark unsupported claims in final answer

This ensures every factual claim is grounded in the corpus, preventing hallucinations and maintaining trust.

