# Part A: RAG Design for Kerala Ayurveda Content System

## 1. Chunking Plan

### Document Types and Chunking Strategy

**Markdown Documents (Foundational & Product Dossiers):**
- **Chunk size:** 512 tokens (approximately 400-450 words)
- **Overlap:** 128 tokens (approximately 100 words) between adjacent chunks
- **Special handling:**
  - Preserve heading hierarchy (H1, H2, H3) as metadata
  - Each chunk includes its parent heading path (e.g., "Ayurveda Foundations > The Tridosha Model > Vata")
  - Split at natural boundaries (section breaks, not mid-paragraph)
  - Preserve markdown formatting for context

**CSV Files (products_catalog.csv):**
- **Strategy:** Each row is a self-contained chunk
- **Metadata:** Include column headers as context for each row
- **Size:** ~200-300 tokens per row (with headers)
- **No overlap needed** (rows are independent)

**FAQ Documents (faq_general_ayurveda_patients.md):**
- **Chunk size:** One Q&A pair per chunk (typically 200-400 tokens)
- **Overlap:** None (Q&A pairs are independent)
- **Metadata:** Question text stored as `question` field, answer as `content`

**Product Dossiers (product_*.md):**
- **Chunk size:** 512 tokens with 128-token overlap
- **Special sections preserved:**
  - "Basic Info" → single chunk
  - "Traditional Positioning" → single chunk
  - "Key Messages" → may span 1-2 chunks
  - "Safety & Precautions" → single chunk (critical for retrieval)

### Chunk Metadata Schema

Each chunk includes:
```json
{
  "doc_id": "product_ashwagandha_tablets_internal.md",
  "chunk_id": "chunk_001",
  "section_path": "Traditional Positioning",
  "chunk_index": 0,
  "total_chunks": 5,
  "doc_type": "product_dossier",
  "tokens": 487,
  "char_start": 0,
  "char_end": 2156
}
```

### Rationale

- **512 tokens:** Balances context richness with retrieval precision. Large enough for complete thoughts, small enough for focused retrieval.
- **128-token overlap:** Ensures section boundaries don't split critical information (e.g., a safety warning that spans two sections).
- **Row-level CSV chunking:** Products are independent entities; row-level retrieval matches user queries about specific products.

**Corpus reference:** Based on structure observed in `product_ashwagandha_tablets_internal.md`, `products_catalog.csv`, and `faq_general_ayurveda_patients.md`.

---

## 2. Retriever Choice and Rationale

### Hybrid Retrieval: BM25 + Dense Embeddings

**Primary Strategy:** Hybrid retrieval combining:
- **BM25 (sparse):** For exact term matching (product names, herb names, Sanskrit terms)
- **Dense embeddings (sentence-transformers):** For semantic similarity (conceptual queries about stress, digestion, etc.)

### Exact Parameters

**For Short Q&A Queries (< 20 words):**
- **Top-K retrieval:** 5 chunks from dense, 5 chunks from BM25
- **Deduplication:** Remove chunks with >80% token overlap
- **Final set:** 5-8 unique chunks
- **Rationale:** Short queries need focused, precise matches

**For Long Article Generation Queries:**
- **Top-K retrieval:** 10 chunks from dense, 10 chunks from BM25
- **Deduplication:** Remove chunks with >80% token overlap
- **Final set:** 12-15 unique chunks
- **Rationale:** Longer content needs broader context

**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Rationale:** Fast, good for domain-agnostic semantic search. For production, consider fine-tuning on Ayurveda corpus.

**BM25 Parameters:**
- **k1:** 1.5 (term frequency saturation)
- **b:** 0.75 (length normalization)
- **Fields:** Title (boost 2.0), Content (boost 1.0), Tags (boost 1.5)

**Hybrid Scoring:**
```
final_score = 0.4 * normalized_bm25_score + 0.6 * normalized_embedding_score
```

**Corpus reference:** Product names like "Ashwagandha Stress Balance Tablets" [source:product_ashwagandha_tablets_internal.md#Basic Info] benefit from BM25 exact matching, while queries like "herbs for stress" benefit from semantic search.

---

## 3. Cursor-Based Retrieval Plan

### Streaming Scan Strategy

**Purpose:** Use cursor-based retrieval for fact-checking and incremental evidence gathering.

**Implementation:**
1. **Initial retrieval:** Get top-K chunks via hybrid search
2. **Cursor scan:** For each factual claim in draft answer, scan corpus with cursor
3. **Batch size:** Process 10 chunks per batch
4. **Early-stop heuristics:**
   - Stop if 3+ chunks support the claim (high confidence)
   - Stop if cursor has scanned 50 chunks without support (likely unsupported)
   - Stop if cursor reaches end of relevant document section

### Fact-Checking Workflow

```
1. Generate draft answer from initial retrieval
2. Extract candidate factual sentences (using simple NLP: subject-verb-object patterns)
3. For each sentence:
   a. Extract key entities (product names, herb names, conditions)
   b. Cursor scan: search corpus for these entities + context
   c. Retrieve top-10 chunks for this specific claim
   d. If max similarity < 0.6 threshold → flag as "UNSUPPORTED"
4. Re-generate answer with unsupported claims removed or marked
```

### Cursor Implementation Details

**Cursor state:**
```python
{
  "current_doc": "product_ashwagandha_tablets_internal.md",
  "current_section": "Safety & Precautions",
  "chunk_index": 3,
  "scanned_chunks": 12,
  "supporting_chunks": [chunk_2, chunk_3, chunk_5]
}
```

**Batch processing:** Process 10 chunks per iteration to balance speed and thoroughness.

**Corpus reference:** Safety information in `product_triphala_capsules_internal.md#Safety & Precautions` requires precise cursor-based verification to avoid hallucinated contraindications.

---

## 4. Prompt Templates

### Template 1: Q&A (Concise Answers)

```
You are a helpful assistant for Kerala Ayurveda, providing accurate information based solely on the provided context documents.

**Context Documents:**
{retrieved_chunks}

**User Query:**
{user_query}

**Instructions:**
1. Answer the query using ONLY information from the context documents above.
2. If the information is not present in the context, respond: "I don't find this in the provided corpus."
3. Include inline citations in square brackets: [source:filename.md#Section Name]
4. Use warm, reassuring, and grounded language (avoid "miracle cure" or "guaranteed" claims).
5. For safety information, be explicit and conservative.
6. Keep the answer concise (2-4 paragraphs for short queries).

**Answer:**
```

**Placeholders:**
- `{retrieved_chunks}`: Formatted chunks with doc_id, section, and content
- `{user_query}`: User's question

### Template 2: Generation/Authoring (Long-Form Content)

```
You are a content writer for Kerala Ayurveda, creating educational articles based on the provided context documents.

**Context Documents:**
{retrieved_chunks}

**Content Request:**
{content_request}

**Instructions:**
1. Use ONLY information from the context documents. Do not invent facts.
2. Structure the article with clear H2/H3 headings.
3. Use short paragraphs (2-4 sentences).
4. Include bulleted lists for practical points.
5. Add inline citations: [source:filename.md#Section Name]
6. Follow the brand voice: warm, reassuring, grounded, tradition-aware, science-friendly.
7. Avoid: disease-cure claims, specific dosing, "guaranteed results."
8. Include a safety note if discussing herbs or therapies.
9. If information is missing, mark it: "MISSING: [description] - requires clinical/editor review."

**Article:**
```

**Placeholders:**
- `{retrieved_chunks}`: 12-15 chunks for broader context
- `{content_request}`: Article topic and requirements

**Corpus reference:** Style guidelines from `content_style_and_tone_guide.md#Brand Voice` and `content_style_and_tone_guide.md#Medical & Legal Boundaries`.

---

## 5. Citation Format and UI Behavior

### Inline Citation Format

**Pattern:** `[source:doc_id#section_name]`

**Examples:**
- `[source:product_ashwagandha_tablets_internal.md#Traditional Positioning]`
- `[source:faq_general_ayurveda_patients.md#2. How long does it take to see results?]`
- `[source:products_catalog.csv#KA-P001]` (for CSV rows)

**UI Behavior:**
- Citations appear as clickable superscripts or inline links
- Hover shows: document name, section, and excerpt
- Click opens source document at relevant section

### JSON Return Schema

```json
{
  "answer": "Ashwagandha is traditionally used in Ayurveda to support the body's ability to adapt to stress [source:product_ashwagandha_tablets_internal.md#Traditional Positioning]. Many people notice changes in sleep and stress resilience within a few weeks [source:faq_general_ayurveda_patients.md#2. How long does it take to see results?].",
  "citations": [
    {
      "doc_id": "product_ashwagandha_tablets_internal.md",
      "section": "Traditional Positioning",
      "excerpt": "In Ayurveda, Ashwagandha is traditionally used to:\n- Support the body's ability to adapt to stress\n- Promote calmness and emotional balance",
      "score_note": "dense_similarity: 0.82, bm25_score: 0.45"
    },
    {
      "doc_id": "faq_general_ayurveda_patients.md",
      "section": "2. How long does it take to see results?",
      "excerpt": "Some people may feel changes in sleep, digestion, or energy in a few weeks.",
      "score_note": "dense_similarity: 0.71, bm25_score: 0.38"
    }
  ],
  "unsupported_claims": [],
  "confidence_score": 0.85
}
```

### Citation Extraction Process

1. **During answer generation:** LLM includes inline citations in square brackets
2. **Post-processing:** Extract citation markers, match to retrieved chunks
3. **Validation:** Verify cited chunks were in retrieval set (prevent hallucinated citations)
4. **Enrichment:** Add excerpt and score metadata to citation objects

**Corpus reference:** Citation format aligns with requirement to ground every factual claim, as specified in the assignment instructions.

