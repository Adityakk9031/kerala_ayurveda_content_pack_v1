# Kerala Ayurveda Agentic AI Assignment - Deliverables

This directory contains the complete assignment deliverables for building a RAG-based agentic AI system for Kerala Ayurveda content.

## Directory Structure

```
final_assignment/
├── README.md                          # This file
├── partA_rag_design.md                # Part A: RAG design document
├── partA_code_and_examples.md         # Part A: Function design and examples
├── partB_agent_workflow_and_eval.md   # Part B: Agentic workflow and evaluation
├── final_combined_assignment.md        # Combined submission document
└── code/
    ├── rag_system.py                  # Python implementation
    ├── requirements.txt               # Python dependencies
    └── README.md                      # Code-specific instructions
```

## File Descriptions

### Part A Documents

**`partA_rag_design.md`**
- Chunking plan (512 tokens, 128 overlap for MD; row-level for CSV)
- Retriever choice (hybrid BM25 + dense embeddings)
- Cursor-based retrieval strategy for fact-checking
- Prompt templates (Q&A and long-form)
- Citation format and JSON schema

**`partA_code_and_examples.md`**
- Python implementation of `answer_user_query()`
- 3 example queries with:
  - Expected retrieval results
  - Generated prompts
  - Sample answers with citations
  - Failure modes and mitigations

### Part B Document

**`partB_agent_workflow_and_eval.md`**
- 5-step agentic workflow (query understanding, retrieval, generation, fact-checking, formatting)
- Each step includes: input/output JSON, failure modes, guardrails
- Minimal evaluation loop: 10-query golden set, scoring rubric (100 points)
- Metrics to track (daily/weekly)
- 2-week prioritization plan

### Combined Document

**`final_combined_assignment.md`**
- Compiles Part A and Part B
- Includes reflection (5 key insights, challenges, future improvements)
- Suitable for direct submission

### Code

**`code/rag_system.py`**
- Runnable Python implementation
- Implements hybrid retrieval, fact-checking, citation extraction
- Includes demo function with example queries
- See `code/README.md` for usage instructions

## How to Run the Code

### Prerequisites

1. Python 3.8 or higher
2. Install dependencies:
   ```bash
   cd code
   pip install -r requirements.txt
   ```

### Running the Demo

```bash
cd code
python rag_system.py
```

This will:
1. Load all markdown and CSV files from the parent directory
2. Build BM25 and embedding indices
3. Run 3 example queries
4. Display answers with citations

### Using in Your Own Code

```python
from rag_system import KeralaAyurvedaRAG

# Initialize (point to directory with corpus files)
rag = KeralaAyurvedaRAG(corpus_path="../")

# Answer a query
result = rag.answer_user_query("What are the benefits of Ashwagandha?")

print(result['answer'])
print(f"Confidence: {result['confidence_score']}")
for citation in result['citations']:
    print(f"  - {citation['doc_id']}#{citation['section']}")
```

### Integration with LLM

The current implementation includes a mock answer generator. To use with an actual LLM:

1. Replace `_generate_mock_answer()` in `rag_system.py` with your LLM API call
2. Set `use_llm=True` in `answer_user_query()`
3. Add API key configuration

See `code/README.md` for detailed integration instructions.

## Mapping to Corpus Sources

All design decisions and examples are grounded in the provided corpus files:

### Foundational Documents
- **`ayurveda_foundations.md`**: Informs positioning, boundaries, tridosha model
- **`content_style_and_tone_guide.md`**: Guides prompt templates and style enforcement
- **`dosha_guide_vata_pitta_kapha.md`**: Used in dosha-related queries

### Product Information
- **`product_ashwagandha_tablets_internal.md`**: Example product dossier structure
- **`product_brahmi_tailam_internal.md`**: Topical product example
- **`product_triphala_capsules_internal.md`**: Digestive product with safety info
- **`products_catalog.csv`**: Structured product data (row-level chunking)

### FAQ and Treatment
- **`faq_general_ayurveda_patients.md`**: Q&A pairs (one per chunk)
- **`treatment_stress_support_program.md`**: Clinic program structure

### Specific Citations in Deliverables

Examples throughout the deliverables cite specific corpus sections:
- `[source:product_ashwagandha_tablets_internal.md#Traditional Positioning]`
- `[source:faq_general_ayurveda_patients.md#2. How long does it take to see results?]`
- `[source:ayurveda_foundations.md#Kerala Ayurveda Content Boundaries]`

All retrieval examples, prompt templates, and guardrails are derived from actual corpus content.

## Missing Information Handling

The corpus does not contain:
- Clinical studies or scientific evidence
- Specific drug interaction data (e.g., blood thinners + Triphala)
- Exact dosage instructions (intentionally excluded per corpus boundaries)
- User testimonials or case studies

The system handles these gaps by:
1. Explicitly stating "I don't find this in the provided corpus"
2. Providing general guidance from related sections (e.g., "consult healthcare provider")
3. Marking missing information as `MISSING: [description] - requires clinical/editor review`

## Time Spent & Tools Used

### Time Estimate
- **Part A Design:** ~1.5 hours (chunking strategy, retrieval design, prompt templates)
- **Part A Code/Examples:** ~1.5 hours (implementation, example queries, failure modes)
- **Part B Workflow:** ~1 hour (5-step workflow, evaluation loop, prioritization)
- **Code Implementation:** ~1 hour (Python RAG system, demo function)
- **Documentation:** ~0.5 hours (README, combined document, reflection)
- **Total:** ~5.5 hours

### Tools Used
- **Cursor AI:** Primary IDE and code generation
- **Python:** Implementation language
- **Libraries:**
  - `sentence-transformers`: Dense embeddings
  - `rank-bm25`: Sparse retrieval
  - `scikit-learn`: Cosine similarity
  - `numpy`: Numerical operations
- **Markdown:** Documentation format

### Approach
1. Read and indexed all corpus files
2. Designed RAG system based on corpus structure
3. Implemented hybrid retrieval with concrete parameters
4. Created example queries grounded in corpus content
5. Designed agentic workflow with guardrails
6. Built evaluation loop with golden set
7. Prioritized features for 2-week sprint

## Key Design Decisions

1. **Hybrid Retrieval (40% BM25, 60% Dense):** Balances exact term matching (product names) with semantic search (concepts)

2. **512-Token Chunks with 128-Token Overlap:** Preserves context while maintaining focused retrieval

3. **Cursor-Based Fact-Checking:** Validates each factual claim with similarity threshold 0.6 (0.75 for safety)

4. **Explicit "I don't find this" Response:** Prevents hallucination, maintains trust

5. **Safety-First Guardrails:** Detects safety-critical queries, applies stricter validation

## Next Steps

To deploy this system:

1. **Week 1:** Implement core RAG pipeline and safety guardrails
2. **Week 2:** Add style guide enforcement and evaluation loop
3. **Week 3+:** Add intent classification, multi-turn conversation, fine-tuning

See `partB_agent_workflow_and_eval.md#3. Prioritization for First 2 Weeks` for detailed timeline.

## Questions or Issues

If you encounter issues:
1. Check `code/README.md` for code-specific help
2. Review example queries in `partA_code_and_examples.md`
3. Verify corpus files are in the correct directory structure

---

**Generated by:** Cursor AI Assistant  
**Date:** [Current Date]  
**Corpus Version:** Kerala Ayurveda Content Pack v1


