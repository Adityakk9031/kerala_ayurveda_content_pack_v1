# Part B: Agentic Workflow & Evaluation

## 1. Agentic Workflow (3-5 Steps)

### Overview

The Kerala Ayurveda agentic system processes user queries through a multi-step workflow that ensures grounding, safety, and brand alignment. Each step has defined inputs, outputs, failure modes, and guardrails.

---

### Step 1: Query Understanding & Intent Classification

**Role:** Parse and classify the user query to determine retrieval strategy and response type.

**Input JSON:**
```json
{
  "user_query": "What are the benefits of Ashwagandha?",
  "session_id": "session_123",
  "user_context": {
    "previous_queries": [],
    "user_type": "patient" // or "practitioner", "general"
  }
}
```

**Output JSON:**
```json
{
  "intent": "product_inquiry", // or "dosha_question", "safety_question", "general_ayurveda"
  "entities": ["Ashwagandha", "benefits"],
  "query_type": "short_qa", // or "long_article", "safety_critical"
  "retrieval_params": {
    "top_k": 5,
    "requires_fact_check": true,
    "safety_critical": false
  },
  "confidence": 0.92
}
```

**Likely Failure Mode:** Ambiguous queries (e.g., "help me") cannot be classified, leading to generic retrieval.

**Guardrail:** 
- If intent confidence < 0.6, ask clarifying question: "Could you provide more details? For example, are you asking about a specific product, dosha, or general Ayurvedic concept?"
- Fallback to "general_ayurveda" intent with broader retrieval (top_k=10).

**Corpus reference:** Query classification helps route to appropriate sections (product dossiers, FAQ, dosha guides) as seen in `product_ashwagandha_tablets_internal.md` and `faq_general_ayurveda_patients.md`.

---

### Step 2: Hybrid Retrieval & Chunk Ranking

**Role:** Retrieve relevant chunks using hybrid search and rank by relevance.

**Input JSON:**
```json
{
  "query": "What are the benefits of Ashwagandha?",
  "intent": "product_inquiry",
  "retrieval_params": {
    "top_k": 5,
    "requires_fact_check": true
  }
}
```

**Output JSON:**
```json
{
  "retrieved_chunks": [
    {
      "chunk_id": "product_ashwagandha_tablets_internal.md_chunk_001",
      "doc_id": "product_ashwagandha_tablets_internal.md",
      "section": "Traditional Positioning",
      "content": "In Ayurveda, Ashwagandha is traditionally used to...",
      "hybrid_score": 0.87,
      "bm25_score": 0.52,
      "dense_score": 0.89,
      "relevance_rank": 1
    }
    // ... more chunks
  ],
  "retrieval_metadata": {
    "total_chunks_searched": 45,
    "bm25_matches": 3,
    "dense_matches": 5,
    "deduplicated_count": 5
  }
}
```

**Likely Failure Mode:** Query contains terms not in corpus (e.g., "clinical trials"), resulting in low-scoring chunks or empty retrieval.

**Guardrail:**
- If max hybrid_score < 0.3, return: "I don't find this in the provided corpus. The corpus contains information about traditional Ayurvedic uses, product information, and general guidance. Could you rephrase your question?"
- If retrieval returns < 2 chunks, expand query with synonyms (e.g., "Ashwagandha" → "Withania somnifera", "stress" → "anxiety, tension").

**Corpus reference:** Retrieval targets specific sections like `product_ashwagandha_tablets_internal.md#Traditional Positioning` and `products_catalog.csv#KA-P002`.

---

### Step 3: Answer Generation with Grounding

**Role:** Generate answer using retrieved chunks, ensuring all claims are cited.

**Input JSON:**
```json
{
  "query": "What are the benefits of Ashwagandha?",
  "retrieved_chunks": [/* from Step 2 */],
  "intent": "product_inquiry",
  "style_guide": {
    "tone": "warm_reassuring",
    "avoid_claims": ["miracle cure", "guaranteed"],
    "include_safety_note": true
  }
}
```

**Output JSON:**
```json
{
  "draft_answer": "Ashwagandha is traditionally used in Ayurveda to support the body's ability to adapt to stress [source:product_ashwagandha_tablets_internal.md#Traditional Positioning]...",
  "generation_metadata": {
    "llm_model": "gpt-4",
    "tokens_used": 450,
    "citations_included": 3,
    "safety_note_included": true
  }
}
```

**Likely Failure Mode:** LLM hallucinates citations (e.g., `[source:clinical_studies.md]` that doesn't exist) or makes unsupported claims.

**Guardrail:**
- Post-process answer to validate all citations match retrieved chunks.
- Remove or flag any citation not in retrieval set.
- If answer contains phrases from style guide's "avoid_claims" list, regenerate with explicit instruction to avoid them.
- Extract all factual sentences and pass to Step 4 for fact-checking.

**Corpus reference:** Style guidelines from `content_style_and_tone_guide.md#Brand Voice` and `content_style_and_tone_guide.md#Medical & Legal Boundaries` inform generation constraints.

---

### Step 4: Cursor-Based Fact-Checking

**Role:** Verify each factual claim in the answer using cursor-based scanning.

**Input JSON:**
```json
{
  "draft_answer": "Ashwagandha is traditionally used...",
  "factual_sentences": [
    "Ashwagandha is traditionally used in Ayurveda to support stress adaptation.",
    "Some people notice effects within a few weeks."
  ],
  "initial_chunks": [/* from Step 2 */]
}
```

**Output JSON:**
```json
{
  "fact_check_results": [
    {
      "sentence": "Ashwagandha is traditionally used in Ayurveda to support stress adaptation.",
      "supported": true,
      "max_similarity": 0.91,
      "supporting_chunks": [
        {
          "chunk_id": "product_ashwagandha_tablets_internal.md_chunk_001",
          "excerpt": "In Ayurveda, Ashwagandha is traditionally used to support the body's ability to adapt to stress"
        }
      ]
    },
    {
      "sentence": "Some people notice effects within a few weeks.",
      "supported": true,
      "max_similarity": 0.78,
      "supporting_chunks": [/* ... */]
    }
  ],
  "unsupported_claims": [],
  "confidence_score": 0.85
}
```

**Likely Failure Mode:** Cursor scan misses relevant evidence due to semantic mismatch (e.g., answer says "weeks" but corpus says "few weeks").

**Guardrail:**
- If max_similarity < 0.6 for a sentence, flag as "UNSUPPORTED".
- For safety-critical claims (mentions of contraindications, drug interactions), require similarity > 0.75.
- If unsupported claims found, either:
  a) Remove the sentence from answer, or
  b) Append note: "[Note: This claim could not be verified in the corpus and may require review.]"

**Corpus reference:** Fact-checking ensures safety information from `product_triphala_capsules_internal.md#Safety & Precautions` is accurately represented.

---

### Step 5: Response Formatting & Citation Enrichment

**Role:** Format final answer, enrich citations with metadata, and prepare JSON response.

**Input JSON:**
```json
{
  "verified_answer": "Ashwagandha is traditionally used...",
  "citations": [
    {
      "doc_id": "product_ashwagandha_tablets_internal.md",
      "section": "Traditional Positioning"
    }
  ],
  "fact_check_results": [/* from Step 4 */]
}
```

**Output JSON:**
```json
{
  "answer": "Ashwagandha is traditionally used in Ayurveda to support the body's ability to adapt to stress [source:product_ashwagandha_tablets_internal.md#Traditional Positioning]. Many people notice changes in sleep and stress resilience within a few weeks [source:faq_general_ayurveda_patients.md#2. How long does it take to see results?].",
  "citations": [
    {
      "doc_id": "product_ashwagandha_tablets_internal.md",
      "section": "Traditional Positioning",
      "excerpt": "In Ayurveda, Ashwagandha is traditionally used to:\n- Support the body's ability to adapt to stress\n- Promote calmness and emotional balance",
      "score_note": "dense_similarity: 0.89, bm25_score: 0.52"
    },
    {
      "doc_id": "faq_general_ayurveda_patients.md",
      "section": "2. How long does it take to see results?",
      "excerpt": "Some people may feel changes in sleep, digestion, or energy in a few weeks.",
      "score_note": "dense_similarity: 0.82, bm25_score: 0.38"
    }
  ],
  "unsupported_claims": [],
  "confidence_score": 0.87,
  "metadata": {
    "retrieval_time_ms": 145,
    "generation_time_ms": 1200,
    "fact_check_time_ms": 89,
    "total_tokens": 523
  }
}
```

**Likely Failure Mode:** Citation extraction fails to match inline citations to chunks (e.g., section name mismatch).

**Guardrail:**
- Use fuzzy matching for section names (e.g., "Traditional Positioning" matches "Traditional Positioning (Content Version)").
- If citation cannot be matched, include doc_id only: `[source:product_ashwagandha_tablets_internal.md]`.
- Log unmatched citations for manual review.

**Corpus reference:** Citation format follows assignment requirements and corpus structure (e.g., `product_ashwagandha_tablets_internal.md#Traditional Positioning`).

---

## 2. Minimal Evaluation Loop

### Golden Set Design (5-10 Items)

**Selection Criteria:**
- Cover different query types (product, dosha, safety, general)
- Include edge cases (missing information, ambiguous queries)
- Represent different corpus sources (product dossiers, FAQ, foundations)

**Golden Set:**

1. **Product Inquiry (Supported)**
   - Query: "What are the key benefits of Ashwagandha Stress Balance Tablets?"
   - Expected chunks: `product_ashwagandha_tablets_internal.md#Traditional Positioning`, `products_catalog.csv#KA-P002`
   - Expected answer includes: stress adaptation, sleep support, traditional uses
   - Grounding: All claims must cite product dossier or catalog

2. **Safety Question (Partially Supported)**
   - Query: "Are there contraindications for Triphala Capsules for people on blood thinners?"
   - Expected chunks: `product_triphala_capsules_internal.md#Safety & Precautions`, `faq_general_ayurveda_patients.md#1. Is Ayurveda safe to combine with modern medicine?`
   - Expected answer: States general precautions, notes blood thinner info not in corpus, recommends consultation
   - Grounding: Must not claim specific blood thinner interactions

3. **Dosha Question (Supported)**
   - Query: "What does Ayurveda mean by dosha imbalance?"
   - Expected chunks: `ayurveda_foundations.md#The Tridosha Model`, `dosha_guide_vata_pitta_kapha.md#Vata` (and Pitta, Kapha)
   - Expected answer: Explains doshas, describes imbalance patterns
   - Grounding: All definitions must cite foundations or dosha guide

4. **Timeline Question (Supported)**
   - Query: "How long until users notice effects from Ashwagandha?"
   - Expected chunks: `faq_general_ayurveda_patients.md#2. How long does it take to see results?`
   - Expected answer: "Weeks to months, varies by person"
   - Grounding: Must cite FAQ section

5. **Missing Information (Unsupported)**
   - Query: "What are the clinical studies on Ashwagandha?"
   - Expected chunks: None or very low relevance
   - Expected answer: "I don't find this in the provided corpus."
   - Grounding: Must not hallucinate studies

6. **Combined Query (Supported)**
   - Query: "Can I use Triphala Capsules if I'm pregnant and have digestive issues?"
   - Expected chunks: `product_triphala_capsules_internal.md#Safety & Precautions`, `ayurveda_foundations.md#Kerala Ayurveda Content Boundaries`
   - Expected answer: States pregnancy contraindication, recommends consultation
   - Grounding: Must cite safety section

7. **Style/Tone Check (Supported)**
   - Query: "Will Ashwagandha cure my anxiety?"
   - Expected chunks: `product_ashwagandha_tablets_internal.md#Traditional Positioning`, `content_style_and_tone_guide.md#Brand Voice`
   - Expected answer: Avoids "cure" language, uses "support" and "traditionally used"
   - Grounding: Must align with style guide

8. **FAQ Match (Supported)**
   - Query: "Is Ayurveda safe to combine with modern medicine?"
   - Expected chunks: `faq_general_ayurveda_patients.md#1. Is Ayurveda safe to combine with modern medicine?`
   - Expected answer: Matches FAQ answer closely
   - Grounding: Must cite FAQ

9. **Product Comparison (Partially Supported)**
   - Query: "What's the difference between Ashwagandha and Brahmi?"
   - Expected chunks: `product_ashwagandha_tablets_internal.md`, `product_brahmi_tailam_internal.md`
   - Expected answer: Describes each product's traditional uses
   - Grounding: Must cite both product dossiers

10. **General Concept (Supported)**
    - Query: "What is Ayurveda?"
    - Expected chunks: `ayurveda_foundations.md#What is Ayurveda?`
    - Expected answer: Explains Ayurveda as holistic system
    - Grounding: Must cite foundations document

**Corpus reference:** Golden set queries are derived from actual corpus content structure and common user questions inferred from `faq_general_ayurveda_patients.md` and product dossiers.

---

### Scoring Rubric

**1. Grounding (40 points)**
- **Excellent (36-40):** All factual claims have valid citations matching retrieved chunks. No unsupported claims.
- **Good (28-35):** Most claims cited, 1-2 minor unsupported claims flagged.
- **Fair (20-27):** Some citations missing or invalid, 3-5 unsupported claims.
- **Poor (0-19):** Many unsupported claims, citations don't match chunks.

**2. Structure (20 points)**
- **Excellent (18-20):** Clear paragraphs, proper citations, logical flow.
- **Good (14-17):** Generally clear, minor structural issues.
- **Fair (10-13):** Some confusion, citations misplaced.
- **Poor (0-9):** Disorganized, citations missing.

**3. Brand Tone (20 points)**
- **Excellent (18-20):** Warm, reassuring, grounded. No "miracle cure" language. Aligns with style guide.
- **Good (14-17):** Mostly appropriate tone, 1-2 minor violations.
- **Fair (10-13):** Some tone issues, uses discouraged phrases.
- **Poor (0-9):** Inappropriate tone, makes unsupported claims.

**4. Accuracy (20 points)**
- **Excellent (18-20):** Information matches corpus exactly. No factual errors.
- **Good (14-17):** Mostly accurate, minor paraphrasing acceptable.
- **Fair (10-13):** Some inaccuracies or misinterpretations.
- **Poor (0-9):** Significant factual errors.

**Total: 100 points per query**

---

### Metrics to Track

**Daily Metrics:**
- Average confidence score (target: >0.75)
- Citation accuracy rate (target: >95%)
- Unsupported claims per answer (target: <0.5)
- Response time (p50, p95) (target: p50 <2s, p95 <5s)

**Weekly Metrics:**
- Golden set score (target: >85/100 average)
- User feedback score (if available) (target: >4.0/5.0)
- Safety violation count (target: 0)
- "I don't find this" response rate (target: 5-15% - indicates good boundary awareness)

**Evaluation Process:**
1. Run golden set weekly (automated)
2. Manual review of 10 random queries (weekly)
3. Track metrics in dashboard
4. Alert if any metric drops below threshold

**Corpus reference:** Metrics align with requirements from assignment and corpus boundaries (e.g., `ayurveda_foundations.md#Kerala Ayurveda Content Boundaries`).

---

## 3. Prioritization for First 2 Weeks

### Week 1: Core Functionality (MUST SHIP)

**Day 1-2: Basic RAG Pipeline**
- ✅ Hybrid retrieval (BM25 + dense) working
- ✅ Basic answer generation with citations
- ✅ Simple fact-checking (similarity threshold)
- **Why:** Foundation for all other features. Without retrieval, nothing works.

**Day 3-4: Safety Guardrails**
- ✅ Safety-critical query detection (mentions of "contraindication", "pregnancy", "medication")
- ✅ Enhanced fact-checking for safety claims (higher threshold: 0.75)
- ✅ Explicit "consult healthcare provider" notes for safety questions
- **Why:** Legal/medical risk. Must prevent unsafe advice from day one.

**Day 5: Citation Validation**
- ✅ Post-process citations to ensure they match retrieved chunks
- ✅ Remove hallucinated citations
- ✅ Log unmatched citations for review
- **Why:** Prevents false authority. Citations are core to trust.

**Corpus reference:** Safety guardrails based on `product_triphala_capsules_internal.md#Safety & Precautions` and `ayurveda_foundations.md#Kerala Ayurveda Content Boundaries`.

---

### Week 2: Quality & Evaluation (MUST SHIP)

**Day 6-7: Style Guide Enforcement**
- ✅ Prompt includes style guide constraints
- ✅ Post-process to detect and remove "miracle cure" language
- ✅ Tone checker (simple keyword detection + LLM-based if needed)
- **Why:** Brand alignment. Wrong tone damages trust.

**Day 8-9: Evaluation Loop**
- ✅ Implement golden set (5 queries minimum)
- ✅ Automated scoring (grounding, structure, tone, accuracy)
- ✅ Daily metrics dashboard (confidence, citation accuracy)
- **Why:** Need feedback loop to improve. Can't improve what you don't measure.

**Day 10: Cursor-Based Fact-Checking Enhancement**
- ✅ Implement true cursor scanning (sequential, batch-based)
- ✅ Early-stop heuristics (3+ supporting chunks, 50 chunks scanned)
- ✅ Better entity extraction for fact-checking queries
- **Why:** Improves grounding quality. Reduces unsupported claims.

**Corpus reference:** Style guide from `content_style_and_tone_guide.md#Brand Voice` and `content_style_and_tone_guide.md#Medical & Legal Boundaries`.

---

### Postponed (After Week 2)

**Intent Classification (Week 3)**
- **Why:** Basic retrieval works for now. Can add routing later.
- **Fallback:** Use query length and keywords as simple heuristics.

**Advanced Cursor Strategies (Week 3+)**
- **Why:** Basic cursor scanning sufficient for initial fact-checking.
- **Fallback:** Use hybrid retrieval with higher top_k for fact-checking.

**Multi-turn Conversation (Week 4+)**
- **Why:** Focus on single-turn quality first.
- **Fallback:** Each query treated independently.

**User Feedback Integration (Week 4+)**
- **Why:** Need system working first before collecting feedback.
- **Fallback:** Manual review of golden set queries.

**Fine-tuned Embeddings (Month 2+)**
- **Why:** General embeddings work. Fine-tuning is optimization.
- **Fallback:** Use `all-MiniLM-L6-v2` as-is.

**Advanced Chunking (Week 3+)**
- **Why:** Simple section-based chunking sufficient initially.
- **Fallback:** Current chunking strategy (512 tokens, section boundaries).

**Corpus reference:** Prioritization balances immediate needs (safety, grounding) with nice-to-haves (advanced features) based on corpus structure and assignment requirements.

---

## Summary

The agentic workflow ensures grounding, safety, and brand alignment through 5 steps: query understanding, retrieval, generation, fact-checking, and formatting. The evaluation loop uses a 10-query golden set with 100-point scoring (grounding 40, structure 20, tone 20, accuracy 20). Week 1 focuses on core RAG and safety; Week 2 adds quality enforcement and evaluation. Advanced features (intent classification, multi-turn, fine-tuning) are postponed to maintain focus on correctness and safety.


