# Part A: Function Design and Examples

## Function Design: `answer_user_query`

### Python Implementation

```python
from typing import Dict, List, Optional
import json
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class KeralaAyurvedaRAG:
    def __init__(self, corpus_path: str = "./"):
        self.corpus_path = corpus_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []  # List of chunk dictionaries
        self.chunk_embeddings = None
        self.bm25_index = None
        self._load_and_index_corpus()
    
    def _load_and_index_corpus(self):
        """Load all markdown and CSV files, chunk them, and build indices."""
        # Implementation would:
        # 1. Load all .md files and products_catalog.csv
        # 2. Chunk according to plan (512 tokens, 128 overlap for MD; row-level for CSV)
        # 3. Generate embeddings for all chunks
        # 4. Build BM25 index from chunk texts
        pass
    
    def _hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid retrieval: BM25 + dense embeddings."""
        # BM25 retrieval
        bm25_scores = self.bm25_index.get_scores(query.split())
        bm25_top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        
        # Dense retrieval
        query_embedding = self.embedding_model.encode([query])[0]
        dense_scores = cosine_similarity(
            [query_embedding], 
            self.chunk_embeddings
        )[0]
        dense_top_indices = np.argsort(dense_scores)[-top_k:][::-1]
        
        # Combine and deduplicate
        combined_indices = list(set(bm25_top_indices.tolist() + dense_top_indices.tolist()))
        
        # Score and rank
        scored_chunks = []
        for idx in combined_indices:
            chunk = self.chunks[idx]
            normalized_bm25 = bm25_scores[idx] / (max(bm25_scores) + 1e-8)
            normalized_dense = dense_scores[idx]
            hybrid_score = 0.4 * normalized_bm25 + 0.6 * normalized_dense
            
            scored_chunks.append({
                **chunk,
                'hybrid_score': hybrid_score,
                'bm25_score': bm25_scores[idx],
                'dense_score': dense_scores[idx]
            })
        
        # Sort by hybrid score and return top_k
        scored_chunks.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return scored_chunks[:top_k]
    
    def _fact_check_with_cursor(self, answer: str, initial_chunks: List[Dict]) -> Dict:
        """Use cursor-based retrieval to fact-check the answer."""
        # Extract factual sentences
        sentences = self._extract_factual_sentences(answer)
        
        unsupported_claims = []
        supporting_evidence = []
        
        for sentence in sentences:
            # Extract key entities
            entities = self._extract_entities(sentence)
            
            # Cursor scan: search for entities in corpus
            cursor_chunks = self._cursor_scan(entities, batch_size=10, max_chunks=50)
            
            # Check if sentence is supported
            max_similarity = max([
                self._semantic_similarity(sentence, chunk['content'])
                for chunk in cursor_chunks
            ]) if cursor_chunks else 0.0
            
            if max_similarity < 0.6:
                unsupported_claims.append({
                    'sentence': sentence,
                    'max_similarity': max_similarity,
                    'scanned_chunks': len(cursor_chunks)
                })
            else:
                supporting_evidence.append({
                    'sentence': sentence,
                    'supporting_chunks': cursor_chunks[:3]
                })
        
        return {
            'unsupported_claims': unsupported_claims,
            'supporting_evidence': supporting_evidence
        }
    
    def _cursor_scan(self, entities: List[str], batch_size: int = 10, max_chunks: int = 50) -> List[Dict]:
        """Cursor-based scanning of corpus for specific entities."""
        # Implementation would:
        # 1. Start from beginning of relevant documents
        # 2. Process chunks in batches
        # 3. Early stop if 3+ supporting chunks found
        # 4. Early stop if max_chunks scanned without support
        # Returns list of relevant chunks
        pass
    
    def _build_qa_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Build the Q&A prompt with retrieved chunks."""
        context = "\n\n---\n\n".join([
            f"**Source:** {chunk['doc_id']}#{chunk.get('section', 'N/A')}\n"
            f"{chunk['content']}"
            for chunk in chunks
        ])
        
        prompt = f"""You are a helpful assistant for Kerala Ayurveda, providing accurate information based solely on the provided context documents.

**Context Documents:**
{context}

**User Query:**
{query}

**Instructions:**
1. Answer the query using ONLY information from the context documents above.
2. If the information is not present in the context, respond: "I don't find this in the provided corpus."
3. Include inline citations in square brackets: [source:filename.md#Section Name]
4. Use warm, reassuring, and grounded language (avoid "miracle cure" or "guaranteed" claims).
5. For safety information, be explicit and conservative.
6. Keep the answer concise (2-4 paragraphs for short queries).

**Answer:**"""
        
        return prompt
    
    def answer_user_query(self, query: str) -> Dict:
        """Main function: answer user query with citations."""
        # Step 1: Hybrid retrieval
        retrieved_chunks = self._hybrid_retrieve(query, top_k=5 if len(query.split()) < 20 else 10)
        
        # Step 2: Build prompt
        prompt = self._build_qa_prompt(query, retrieved_chunks)
        
        # Step 3: Generate answer (using LLM - placeholder here)
        answer = self._generate_answer(prompt)  # Would call OpenAI/Anthropic/etc.
        
        # Step 4: Fact-check with cursor
        fact_check_result = self._fact_check_with_cursor(answer, retrieved_chunks)
        
        # Step 5: Extract citations and build response
        citations = self._extract_citations(answer, retrieved_chunks)
        
        # Step 6: Remove or flag unsupported claims
        if fact_check_result['unsupported_claims']:
            answer = self._flag_unsupported_claims(answer, fact_check_result['unsupported_claims'])
        
        return {
            'answer': answer,
            'citations': citations,
            'unsupported_claims': fact_check_result['unsupported_claims'],
            'confidence_score': self._calculate_confidence(retrieved_chunks, fact_check_result)
        }
    
    def _generate_answer(self, prompt: str) -> str:
        """Placeholder for LLM call."""
        # Would integrate with OpenAI, Anthropic, or local LLM
        pass
    
    def _extract_citations(self, answer: str, chunks: List[Dict]) -> List[Dict]:
        """Extract citations from answer and match to chunks."""
        # Extract [source:...] patterns
        citation_pattern = r'\[source:([^\]]+)\]'
        citations = []
        
        for match in re.finditer(citation_pattern, answer):
            citation_ref = match.group(1)
            # Parse doc_id and section
            if '#' in citation_ref:
                doc_id, section = citation_ref.split('#', 1)
            else:
                doc_id = citation_ref
                section = None
            
            # Find matching chunk
            matching_chunk = next(
                (c for c in chunks if c['doc_id'] == doc_id and c.get('section') == section),
                None
            )
            
            if matching_chunk:
                citations.append({
                    'doc_id': doc_id,
                    'section': section,
                    'excerpt': matching_chunk['content'][:200] + '...',
                    'score_note': f"dense_similarity: {matching_chunk.get('dense_score', 0):.2f}, bm25_score: {matching_chunk.get('bm25_score', 0):.2f}"
                })
        
        return citations
```

---

## Example Queries and Responses

### Example 1: "What are the key benefits of Ashwagandha Stress Balance Tablets? How long until users notice effects?"

#### Expected Retrieval Results

**Chunks retrieved (top 5):**

1. **File:** `product_ashwagandha_tablets_internal.md`  
   **Section:** "Traditional Positioning"  
   **Content:** "In Ayurveda, Ashwagandha is traditionally used to: Support the body's ability to adapt to stress, Promote calmness and emotional balance, Support strength and stamina, Help maintain restful sleep"

2. **File:** `product_ashwagandha_tablets_internal.md`  
   **Section:** "Key Messages for Content"  
   **Content:** "Stress resilience, not sedation - Emphasise adaptation, steadiness, and recovery. Avoid implying that tablets replace rest, boundaries, or therapy."

3. **File:** `faq_general_ayurveda_patients.md`  
   **Section:** "2. How long does it take to see results?"  
   **Content:** "Timelines vary from person to person. In general: Some people may feel changes in sleep, digestion, or energy in a few weeks. Deeper changes in patterns (stress, lifestyle, long-standing discomforts) often take longer. We recommend thinking in terms of weeks to months, not overnight fixes."

4. **File:** `products_catalog.csv`  
   **Row:** KA-P002  
   **Content:** "Ashwagandha Stress Balance Tablets, Stress & Sleep, Tablets, Stress resilience; restful sleep, Ashwagandha root extract"

5. **File:** `product_ashwagandha_tablets_internal.md`  
   **Section:** "Basic Info"  
   **Content:** "Product name: Ashwagandha Stress Balance Tablets, Category: Stress & sleep support, Format: Tablets, Key herb: Ashwagandha (Withania somnifera) root extract"

#### Prompt Sent to LLM

```
You are a helpful assistant for Kerala Ayurveda, providing accurate information based solely on the provided context documents.

**Context Documents:**

**Source:** product_ashwagandha_tablets_internal.md#Traditional Positioning
In Ayurveda, Ashwagandha is traditionally used to:
- Support the body's ability to adapt to stress
- Promote calmness and emotional balance
- Support strength and stamina
- Help maintain restful sleep

Our positioning:
> "Daily support for stress resilience and restful sleep, inspired by a classical adaptogenic herb."

---

**Source:** product_ashwagandha_tablets_internal.md#Key Messages for Content
1. **Stress resilience, not sedation**
   - Emphasise adaptation, steadiness, and recovery.
   - Avoid implying that tablets replace rest, boundaries, or therapy.

---

**Source:** faq_general_ayurveda_patients.md#2. How long does it take to see results?
Timelines vary from person to person. In general:
- Some people may feel changes in sleep, digestion, or energy in a few weeks.
- Deeper changes in patterns (stress, lifestyle, long-standing discomforts) often take longer.

We recommend thinking in terms of **weeks to months**, not overnight fixes.

---

**Source:** products_catalog.csv#KA-P002
Ashwagandha Stress Balance Tablets, Stress & Sleep, Tablets, Stress resilience; restful sleep, Ashwagandha root extract

---

**Source:** product_ashwagandha_tablets_internal.md#Basic Info
Product name: Ashwagandha Stress Balance Tablets
Category: Stress & sleep support
Format: Tablets
Key herb: Ashwagandha (Withania somnifera) root extract

**User Query:**
What are the key benefits of Ashwagandha Stress Balance Tablets? How long until users notice effects?

**Instructions:**
1. Answer the query using ONLY information from the context documents above.
2. If the information is not present in the context, respond: "I don't find this in the provided corpus."
3. Include inline citations in square brackets: [source:filename.md#Section Name]
4. Use warm, reassuring, and grounded language (avoid "miracle cure" or "guaranteed" claims).
5. For safety information, be explicit and conservative.
6. Keep the answer concise (2-4 paragraphs for short queries).

**Answer:**
```

#### Sample Answer (with citations)

```
Ashwagandha Stress Balance Tablets are designed to support stress resilience and restful sleep [source:product_ashwagandha_tablets_internal.md#Traditional Positioning]. In Ayurveda, Ashwagandha is traditionally used to help the body adapt to stress, promote calmness and emotional balance, and support strength and stamina [source:product_ashwagandha_tablets_internal.md#Traditional Positioning].

The tablets focus on building resilience and adaptation rather than providing sedation [source:product_ashwagandha_tablets_internal.md#Key Messages for Content]. This means they support your body's natural ability to recover and maintain steadiness during challenging times.

Regarding timelines, experiences vary from person to person [source:faq_general_ayurveda_patients.md#2. How long does it take to see results?]. Some people may notice changes in sleep, digestion, or energy within a few weeks, while deeper changes in stress patterns and lifestyle often take longer. We recommend thinking in terms of weeks to months, rather than expecting overnight fixes [source:faq_general_ayurveda_patients.md#2. How long does it take to see results?].
```

#### Citations JSON

```json
{
  "answer": "Ashwagandha Stress Balance Tablets are designed to support stress resilience and restful sleep...",
  "citations": [
    {
      "doc_id": "product_ashwagandha_tablets_internal.md",
      "section": "Traditional Positioning",
      "excerpt": "In Ayurveda, Ashwagandha is traditionally used to:\n- Support the body's ability to adapt to stress\n- Promote calmness and emotional balance...",
      "score_note": "dense_similarity: 0.89, bm25_score: 0.52"
    },
    {
      "doc_id": "product_ashwagandha_tablets_internal.md",
      "section": "Key Messages for Content",
      "excerpt": "1. **Stress resilience, not sedation**\n   - Emphasise adaptation, steadiness, and recovery...",
      "score_note": "dense_similarity: 0.76, bm25_score: 0.41"
    },
    {
      "doc_id": "faq_general_ayurveda_patients.md",
      "section": "2. How long does it take to see results?",
      "excerpt": "Timelines vary from person to person. In general:\n- Some people may feel changes in sleep, digestion, or energy in a few weeks...",
      "score_note": "dense_similarity: 0.82, bm25_score: 0.38"
    }
  ],
  "unsupported_claims": [],
  "confidence_score": 0.87
}
```

#### Likely Failure Mode

**Failure:** Query asks for "clinical studies" or "scientific evidence" which is not present in the corpus.

**Mitigation:** System should respond: "I don't find information about clinical studies or scientific evidence for Ashwagandha Stress Balance Tablets in the provided corpus. The corpus contains traditional Ayurvedic uses and positioning information."

---

### Example 2: "Are there any contraindications or precautions for Triphala Capsules, especially for people on blood thinners?"

#### Expected Retrieval Results

**Chunks retrieved (top 5):**

1. **File:** `product_triphala_capsules_internal.md`  
   **Section:** "Safety & Precautions (Content Version)"  
   **Content:** "Not intended to diagnose, treat, cure, or prevent any disease. Individuals with: Chronic digestive disease, Recent abdominal surgery, Pregnancy or breastfeeding should consult a qualified healthcare provider before using Triphala."

2. **File:** `products_catalog.csv`  
   **Row:** KA-P001  
   **Content:** "Triphala Capsules, Digestive support, Capsules, Digestive comfort; regular elimination, Amalaki; Bibhitaki; Haritaki, No, Consult doctor in chronic digestive disease, pregnancy, or post-surgery"

3. **File:** `faq_general_ayurveda_patients.md`  
   **Section:** "1. Is Ayurveda safe to combine with modern medicine?"  
   **Content:** "Ayurveda is often used alongside modern medicine. However, herb–drug interactions are possible, and individual health situations vary. We encourage readers to: Inform their doctor about any Ayurvedic supplements or therapies they use. Consult both their Ayurvedic practitioner and medical doctor if they have chronic conditions or take prescription medicines."

4. **File:** `product_triphala_capsules_internal.md`  
   **Section:** "Traditional Positioning"  
   **Content:** "In Ayurveda, Triphala is traditionally used to support: Digestive comfort, Regular elimination, Gentle internal cleansing"

5. **File:** `ayurveda_foundations.md`  
   **Section:** "Kerala Ayurveda Content Boundaries (for writers & agents)"  
   **Content:** "No suggesting stopping or changing prescribed medication. Always add a gentle safety note for herbs and therapies: 'Individuals with existing medical conditions or those on medication should consult their healthcare provider before starting any new supplement or therapy.'"

#### Sample Answer (with citations)

```
Triphala Capsules are generally considered gentle for digestive support, but there are important precautions to consider [source:product_triphala_capsules_internal.md#Safety & Precautions (Content Version)].

Individuals with chronic digestive disease, recent abdominal surgery, or those who are pregnant or breastfeeding should consult a qualified healthcare provider before using Triphala [source:product_triphala_capsules_internal.md#Safety & Precautions (Content Version)].

Regarding blood thinners specifically, I don't find explicit information about blood thinner interactions in the provided corpus. However, the corpus emphasizes that herb–drug interactions are possible [source:faq_general_ayurveda_patients.md#1. Is Ayurveda safe to combine with modern medicine?]. If you are taking prescription medications, including blood thinners, we strongly recommend consulting both your medical doctor and Ayurvedic practitioner before starting any new supplement [source:faq_general_ayurveda_patients.md#1. Is Ayurveda safe to combine with modern medicine?]. The corpus also states that individuals with existing medical conditions or those on medication should consult their healthcare provider before starting any new supplement or therapy [source:ayurveda_foundations.md#Kerala Ayurveda Content Boundaries (for writers & agents)].
```

#### Citations JSON

```json
{
  "answer": "Triphala Capsules are generally considered gentle...",
  "citations": [
    {
      "doc_id": "product_triphala_capsules_internal.md",
      "section": "Safety & Precautions (Content Version)",
      "excerpt": "Not intended to diagnose, treat, cure, or prevent any disease. Individuals with: Chronic digestive disease, Recent abdominal surgery...",
      "score_note": "dense_similarity: 0.91, bm25_score: 0.67"
    },
    {
      "doc_id": "faq_general_ayurveda_patients.md",
      "section": "1. Is Ayurveda safe to combine with modern medicine?",
      "excerpt": "Ayurveda is often used alongside modern medicine. However, herb–drug interactions are possible...",
      "score_note": "dense_similarity: 0.78, bm25_score: 0.43"
    },
    {
      "doc_id": "ayurveda_foundations.md",
      "section": "Kerala Ayurveda Content Boundaries (for writers & agents)",
      "excerpt": "Individuals with existing medical conditions or those on medication should consult their healthcare provider...",
      "score_note": "dense_similarity: 0.72, bm25_score: 0.39"
    }
  ],
  "unsupported_claims": [],
  "confidence_score": 0.83
}
```

#### Likely Failure Mode

**Failure:** Query asks for specific information about blood thinner interactions, which is not explicitly mentioned in the corpus.

**Mitigation:** System correctly identifies the gap and provides general guidance from related safety sections, explicitly stating "I don't find explicit information about blood thinner interactions in the provided corpus" while still offering relevant safety guidance.

---

### Example 3: "What does Ayurveda mean by 'dosha imbalance' and how is it addressed?"

#### Expected Retrieval Results

**Chunks retrieved (top 5):**

1. **File:** `ayurveda_foundations.md`  
   **Section:** "The Tridosha Model (Vata, Pitta, Kapha)"  
   **Content:** "Ayurveda groups functions of the body–mind into three broad principles called *doshas*: Vata – associated with movement, communication, adaptability; Pitta – associated with transformation, digestion, metabolism, focus; Kapha – associated with stability, structure, nourishment, lubrication. Every person has all three, in unique proportions. Imbalance can show up as discomfort in energy, digestion, sleep, mood, or other functions."

2. **File:** `dosha_guide_vata_pitta_kapha.md`  
   **Section:** "Vata"  
   **Content:** "Imbalance may show as: Restlessness, worry, scattered focus, Irregular appetite or digestion, Sleep disturbance (difficulty falling or staying asleep), Feeling cold, dryness in skin or hair"

3. **File:** `dosha_guide_vata_pitta_kapha.md`  
   **Section:** "Pitta"  
   **Content:** "Imbalance may show as: Irritability, impatience, Overheating, skin sensitivity, Tendency to 'overdo' and burn out"

4. **File:** `dosha_guide_vata_pitta_kapha.md`  
   **Section:** "Kapha"  
   **Content:** "Imbalance may show as: Lethargy, low motivation, Feeling 'stuck' or resistant to change, Tendency to feel heavy or sluggish"

5. **File:** `treatment_stress_support_program.md`  
   **Section:** "Core Components"  
   **Content:** "Initial Ayurvedic Consultation: Detailed history of lifestyle, sleep, digestion, stressors. Assessment of dosha tendencies and current imbalance. Collaborative goal-setting"

#### Sample Answer (with citations)

```
In Ayurveda, doshas are three broad principles that describe functions of the body and mind: Vata (movement, communication, adaptability), Pitta (transformation, digestion, metabolism, focus), and Kapha (stability, structure, nourishment, lubrication) [source:ayurveda_foundations.md#The Tridosha Model (Vata, Pitta, Kapha)]. Every person has all three doshas in unique proportions [source:ayurveda_foundations.md#The Tridosha Model (Vata, Pitta, Kapha)].

Dosha imbalance occurs when one or more doshas become excessive or disturbed, which can show up as discomfort in energy, digestion, sleep, mood, or other functions [source:ayurveda_foundations.md#The Tridosha Model (Vata, Pitta, Kapha)]. For example, Vata imbalance may appear as restlessness, worry, scattered focus, irregular appetite, or sleep disturbances [source:dosha_guide_vata_pitta_kapha.md#Vata]. Pitta imbalance might show as irritability, overheating, or a tendency to overdo and burn out [source:dosha_guide_vata_pitta_kapha.md#Pitta]. Kapha imbalance could manifest as lethargy, low motivation, or feeling stuck or heavy [source:dosha_guide_vata_pitta_kapha.md#Kapha].

Ayurveda addresses dosha imbalance through personalized approaches that may include dietary choices, lifestyle routines, herbs, and therapies [source:treatment_stress_support_program.md#Core Components]. An initial Ayurvedic consultation typically involves assessing dosha tendencies and current imbalance, followed by collaborative goal-setting and personalized recommendations [source:treatment_stress_support_program.md#Core Components]. The approach emphasizes supporting natural processes and balance rather than treating diseases [source:ayurveda_foundations.md#How we position Ayurveda at Kerala Ayurveda].
```

#### Likely Failure Mode

**Failure:** Query asks for specific "dosha balancing recipes" or "exact protocols" which may not be detailed in the corpus.

**Mitigation:** System should provide general principles from the corpus and note that specific protocols require personalized consultation, as indicated in the treatment program documentation.

---

## Failure Modes Summary

1. **Missing specific information:** Corpus may lack clinical studies, exact dosages, or specific drug interactions. System should explicitly state "I don't find this in the provided corpus" rather than hallucinating.

2. **Ambiguous queries:** Queries like "best product for me" require personalization not available in static corpus. System should provide general information and recommend consultation.

3. **Safety-critical gaps:** When safety information is incomplete (e.g., blood thinner interactions), system should err on the side of caution and recommend professional consultation.

4. **Citation hallucination:** LLM might generate citations that don't match retrieved chunks. Post-processing validation ensures citations only reference actually retrieved chunks.


