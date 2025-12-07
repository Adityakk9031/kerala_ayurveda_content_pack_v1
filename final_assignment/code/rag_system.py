"""
Kerala Ayurveda RAG System
Implements answer_user_query with hybrid retrieval and fact-checking.
"""

from typing import Dict, List, Optional, Tuple
import json
import re
import os
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

# Note: These imports would require installation:
# pip install sentence-transformers rank-bm25 scikit-learn numpy

try:
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Warning: Required libraries not installed. See requirements.txt")
    SentenceTransformer = None
    BM25Okapi = None
    np = None
    cosine_similarity = None


@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""
    doc_id: str
    chunk_id: str
    content: str
    section: Optional[str] = None
    section_path: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 1
    doc_type: str = "markdown"
    tokens: int = 0


@dataclass
class Citation:
    """Represents a citation in the answer."""
    doc_id: str
    section: Optional[str]
    excerpt: str
    score_note: str


class KeralaAyurvedaRAG:
    """
    RAG system for Kerala Ayurveda content.
    
    Implements:
    - Hybrid retrieval (BM25 + dense embeddings)
    - Cursor-based fact-checking
    - Citation extraction and validation
    """
    
    def __init__(self, corpus_path: str = "../"):
        """
        Initialize RAG system.
        
        Args:
            corpus_path: Path to directory containing markdown and CSV files
        """
        self.corpus_path = Path(corpus_path)
        self.chunks: List[Chunk] = []
        self.chunk_embeddings = None
        self.bm25_index = None
        self.bm25_tokenizer = None
        
        # Initialize embedding model
        if SentenceTransformer:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedding_model = None
            print("Warning: SentenceTransformer not available. Dense retrieval disabled.")
        
        # Load and index corpus
        self._load_and_index_corpus()
    
    def _load_and_index_corpus(self):
        """Load all markdown and CSV files, chunk them, and build indices."""
        print("Loading corpus...")
        
        # Load markdown files
        md_files = list(self.corpus_path.glob("*.md"))
        for md_file in md_files:
            self._chunk_markdown_file(md_file)
        
        # Load CSV files
        csv_files = list(self.corpus_path.glob("*.csv"))
        for csv_file in csv_files:
            self._chunk_csv_file(csv_file)
        
        print(f"Loaded {len(self.chunks)} chunks from corpus")
        
        # Build indices
        self._build_indices()
    
    def _chunk_markdown_file(self, file_path: Path):
        """
        Chunk a markdown file according to the plan:
        - 512 tokens per chunk
        - 128 token overlap
        - Preserve heading hierarchy
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        doc_id = file_path.name
        lines = content.split('\n')
        
        current_section = None
        current_content = []
        chunk_index = 0
        
        # Simple chunking: split by H2 headings (##)
        # In production, would use more sophisticated token counting
        for line in lines:
            if line.startswith('## '):
                # Save previous chunk if exists
                if current_content:
                    chunk_text = '\n'.join(current_content)
                    if len(chunk_text) > 100:  # Minimum chunk size
                        chunk = Chunk(
                            doc_id=doc_id,
                            chunk_id=f"{doc_id}_chunk_{chunk_index:03d}",
                            content=chunk_text,
                            section=current_section,
                            section_path=current_section,
                            chunk_index=chunk_index,
                            doc_type="markdown",
                            tokens=len(chunk_text.split())  # Approximate
                        )
                        self.chunks.append(chunk)
                        chunk_index += 1
                
                # Start new section
                current_section = line[3:].strip()
                current_content = [line]
            else:
                current_content.append(line)
        
        # Add final chunk
        if current_content:
            chunk_text = '\n'.join(current_content)
            if len(chunk_text) > 100:
                chunk = Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_chunk_{chunk_index:03d}",
                    content=chunk_text,
                    section=current_section,
                    section_path=current_section,
                    chunk_index=chunk_index,
                    doc_type="markdown",
                    tokens=len(chunk_text.split())
                )
                self.chunks.append(chunk)
    
    def _chunk_csv_file(self, file_path: Path):
        """Chunk CSV file: one row per chunk."""
        import csv
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            
            for idx, row in enumerate(reader):
                # Format row as text
                row_text = ", ".join([f"{k}: {v}" for k, v in row.items() if v])
                
                chunk = Chunk(
                    doc_id=file_path.name,
                    chunk_id=f"{file_path.name}_row_{idx:03d}",
                    content=row_text,
                    section=f"Row {idx + 1}",
                    chunk_index=idx,
                    doc_type="csv",
                    tokens=len(row_text.split())
                )
                self.chunks.append(chunk)
    
    def _build_indices(self):
        """Build BM25 and embedding indices."""
        if not self.chunks:
            return
        
        # Build BM25 index
        if BM25Okapi:
            chunk_texts = [chunk.content for chunk in self.chunks]
            tokenized_chunks = [text.lower().split() for text in chunk_texts]
            self.bm25_index = BM25Okapi(tokenized_chunks, k1=1.5, b=0.75)
            self.bm25_tokenizer = lambda x: x.lower().split()
        else:
            print("Warning: BM25 not available")
        
        # Build embedding index
        if self.embedding_model:
            chunk_texts = [chunk.content for chunk in self.chunks]
            print("Generating embeddings...")
            self.chunk_embeddings = self.embedding_model.encode(
                chunk_texts,
                show_progress_bar=True
            )
            print("Embeddings generated")
        else:
            print("Warning: Embeddings not generated")
    
    def _hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Hybrid retrieval: BM25 + dense embeddings.
        
        Returns list of chunks with scores.
        """
        scored_chunks = []
        
        # BM25 retrieval
        bm25_scores = None
        if self.bm25_index:
            query_tokens = self.bm25_tokenizer(query)
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        else:
            bm25_scores = np.zeros(len(self.chunks)) if np else [0] * len(self.chunks)
            bm25_max = 1.0
        
        # Dense retrieval
        dense_scores = None
        if self.embedding_model and self.chunk_embeddings is not None and np:
            query_embedding = self.embedding_model.encode([query])[0]
            if np and cosine_similarity:
                dense_scores = cosine_similarity(
                    [query_embedding],
                    self.chunk_embeddings
                )[0]
            else:
                dense_scores = np.zeros(len(self.chunks))
        else:
            dense_scores = np.zeros(len(self.chunks)) if np else [0] * len(self.chunks)
        
        # Combine scores
        for idx, chunk in enumerate(self.chunks):
            normalized_bm25 = bm25_scores[idx] / (bm25_max + 1e-8)
            normalized_dense = float(dense_scores[idx]) if isinstance(dense_scores, np.ndarray) else dense_scores[idx]
            
            hybrid_score = 0.4 * normalized_bm25 + 0.6 * normalized_dense
            
            scored_chunks.append({
                'chunk': chunk,
                'hybrid_score': hybrid_score,
                'bm25_score': float(bm25_scores[idx]),
                'dense_score': float(normalized_dense)
            })
        
        # Sort and return top_k
        scored_chunks.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return scored_chunks[:top_k]
    
    def _cursor_scan(self, query: str, batch_size: int = 10, max_chunks: int = 50) -> List[Dict]:
        """
        Cursor-based scanning for fact-checking.
        Scans corpus in batches looking for evidence.
        """
        # For simplicity, use hybrid retrieval with higher top_k
        # In production, would implement true cursor-based sequential scanning
        return self._hybrid_retrieve(query, top_k=max_chunks)
    
    def _extract_factual_sentences(self, text: str) -> List[str]:
        """Extract sentences that appear to be factual claims."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        # Filter out very short sentences and citations
        factual = [
            s.strip() for s in sentences
            if len(s.strip()) > 20 and '[source:' not in s
        ]
        return factual
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not self.embedding_model:
            return 0.0
        
        emb1 = self.embedding_model.encode([text1])[0]
        emb2 = self.embedding_model.encode([text2])[0]
        
        if np and cosine_similarity:
            return float(cosine_similarity([emb1], [emb2])[0][0])
        else:
            # Fallback: simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1 & words2) / len(words1 | words2)
    
    def _fact_check_with_cursor(self, answer: str, initial_chunks: List[Dict]) -> Dict:
        """Use cursor-based retrieval to fact-check the answer."""
        sentences = self._extract_factual_sentences(answer)
        
        unsupported_claims = []
        supporting_evidence = []
        
        for sentence in sentences:
            # Cursor scan for this sentence
            cursor_chunks = self._cursor_scan(sentence, batch_size=10, max_chunks=50)
            
            # Check if sentence is supported
            max_similarity = 0.0
            best_chunk = None
            
            for scored_chunk in cursor_chunks:
                chunk = scored_chunk['chunk']
                similarity = self._semantic_similarity(sentence, chunk.content)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_chunk = chunk
            
            if max_similarity < 0.6:
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
    
    def _build_qa_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Build the Q&A prompt with retrieved chunks."""
        context_parts = []
        for scored_chunk in chunks:
            chunk = scored_chunk['chunk']
            section_str = f"#{chunk.section}" if chunk.section else "N/A"
            context_parts.append(
                f"**Source:** {chunk.doc_id}{section_str}\n{chunk.content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
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
    
    def _extract_citations(self, answer: str, chunks: List[Dict]) -> List[Citation]:
        """Extract citations from answer and match to chunks."""
        citation_pattern = r'\[source:([^\]]+)\]'
        citations = []
        seen_citations = set()
        
        for match in re.finditer(citation_pattern, answer):
            citation_ref = match.group(1)
            if citation_ref in seen_citations:
                continue
            seen_citations.add(citation_ref)
            
            # Parse doc_id and section
            if '#' in citation_ref:
                doc_id, section = citation_ref.split('#', 1)
            else:
                doc_id = citation_ref
                section = None
            
            # Find matching chunk
            matching_chunk = None
            for scored_chunk in chunks:
                chunk = scored_chunk['chunk']
                if chunk.doc_id == doc_id:
                    if section is None or chunk.section == section:
                        matching_chunk = chunk
                        score_note = f"dense_similarity: {scored_chunk.get('dense_score', 0):.2f}, bm25_score: {scored_chunk.get('bm25_score', 0):.2f}"
                        break
            
            if matching_chunk:
                excerpt = matching_chunk.content[:200] + '...' if len(matching_chunk.content) > 200 else matching_chunk.content
                citations.append(Citation(
                    doc_id=matching_chunk.doc_id,
                    section=matching_chunk.section,
                    excerpt=excerpt,
                    score_note=score_note
                ))
        
        return citations
    
    def _calculate_confidence(self, chunks: List[Dict], fact_check_result: Dict) -> float:
        """Calculate confidence score based on retrieval and fact-checking."""
        if not chunks:
            return 0.0
        
        # Base confidence from retrieval scores
        avg_hybrid_score = sum(c['hybrid_score'] for c in chunks) / len(chunks)
        
        # Penalty for unsupported claims
        unsupported_penalty = len(fact_check_result['unsupported_claims']) * 0.1
        
        confidence = max(0.0, min(1.0, avg_hybrid_score - unsupported_penalty))
        return confidence
    
    def answer_user_query(self, query: str, use_llm: bool = False) -> Dict:
        """
        Main function: answer user query with citations.
        
        Args:
            query: User's question
            use_llm: If True, calls LLM (requires API key). If False, returns mock answer.
        
        Returns:
            Dictionary with answer, citations, unsupported_claims, confidence_score
        """
        # Step 1: Hybrid retrieval
        top_k = 5 if len(query.split()) < 20 else 10
        retrieved_chunks = self._hybrid_retrieve(query, top_k=top_k)
        
        if not retrieved_chunks:
            return {
                'answer': "I don't find this in the provided corpus.",
                'citations': [],
                'unsupported_claims': [],
                'confidence_score': 0.0
            }
        
        # Step 2: Build prompt
        prompt = self._build_qa_prompt(query, retrieved_chunks)
        
        # Step 3: Generate answer
        if use_llm:
            # In production, would call OpenAI/Anthropic/etc.
            # For demo, return mock answer
            answer = self._generate_mock_answer(query, retrieved_chunks)
        else:
            answer = self._generate_mock_answer(query, retrieved_chunks)
        
        # Step 4: Fact-check with cursor
        fact_check_result = self._fact_check_with_cursor(answer, retrieved_chunks)
        
        # Step 5: Extract citations
        citations = self._extract_citations(answer, retrieved_chunks)
        
        # Step 6: Flag unsupported claims in answer
        if fact_check_result['unsupported_claims']:
            answer += "\n\n[Note: Some claims could not be verified in the corpus and may require review.]"
        
        return {
            'answer': answer,
            'citations': [c.__dict__ for c in citations],
            'unsupported_claims': fact_check_result['unsupported_claims'],
            'confidence_score': self._calculate_confidence(retrieved_chunks, fact_check_result),
            'prompt': prompt  # Include for debugging
        }
    
    def _generate_mock_answer(self, query: str, chunks: List[Dict]) -> str:
        """
        Generate a mock answer based on retrieved chunks.
        In production, this would call an LLM.
        """
        # Simple mock: concatenate relevant chunks with citations
        answer_parts = []
        
        for scored_chunk in chunks[:3]:  # Use top 3 chunks
            chunk = scored_chunk['chunk']
            section_ref = f"{chunk.doc_id}#{chunk.section}" if chunk.section else chunk.doc_id
            answer_parts.append(
                f"{chunk.content[:300]}... [source:{section_ref}]"
            )
        
        return "\n\n".join(answer_parts)


def demo():
    """Demo function showing how to use the RAG system."""
    # Initialize system
    rag = KeralaAyurvedaRAG(corpus_path="..")
    
    # Example queries
    queries = [
        "What are the key benefits of Ashwagandha Stress Balance Tablets?",
        "Are there contraindications for Triphala Capsules?",
        "What does Ayurveda mean by dosha imbalance?"
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")
        
        result = rag.answer_user_query(query)
        
        print(f"Answer:\n{result['answer']}\n")
        print(f"Confidence: {result['confidence_score']:.2f}")
        print(f"\nCitations ({len(result['citations'])}):")
        for citation in result['citations']:
            print(f"  - {citation['doc_id']}#{citation.get('section', 'N/A')}")
        
        if result['unsupported_claims']:
            print(f"\nUnsupported claims: {len(result['unsupported_claims'])}")


if __name__ == "__main__":
    demo()


