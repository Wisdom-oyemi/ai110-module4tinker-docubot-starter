"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import string

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Create chunks from documents (markdown section-based)
        self.chunks = self.create_chunks(self.documents)  # List of (filename, chunk_text)

        # Build a retrieval index over chunks (implemented in Phase 1)
        self.index = self.build_index(self.chunks)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Document Chunking
    # -----------------------------------------------------------

    def create_chunks(self, documents):
        """
        Split documents into chunks at markdown ## headers.
        Returns a list of tuples: (filename, chunk_text)
        """
        chunks = []
        for filename, text in documents:
            sections = text.split("## ")
            for i, section in enumerate(sections):
                if i == 0 and section.strip():
                    # First section (before any ##) is treated as a chunk
                    chunks.append((filename, section))
                elif i > 0:
                    # Re-add ## prefix to maintain section header
                    chunk = "## " + section
                    chunks.append((filename, chunk))
        return chunks

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Build a tiny inverted index mapping lowercase words to the document chunks
        they appear in.

        Example structure:
        {
            "token": [0, 1, 3],
            "database": [2]
        }

        Keep this simple: split on whitespace, lowercase tokens,
        ignore punctuation if needed.
        """
        index = {}
        for chunk_id, (filename, text) in enumerate(documents):
            chunk_tokens = set()
            for raw in text.lower().split():
                token = raw.strip(string.punctuation)
                if token:
                    chunk_tokens.add(token)
            for token in chunk_tokens:
                if token not in index:
                    index[token] = []
                index[token].append(chunk_id)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        TODO (Phase 1):
        Return a simple relevance score for how well the text matches the query.

        Suggested baseline:
        - Convert query into lowercase words
        - Count how many appear in the text
        - Return the count as the score
        """
        query_tokens = []
        for raw in query.lower().split():
            token = raw.strip(string.punctuation)
            if token:
                query_tokens.append(token)
        if not query_tokens:
            return 0

        token_counts = {}
        for raw in text.lower().split():
            token = raw.strip(string.punctuation)
            if token:
                token_counts[token] = token_counts.get(token, 0) + 1

        score = 0
        for token in query_tokens:
            score += token_counts.get(token, 0)
        return score

    def retrieve(self, query, top_k=3):
        """
        Use the index and scoring function to select top_k relevant document chunks.
        Apply score-per-token guardrail: only return chunks with average score >= 1.0 per query token.

        Return a list of (filename, text) sorted by score descending.
        """
        if top_k <= 0:
            return []

        query_tokens = []
        for raw in query.lower().split():
            token = raw.strip(string.punctuation)
            if token:
                query_tokens.append(token)
        if not query_tokens:
            return []

        candidates = set()
        for token in query_tokens:
            for chunk_id in self.index.get(token, []):
                candidates.add(chunk_id)

        if not candidates:
            return []

        scored = []
        for chunk_id in candidates:
            filename, text = self.chunks[chunk_id]
            score = self.score_document(query, text)
            
            # Apply guardrail: score per token must be >= 1.0
            score_per_token = score / len(query_tokens)
            if score_per_token >= 1.0:
                scored.append((score, chunk_id, filename, text))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(filename, text) for _, _, filename, text in scored[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
