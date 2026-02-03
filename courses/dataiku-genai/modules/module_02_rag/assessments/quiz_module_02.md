# Module 2 Quiz: RAG with Knowledge Banks

**Course:** Gen AI & Dataiku: LLM Mesh Use Cases
**Module:** 2 - RAG with Knowledge Banks
**Time Limit:** 20 minutes
**Total Points:** 100
**Passing Score:** 70/100

## Instructions

- This quiz has 14 questions covering Knowledge Banks, document ingestion, RAG architecture, and retrieval strategies
- Select the best answer for each question
- Point values are indicated for each question
- You have 2 attempts per question
- Refer to module guides and notebooks if needed

---

## Section 1: Knowledge Banks Fundamentals (28 points)

### Question 1 (7 points)

What is the PRIMARY purpose of a Knowledge Bank in Dataiku?

A) To store structured database records for SQL queries
B) To create a searchable vector store from unstructured documents
C) To backup important project files automatically
D) To version control datasets and models

**Answer:** B

**Explanation:**
- **B (Correct):** Knowledge Banks convert documents into embeddings and store them in vector databases for semantic search and RAG applications.
- **A:** Knowledge Banks are for unstructured data; structured data uses regular datasets.
- **C:** Knowledge Banks are not backup systems; they're for semantic retrieval.
- **D:** Version control is handled separately; Knowledge Banks focus on document retrieval.

---

### Question 2 (7 points)

In a Knowledge Bank, what is "chunking"?

A) Compressing documents to reduce storage costs
B) Dividing documents into smaller segments for processing
C) Removing duplicates from the document collection
D) Organizing documents by file type

**Answer:** B

**Explanation:**
- **B (Correct):** Chunking splits documents into smaller pieces that fit within embedding model context windows and enable precise retrieval.
- **A:** Chunking is about processing strategy, not compression.
- **C:** Deduplication is a separate process.
- **D:** Organization by file type is metadata management, not chunking.

---

### Question 3 (7 points)

Which file formats can be ingested into a Dataiku Knowledge Bank?

A) Only plain text (.txt) files
B) Only PDF and Word documents
C) PDF, Word, HTML, Markdown, plain text, and other common formats
D) Only formats that have been manually converted to JSON

**Answer:** C

**Explanation:**
- **C (Correct):** Knowledge Banks support automatic parsing of many document formats including PDF, DOCX, HTML, MD, TXT, and more.
- **A:** Many formats beyond plain text are supported.
- **B:** HTML, Markdown, and other formats are also supported.
- **D:** Automatic parsing handles format conversion; manual JSON conversion isn't required.

---

### Question 4 (7 points)

What happens to documents after they are ingested into a Knowledge Bank?

A) They are stored as-is in their original format
B) They are chunked, embedded, and indexed in a vector store
C) They are converted to structured SQL tables
D) They are compressed and archived

**Answer:** B

**Explanation:**
- **B (Correct):** The ingestion pipeline chunks documents, generates embeddings using a specified model, and indexes them for retrieval.
- **A:** Documents are processed and transformed, not stored as-is.
- **C:** Knowledge Banks use vector stores, not SQL tables.
- **D:** The focus is on searchability, not archival compression.

---

## Section 2: RAG Architecture (35 points)

### Question 5 (9 points)

In a RAG (Retrieval-Augmented Generation) application, what is the retrieval step?

A) Downloading the LLM model weights from the provider
B) Searching the Knowledge Bank for relevant document chunks
C) Extracting structured data from the LLM response
D) Retrieving cached responses to save API calls

**Answer:** B

**Explanation:**
- **B (Correct):** The retrieval step queries the Knowledge Bank to find document chunks most relevant to the user's question.
- **A:** Model weights are managed by the provider; retrieval refers to document search.
- **C:** Extraction happens after generation, not during retrieval.
- **D:** Caching is a separate optimization; retrieval specifically means searching the knowledge base.

---

### Question 6 (9 points)

What is the typical flow in a RAG application?

A) Generate → Retrieve → Respond
B) Retrieve → Generate → Validate
C) Query → Retrieve → Augment Prompt → Generate
D) Embed → Search → Cache → Generate

**Answer:** C

**Explanation:**
- **C (Correct):** RAG flow: user query → retrieve relevant chunks → augment prompt with retrieved context → generate response.
- **A:** Retrieval must happen before generation to provide context.
- **B:** Generation happens after retrieval, but validation isn't a core RAG step.
- **D:** This mixes implementation details; the conceptual flow is Query → Retrieve → Augment → Generate.

---

### Question 7 (8 points)

Why is RAG useful compared to relying solely on an LLM's pre-trained knowledge?

A) RAG is always faster than using LLM knowledge
B) RAG provides access to current, proprietary, or specialized information
C) RAG eliminates the need for prompt engineering
D) RAG models are cheaper to run than standard LLMs

**Answer:** B

**Explanation:**
- **B (Correct):** RAG enables LLMs to access up-to-date, domain-specific, or private information not in their training data.
- **A:** RAG adds retrieval overhead; speed isn't the primary benefit.
- **C:** Prompt engineering is still important in RAG applications.
- **D:** RAG adds costs (embeddings, retrieval); the benefit is information access, not cost savings.

---

### Question 8 (9 points)

In a RAG system, what is "source attribution"?

A) Tracking which LLM model generated each response
B) Identifying which documents/chunks were used to generate the answer
C) Recording which user made each query
D) Monitoring the cost of each API call

**Answer:** B

**Explanation:**
- **B (Correct):** Source attribution links generated content back to specific source documents, enabling verification and transparency.
- **A:** Model tracking is different from document source attribution.
- **C:** User tracking is about access logs, not source attribution.
- **D:** Cost monitoring is a governance feature, not source attribution.

---

## Section 3: Configuration and Optimization (37 points)

### Question 9 (9 points)

What is the trade-off when choosing chunk size for a Knowledge Bank?

A) Larger chunks provide more context but may dilute relevance; smaller chunks are more precise but may lack context
B) Larger chunks are always better because they contain more information
C) Smaller chunks are always better because they reduce cost
D) Chunk size doesn't affect retrieval quality, only storage size

**Answer:** A

**Explanation:**
- **A (Correct):** Chunk size involves balancing context (larger) vs. precision (smaller). Optimal size depends on document type and use case.
- **B:** Very large chunks can include irrelevant information, reducing retrieval precision.
- **C:** Very small chunks may lack sufficient context for understanding.
- **D:** Chunk size significantly affects both retrieval quality and storage.

---

### Question 10 (9 points)

What does the "overlap" setting in chunking control?

A) How many duplicate documents are allowed in the Knowledge Bank
B) How many tokens are shared between consecutive chunks
C) How many chunks can be retrieved simultaneously
D) How many embedding models can process chunks concurrently

**Answer:** B

**Explanation:**
- **B (Correct):** Overlap specifies how many tokens from the end of one chunk are repeated at the start of the next, preventing information loss at boundaries.
- **A:** Overlap is about chunk boundaries, not document deduplication.
- **C:** This would be a "top-k" or retrieval parameter, not overlap.
- **D:** Overlap doesn't relate to parallel processing.

---

### Question 11 (8 points)

Your RAG application is returning irrelevant results. Which parameter should you adjust FIRST?

A) Increase max_tokens in the generation step
B) Adjust the top-k retrieval parameter (number of chunks retrieved)
C) Change the embedding model to a more expensive one
D) Increase the chunk overlap percentage

**Answer:** B

**Explanation:**
- **B (Correct):** Top-k directly controls retrieval relevance. Start by experimenting with retrieving more or fewer chunks.
- **A:** Max_tokens affects generation length, not retrieval relevance.
- **C:** Embedding model changes are significant; try simpler adjustments first.
- **D:** Overlap affects chunk boundaries, not immediate relevance.

---

### Question 12 (7 points)

What is the purpose of the embedding model in a Knowledge Bank?

A) To compress documents for efficient storage
B) To convert text into numerical vectors for semantic search
C) To generate summaries of ingested documents
D) To translate documents into multiple languages

**Answer:** B

**Explanation:**
- **B (Correct):** Embedding models convert text into high-dimensional vectors that capture semantic meaning, enabling similarity search.
- **A:** Embeddings enable search, not compression.
- **C:** Summarization is a separate LLM task.
- **D:** Translation is a separate task; embeddings capture semantic meaning.

---

### Question 13 (Bonus 10 points)

When should you update/rebuild a Knowledge Bank?

A) Every time any user makes a query
B) On a fixed schedule regardless of content changes
C) When source documents are added, modified, or removed
D) Only when the LLM Mesh connection changes

**Answer:** C

**Explanation:**
- **C (Correct):** Knowledge Banks should be refreshed when underlying documents change to keep retrieved information current.
- **A:** Queries don't require rebuilding; the index is used as-is.
- **B:** Schedule-based updates may be appropriate, but content changes are the trigger.
- **D:** LLM connections don't affect the Knowledge Bank; they're separate systems.

---

## Section 4: Best Practices (Bonus)

### Question 14 (Bonus 8 points)

You're building a RAG application for customer support. Which strategy is MOST effective?

A) Ingest all company documents into a single massive Knowledge Bank
B) Create focused Knowledge Banks for different document types or topics
C) Use the smallest possible chunk size to maximize precision
D) Disable source attribution to improve response speed

**Answer:** B

**Explanation:**
- **B (Correct):** Focused Knowledge Banks improve retrieval precision by reducing noise and allowing optimized configurations per domain.
- **A:** Large, unfocused Knowledge Banks can return less relevant results due to topic mixing.
- **C:** Very small chunks may lack context; size should be optimized for content type.
- **D:** Source attribution is valuable for trust and debugging; the speed impact is minimal.

---

## Answer Key Summary

| Question | Answer | Points | Topic |
|----------|--------|--------|-------|
| 1 | B | 7 | Knowledge Bank purpose |
| 2 | B | 7 | Chunking definition |
| 3 | C | 7 | File format support |
| 4 | B | 7 | Document processing |
| 5 | B | 9 | Retrieval step |
| 6 | C | 9 | RAG flow |
| 7 | B | 8 | RAG benefits |
| 8 | B | 9 | Source attribution |
| 9 | A | 9 | Chunk size trade-offs |
| 10 | B | 9 | Chunk overlap |
| 11 | B | 8 | Retrieval optimization |
| 12 | B | 7 | Embedding models |
| 13 | C | 10 | Knowledge Bank updates |
| 14 | B | 8 | Best practices |

**Total:** 118 points (100 base + 18 bonus)

---

## Performance Indicators

**100-118:** Expert-level understanding of RAG and Knowledge Banks
**85-99:** Strong grasp of RAG concepts and configuration
**70-84:** Adequate understanding, practice more with retrieval optimization
**Below 70:** Review module materials, especially RAG architecture and chunking strategies

## Common Mistakes to Avoid

1. **Wrong chunk size**: Test different sizes; one size doesn't fit all document types
2. **Ignoring overlap**: Overlap prevents losing information at chunk boundaries
3. **Poor top-k tuning**: Retrieving too many or too few chunks affects quality
4. **Stale Knowledge Banks**: Remember to refresh when documents change
5. **Unfocused Knowledge Banks**: Create domain-specific banks for better precision

## Key Concepts to Master

- **RAG Pipeline**: Query → Retrieve → Augment → Generate
- **Chunking**: Splitting documents for effective retrieval
- **Embeddings**: Converting text to vectors for semantic search
- **Top-K**: Number of chunks to retrieve
- **Source Attribution**: Linking answers back to source documents

## Next Steps

After completing this quiz:
1. Review any questions you missed
2. Complete the Knowledge Bank hands-on exercises
3. Experiment with different chunking and retrieval parameters
4. Build a simple RAG application using the module notebooks
5. Proceed to Module 3: Custom LLM Applications when ready
