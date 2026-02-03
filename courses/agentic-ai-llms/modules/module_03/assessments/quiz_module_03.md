# Quiz: Module 3 - Memory & Context Management

**Estimated Time:** 20 minutes
**Total Points:** 100
**Passing Score:** 70%

---

## Instructions

This quiz assesses your understanding of conversation memory, RAG fundamentals, and vector stores. Focus on practical application and architectural decisions.

---

## Part A: Conversation Memory (30 points)

### Question 1 (8 points)
Explain why LLMs have no inherent memory and what this means for multi-turn conversations.

**Your Answer:**

---

### Question 2 (10 points)
Compare and contrast buffer memory and summary memory. For each, describe:
- How it works
- When to use it
- One key limitation

**Your Answer:**

---

### Question 3 (7 points)
You're building a customer support agent with conversations averaging 20 messages but occasionally reaching 200+ messages. Which memory strategy would you use and why?

A) Buffer memory - keep all messages
B) Summary memory - summarize periodically
C) Sliding window - keep last N messages
D) Hybrid - buffer recent messages + summary of older ones

**Your Answer:**

---

### Question 4 (5 points)
**True or False:** When implementing conversation memory, you should always keep the system prompt as the first message to maintain consistent behavior.

**Your Answer:**

---

## Part B: RAG Fundamentals (40 points)

### Question 5 (10 points)
Describe the complete RAG (Retrieval-Augmented Generation) pipeline from user query to final response. Include all major steps.

**Your Answer:**

---

### Question 6 (8 points)
What problem does RAG solve that cannot be solved by simply putting all your documents into the LLM's context window?

**Your Answer:**

---

### Question 7 (9 points)
Match each chunking strategy with its best use case:

**Strategies:**
1. Fixed size (500 tokens per chunk)
2. Semantic splitting (split at natural boundaries)
3. Recursive splitting (hierarchical)
4. Document-aware (respect structure)

**Use Cases:**
A) Processing API documentation with clear sections and headers
B) Processing general web content with varying structure
C) Processing legal documents where preserving exact paragraph structure matters
D) Quick implementation for simple use case

**Your Answer:**

---

### Question 8 (8 points)
You implement a RAG system but find that retrieved chunks often lack necessary context (e.g., missing the heading that explains what the paragraph is about). What is this problem called, and name two strategies to address it?

**Your Answer:**

---

### Question 9 (5 points)
**True or False:** In a RAG system, you should always retrieve as many documents as possible to give the LLM maximum context.

**Your Answer:**

---

## Part C: Vector Stores (30 points)

### Question 10 (8 points)
Explain in simple terms what an embedding is and why vectors are useful for semantic search.

**Your Answer:**

---

### Question 11 (10 points)
Given these three queries and three document chunks, explain which document would likely have the highest cosine similarity with Query 1 and why:

**Query 1:** "How do I reset my password?"

**Documents:**
A) "Password authentication uses bcrypt hashing with salt rounds configured in the security settings file."
B) "To reset your password, click 'Forgot Password' on the login page and follow the email instructions."
C) "User accounts are secured with two-factor authentication and regular password rotation policies."

**Your Answer:**

---

### Question 12 (7 points)
What is the purpose of a "reranker" in advanced retrieval systems?

A) To reorder chunks alphabetically
B) To score retrieved chunks with a more sophisticated model and reorder by relevance
C) To remove duplicate chunks
D) To compress chunks to fit context limits

**Your Answer:**

---

### Question 13 (5 points)
**True or False:** The same embedding model must be used for both indexing documents and embedding queries for vector search to work correctly.

**Your Answer:**

---

## Part D: Advanced Memory Patterns (Bonus - 10 points)

### Question 14 (5 points)
What is entity memory, and when is it more useful than conversation buffer memory?

**Your Answer:**

---

### Question 15 (5 points)
You're building an agent that needs to remember user preferences across multiple sessions (e.g., preferred language, notification settings). Which combination of memory types would you use?

**Your Answer:**

---

# Answer Key

## Part A: Conversation Memory

**Question 1:** (8 points)
*Sample Answer:* LLMs are stateless—each API call is independent with no inherent memory of previous interactions. For multi-turn conversations, you must explicitly pass conversation history with each request. The model sees the entire conversation history as fresh input each time, treating it like a new prompt. This means memory management is your responsibility, not the model's.

**Question 2:** (10 points)
*Sample Answer:*

**Buffer Memory:**
- How: Keeps all messages in full verbatim
- When: Short conversations (under ~50 messages or context limit)
- Limitation: Grows unbounded, eventually exceeds context window

**Summary Memory:**
- How: Periodically summarizes old messages, keeps recent ones full
- When: Long conversations where full history isn't needed
- Limitation: Loses fine-grained details, summarization can miss important context

*Grading:* 5 points per memory type, all three aspects needed

**Question 3:** Answer: D (7 points)
*Explanation:* Hybrid approach works best - buffer keeps recent messages (last 10-20) for immediate context, while summaries preserve key information from earlier parts of long conversations. This balances detail retention with context window management.

**Question 4:** True (5 points)
*Explanation:* System prompt should always be first (index 0) to maintain consistent agent behavior across the conversation. When trimming old messages, preserve the system prompt.

## Part B: RAG Fundamentals

**Question 5:** (10 points)
*Sample Answer:*
1. User submits query
2. Query is embedded into vector representation
3. Vector search finds most similar document chunks in vector store
4. Retrieved chunks are retrieved (typically top 3-10)
5. Chunks are added to LLM prompt as context
6. LLM generates response using retrieved context
7. Response includes citations to source documents
8. Return response to user

*Grading:* 1.25 points per step, minimum 6 steps for full credit

**Question 6:** (8 points)
*Sample Answer:* RAG solves several problems: (1) Scale - you can search millions of documents, far exceeding context window limits, (2) Cost - only relevant chunks consume tokens, not entire document corpus, (3) Freshness - update documents without retraining, (4) Precision - semantic search finds specifically relevant information rather than hoping it's noticed in a massive context dump.

**Question 7:** (9 points)
*Answer:*
1. Fixed size → D (Quick implementation)
2. Semantic splitting → B (Varying structure)
3. Recursive splitting → C (Legal documents)
4. Document-aware → A (API documentation)

*Grading:* 2.25 points per correct match

**Question 8:** (8 points)
*Answer:* This is the **"lost context" or "chunk boundary"** problem.

*Strategies:*
1. **Overlapping chunks** - Include overlap between chunks (e.g., 50-100 tokens) to preserve context
2. **Metadata enrichment** - Store section headers/titles with each chunk
3. **Larger chunk sizes** - Increase chunk size to include more surrounding context
4. **Parent-child chunking** - Store small chunks for search but return larger parent chunk with context

*Grading:* 3 points for naming problem, 2.5 points per strategy

**Question 9:** False (5 points)
*Explanation:* More isn't always better. Too many chunks dilute relevance, increase costs, and cause "lost in the middle" problems. Typically 3-10 highly relevant chunks outperform 50 marginally relevant ones.

## Part C: Vector Stores

**Question 10:** (8 points)
*Sample Answer:* An embedding is a numerical representation (vector) of text where semantically similar content has similar numbers. Vectors are useful for semantic search because we can calculate mathematical similarity (like cosine similarity) between the query vector and document vectors to find conceptually related content, not just keyword matches. For example, "reset password" and "forgot login credentials" would have similar vectors even though they share no words.

**Question 11:** (10 points)
*Answer:* Document B would have the highest similarity.

*Explanation:* Document B directly addresses the user's intent (resetting password) with actionable instructions. While Document A mentions passwords, it focuses on the technical implementation (hashing), not the user action. Document C discusses related security topics but not the specific task. Semantic embeddings capture meaning and intent, so B's direct relevance to the query would produce the highest cosine similarity.

*Grading:* 4 points for correct answer, 6 points for sound explanation

**Question 12:** Answer: B (7 points)
*Explanation:* A reranker takes the initial retrieved results and rescores them with a more sophisticated cross-encoder model that can assess query-document relevance more accurately than vector similarity alone. This improves precision by reordering results by true relevance.

**Question 13:** True (5 points)
*Explanation:* Query and document embeddings must come from the same model and exist in the same vector space for similarity calculations to be meaningful. Using different models would produce incomparable vectors.

## Part D: Advanced Memory Patterns

**Question 14:** (5 points)
*Sample Answer:* Entity memory tracks specific entities (people, products, companies) and their attributes across conversations. It's more useful than buffer memory when you need to maintain structured facts about specific entities that might be discussed across many turns or sessions. For example, tracking customer account details, project names, or relationship information.

**Question 15:** (5 points)
*Sample Answer:* Combination of:
1. **Vector store/database** - For long-term persistent storage of user preferences
2. **Entity memory** - To track user-specific attributes in structured form
3. **Short-term buffer** - For current session context

When user starts session, load preferences from persistent store, use entity memory during conversation, save updates back to persistent store at session end.

*Acceptable alternatives mentioning persistent storage + session memory*

---

## Scoring Guide

- **90-100 points:** Excellent - Strong grasp of memory and retrieval systems
- **80-89 points:** Good - Solid understanding with minor gaps
- **70-79 points:** Passing - Core concepts understood, practice implementation
- **Below 70:** Review module materials and build a RAG system

**Key Topics to Review if Struggling:**
- Conversation memory strategies (buffer vs summary)
- RAG pipeline architecture
- Chunking strategies and chunk boundary problems
- Embedding and vector similarity concepts
- When to use different memory types
- Reranking and hybrid search
