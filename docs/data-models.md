# NeuroFlow Data Models

## Overview

This document defines the logical data model for the NeuroFlow platform. It describes the primary entities, their attributes, relationships, and storage mechanisms used throughout the system.

---

# Entity Relationship Diagram

```text
Document
    │
    │ 1
    │
    └───────────────┐
                    │
                    ▼
                Chunk
                    │
                    │
                    ▼
              Vector Embedding

User Query
      │
      ▼
Retrieved Chunks
      │
      ▼
Generated Response
      │
      ▼
Evaluation
      │
      ▼
User Feedback
```

---

# 1. Document

Represents a file uploaded by the user.

| Field | Type | Description |
|--------|------|-------------|
| document_id | UUID | Unique document identifier |
| filename | String | Original file name |
| file_type | String | PDF, DOCX, Image, CSV, URL |
| source | String | Upload source |
| upload_time | Timestamp | Upload date and time |
| status | String | Uploaded, Processing, Indexed |

---

# 2. Chunk

Represents a small section of a document.

| Field | Type | Description |
|--------|------|-------------|
| chunk_id | UUID | Unique chunk identifier |
| document_id | UUID | Parent document |
| chunk_text | Text | Extracted text |
| chunk_index | Integer | Position within the document |
| token_count | Integer | Number of tokens |

---

# 3. Vector Embedding

Represents the numerical embedding generated for a chunk.

| Field | Type | Description |
|--------|------|-------------|
| embedding_id | UUID | Unique embedding identifier |
| chunk_id | UUID | Related chunk |
| embedding_vector | Vector | High-dimensional embedding |
| embedding_model | String | Model used to generate embeddings |

---

# 4. User Query

Represents a user's question submitted to the system.

| Field | Type | Description |
|--------|------|-------------|
| query_id | UUID | Unique query identifier |
| user_query | Text | User's question |
| query_time | Timestamp | Time of submission |

---

# 5. Generated Response

Represents the answer produced by the language model.

| Field | Type | Description |
|--------|------|-------------|
| response_id | UUID | Unique response identifier |
| query_id | UUID | Related query |
| model_name | String | LLM used |
| generated_text | Text | Generated response |
| generation_time | Timestamp | Time of generation |

---

# 6. Evaluation

Stores quality metrics for each generated response.

| Field | Type | Description |
|--------|------|-------------|
| evaluation_id | UUID | Unique evaluation identifier |
| query_id | UUID | Related query |
| faithfulness | Float | Faithfulness score |
| answer_relevance | Float | Answer relevance score |
| context_precision | Float | Context precision score |
| context_recall | Float | Context recall score |
| evaluated_at | Timestamp | Evaluation time |

---

# 7. User Feedback

Stores feedback submitted by users.

| Field | Type | Description |
|--------|------|-------------|
| feedback_id | UUID | Unique feedback identifier |
| query_id | UUID | Related query |
| rating | Integer | Rating (1–5) |
| comment | Text | Optional feedback |
| submitted_at | Timestamp | Submission time |

---

# Relationships

| Parent Entity | Child Entity | Relationship |
|---------------|--------------|--------------|
| Document | Chunk | One-to-Many |
| Chunk | Vector Embedding | One-to-One |
| User Query | Generated Response | One-to-One |
| User Query | Evaluation | One-to-One |
| User Query | User Feedback | One-to-Many |

---

# Storage Strategy

| Entity | Storage |
|--------|---------|
| Document Metadata | PostgreSQL |
| Chunks | PostgreSQL |
| Vector Embeddings | pgvector |
| Queries | PostgreSQL |
| Generated Responses | PostgreSQL |
| Evaluation Scores | PostgreSQL |
| User Feedback | PostgreSQL |

---

# Data Lifecycle

```text
Document Upload
       │
       ▼
Document Metadata
       │
       ▼
Chunk Creation
       │
       ▼
Embedding Generation
       │
       ▼
Vector Database
       │
       ▼
User Query
       │
       ▼
Generated Response
       │
       ▼
Evaluation
       │
       ▼
User Feedback
```