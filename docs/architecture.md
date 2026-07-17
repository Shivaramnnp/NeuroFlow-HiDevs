# NeuroFlow System Architecture

## Overview

NeuroFlow is a Retrieval-Augmented Generation (RAG) platform that enables users to ingest data from multiple sources, retrieve relevant information using hybrid search techniques, generate context-aware responses using Large Language Models (LLMs), evaluate response quality, and continuously improve through fine-tuning.

The system is organized into five major subsystems:

1. Ingestion Subsystem
2. Retrieval Subsystem
3. Generation Subsystem
4. Evaluation Subsystem
5. Fine-Tuning Subsystem

---

# 1. Ingestion Subsystem

## Purpose

The Ingestion Subsystem is responsible for converting raw data from different sources into searchable vector representations. It accepts multiple input formats, extracts their content, processes the extracted data, generates embeddings, and stores them in the vector database for efficient retrieval.

## Supported Input Sources

- PDF Documents
- Microsoft Word (DOCX)
- Images
- CSV Files
- Web URLs

## Processing Pipeline

The ingestion pipeline performs the following steps:

1. Accept the uploaded file or URL.
2. Extract text based on the input type.
3. Split the extracted text into smaller chunks.
4. Generate embeddings for every chunk using an embedding model.
5. Store the embeddings, original chunk text, and metadata in the vector database.

## Data Flow

```text
User Upload
      │
      ▼
Content Extraction
      │
      ▼
Text Chunking
      │
      ▼
Embedding Generation
      │
      ▼
Vector Store (pgvector)
```

## Component Description

### Content Extraction

Different extractors are used depending on the uploaded data type.

| Input Type | Processing Method |
|------------|-------------------|
| PDF | Text Extraction |
| DOCX | Document Parsing |
| Images | OCR |
| CSV | Structured Data Parsing |
| Web URL | Web Page Content Extraction |

After extraction, all data is converted into plain text.

### Text Chunking

Large documents are divided into smaller chunks before generating embeddings. Chunking improves retrieval accuracy by allowing the system to search only the relevant portions of a document instead of processing the entire document.

### Embedding Generation

Each chunk is converted into a numerical vector (embedding) using an embedding model. These vectors capture the semantic meaning of the text, enabling similarity-based search instead of exact keyword matching.

### Vector Storage

The generated embeddings, original text chunks, and metadata (such as document name, page number, and source) are stored in the vector database. Once stored, the document becomes available for retrieval during user queries.

---

# 2. Retrieval Subsystem

## Purpose

The Retrieval Subsystem is responsible for identifying and ranking the most relevant information for a user's query. It combines semantic search, keyword search, and metadata filtering to retrieve high-quality context before passing it to the Large Language Model (LLM).

## Retrieval Pipeline

The retrieval pipeline performs the following steps:

1. Accept the user's query.
2. Convert the query into an embedding.
3. Execute embedding similarity search, keyword search, and metadata filtering in parallel.
4. Merge the results using Reciprocal Rank Fusion (RRF).
5. Re-rank the retrieved chunks using a cross-encoder reranker.
6. Select the highest-ranked chunks.
7. Return the final context window for response generation.

## Data Flow

```text
User Query
      │
      ▼
Generate Query Embedding
      │
      ▼
 ┌──────────────┬────────────────┬─────────────────┐
 │              │                │                 │
 ▼              ▼                ▼
Vector Search  Keyword Search  Metadata Filter
 │              │                │
 └──────────────┴────────────────┴─────────────────┘
                │
                ▼
     Reciprocal Rank Fusion (RRF)
                │
                ▼
     Cross-Encoder Re-ranker
                │
                ▼
      Ranked Context Window
```

## Component Description

### Query Embedding

The user's query is converted into an embedding using the same embedding model used during document ingestion. This enables semantic similarity comparison with stored document embeddings.

### Embedding Similarity Search

The vector database searches for document chunks whose embeddings are closest to the query embedding, allowing retrieval based on semantic meaning rather than exact keyword matches.

### Keyword Search

Keyword search identifies chunks containing exact words or phrases from the user's query. This improves retrieval for technical terms, names, and exact matches.

### Metadata Filtering

Metadata filtering narrows the search using document attributes such as source, document type, author, upload date, or custom tags.

### Reciprocal Rank Fusion (RRF)

Results from vector search, keyword search, and metadata filtering are combined using Reciprocal Rank Fusion (RRF). This ranking strategy improves retrieval quality by considering the ranking from multiple retrieval methods instead of relying on a single search technique.

### Cross-Encoder Re-ranker

The retrieved chunks are passed through a cross-encoder model that evaluates each chunk together with the user's query. This produces more accurate relevance scores and improves the final ranking.

### Ranked Context Window

The highest-ranked chunks are selected and combined into a context window. This context is forwarded to the Generation Subsystem for response generation.

---

# 3. Generation Subsystem

## Purpose

The Generation Subsystem is responsible for producing responses to user queries. It receives the ranked context from the Retrieval Subsystem, constructs a prompt, selects the appropriate Large Language Model (LLM), generates the response, streams it to the user, and logs the interaction for future evaluation.

## Generation Pipeline

The generation pipeline performs the following steps:

1. Receive the ranked context window from the Retrieval Subsystem.
2. Construct a prompt by combining the user's query with the retrieved context.
3. Route the request to the most appropriate LLM based on cost, capability, or domain.
4. Generate the response.
5. Stream the generated response to the client in real time.
6. Store the complete request and response for evaluation.

## Data Flow

```text
Ranked Context Window
          │
          ▼
     Prompt Builder
          │
          ▼
      Model Router
          │
          ▼
     Selected LLM
          │
          ▼
Response Generation
          │
          ▼
 Response Streaming
          │
          ▼
 Interaction Logging
```

## Component Description

### Prompt Builder

The Prompt Builder combines the user's query with the retrieved context to create a structured prompt. This ensures the LLM generates responses using relevant information from the knowledge base.

### Model Router

The Model Router selects the most suitable LLM based on predefined routing rules such as cost, latency, model capability, or domain-specific requirements.

### Response Generation

The selected LLM processes the prompt and generates a response using the provided context.

### Response Streaming

Instead of waiting for the complete response, the generated text is streamed to the user token by token, providing a faster and more interactive experience.

### Interaction Logging

Every request and generated response is logged, including the user query, retrieved context, selected model, generated answer, and timestamps. These logs are used by the Evaluation Subsystem to measure system performance and identify opportunities for improvement.
---

# 4. Evaluation Subsystem

## Purpose

The Evaluation Subsystem is responsible for measuring the quality of every response generated by the system. It evaluates whether the response is accurate, relevant, and supported by the retrieved context. Evaluation results are stored for monitoring, analysis, and future model improvement.

## Evaluation Pipeline

The evaluation pipeline performs the following steps:

1. Receive the complete interaction log from the Generation Subsystem.
2. Evaluate the generated response using predefined quality metrics.
3. Store individual evaluation scores in PostgreSQL.
4. Calculate rolling performance metrics.
5. Provide evaluation reports for monitoring and fine-tuning.

## Data Flow

```text
Interaction Log
       │
       ▼
Evaluation Engine
       │
       ▼
Quality Metrics
       │
       ▼
Store Scores (PostgreSQL)
       │
       ▼
Rolling Aggregation
       │
       ▼
Evaluation Dashboard
```

## Evaluation Metrics

### Faithfulness

Measures whether the generated response is supported by the retrieved context. Responses containing unsupported or hallucinated information receive lower scores.

### Answer Relevance

Measures how well the generated response answers the user's question.

### Context Precision

Measures whether the retrieved chunks were actually useful for generating the final response. Higher precision indicates that most retrieved chunks contributed meaningful information.

### Context Recall

Measures whether all important information required to answer the user's question was successfully retrieved from the knowledge base.

## Score Storage

Each evaluation stores:

- Query ID
- User Query
- Retrieved Context
- Generated Response
- Faithfulness Score
- Answer Relevance Score
- Context Precision Score
- Context Recall Score
- Evaluation Timestamp

All evaluation records are stored in PostgreSQL for future analysis.

## Rolling Aggregates

The system continuously computes aggregated metrics such as:

- Average Faithfulness
- Average Answer Relevance
- Average Context Precision
- Average Context Recall
- Overall System Quality Score

These metrics help monitor system performance over time and identify areas for improvement.

---

# 5. Fine-Tuning Subsystem

## Purpose

The Fine-Tuning Subsystem is responsible for continuously improving the performance of the NeuroFlow platform by utilizing evaluation results, user feedback, and interaction logs. It identifies areas where the system performs poorly and uses this information to enhance retrieval strategies, prompts, model routing, or fine-tune language models.

## Fine-Tuning Pipeline

The fine-tuning pipeline performs the following steps:

1. Collect interaction logs and evaluation scores.
2. Analyze low-performing responses and user feedback.
3. Identify patterns and failure cases.
4. Prepare high-quality datasets for training.
5. Fine-tune or update the selected language model.
6. Validate the updated model using benchmark datasets.
7. Deploy the improved model after successful evaluation.

## Data Flow

```text
Interaction Logs
        │
        ▼
Evaluation Scores
        │
        ▼
Failure Analysis
        │
        ▼
Dataset Preparation
        │
        ▼
Model Fine-Tuning
        │
        ▼
Model Validation
        │
        ▼
Production Deployment
```

## Component Description

### Interaction Log Collection

The subsystem gathers user queries, retrieved context, generated responses, evaluation metrics, and user feedback from previous interactions.

### Failure Analysis

Responses with low faithfulness, low relevance, or poor retrieval quality are analyzed to identify recurring issues and system weaknesses.

### Dataset Preparation

High-quality examples are cleaned, labeled, and formatted into training datasets suitable for supervised fine-tuning or preference optimization.

### Model Fine-Tuning

The selected language model is trained on the prepared dataset to improve response quality, domain knowledge, and overall system performance.

### Model Validation

The updated model is evaluated using benchmark datasets and predefined quality metrics to ensure that performance has improved without introducing regressions.

### Production Deployment

After successful validation, the updated model is deployed to production where it replaces or complements the previous version.

---

# Overall System Workflow

The complete NeuroFlow architecture follows the sequence below:

```text
User Upload
      │
      ▼
Ingestion Subsystem
      │
      ▼
Vector Database
      │
      ▼
User Query
      │
      ▼
Retrieval Subsystem
      │
      ▼
Generation Subsystem
      │
      ▼
Generated Response
      │
      ▼
Evaluation Subsystem
      │
      ▼
Fine-Tuning Subsystem
      │
      ▼
Improved AI System
```

## Summary

NeuroFlow is designed as a modular Retrieval-Augmented Generation (RAG) platform. Each subsystem has a clearly defined responsibility, allowing the platform to efficiently process documents, retrieve relevant information, generate accurate responses, evaluate performance, and continuously improve through fine-tuning. This modular architecture supports scalability, maintainability, and future enhancements.