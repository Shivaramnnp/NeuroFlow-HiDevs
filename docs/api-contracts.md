# NeuroFlow API Contracts

## Overview

This document defines the REST API contracts for the NeuroFlow platform. The APIs enable document ingestion, query processing, evaluation, and user feedback. All requests and responses use JSON unless otherwise specified.

---

# Base URL

```
http://localhost:8000/api/v1
```

---

# Authentication

All protected endpoints require a Bearer Token.

```
Authorization: Bearer <access_token>
```

---

# 1. Upload Document

**Endpoint**

```
POST /documents/upload
```

**Description**

Uploads a document to the ingestion pipeline.

### Request

**Content-Type**

```
multipart/form-data
```

| Field | Type | Required | Description |
|--------|------|----------|-------------|
| file | File | Yes | PDF, DOCX, Image, or CSV |
| source | String | No | Document source |

### Success Response

**201 Created**

```json
{
  "document_id": "doc_001",
  "status": "uploaded",
  "message": "Document uploaded successfully."
}
```

### Error Response

**400 Bad Request**

```json
{
  "error": "Unsupported file format."
}
```

---

# 2. Query Documents

**Endpoint**

```
POST /query
```

**Description**

Retrieves relevant context and generates an AI response.

### Request

```json
{
  "query": "What is Retrieval-Augmented Generation?"
}
```

### Success Response

**200 OK**

```json
{
  "answer": "Retrieval-Augmented Generation (RAG) combines information retrieval with large language models...",
  "sources": [
    {
      "document_id": "doc_001",
      "chunk_id": "chunk_15"
    }
  ]
}
```

---

# 3. Retrieve Document Information

**Endpoint**

```
GET /documents/{document_id}
```

### Success Response

```json
{
  "document_id": "doc_001",
  "filename": "rag_notes.pdf",
  "status": "indexed",
  "uploaded_at": "2026-07-17T10:30:00Z"
}
```

---

# 4. Delete Document

**Endpoint**

```
DELETE /documents/{document_id}
```

### Success Response

```json
{
  "message": "Document deleted successfully."
}
```

---

# 5. Evaluation Results

**Endpoint**

```
GET /evaluations/{query_id}
```

### Success Response

```json
{
  "query_id": "query_101",
  "faithfulness": 0.95,
  "answer_relevance": 0.93,
  "context_precision": 0.91,
  "context_recall": 0.94
}
```

---

# 6. Submit User Feedback

**Endpoint**

```
POST /feedback
```

### Request

```json
{
  "query_id": "query_101",
  "rating": 5,
  "comment": "Very helpful response."
}
```

### Success Response

```json
{
  "message": "Feedback submitted successfully."
}
```

---

# Common HTTP Status Codes

| Status Code | Meaning |
|-------------|---------|
| 200 | Request successful |
| 201 | Resource created |
| 400 | Bad request |
| 401 | Unauthorized |
| 404 | Resource not found |
| 500 | Internal server error |

---

# API Workflow

```text
Client
   │
   ▼
REST API
   │
   ▼
Authentication
   │
   ▼
Business Logic
   │
   ▼
Database / Vector Store
   │
   ▼
JSON Response
```