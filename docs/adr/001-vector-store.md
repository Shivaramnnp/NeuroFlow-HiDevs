# ADR-001: Selection of Vector Store

## Status

Accepted

---

## Context

NeuroFlow requires a vector database to store document embeddings and perform semantic similarity searches. The vector store must integrate well with the existing PostgreSQL database while providing efficient nearest-neighbor search capabilities.

The system should support:

- High-dimensional vector storage
- Fast similarity search
- Metadata filtering
- Scalability
- Easy integration with the backend

---

## Decision

The project will use **pgvector** as the vector storage solution.

pgvector extends PostgreSQL with vector data types and similarity search functions, allowing vector embeddings and relational data to be managed within a single database.

---

## Alternatives Considered

### ChromaDB

**Pros**

- Easy to set up
- Designed specifically for vector search
- Good for rapid prototyping

**Cons**

- Separate database service
- Additional infrastructure
- Less integrated with PostgreSQL

---

### Pinecone

**Pros**

- Fully managed service
- Highly scalable
- Excellent performance

**Cons**

- Cloud dependency
- Additional operational cost
- Vendor lock-in

---

### Weaviate

**Pros**

- Rich feature set
- Hybrid search support
- GraphQL API

**Cons**

- More complex deployment
- Higher operational overhead

---

## Consequences

### Advantages

- Single database for structured and vector data
- Simplified deployment
- Reduced infrastructure complexity
- Native PostgreSQL integration
- Supports metadata filtering alongside vector search

### Disadvantages

- Lower scalability compared to specialized cloud vector databases
- Performance may decrease with extremely large datasets
- Advanced vector indexing options are more limited than dedicated vector databases

---

## Rationale

pgvector provides the best balance between simplicity, performance, and maintainability for the NeuroFlow platform. It minimizes infrastructure complexity while delivering the semantic search capabilities required by the Retrieval-Augmented Generation (RAG) pipeline.