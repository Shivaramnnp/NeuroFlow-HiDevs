# ADR-002: Chunking Strategy

## Status

Accepted

---

## Context

Large Language Models (LLMs) have a limited context window and cannot efficiently process entire documents during retrieval. To improve retrieval accuracy and reduce computational cost, documents must be divided into smaller, meaningful segments before generating embeddings.

An effective chunking strategy should:

- Preserve the semantic meaning of the text
- Improve retrieval accuracy
- Minimize information loss
- Support efficient embedding generation
- Work across multiple document formats

---

## Decision

NeuroFlow will use a **fixed-size chunking strategy with overlapping chunks**.

Each document will be split into chunks of approximately **500 tokens** with an overlap of **100 tokens** between consecutive chunks.

The overlap ensures that important information spanning chunk boundaries is preserved during retrieval.

---

## Alternatives Considered

### Fixed-Size Chunking (Without Overlap)

**Pros**

- Simple implementation
- Fast processing
- Consistent chunk sizes

**Cons**

- Context may be lost at chunk boundaries
- Reduced retrieval accuracy

---

### Sentence-Based Chunking

**Pros**

- Preserves natural language structure
- Easy to understand

**Cons**

- Uneven chunk sizes
- May exceed model token limits

---

### Semantic Chunking

**Pros**

- Best preservation of meaning
- Higher retrieval quality

**Cons**

- More computationally expensive
- Complex implementation

---

## Consequences

### Advantages

- Maintains context across chunk boundaries
- Improves semantic retrieval
- Produces consistent embedding sizes
- Easy to implement and maintain

### Disadvantages

- Slight increase in storage due to overlapping text
- Additional embedding computations for overlapping regions

---

## Rationale

A fixed-size chunking strategy with overlap provides a balance between implementation simplicity, retrieval performance, and computational efficiency. It reduces the risk of losing important contextual information while maintaining predictable chunk sizes for embedding generation and vector search.