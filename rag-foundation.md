# RAG (Retrieval-Augmented Generation)
- Allows language models to retrieve relevant information from external knowledge sources at runtime.
- Reduces hallucinations by grounding responses in retrieved data.
- Uses embeddings to map text into vector space for semantic search.
- Employs semantic search to find relevant documents, which can then be summarized or used for generation.
- Combines **retriever** (fetches relevant documents) and **generator** (produces final answer) in a pipeline.
- Commonly used in chatbots, question answering systems, and knowledge assistants.

# Embeddings
- Embedding models transform text into numerical vectors in high-dimensional spaces.
- Created using machine learning or deep learning models trained on large text corpora to capture semantic relationships.
- Dimensionality of embeddings affects expressiveness and performance.
- Similar vectors are close in high-dimensional space, indicating semantic similarity.
- **Examples**: word2vec, OpenAI’s Ada/embed-3 models, sentence-transformers, BERT embeddings.

# Similarity Score
- Comparing embeddings is meaningful only if they are in the same vector space.
- Similarity scores can be calculated using:
  - **Cosine similarity**
  - **Inner product**
  - **Squared L2 (Euclidean) distance**
- Cosine similarity measures the angle between vectors; inner product considers magnitude.

# Similarity Search Indexing
- Compares a query vector with vectors stored in a database to find similar items.
- Indexing improves search efficiency by grouping similar vectors, reducing the search space.
- Common indexing techniques:
  - **Inverted File Indexes (IVF)**
  - **Hierarchical Navigable Small World (HNSW)** graphs
- Approximate nearest neighbor (ANN) algorithms are widely used for large datasets; flat search is practical for small datasets.

# Vector Database
- Vector databases store, index, and manage embeddings for efficient similarity search.
- Enable storage, retrieval, and querying of embeddings at scale.
- **Examples**: Chroma (open-source), Pinecone, Weaviate, Milvus, Qdrant.

# Document Chunking
- Chunk length affects how much context the model receives; longer chunks give more context but risk exceeding token limits.
- Methods:
  - **Fixed size**: split text into uniform-length chunks.
  - **Dynamic**: split at sentence or semantic boundaries.
  - **Advanced**: use algorithms for overlapping or relevance-based splitting.
- Overlapping chunks preserve context across splits.

# LangChain
- Implements document chunk overlapping to maintain context.
- Enhances chunks with metadata for citations and source attribution.
- Supports version control for documents and data sources.
- Uses LLMs to summarize retrieved documents.
- Can implement advanced techniques such as keyword extraction to improve relevance.

# Prompt Engineering Considerations for RAG
- Include chunk metadata in prompts to provide context and source information.
- XML formatting can improve structure but is optional.
- Frame instructions positively (what to do) rather than negatively (what not to do) for better LLM performance.
- Iteratively tweak and test prompts for optimal results.
- Provide clear, concise instructions to guide the LLM effectively.
