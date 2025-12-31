```retriever = vectordb.as_retriever(search_kwargs={"k": 1})
docs = retriever.invoke(query)```



## MMR search
MMR in vector stores is a technique used to balance the relevance and diversity of retrieved results. It selects documents that are both highly relevant to the query and minimally similar to previously selected documents. This approach helps to avoid redundancy and ensures a more comprehensive coverage of different aspects of the query.

The following code is showing how to conduct an MMR search in a vector database. You just need to sepecify search_type="mmr".

```python
retriever = vectordb.as_retriever(search_kwargs={"k": 1, "search_type": "mmr"})
docs = retriever.invoke(query)
```

## Similarity score threshold retrieval

Similarity score threshold retrieval is a method used to filter retrieved documents based on a minimum similarity score. This approach ensures that only documents with a certain level of relevance are returned, helping to reduce noise and focus on the most pertinent results.

```python
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4}
)
docs = retriever.invoke(query)
```

## Multi-Query Retriever

Multi-Query Retriever is a method used to retrieve documents based on multiple queries. This approach ensures that only documents with a certain level of relevance are returned, helping to reduce noise and focus on the most pertinent results.

```python
retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm_model_langchain()
)
docs = retriever.invoke(query)
```

## Self-Querying Retriever
Self-Querying Retriever is a method used to retrieve documents based on a self-querying retriever. This approach ensures that only documents with a certain level of relevance are returned, helping to reduce noise and focus on the most pertinent results.   

```python
retriever = SelfQueryRetriever.from_llm(
        llm=llm_model_langchain(),
        vectorstore=vectordb, # Vector store
        metadata_field_info=metadata_field_info, # Metadata field info
        document_contents=document_content_description, # Document content description
    )
docs = retriever.invoke(query)
Example: src/modules/self-querying-retriever.py
```

## Parent Document Retriever
When splitting documents for retrieval, there are often conflicting desires:

- You may want to have small documents so that their embeddings can most accurately reflect their meaning. If the documents are too long, the embeddings can lose meaning.
- You want to have long enough documents so that the context of each chunk is retained.

Parent Document Retriever is a method used to retrieve documents based on a parent document. This approach ensures that only documents with a certain level of relevance are returned, helping to reduce noise and focus on the most pertinent results.