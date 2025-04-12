# AI Solutions Explorer

Developed by: Michelle Naa Kwarley Owoo
Index Number: 10211100296


Deployed Solution Link:https://ai10211100296-kekp3v7gluvwtq22ddddn6.streamlit.app/

## Project Overview

This Streamlit application allows users to explore and solve diverse machine learning and AI problems, including regression, clustering, neural networks, and large language models. The application includes interactive interfaces for data upload, model training, and result visualization.

## Features

### 1. Regression Analysis
- Upload regression datasets
- Train linear regression models
- Visualize regression results
- Make custom predictions

### 2. Clustering
- Upload datasets for clustering
- Perform K-means clustering
- Visualize clusters and centroids
- Download clustered data

### 3. Neural Networks
- Upload classification/regression datasets
- Train customizable neural networks
- Monitor training progress in real-time
- Evaluate model performance
- Make predictions with trained models

### 4. Large Language Models (RAG Implementation)
- Question answering on Information Extracted from Academic City Handbook
- Retrieval-Augmented Generation architecture
- Context-aware responses
- Comparison with ChatGPT

  ## 📊 Evaluation and Comparison: Mistral RAG vs ChatGPT

As part of Deliverable (d), we evaluated the performance of our custom RAG pipeline (powered by Mistral-7B-Instruct) against ChatGPT (gpt-3.5-turbo) using a series of real queries from the Academic City student handbook.

---

###  Evaluation Metrics

Each response was assessed on the following:
- **Accuracy**: Faithfulness to the handbook content
- **Clarity**: How clear and understandable the response is
- **Relevance**: Appropriateness and specificity of the content
- **Conciseness**: Ability to avoid unnecessary verbosity

---

###  Summary Table

| Query Topic                           | Model       | Accuracy | Clarity | Relevance | Conciseness | Avg |
|--------------------------------------|-------------|----------|---------|-----------|-------------|-----|
| Appeal Process                       | Mistral     | 5        | 5       | 5         | 3           | 4.5 |
|                                      | ChatGPT     | 4        | 5       | 4         | 5           | 4.5 |
| Dress Code                           | Mistral     | 5        | 4       | 5         | 3           | 4.25 |
|                                      | ChatGPT     | 4        | 5       | 4         | 5           | 4.5 |
| Disruptive Conduct                   | Mistral     | 5        | 4       | 5         | 4           | 4.5 |
|                                      | ChatGPT     | 5        | 5       | 5         | 4           | 4.75 |
| Drug Policy                          | Mistral     | 5        | 5       | 5         | 4           | 4.75 |
|                                      | ChatGPT     | 5        | 5       | 5         | 5           | 5   |
| Prohibited Conduct & Items           | Mistral     | 5        | 4       | 5         | 3           | 4.25 |
|                                      | ChatGPT     | 4        | 5       | 4         | 5           | 4.5 |
| Respect & Regard for Others          | Mistral     | 5        | 4       | 5         | 4           | 4.5 |
|                                      | ChatGPT     | 5        | 5       | 5         | 5           | 5   |
| Field Trip Policy                    | Mistral     | 4        | 5       | 4         | 4           | 4.25 |
|                                      | ChatGPT     | 5        | 5       | 5         | 5           | 5   |
| Housing Policy (Detailed)            | Mistral     | 4        | 4       | 4         | 3           | 3.75 |
|                                      | ChatGPT     | 5        | 5       | 5         | 5           | 5   |

---

###  Analysis

#### 🔹 Mistral RAG (Custom Model)
-  Excellent factual grounding — highly accurate when answering policy-based questions.
-  Always references the actual document and section numbers.
-  Can be overly verbose when brief responses are preferred.
-  Longer answers tend to be more "listy" and less structured than ChatGPT.

#### 🔹 ChatGPT (gpt-3.5-turbo)
-  Extremely fluent, human-like, and well-structured answers.
-  Very concise when needed and detailed when requested.
-  Occasionally generalizes or lacks explicit citations (e.g., section numbers).
-  Higher risk of “hallucinated” content in complex queries if not grounded in source text.

---

###  Conclusion

While ChatGPT offers smoother, more natural responses, the custom Mistral-based RAG system provides **superior factual reliability** and document-specific accuracy. This makes it better suited for tasks where answers must be strictly grounded in a source, such as policy manuals or academic regulations.

>  Final verdict: **Mistral RAG is preferred for accuracy-critical, document-anchored Q&A systems**, while ChatGPT excels at tone, structure, and user-friendly output.

