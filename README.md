# RAG based on NFL box scores

- loads an nfl data set from kaggle
- creates "text" from the box scores for ease of use with llms
- implements faiss to enable natural language queries into the databse
- finds top 3 relevant documents for each question
- uses qwen2.5 to generate a response

## Future

- implement hallucination checks and verification