import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from rag_query import answer_question
from embed_data import EmbeddedData

embedded_data = EmbeddedData()

q = ""
while (q := input("Ask a question (or 'exit' to quit): ")) and q.lower() != "exit":
    print("Answer:")
    print(answer_question(q, embedded_data))
    print("\n")
