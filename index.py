import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import kagglehub
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

OUTPUT_DIR = "dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

path = kagglehub.dataset_download("grayengineering425/nfl-box-scores")
print("Path to dataset files:", path)
print(os.listdir(path))

data = pd.read_csv(path + "/box_scores.csv")
data["date"] = pd.to_datetime(data["date"], format="%B %d, %Y")
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day
data["game_id"] = data["date"].astype(str) + "_" + data["home"] + "_" + data["visitor"]

data = data[data["year"] == 2016]

for index, row in data.iterrows():
    lines = []

    lines.append(f"{row['date']}, {row['home']} at {row['visitor']}")

    lines.append(f"{row['home']}: {row['home_score']}")
    lines.append(f"{row['visitor']}: {row['visitor_score']}")

    lines.append(f"Home First Downs: {row['home_first_downs']}")
    lines.append(f"Visitor First Downs: {row['visitor_first_downs']}")
    lines.append(f"Home Net Yards: {row['home_net_yards']}")
    lines.append(f"Visitor Net Yards: {row['visitor_net_yards']}")
    lines.append(f"Home Time of Possession: {row['home_time_of_possession']}")
    lines.append(f"Visitor Time of Possession: {row['visitor_time_of_possession']}")

    output = "\n".join(lines)
    filename = f"{OUTPUT_DIR}/{row['game_id']}.txt"
    with open(filename, "w") as f:
        f.write(output)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

docs = []
filenames = os.listdir(OUTPUT_DIR)
for file in filenames:
    with open(os.path.join(OUTPUT_DIR, file), "r") as f:
        docs.append(f.read())

# convert all the docs into embeddings
# convert to a tensor to make it easier to work with machine learning
embeddings = embedding_model.encode(docs, convert_to_tensor=True)

dimensions = embeddings.shape[1]
# store the embeddings flat - without clustering
# L2 distance - euclidean
index = faiss.IndexFlatL2(dimensions)

# convert the embeddings to numpy arryas
# wouldn't be neccessary if convert_to_tensor was false
embeddings_np = embeddings.cpu().detach().numpy()

# store the embeddings in the index
index.add(embeddings_np)

# index stores no meta data
# create a list of embeddings with dictionaries
# key - index in the embeddings list
# value - file name
index_to_file = {i: f for i, f in enumerate(filenames)}


def retrieve_docs(query, k=3):
    query_emb = embedding_model.encode([query])
    # search using a query
    k = 3
    D, I = index.search(np.array(query_emb), k)
    docs = []
    # iterate through all of the resulting indexes
    for idx in I[0]:
        filename = index_to_file[idx]
        with open(os.path.join(OUTPUT_DIR, filename), "r") as f:
            docs.append(f.read())
    return docs


# load the model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
large_lang_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")


def answer_question(query, top_k=3, max_tokens=512):
    # find the relevant information
    context = ",".join(retrieve_docs(query, k=top_k))

    # build the llm prompt
    prompt = f"""
        You are a fact-based sports assistant.
        Answer the question using ONLY the information below. Do NOT hallucinate.

        Context:
        {context}

        Question:
        {query}
    """

    # query the model
    print("tokenizing")
    inputs = tokenizer(prompt, return_tensors="pt").to(large_lang_model.device)
    print("generating")
    outputs = large_lang_model.generate(**inputs, max_new_tokens=max_tokens)
    print("detokenizing")
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


print(
    answer_question(
        "What was the final score of New England vs Miami on September 18 2016?"
    )
)


del large_lang_model, tokenizer
import gc

gc.collect()
torch.cuda.empty_cache()  # if using GPU
