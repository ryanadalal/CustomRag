import kagglehub
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

DATA_DIR = "dataset"


class EmbeddedData:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.index, self.index_to_file = self.load_embeddings()

    def load_embeddings(self):
        if os.path.exists("./nfl.index") and os.path.exists("./index_to_file.json"):
            print("Loading existing FAISS index...")
            with open("./index_to_file.json", "r") as f:
                index_to_file = json.load(f)
                index_to_file = {int(k): v for k, v in index_to_file.items()}
            index = faiss.read_index("./nfl.index")
            return index, index_to_file
        else:
            if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
                print("data files not found, downloading dataset...")
                data = self.download_data()
                self.generate_data_text_files(data)

            print("Creating FAISS index...")
            docs, filenames = self.load_data_text_files()
            # convert all the docs into embeddings
            # convert to a tensor to make it easier to work with machine learning
            embeddings = self.embedding_model.encode(docs, convert_to_tensor=True)
            dimensions = embeddings.shape[1]
            # store the embeddings flat - without clustering
            # L2 distance - euclidean
            index = faiss.IndexFlatL2(dimensions)

            # convert the embeddings to numpy arryas
            # wouldn't be neccessary if convert_to_tensor was false
            embeddings_np = embeddings.cpu().detach().numpy()

            # store the embeddings in the index
            index.add(embeddings_np)
            faiss.write_index(index, "./nfl.index")

            # index stores no meta data
            # create a list of embeddings with dictionaries
            # key - index in the embeddings list
            # value - file name
            index_to_file = {i: f for i, f in enumerate(filenames)}

            with open("./index_to_file.json", "w") as f:
                json.dump(index_to_file, f)

            return index, index_to_file

    def download_data(self):
        path = kagglehub.dataset_download("grayengineering425/nfl-box-scores")
        data = pd.read_csv(path + "/box_scores.csv")
        data["date"] = pd.to_datetime(data["date"], format="%B %d, %Y")
        data["year"] = data["date"].dt.year
        data["month"] = data["date"].dt.month
        data["day"] = data["date"].dt.day
        data["game_id"] = (
            data["date"].astype(str) + "_" + data["home"] + "_" + data["visitor"]
        )
        data = data[data["year"] == 2016]
        return data

    def generate_data_text_files(self, data):
        os.makedirs(DATA_DIR, exist_ok=True)
        for self.index, row in data.iterrows():
            lines = []

            lines.append(f"{row['date']}, {row['home']} at {row['visitor']}")

            lines.append(f"{row['home']}: {row['home_score']}")
            lines.append(f"{row['visitor']}: {row['visitor_score']}")

            lines.append(f"Home First Downs: {row['home_first_downs']}")
            lines.append(f"Visitor First Downs: {row['visitor_first_downs']}")
            lines.append(f"Home Net Yards: {row['home_net_yards']}")
            lines.append(f"Visitor Net Yards: {row['visitor_net_yards']}")
            lines.append(f"Home Time of Possession: {row['home_time_of_possession']}")
            lines.append(
                f"Visitor Time of Possession: {row['visitor_time_of_possession']}"
            )

            output = "\n".join(lines)
            filename = f"{DATA_DIR}/{row['game_id']}.txt"
            with open(filename, "w") as f:
                f.write(output)

    def load_data_text_files(self):
        docs = []
        filenames = os.listdir(DATA_DIR)
        for file in filenames:
            with open(os.path.join(DATA_DIR, file), "r") as f:
                docs.append(f.read())
        return docs, filenames

    def retrieve_docs(self, query, k=3):
        query_emb = self.embedding_model.encode([query])
        # search using a query
        k = 3
        D, I = self.index.search(np.array(query_emb), k)
        docs = []
        # iterate through all of the resulting indexes
        for idx in I[0]:
            filename = self.index_to_file[idx]
            with open(os.path.join(DATA_DIR, filename), "r") as f:
                docs.append(f.read())
        return docs
