import kagglehub
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

DATA_DIR = "data.json"


class EmbeddedData:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.index, self.dataframe = self.load_embeddings()

    def load_embeddings(self):
        if not os.path.exists(DATA_DIR):
            print("data files not found, downloading dataset...")
            data = self.download_data()
            self.generate_data_text_files(data)
        dataframe = pd.read_json(DATA_DIR)
        if os.path.exists("./nfl.index"):
            print("Loading existing FAISS index...")
            index = faiss.read_index("./nfl.index")
        else:
            # convert all the docs into embeddings
            # convert to a tensor to make it easier to work with machine learning
            embeddings = self.embedding_model.encode(
                dataframe["text"], convert_to_tensor=True
            )
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

        return index, dataframe

    def download_data(self):
        path = kagglehub.dataset_download("grayengineering425/nfl-box-scores")
        data = pd.read_csv(path + "/box_scores.csv")
        data["date"] = pd.to_datetime(data["date"], format="%B %d, %Y")
        data["year"] = data["date"].dt.year
        data["month"] = data["date"].dt.month
        data["day"] = data["date"].dt.day
        data["date"] = data["date"].dt.strftime("%B %d, %Y")

        # remove this later for faster testing
        data = data[data["year"] == 2016]

        return data

    def generate_data_text_files(self, data):
        def build_game_text(row):
            lines = []

            lines.append(f"{row['date']}, {row['home']} at {row['visitor']}")
            lines.append(f"Home Team: {row['home']} | Visitor Team: {row['visitor']}")
            lines.append(f"{row['home']} Score: {row['home_score']}")
            lines.append(f"{row['visitor']} Score: {row['visitor_score']}")
            lines.append(f"{row['home']} First Downs: {row['home_first_downs']}")
            lines.append(f"{row['visitor']} First Downs: {row['visitor_first_downs']}")
            lines.append(f"{row['home']} Net Yards: {row['home_net_yards']}")
            lines.append(f"{row['visitor']} Net Yards: {row['visitor_net_yards']}")
            lines.append(
                f"{row['home']} Time of Possession: {row['home_time_of_possession']}"
            )
            lines.append(
                f"{row['visitor']} Time of Possession: {row['visitor_time_of_possession']}"
            )

            return "\n".join(lines)

        data["text"] = data.apply(build_game_text, axis=1)
        data["index"] = range(len(data))

        data = data.to_dict(orient="records")
        with open(DATA_DIR, "w") as f:
            json.dump(data, f, indent=2)

    def retrieve_docs(self, query, k=3):
        query_emb = self.embedding_model.encode([query])
        # search using a query
        k = 3
        D, I = self.index.search(np.array(query_emb), k)
        docs = []
        # iterate through all of the resulting indexes
        for idx in I[0]:
            docs.append(self.dataframe.iloc[idx]["text"])
        return docs
