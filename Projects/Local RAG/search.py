import pandas as pd
import numpy as np
import torch
from sentence_transformers import util, SentenceTransformer
import textwrap

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

df = pd.read_csv("chunks_embeddings.csv")
df["embeddings"] = df["embeddings"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
pages_and_chunks = df.to_dict(orient="records")

embeddings = torch.tensor(np.array(df["embeddings"].tolist()), dtype=torch.float32)
print(embeddings.shape)

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")

query = "Using Azure Kubernetes Service"
print(f"Query: {query}")
query_embedding = embedding_model.encode(query, convert_to_tensor=True)

from time import perf_counter as timer
start_time = timer()
dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
end_time = timer()

print(f"Time take to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

# 4. Get the top-k results (we'll keep this to 5)
top_results_dot_product = torch.topk(dot_scores, k=5)
# print(top_results_dot_product)


print(f"Query: '{query}'\n")
print("Results:")
# Loop through zipped together scores and indicies from torch.topk
for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
    print(f"Score: {score:.4f}")
    # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
    print("Text:")
    print_wrapped(pages_and_chunks[idx]["chunk_sentence"])
    # Print the page number too so we can reference the textbook further (and check the results)
    print(f"Page number: {pages_and_chunks[idx]['page_number']}")
    print("\n")