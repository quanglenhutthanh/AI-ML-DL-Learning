import re
import random
import fitz
from tqdm.auto import tqdm
import pandas as pd
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer

def text_format(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def open_and_read_pdf(file_path) -> list[dict]:
    doc = fitz.open(file_path)
    pages_and_texts = []
    for page_number, page in enumerate(tqdm(doc)):
        text = page.get_text()
        text = text_format(text=text)
        pages_and_texts.append({
            "page_number": page_number,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count": len(text.split(". ")),
            "page_token_count": len(text)/4 ,
            "text": text                       
        })
    return pages_and_texts

def split_page_into_sentences():
    return []

def split_list(input_list: list[str],
               slice_size: int=10):
    return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]

def embedding_text(sentences):
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")
    
    embeddings = embedding_model.encode(sentences=sentences)
    
    return embeddings
    # embedding_dict = dict(zip(sentences, embeddings))
    # for sentence, embedding in embedding_dict.items():
    #     print(f"Sentence: {sentence}")
    #     print(f"Embedding: {embedding}")

def test_pdf_processing():
    pages_and_texts = open_and_read_pdf("test.pdf")
    # print(pages_and_texts[1])
    # df = pd.DataFrame(pages_and_texts)
    # print(df.head(2))
    # print(df.describe())

    nlp = English()
    nlp.add_pipe("sentencizer")

    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])
        item["sentence_chunks"] = split_list(item["sentences"])
    # print(pages_and_texts[1])

    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
            chunk_dict["chunk_sentence"] = joined_sentence_chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4
            chunk_dict["embeddings"] = embedding_text(joined_sentence_chunk)
            pages_and_chunks.append(chunk_dict)

    # df = pd.DataFrame(pages_and_chunks)
    # print(df.describe().round(2))
    
    print(random.sample(pages_and_chunks, k=1))
    print(f"Total sample chunks: {len(pages_and_chunks)}")

    df_chunks_embeddings = pd.DataFrame(pages_and_chunks)
    df_chunks_embeddings.to_csv("chunks_embeddings.csv")

if __name__ == "__main__":
    # test_list = list(range(25))
    # print(split_list(test_list))
    test_pdf_processing()
    # sentences = ["I like cat!","she is hot.","Embedding are one of my favorites."]
    # embedding_text(sentences)