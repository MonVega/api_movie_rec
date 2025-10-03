import pandas as pd
import re
import spacy
from sklearn.preprocessing import FunctionTransformer
import nltk
from nltk.corpus import stopwords
import os

# Indicar a NLTK dónde buscar los datos
nltk.data.path.append(os.path.join(os.getcwd(),"nltk_data"))

# Cargar stopwords
stop_words = [x.lower() for x in stopwords.words("english")]

# Cargar modelo de Spacy
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

def process_text(df):
    df = df.copy()
    df["title_str"] = df["title"].astype(str)
    df["description_str"] = df["description"].astype(str)
    
    df["title_symbol_count"] = df["title_str"].apply(lambda x: len(re.findall(r"[^\w\s]", x)))
    df["desc_symbol_count"] = df["description_str"].apply(lambda x: len(re.findall(r"[^\w\s]", x)))
    
    df["description_clean"] = (
        df["description_str"]
        .str.lower()
        .str.normalize("NFKD")
        .str.encode("ascii", "ignore").str.decode("utf-8")
        .str.replace(r"<br\s*/?>", " ", regex=True)
        .str.replace(r"https?://\S+|www\.\S+", "", regex=True)
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["description_clean"] = df["description_clean"].map(
        lambda sentence: " ".join([word for word in sentence.split() if word not in stop_words])
    )
    df["description_lem"] = df["description_clean"].apply(
        lambda text: " ".join([token.lemma_ for token in nlp(text)])
    )
    
    return df[["description_lem", "desc_symbol_count", "title_symbol_count"]]


def split_text_numeric(df):
    X_text = df["description_lem"].tolist()
    X_num = df[["title_symbol_count", "desc_symbol_count"]]
    return {"text": X_text, "numeric": X_num}


