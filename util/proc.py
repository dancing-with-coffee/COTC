import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

###

def read_texts(data_name, max_features):
    label_text_augmentation_csv = pd.read_csv("data/" + data_name + ".csv", sep="\t", header=None, keep_default_na=False, quoting=3, encoding="utf-8", on_bad_lines="skip")

    labels = label_text_augmentation_csv[0].tolist()
    texts = label_text_augmentation_csv[1].tolist()
    texts_0 = label_text_augmentation_csv[2].tolist()
    texts_1 = label_text_augmentation_csv[3].tolist()

    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)

    tfidfs = vectorizer.fit_transform(texts).toarray()

    return labels, texts, texts_0, texts_1, tfidfs

###
