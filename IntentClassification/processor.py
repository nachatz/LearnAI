import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import TweetTokenizer
from sklearn.preprocessing import OrdinalEncoder
from collections import Counter


class Processor:
    def __init__(self):
        nltk.download("stopwords")
        nltk.download("word_tokenize")
        nltk.download("punkt")
        self.stemmer = PorterStemmer()
        self.tknzr = TweetTokenizer()
        self.encoder = OrdinalEncoder()
        self.token_counts = Counter()
        self.token_to_id = {}
        self.next_id = 1

    def tokenize(self, data, stop_words):
        stop_words = set(stopwords.words("english"))
        punctuation = [",", ".", "!", '"', "'", "`", "``", "''"]

        for idx, i in enumerate(data["text"]):
            text = re.sub(r"\d+", "", data["text"][idx])
            tokens = self.tknzr.tokenize(text)
            text = [self.stemmer.stem(token) for token in tokens]
            data["text"][idx] = [
                word
                for word in text
                if word not in stop_words and word not in punctuation
            ]

    def ordinal_encode(self, data):
        ordinal_encoding = {}
        encoded_df = self.encoder.fit_transform(data)
        for i, col in enumerate(data.columns):
            categories = self.encoder.categories_[i]
            ordinal_encoding[col] = {
                category: encoding
                for category, encoding in zip(categories, encoded_df[:, i])
            }

        return ordinal_encoding

    def encode_tokens(self, tokens):
        return [
            self.token_to_id[token] for token in tokens if token in self.token_to_id
        ]

    def preprocess_text_data(self, data):
        # encode text field
        data["text"] = data["text"].apply(self.encode_tokens)

        # Update token counts based on the encoded tokens
        for tokens in data["text"]:
            self.token_counts.update(tokens)

        # Determine token IDs based on their counts
        for token, count in self.token_counts.items():
            if count < 10:
                self.token_to_id[token] = 0
            elif token not in self.token_to_id:
                self.token_to_id[token] = self.next_id
                self.next_id += 1

        # Re-encode
        data["text"] = data["text"].apply(
            lambda tokens: [self.token_to_id[token] for token in tokens]
        )

        # Encode sentiment labels as numeric values
        sentiment_labels = data["sentiment"].unique()
        sentiment_to_id = {label: i for i, label in enumerate(sentiment_labels)}
        data["sentiment"] = data["sentiment"].map(sentiment_to_id)

        return data
