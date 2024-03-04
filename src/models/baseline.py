import logging
import argparse
from pathlib import Path
from src.data.read_dataset import get_data
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

project_dir = Path(__file__).resolve().parents[2]


class BaselinePredict:
    """
    Provides a classic baseline for comparison
    Usage:
    python -m src.models.baseline <sentence>
    """

    def __init__(self, model_name):
        self._model = self.train(model_name)

    def predict(self, sentence: str):
        """Predict the binary class of a sentence using a Logistic Regression
        Args:
            sentence (str): sentence
        Returns:
            multi class (str): Dance (class 0) | Heavy Metal (class 1) | Hip Hop (class 2) | Indie (class 3) | Pop (class 4) | Rock (class 5)
        """
        # predict
        return self._model.predict([sentence])

    def train(self, model) -> str:
        """Train a logistic regression method"""
        try:
            # pipeline
            pipeline = Pipeline([
                ('vect', CountVectorizer()),
                #('tfidf', TfidfTransformer()),
                ('lr', LogisticRegression(multi_class='auto', max_iter=10000))
            ])

            # data
            df_train, _ = get_data()

            # text
            train_texts = df_train['text']

            # fit
            pipeline.fit(train_texts, df_train['label'])

            return pipeline

        except Exception:
            logging.error(f'directory or model is invalid or does not exist: {self._model_name}')
