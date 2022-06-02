
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import exists
from model_io import ensure_stop_words
import config
import joblib

from stop_words_provider import StopWordsProvider


class TfIdfizer(StopWordsProvider):
    def __init__(self, get_train_data, tf_idf_model_path):
        StopWordsProvider.__init__(self)
        self.__vectors = None
        self.__get_train_data = get_train_data
        self.__model_path = tf_idf_model_path

    def tf_idf_ize(self, origin: list):
        return self.tf_idf_vectors.transform(origin)

    @property
    def tf_idf_vectors(self):
        if self.__vectors == None:
            print('Lazy Load: TF-IDF vectors')
            self.__vectors = self.__ensure_tf_idf_vectors(
                self.__get_train_data, self.__model_path)
        return self.__vectors

    @staticmethod
    def __ensure_tf_idf_vectors(train_set_getter, model_path, force=False):
        if force or not exists(model_path):
            train_set = train_set_getter()
            vectors = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            vectors.fit_transform(train_set)
            joblib.dump(vectors, model_path)
            return vectors
        else:
            return joblib.load(model_path)
