
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import exists
from model_io import ensure_stop_words,cut_text
import config
import joblib

class TfIdfizer(object):
    def __init__(self, get_train_data, tf_idf_model_path):
        self.__vectors = None
        self.__get_train_data=get_train_data
        self.__model_path=tf_idf_model_path
        self.__p_stop_words = None


    @property
    def _stop_words(self):
        if self.__p_stop_words == None:
            self.__p_stop_words = ensure_stop_words()
        return self.__p_stop_words

    def tf_idf_ize(self, origin: list[str]):
        return self.tf_idf_vectors.transform(origin)

    @property
    def tf_idf_vectors(self):
        if self.__vectors == None:
            print('Lazy Load: TF-IDF vectors')
            self.__vectors = self.__ensure_tf_idf_vectors(self.__get_train_data(),self.__model_path)
        return self.__vectors

    @staticmethod
    def __ensure_tf_idf_vectors(train_set, model_path, force=False):
        if force or not exists(model_path):
            vectors = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            vectors.fit_transform(train_set)
            joblib.dump(vectors, model_path)
            return vectors
        else:
            return joblib.load(model_path)
