'''
问题分类器，TF-IDF向量+逻辑斯蒂回归方式实现
各部分数据和模型是懒加载的
'''

import config
from model_io import cut_text
from tf_idf_izer import TfIdfizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import joblib

from os.path import exists

class QuestionClassifier(TfIdfizer):
    def __init__(self, train_set_replication=1):
        TfIdfizer.__init__(self, lambda: self.train_data,config.question_classification_tf_idf_vectors_path)
        self.__train_data = None
        self.__train_label = None
        self.__train_data_vec = None
        self.__model = None
        self.__train_set_replication = train_set_replication

    def run(self, origin: list):
        test_data_vec = self.tf_idf_ize(origin)
        return self.model.predict(test_data_vec)

    def validate(self):
        validate_data, validate_label = self.load_question_data_set(
            config.validate_question_path)
        validate_data_vec = self.tf_idf_ize(validate_data)
        return self.model.score(validate_data_vec, validate_label)

    @property
    def train_data_vec(self):
        if self.__train_data_vec is None:
            self.__train_data_vec = self.tf_idf_ize(
                self.train_data)
        return self.__train_data_vec

    def load_question_data_set(self, path: str, replication=1):
        labels = []
        data = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) > 1:
                    [label, line] = line.strip().split('\t')
                    for _ in range(replication):
                        data.append(
                            ' '.join(cut_text(line, self._stop_words)))
                        labels.append(label)
        return data, labels

    @property
    def model(self):
        if self.__model == None:
            print('Lazy Load: logistic model')
            self.__model = self.__ensure_logistic_regression(
                self.train_data_vec, self.train_label)
        return self.__model

    @property
    def train_data(self):
        if self.__train_data == None:
            print('Lazy Load: classifier train set')
            (self.__train_data, self.__train_label) = self.load_question_data_set(
                config.train_question_path, self.__train_set_replication)
        return self.__train_data

    @property
    def train_label(self):
        if self.__train_label == None:
            print('Lazy Load: classifier train set')
            (self.__train_data, self.__train_label) = self.load_question_data_set(
                config.train_question_path, self.__train_set_replication)
        return self.__train_label

    @staticmethod
    def __ensure_logistic_regression(train_data, train_label, force=False):
        if force or not exists(config.logistic_regression_path):
            print('Training: Logistic Regression.')
            model = LogisticRegression(max_iter=1000)
            # 默认解法：L-BFGS算法
            param_grid = [
                {'C': [1, 10, 100,  1000,  10000,  100000]}]
            # 利用基于交叉验证的网格搜索算法自动调参
            grid_search = GridSearchCV(
                model, param_grid, cv=3, n_jobs=-1)
            best_param = grid_search.fit(
                train_data, train_label)
            # 自动调参，10轮无长进自动停止
            joblib.dump(best_param.best_estimator_,
                        config.logistic_regression_path, compress=3)
            print('Completed, params:', best_param.best_params_)
            return best_param.best_estimator_
        else:
            return joblib.load(config.logistic_regression_path)
