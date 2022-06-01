'''
向量空间模型
'''

import json
import numpy as np
from math import log10


class MeasureResult(object):
    def __init__(self, document_id, similarity):
        self.document_id = document_id
        self.similarity = similarity


class Model(object):
    def __init__(self, weight, idf):
        self.weight = weight
        self.idf = idf

    @staticmethod
    def _max_of(sequence: list) -> MeasureResult:
        max = MeasureResult(-1, 0)
        for current in sequence:
            if max.similarity < current.similarity:
                max = current
        return max

    def documents_to_vec(self, question: dict):
        document_vec = {}
        for (id, weight) in self.weight.items():
            document_vec[id] = np.array(
                [weight.get(word,0) for (word, _) in question.items()], dtype=np.double)
        return document_vec

    def question_to_vec(self, question: dict):
        return np.array([question_weight for (_, question_weight) in question.items()], dtype=np.double)

    def measure_inner_product(self, question: dict) -> MeasureResult:
        doc = self.documents_to_vec(question)
        q = self.question_to_vec(question).T
        return Model._max_of(
            [MeasureResult(id, np.dot(vec, q)) for (id, vec) in doc.items()])

    def measure_cosine(self, question: dict) -> MeasureResult:
        doc = self.documents_to_vec(question)
        q = self.question_to_vec(question).T
        q_norm = np.linalg.norm(q)
        return Model._max_of(
            [MeasureResult(id, np.dot(vec, q)/(np.linalg.norm(vec)*q_norm)) for (id, vec) in doc.items()])

    @staticmethod
    def __jaccard(doc_vec, q_vec_T, q_norm2):
        dot_prod = np.dot(doc_vec, q_vec_T)
        doc_norm2 = np.dot(doc_vec, doc_vec.T)
        return dot_prod/(doc_norm2+q_norm2-dot_prod)

    def measure_jaccard(self, question: dict) -> MeasureResult:
        doc = self.documents_to_vec(question)
        q = self.question_to_vec(question).T
        q_norm_2 = np.dot(q.T, q)
        return Model._max_of(
            [MeasureResult(id, Model.__jaccard(vec, q, q_norm_2)) for (id, vec) in doc.items()])

    @property
    def measures(self):
        return {
            'dot': self.measure_inner_product,
            'cos': self.measure_cosine,
            'jac': self.measure_jaccard
        }

    def __tf_idf(self, question: list) -> dict:
        term_freq = {}
        for word in question:
            term_freq[word] = term_freq.get(word, 0) + 1
        return {word:  (1 + log10(tf))*self.idf.get(word, 0)  # 计算问题的tf-idf
                for (word, tf) in term_freq.items()}

    def evaluate(self, measure, train_set: list) -> float:
        correct = 0
        progress = 0
        total = len(train_set)
        for item in train_set:
            ans = item['pid']
            result = measure(self.__tf_idf(item['question']))
            if result.document_id == ans:
                correct += 1
            progress += 1
            if progress % 100 == 0:
                print('进度: %.2f%%' % (progress * 100/total))
        return correct / total

    def run(self, measure, test_set: list, write_to: str):
        for item in test_set:
            item['pid'] = measure(self.__tf_idf(item['question'])).document_id
        with open(write_to, 'w', encoding='utf-8') as f:
            json.dump(test_set, f, ensure_ascii=False)
