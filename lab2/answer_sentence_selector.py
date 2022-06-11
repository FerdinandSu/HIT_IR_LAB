'''
候选答案句排序器
'''

import json
import numpy as np
from my_wheels import *
from tf_idf_izer import TfIdfizer
from model_io import cut_and_pos_text_array, cut_text, ensure_segmented
from distance import quick_levenshtein
from strange_json import strange_json_to_array
from sklearn.model_selection import train_test_split
import similarities
from os.path import exists
import svm_rank
import config


class AnswerSentenceSelector(TfIdfizer):
    def __init__(self):
        TfIdfizer.__init__(
            self, lambda: [' '.join([str for sub_item in item for str in sub_item]) for (_, item) in self.segmented.items()], config.answer_selected_tf_idf_vectors_path)
        self.__segmented = None

    @property
    def segmented(self):
        if self.__segmented == None:
            self.__segmented = ensure_segmented()
        return self.__segmented

    def __as_vector(self, seq: list) -> np.ndarray:
        vec = self.tf_idf_ize([' '.join(seq)]).toarray()
        return np.array(vec[0])

    def get_features(self, question: list, answer: list):
        """
        实词词性，参考 https://ltp.readthedocs.io/zh_CN/latest/appendix.html#id3
        """
        answer_uni_gram_set = set(answer)
        answer_bi_gram_set = {
            prefix + suffix for (prefix, suffix) in zip(answer[:-1], answer[1:])}
        answer_size = len(answer)
        if answer_size == 0:
            return []
        # 参阅 https://github.com/fxsjy/jieba 和 https://www.biaodianfu.com/pos-tagging-set.html
        function_word_types = {'d', 'p', 'c', 'u', 'xc', 'w'}
        question_sentence = ''.join(question)
        answer_sentence = ''.join(answer)
        answer_sentence_size = len(answer_sentence)
        if answer_sentence_size == 0:
            return []
        result = []
        result.append('1:%d' % answer_size)  # 总长度
        result.append('2:%d' % count_if(cut_and_pos_text_array(answer, self._stop_words),
                                        lambda x: x[1] not in function_word_types))  # 答案句实词数
        result.append('3:%d' % abs(len(question) - len(answer)))  # 问句与答案句长度差异
        result.append('4:%f' % (count_if(
            [prefix + suffix for (prefix, suffix)
             in zip(question[:-1], question[1:])],
            lambda w: w in answer_bi_gram_set)/(1+answer_size)))  # 二元词共现率
        result.append('5:%f' % (count_if(
            question,
            lambda w: w in answer_uni_gram_set)/answer_size))  # 一元词共现率
        result.append('6:%d' % (size_of_longest_common_substring(
            question_sentence, answer_sentence)/answer_sentence_size))  # 最长公共子串长度比例
        result.append('7:%d' % (size_of_longest_common_sequence(
            question_sentence, answer_sentence)/answer_sentence_size))  # 最长公共子序列长度比例
        result.append('8:%d' % quick_levenshtein(
            question_sentence, answer_sentence))  # 编辑距离
        question_vec = self.__as_vector(question)
        answer_vec = self.__as_vector(answer)
        result.append('9:%f' % (similarities.measure_inner_product(
            question_vec, answer_vec)))  # 内积相似度
        result.append('10:%f' % (similarities.measure_cosine(
            question_vec, answer_vec)))  # 余弦相似度
        result.append('11:%f' % (similarities.measure_jaccard(
            question_vec, answer_vec)))  # jaccard相似度
        return result

    def __ensure_train_features(self, test_size=0.1, force=False):
        if force or not exists(config.answer_feature_train_path) or not exists(config.answer_feature_validate_path):
            print('Lazy load: train feature file.')
            train_set = strange_json_to_array(config.train_data_path)
            feature_dict = {}
            for item in train_set:
                qid = item['qid']
                pid = str(item['pid'])
                question = cut_text(item['question'], self._stop_words)
                answers = {' '.join(cut_text(line, self._stop_words))
                           for line in item['answer_sentence']}
                feature_dict[qid] = []

                for origin_passage in self.segmented[pid]:
                    ranking = 3 if ' '.join(origin_passage) in answers else 0
                    features = self.get_features(question, origin_passage)
                    if len(features) == 0:
                        continue
                    feature = ' '.join(features)
                    feature_dict[qid].append(
                        '%d qid:%d %s' % (ranking, qid, feature))
            all_features = [value for (_, value) in feature_dict.items()]
            # 交叉验证......
            train_features, validate_features = train_test_split(
                all_features, test_size=test_size)
            train_features.sort(key=lambda lst: int(
                lst[0].split()[1].split(':')[1]))
            validate_features.sort(key=lambda lst: int(
                lst[0].split()[1].split(':')[1]))
            with open(config.answer_feature_train_path, 'w', encoding='utf-8') as f:
                f.write(
                    '\n'.join([feature for fl in train_features for feature in fl]))

            with open(config.answer_feature_validate_path, 'w', encoding='utf-8') as f:
                f.write(
                    '\n'.join([feature for fl in validate_features for feature in fl]))

    def __predict(self, src: str, dst: str):
        self.__ensure_model()
        svm_rank.predict(src, config.answer_selector_model_path, dst)

    def __ensure_test_feature(self, force=False):
        if force or not exists(config.answer_feature_test_path):
            print('Lazy load: test feature file.')
            with open(config.question_classification_result_path, 'r', encoding='utf-8') as f:
                test_set = json.load(f)
            feature_dict = {}
            for item in test_set:  # 遍历train.json文件中的每一行query信息
                qid = item['qid']
                pid = str(item['pid'])
                question = item['question']
                feature_dict[qid] = []

                for origin_passage in self.segmented[pid]:
                    if len(origin_passage) <= 1:
                        continue
                    print(f'doing qid=={qid},origin={origin_passage}')
                    try:
                        features = self.get_features(question, origin_passage)

                    except Exception as ex:
                        print(ex)

                    if len(features) == 0:
                        continue
                    feature = ' '.join(features)
                    feature_dict[qid].append(
                        '0 qid:%d %s' % (qid, feature))
            all_features = [value for (_, value) in feature_dict.items()]
            all_features.sort(key=lambda lst: int(
                lst[0].split()[1].split(':')[1]))
            with open(config.answer_feature_test_path, 'w', encoding='utf-8') as f:
                f.write(
                    '\n'.join([feature for fl in all_features for feature in fl]))

    def __ensure_model(self, force=False):
        if force or not exists(config.answer_selector_model_path):
            print('Lazy load: Answer Selection Model')
            self.__ensure_train_features(force=force)
            print('Training......')
            svm_rank.train(config.answer_feature_train_path,
                           config.answer_selector_model_path)
            print('Training Completed.')

    def validate(self):
        print("Predicting validation set......")
        self.__predict(config.answer_feature_validate_path,
                       config.answer_selected_validate_path)
        with open(config.answer_feature_validate_path, 'r', encoding='utf-8') as expected_file, open(config.answer_selected_validate_path, 'r', encoding='utf-8') as actual_file:
            expected_set = {}
            actual_set = {}
            total=0
            correct = 0
            for expected_line, actual_line in zip(expected_file, actual_file):
                if len(expected_line) == 1:
                    continue # 去冗余内容，如“English”，“中文”，空行
                qid = int(expected_line.split()[1].split(':')[1])
                expected, actual = expected_set.get(
                    qid, []), actual_set.get(qid, [])
                expected.append((int(expected_line[0]), len(expected)))
                actual.append((float(actual_line.strip()), len(actual)))
                expected_set[qid], actual_set[qid] = expected, actual

                for qid in expected_set:
                    expected = max_with(
                        expected_set[qid], lambda item: item[0])
                    actual = max_with(actual_set[qid], lambda item: item[0])
                    total+=1
                    if expected == actual:
                        correct += 1
            return correct/total

    def predict(self, sel_num=1):
        self.__ensure_test_feature()
        svm_rank.predict(config.answer_feature_test_path,
                        config.answer_selector_model_path,
                         config.answer_selected_test_path)
        labels = {}
        with open(config.answer_feature_test_path, 'r', encoding='utf-8') as feature_file, open(config.answer_selected_test_path, 'r', encoding='utf-8') as result_file:

            for feature_line, result_line in zip(feature_file, result_file):
                if len(feature_line) == 1:
                    break
                qid = int(feature_line.split()[1].split(':')[1])
                if qid not in labels:
                    labels[qid] = []
                labels[qid].append(
                    (float(result_line.strip()), len(labels[qid])))
        with open(config.question_classification_result_path, 'r', encoding='utf-8') as f:
            test_set = json.load(f)
        for item in test_set:
            qid = item['qid']
            pid = str(item['pid'])
            rank_lst, seg_passage = sorted(
                labels[qid], key=lambda val: val[0], reverse=True), self.segmented[pid]
            item['answer_sentence'] = [seg_passage[rank[1]]
                                       for rank in rank_lst[:sel_num]]
        with open(config.answer_selected_test_result_path, 'w', encoding='utf-8') as f:
            json.dump(test_set, f)
