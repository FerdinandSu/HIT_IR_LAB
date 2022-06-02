"""
基于规则方法实现答案句抽取.
"""
import json
import config
from stop_words_provider import StopWordsProvider
from strange_json import array_to_strange_json, strange_json_to_array
from question_classifier import QuestionClassifier
from extern_wheels import *
from model_io import cut_and_pos_text,cut_text
import re

test_span_path = './answer_span_selection/test_predict.json'  # 处理分号后的答案


class AnswerSpanSelector(StopWordsProvider):
    def __init__(self):
        StopWordsProvider.__init__(self)

    def __select_answer_human(self,answer_sentence):
        # 人名/机构名，ignore 任务描述(就五个......)
        human_types={"nr","nt","nz","nw"}
        result = ''
        tags = cut_and_pos_text(answer_sentence,self._stop_words)
        for (idx, (_, type)) in enumerate(tags):
            if (type in human_types) and answer_sentence[idx] not in result:
                result += answer_sentence[idx]
        return result if result else ''.join(answer_sentence)

    def __select_answer_location(self,answer_sentence):
        result =  ''
        tags = cut_and_pos_text(answer_sentence,self._stop_words)
        for idx, tag in enumerate(tags):
            if (tag == 'ns' or tag == 'f' or tag == 's') and answer_sentence[idx] not in result:
                result += answer_sentence[idx]
        return result if result else ''.join(answer_sentence)

    def __select_answer_number(self,answer_sentence):
        results = []
        tags = cut_and_pos_text(answer_sentence,self._stop_words)
        for idx, tag in enumerate(tags):
            if tag == 'm' and idx < len(tags) - 1 and tags[idx + 1] == 'q':
                results.append(answer_sentence[idx] + answer_sentence[idx + 1])
        return results[0] if results else ''.join(answer_sentence)

    # query_type ref "哈工大 IR 研究室问题分类体系 ver2.doc"
    def __select_answer_time(self,query_type, answer_sentence: list):
        ans, results = ''.join(answer_sentence), []
        if query_type == 'TIME_YEAR':
            results = re.findall(r'\d{2,4}年', ans)
        elif query_type == 'TIME_MONTH':  # xx月或x月
            results = re.findall(r'\d{1,2}月', ans)
        elif query_type == 'TIME_DAY':
            results = re.findall(r'\d{1,2}日}', ans)
        elif query_type == 'TIME_WEEK':
            results = re.findall(r'((周|星期|礼拜)[1-7一二三四五六日])', ans)
            results = [item[0] for item in results]
        elif query_type == 'TIME_RANGE':
            results = re.findall(r'\d{2,4}[年]?[-到至]\d{2,4}[年]?', ans)
        else:
            results = re.findall(
                r'\d{1,4}[年/-]\d{1,2}[月/-]\d{1,2}[日号]?', ans)
            if results:
                results = re.findall(r'\d{1,4}[年/-]\d{1,2}月?', ans)
            if not results:
                results = re.findall(r'\d{1,2}[月/-]\d{1,2}[日号]?', ans)
            if not results:
                results = re.findall(r'\d{2,4}年', ans)
            if not results:
                results = re.findall(r'\d{1,2}月', ans)
        return results[0] if results else ans

    def select_answer(self,query_type: str, answer_sentence: str) -> str:
        if query_type.startswith('HUM'):
            res = self.__select_answer_human( answer_sentence)
        elif query_type.startswith('LOC'):
            res = self.__select_answer_location(answer_sentence)
        elif query_type.startswith('NUM'):
            res = self.__select_answer_number(answer_sentence)
        elif query_type.startswith('TIME'):
            res = self.__select_answer_time(query_type, answer_sentence)
        else:
            res = ''.join(answer_sentence)
        for char in [':', '：']:
            if char in res:
                res = res.split(char)[1]
        return res
    def run(self):
        with open(config.answer_selected_test_result_path,'r',encoding='utf-8') as f:
            result=json.load(f)
        for item in result:
            item['answer'] = self.select_answer(
                 item['class'], ''.join( item['answer_sentence'][0]))
        with open(config.answer_span_test_result_path,'w',encoding='utf-8') as f:
            json.dump(result,f)
        array_to_strange_json(config.final_ans_path,result)
    def validate(self):
        classifier=QuestionClassifier(3)
        with open(config.train_preprocessed_path ,'r',encoding='utf-8') as f:
            train_data_set = json.load(f)
        train_data = [' '.join(item['question']) for item in train_data_set]
        train_label_result = classifier.run(train_data)
        bleu=0
        f1=0
        expected_list=[]
        actual_list=[]
        for item, label in zip(train_data_set, train_label_result):
            expected = item['answer']
            actual = self.select_answer(label,item['answer_sentence'][0]) 
            bleu += bleu1(actual, expected)
            (_,_, this_f1)=precision_recall_f1(actual, expected)
            f1+=this_f1
            expected_list.append(expected)
            actual_list.append(actual)
        bleu /= len(train_data_set)
        f1 /= len(train_data_set)
        x_match=exact_match(actual_list, expected_list)
        return bleu,f1,x_match

