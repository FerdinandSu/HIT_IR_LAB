"""
基于规则方法实现答案句抽取.
"""
import json
from unittest import result
import config
from strange_json import array_to_strange_json
from answer_sentence_selection import test_ans_path as test_select_path
from question_classifier import QuestionClassifier
from extern_wheels import *
from model_io import cut_and_pos_text,cut_text
import re

test_span_path = './answer_span_selection/test_predict.json'  # 处理分号后的答案


class AnswerSpanSelector(object):
    @staticmethod
    def _get_hum_ans(ans_lst):
        human_types={"nr","nt","nz","nw"}
        ans, res = ''.join(ans_lst), ''
        tags = cut_and_pos_text(ans_lst)
        for (idx, (_, type)) in enumerate(tags):
            if (type in human_types) and ans_lst[idx] not in res:
                res += ans_lst[idx]
        return res if res else ans

    @staticmethod
    def _get_loc_ans(ans_lst):
        ans, res = ''.join(ans_lst), ''
        tags = cut_and_pos_text(ans_lst)
        for idx, tag in enumerate(tags):
            if (tag == 'ns' or tag == 'f' or tag == 's') and ans_lst[idx] not in res:
                res += ans_lst[idx]
        return res if res else ans

    @staticmethod
    def _get_num_ans(ans_lst: list):
        ans, res_lst = ''.join(ans_lst), []
        tags = cut_and_pos_text(ans_lst)
        for idx, tag in enumerate(tags):
            if tag == 'm' and idx < len(tags) - 1 and tags[idx + 1] == 'q':
                res_lst.append(ans_lst[idx] + ans_lst[idx + 1])
        return res_lst[0] if res_lst else ans

    # query_type ref "哈工大 IR 研究室问题分类体系 ver2.doc"
    @staticmethod
    def _get_time_ans(query_type, ans_lst: list):
        ans, res_lst = ''.join(ans_lst), []
        if query_type == 'TIME_YEAR':
            res_lst = re.findall(r'\d{2,4}年', ans)
        elif query_type == 'TIME_MONTH':  # xx月或x月
            res_lst = re.findall(r'\d{1,2}月', ans)
        elif query_type == 'TIME_DAY':
            res_lst = re.findall(r'\d{1,2}日}', ans)
        elif query_type == 'TIME_WEEK':
            res_lst = re.findall(r'((周|星期|礼拜)[1-7一二三四五六日])', ans)
            res_lst = [item[0] for item in res_lst]
        elif query_type == 'TIME_RANGE':
            # xxxx年到xxxx年或者xxxx年-xxxx年
            res_lst = re.findall(r'\d{2,4}[年]?[-到至]\d{2,4}[年]?', ans)
        else:
            res_lst = re.findall(
                r'\d{1,4}[年/-]\d{1,2}[月/-]\d{1,2}[日号]?', ans)  # 年月日
            if res_lst:
                res_lst = re.findall(r'\d{1,4}[年/-]\d{1,2}月?', ans)  # 年月
            if not res_lst:
                res_lst = re.findall(r'\d{1,2}[月/-]\d{1,2}[日号]?', ans)  # 月日
            if not res_lst:
                res_lst = re.findall(r'\d{2,4}年', ans)
            if not res_lst:
                res_lst = re.findall(r'\d{1,2}月', ans)
        return res_lst[0] if res_lst else ans  # 当前默认选择第一个数字 todo 优化选取

    @staticmethod
    def get_ans(query_type: str, ans_lst: list) -> str:
        if query_type.startswith('HUM'):
            res = AnswerSpanSelector._get_hum_ans( ans_lst)
        elif query_type.startswith('LOC'):
            res = AnswerSpanSelector._get_loc_ans(ans_lst)
        elif query_type.startswith('NUM'):
            res = AnswerSpanSelector._get_num_ans(ans_lst)
        elif query_type.startswith('TIME'):
            res = AnswerSpanSelector._get_time_ans(query_type, ans_lst)
        else:
            res = ''.join(ans_lst)
        for char in [':', '：']:
            if char in res:
                res = res.split(char)[1]
        return res

    def run():
        with open(config.answer_selected_test_result_path,'r',encoding='utf-8') as f:
            result=json.load(f)
        for item in result:
            item['answer'] = AnswerSpanSelector.get_ans(
                 item['label'], item['answer_sentence'][0])
        with open(config.answer_span_test_result_path,'w',encoding='utf-8') as f:
            json.dump(f,result)
        array_to_strange_json(config.final_ans_path,result)

    def validate():
        classifier=QuestionClassifier(3)
        classifier.run()
        res_lst, bleu_val, predict_lst, truth_lst = get_train_labels(), 0, [], []
        for item in res_lst:
            actual = item['answer_sentence'][0]
            ans_lst, expected = cut_text(
                item), item['answer']
            predict_val = get_ans(item['question'], item['label'], ans_lst)
            bleu = bleu1(predict_val, expacted)
            bleu_val += bleu
            predict_lst.append(predict_val)
            truth_lst.append(expacted)
        return bleu_val / len(res_lst), exact_match(predict_lst, truth_lst)


def main():
    print('*' * 100 + '\n正在对训练集进行答案抽取...')
    bleu_val, exact_val = evaluate()
    print('训练集上平均bleu值为{}\t精确匹配的准确率为{}'.format(bleu_val, exact_val))
    print('*' * 100 + '\n正在对测试集进行答案抽取...')
    predict()
    print('答案抽取完成...\n' + '*' * 100)


if __name__ == '__main__':
    main()
