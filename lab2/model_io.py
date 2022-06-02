import config
from model import Model
from strange_json import strange_json_to_array
from os.path import exists
import json
#from ltp import LTP
from math import log10
import joblib
import jieba
import jieba.posseg as pos_seg

#ltp = LTP()

# 使用Paddle-Paddle
use_paddle = True

# 并行数，0为不并行；Windows下不可并行。
use_parallel = 0

paddle_ready = False

if use_parallel > 0:
    jieba.enable_parallel(use_parallel)

if use_paddle:
    jieba.enable_paddle()


def ensure_stop_words() -> set:
    with open(config.stop_words_path, 'r', encoding='utf-8') as stop_words_file:
        stop_words = set(stop_words_file.read().split('\n'))
    return stop_words


def cut_and_pos_text(origin_line: str, stop_words: set) -> list:
    processed_line = pos_seg.cut(origin_line, use_paddle=use_paddle)
    return [(word, type) for (word, type) in processed_line if word not in stop_words]

def cut_and_pos_text_array(origin_line: list, stop_words: set) -> list:
    return cut_and_pos_text(''.join(origin_line),stop_words)

def cut_text(origin_line: str, stop_words: set) -> list:
    # (string[][],?) ltp.seg(string[])
    # processed_line, _ = ltp.seg([origin_line]) ltp也太慢了.......
    #processed_line = processed_line[0]
    processed_line = jieba.cut(origin_line, use_paddle=use_paddle)
    return [word for word in processed_line if word not in stop_words]

def ensure_preprocessed(force: bool = False):
    if force or not exists(config.test_result_path):
        model=ensure_model(force)
        test_set = ensure_test(force)
        model.run(model.measure_inner_product, test_set, config.test_result_path)
    with open(config.test_result_path, 't', encoding='utf-8') as f:
        return json.load(f)
    

def ensure_segmented(force: bool = False):
    if force or not exists(config.segmented_data_path):
        print('预处理：分词')
        seg_passages = {}
        stop_words = ensure_stop_words()
        origin = strange_json_to_array(config.origin_data_path)
        total = len(origin)
        progress = 0
        for item in origin:
            seg_passages[item['pid']] = [
                cut_text(line, stop_words) for line in item['document']]
            progress += 1
            if progress % 100 == 0:
                print('进度: %.2f%%' % (progress * 100/total))
        with open(config.segmented_data_path, "w", encoding="utf-8") as f:
            json.dump(seg_passages, f, ensure_ascii=False)
        return seg_passages
    else:
        with open(config.segmented_data_path, "r", encoding="utf-8") as f:
            return json.load(f)

def ensure_model(force: bool = False) -> Model:
    '''
    向量空间模型，ref PPT 2-30
    '''
    if force or not exists(config.model_path):
        print('预处理：训练')
        segmented = ensure_segmented(force)
        term_freq = {}  # 词频Dictionary<id,Dictionary<word,int>>
        doc_freq = {}  # 文档频率Dictionary<id,word>
        total = len(segmented)
        progress = 0
        for id, passage in segmented.items():
            term_freq[id] = {}
            for word in [word for sentence in passage for word in sentence]:
                if word not in term_freq[id]:
                    term_freq[id][word] = 0
                    doc_freq[word] = doc_freq[word] + \
                        1 if word in doc_freq else 1  # 计算每一个词项的df，保存在words中
                term_freq[id][word] += 1
                progress += 1
                if progress % 100 == 0:
                    print('进度: %.2f%%' % (progress * 100/total))
        doc_count = len(segmented)
        value_idf = {word: log10(doc_count/df)
                     for (word, df) in doc_freq.items()}  # df -> idf
        weight = {id: {word: (1+log10(tf))*value_idf[word] for word, tf in tf_dic.items(
        )} for id, tf_dic in term_freq.items()}  # tf * idf
        joblib.dump((weight, value_idf), config.model_path, 3)
    else:
        (weight, value_idf) = joblib.load(config.model_path)

    return Model(weight, value_idf)

def preprocess_data(origin: str, target: str) -> list:
    stop_words = ensure_stop_words()
    target_set = strange_json_to_array(origin)
    for item in target_set:
        item['question'] = cut_text(item['question'], stop_words)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(target_set, f, ensure_ascii=False)
    return target_set


def ensure_preprocess_data(origin: str, target: str, force=False) -> list:
    if force or not exists(target):
        return preprocess_data(origin, target)
    else:
        with open(target, "r", encoding="utf-8") as f:
            target_set = json.load(f)
        return target_set


def ensure_train(force=False) -> list:
    return ensure_preprocess_data(config.train_data_path,
                                  config.train_preprocessed_path, force)


def ensure_test(force=False) -> list:
    return ensure_preprocess_data(config.test_data_path,
                                  config.test_preprocessed_path, force)
