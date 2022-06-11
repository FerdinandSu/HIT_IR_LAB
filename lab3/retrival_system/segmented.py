
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
    with open('lab1/segment/stopwords(new).txt', 'r', encoding='utf-8') as stop_words_file:
        stop_words = set(stop_words_file.read().split('\n'))
    return stop_words


def cut_and_pos_text(origin_line: str, stop_words: set) -> list:
    processed_line = pos_seg.cut(origin_line, use_paddle=use_paddle)
    return [(word, type) for (word, type) in processed_line if word not in stop_words]


def cut_and_pos_text_array(origin_line: list, stop_words: set) -> list:
    return cut_and_pos_text(''.join(origin_line), stop_words)


def cut_text(origin_line: str, stop_words: set) -> list:
    # (string[][],?) ltp.seg(string[])
    # processed_line, _ = ltp.seg([origin_line]) ltp也太慢了.......
    #processed_line = processed_line[0]
    processed_line = jieba.cut(origin_line, use_paddle=use_paddle)
    return [word for word in processed_line if word not in stop_words]


origin_path = 'lab1/results/craw.json'
target_path = 'lab1/results/preprocessed_pro.json'


def ensure_segmented(force: bool = False):

    print('预处理：分词')
    target = []
    stop_words = ensure_stop_words()
    with open(origin_path, "r", encoding="utf-8") as f:
        origin = json.load(f)
    total = len(origin)
    progress = 0
    for item in origin:
        if not item['title']:
            continue
        item['segmented_title'] = cut_text(item['title'], stop_words)
        item['segmented_parapraghs']=cut_text(item['paragraghs'],stop_words)
        item['segmented_file_name']={fn:cut_text(fn,stop_words) for fn in item['file_name']}
        target.append(item)
        progress += 1
        if progress % 100 == 0:
            print('进度: %.2f%%' % (progress * 100/total))
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(target, f, ensure_ascii=False)


ensure_segmented()