'''
用于适配奇怪格式的Json文件，
啊，就是一行一个Json那种
（所以说为啥不直接用数组呢😅
'''

from json import loads, dumps


def strange_json_to_array(path: str) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [loads(line) for line in f]


def array_to_strange_json(path: str, array: list) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join([dumps(element, ensure_ascii=False)
                for element in array]))
