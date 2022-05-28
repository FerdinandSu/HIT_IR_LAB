import json
from ltp import LTP

ltp = LTP()

# work dir= project dir(即segment文件夹外侧)

stopwords_path = "segment/stopwords(new).txt"
origin_path = "results/craw.json"
target_path = "results/preprocessed_origin.json"

with open(stopwords_path, 'r', encoding='utf-8') as stopwords_file:
    stopwords = set(stopwords_file.read().split('\n'))


def process_text(origin_line: str) -> list[str]:
    # (string[][],?) ltp.seg(string[])
    processed_line, _ = ltp.seg([origin_line])
    processed_line = processed_line[0]
    return [word for word in processed_line if word not in stopwords]


def process_document(document: dict) -> dict:
    output = dict()
    output['url'] = document['url']
    output['file_name'] = document['file_name']
    output['segmented_title'] = process_text(document['title'])
    output['segmented_parapraghs'] = process_text(document['paragraghs'])
    return output


with open(origin_path, 'r', encoding='utf-8') as origin_file:
    origin = json.load(origin_file)

result = []

total_count = len(origin)
count = 0

for document in origin:
    try:
        processed_document = process_document(document)
        result.append(process_document(document))
    except:
        pass
    count += 1
    if count % 100 == 0:
        print(str(count) + '/'+str(total_count)+'\n')

with open(target_path, 'w+', encoding='utf-8') as target_file:
    json.dump(result, target_file, ensure_ascii=False)
