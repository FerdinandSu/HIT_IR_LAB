'''
使用逻辑斯蒂回归和支持向量机的问题分类
'''

from regex import F
import config
from model_io import process_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from strange_json import strange_json_to_array
import joblib
import json
from os.path import exists

lr_model_path, tf_idf_path = './question_classification/lr_model', './question_classification/tf_idf'
test_label_path = './question_classification/test_predict.json'


def load_question_data_set(path: str):
    labels = []
    data = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) > 1:
                [label, line] = line.strip().split('\t')
                data.append(' '.join(process_text(line)))
                labels.append(label)
    return data, labels


def ensure_tf_idf_vectors(train_set, force=False):
    if force or not exists(config.tf_idf_vectors_path):
        vectors = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        vectors.fit_transform(train_set)
        joblib.dump(vectors, config.tf_idf_vectors_path)
        return vectors
    else:
        return joblib.load(config.tf_idf_vectors_path)


def ensure_logistic_regression(train_data, train_label, force=False):
    if force or not exists(config.logistic_regression_path):
        print('正在通过网格搜索获取最佳模型参数...')
        model = LogisticRegression(max_iter=400)
        # 默认解法：L-BFGS算法
        param_grid = [
            {'C': [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]}]
        # 利用基于交叉验证的网格搜索算法自动调参
        grid_search = GridSearchCV(
            model, param_grid, cv=3, n_jobs=-1)
        best_param = grid_search.fit(
            train_data, train_label, early_stopping_rounds=10)
        # 自动调参，19轮无长进自动停止
        joblib.dump(best_param.best_estimator_,
                    config.logistic_regression_path, compress=3)
        print('合理的超参数为', best_param.best_params_)
    else:
        return joblib.load(config.logistic_regression_path)


def get_train_labels():  # 将train.json文件中的所有问题分类
    tf_idf_vec = joblib.load(tf_idf_path)
    lr = joblib.load(lr_model_path)
    res_lst = read_json('./preprocessed/train_preprocessed.json')
    x_data = [' '.join(item['question']) for item in res_lst]
    y_data = lr.predict(tf_idf_vec.transform(x_data))
    for item, label in zip(res_lst, y_data):
        item['label'] = label
    return res_lst


def main():
    print('*' * 100 + '\n正在加载VSM模型和LR逻辑回归模型...')
    train_data, train_label = load_question_data_set(
        config.train_question_path)
    validate_data, validate_label = load_question_data_set(
        config.validate_question_path)
    tf_idf_vec = ensure_tf_idf_vectors(train_data)
    # 转换为TF-IDF向量
    train_data_vec = tf_idf_vec.transform(        train_data)
    validate_data_vec = tf_idf_vec.transform(validate_data)
    lr = ensure_logistic_regression(train_data_vec, train_label)
    accuracy=lr.score(validate_data_vec, validate_label)
    print('LR模型准确率：%.4f%%' % (accuracy * 100))

    print('*' * 100 + '\n正在对测试集进行问题类别预测...')
    test_data_set = strange_json_to_array(config.test_data_path)  # 对测试集的问题进行类别预测
    test_data = [' '.join(item['question']) for item in test_data_set]
    test_data_vec=tf_idf_vec.transform(test_data)
    test_label_result = lr.predict(test_data_vec)
    for item, label in zip(test_data_set, test_label_result):
        item['label'] = label
    with open(config.question_classfication_result_path,'w',encoding='utf-8') as f:
        json.dump(f,test_data_set)
    write_json(test_label_path, test_data_set)
    print('预测结束\n' + '*' * 100)


if __name__ == '__main__':
    main()
