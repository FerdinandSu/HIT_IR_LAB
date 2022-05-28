'''
使用逻辑斯蒂回归和支持向量机的问题分类
'''

from regex import F
import config
from model_io import process_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from preprocessed import test_predict_path
import joblib
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


#// TODO
def ensure_logistic_regression(x_train, y_train, force=False):  # solver选用默认的lbfgs, multi_class选用多分类问题中的multinomial
    if force or not exists(config.logistic_regression_path):
        print('正在通过网格搜索获取最佳模型参数...')
        lr = LogisticRegression(max_iter=400, n_jobs=-1)
        param_grid = [
            {'C': [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]}]
        grid_search = GridSearchCV(
            lr, param_grid, cv=3, n_jobs=-1).fit(x_train, y_train)
        joblib.dump(grid_search.best_estimator_,
                    config.logistic_regression_path, compress=3)  # 导出模型
        print('合理的超参数为', grid_search.best_params_)
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
    test_data, test_label = load_question_data_set(
        config.train_question_path)
    tf_idf_vec = ensure_tf_idf_vectors(train_data)
    train_data, test_data = tf_idf_vec.transform(
        train_data), tf_idf_vec.transform(test_data)
    lr = ensure_logistic_regression(train_data,train_label)
    print('模型准确率：%.4f%%' % (lr.score(test_data,test_label) * 100))

    print('*' * 100 + '\n正在对测试集进行问题类别预测...')
    json_lst = read_json(test_predict_path)  # 对测试集的问题进行类别预测
    x_data = [' '.join(item['question']) for item in json_lst]
    y_data = lr.predict(tf_idf_vec.transform(x_data))
    for item, label in zip(json_lst, y_data):
        item['label'] = label
    write_json(test_label_path, json_lst)
    print('预测结束\n' + '*' * 100)


if __name__ == '__main__':
    main()
