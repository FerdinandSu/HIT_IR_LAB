'''
预处理。建立向量空间模型VSM
'''

from model_io import *

run_validate=False

run_predict=True

print('Loading Model......')
model = ensure_model()
# 训练集
if run_validate:
    train_set = ensure_train()
    accuracy = model.evaluate(model.measure_inner_product, train_set)
    print('Accuracy: %.2f%%' % (accuracy * 100))
if run_predict:
    print('Predicting....')
    test_set = ensure_test()
    model.run(model.measure_inner_product, test_set, config.test_result_path)
print('Done.')
