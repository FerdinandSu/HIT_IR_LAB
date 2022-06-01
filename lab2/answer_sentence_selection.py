from answer_sentence_selector import AnswerSentenceSelector

# 是否验证模型
run_validate = False

# 是否处理测试集
run_predict = True

selector = AnswerSentenceSelector()

if run_validate:
    print('Validating......')
    accuracy = selector.validate()
    print('Accuracy: ：%.4f%%' % (accuracy * 100))
if run_predict:
    print('Classifying......')
    selector.predict(3)
    print('Done.')
