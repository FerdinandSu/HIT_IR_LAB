from answer_span_selector import AnswerSpanSelector

# 是否验证模型
run_validate = True
# Average BLEU=0.40, F1=0.48; Exact Match=14.4619%

# 是否处理测试集
run_predict = False

selector=AnswerSpanSelector()

if run_validate:
    print('Validating...')
    bleu, f1, x = selector.validate()
    print('Average BLEU=%.2f, F1=%.2f; Exact Match=%.4f%%' % (bleu, f1, x*100))

if run_predict:
    print('Predicting Test Set...')
    selector.run()
    print('Done.')
