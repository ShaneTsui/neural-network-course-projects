from scikit.pipeline.cafe_pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

n_components = [20]
learning_rates = [0.5]
n_epoches = 10
n_repeats = 5
for n_comp in n_components:
    for lr in learning_rates:
        pipeline = Pipeline(facial_expressions=['ht', 'a', 's'], classifier_type="softmax")

        pipeline.build(n_components=n_comp, learning_rate=lr, n_epoches=n_epoches, batch_size=None, n_repeats=n_repeats)
        pipeline.run()

        # plot errors
        pipeline.records.plt_losses(n_components=n_comp, lr=lr, n_epoches=n_epoches)
        pipeline.records.show_accuracies()
