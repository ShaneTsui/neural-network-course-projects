from scikit.pipeline.cafe_pipeline import Pipeline
from utils.display import *
import warnings
warnings.filterwarnings('ignore')

# emotion_dict = {"h": "happy", "ht": "happy with teeth", "m": "maudlin",
#     "s": "surprise", "f": "fear", "a": "anger", "d": "disgust", "n": "neutral"}

# Display six faces

n_components = [10, 20, 40]
learning_rates = [0.2]
n_epoches = 50
n_repeats = 10
classifier_type="softmax"

for n_comp in n_components:
    for lr in learning_rates:
        pipeline = Pipeline(facial_expressions=['ht', 'a', 's', 'm', 'd', 'f'], classifier_type=classifier_type)
        pipeline.build(n_components=n_comp, learning_rate=lr, n_epoches=n_epoches, batch_size=None, n_repeats=n_repeats)
        pipeline.run()
        # plot errors
        pipeline.visualize_weights()
        pipeline.records.plt_losses(n_components=n_comp, std_idx=[10, 20, 30, 40, 50], lr=lr, n_epoches=n_epoches)
        pipeline.records.show_accuracies()
        cm = pipeline.records.print_confusion(classifier_type, n_components, lr, n_epoches, pipeline.classifier.label_set)#, pipeline.classifier.confusion_matrix)
        print(cm)
# print(test(np.array([[1,2,3],[4,5,6]]), np.array([[1,2],[2,3],[3,4]]).T))
# print((np.array([[1,2,3]]) * np.array([[1,2],[2,3],[3,4]]).T))
# print(np.sum((np.array([[1,2,3]]) * np.array([[1,2],[2,3],[3,4]]).T), axis=1))
