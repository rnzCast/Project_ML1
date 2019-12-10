from sklearn.model_selection import StratifiedKFold
import sklearn.metrics
from scipy import interp
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import sklearn


class PredictRfc:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test

    def predict_rfc(self):
        self.pipe = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=1,
                                                                      min_samples_leaf=1,
                                                                      min_samples_split=2,
                                                                      n_estimators=1000,
                                                                      ))
        self.pipe.fit(self.X_train, self.y_train)
        y_pred = self.pipe.predict(self.X_test)
        confmat = confusion_matrix(y_true=self.y_test, y_pred=y_pred)
        print(confmat)

        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j],
                        va='center',
                        ha='center')

        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.show()

        """Optimizing the precision and recall of a classification model"""
        return [print('Precision: %.3f' % precision_score(y_true=self.y_test, y_pred=y_pred)),
                print('Recall: %.3f' % recall_score(y_true=self.y_test, y_pred=y_pred)),
                print('F1: %.3f' % f1_score(y_true=self.y_test, y_pred=y_pred)),self.plot_roc()]


    def plot_roc(self):
        cv = list(StratifiedKFold(n_splits=3,
                                  random_state=1).split(self.X_train, self.y_train))

        fig = plt.figure(figsize=(7, 5))

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []

        for i, (train, test) in enumerate(cv):
            probas = self.pipe.fit(self.X_train[train],
                                 self.y_train[train]).predict_proba(self.X_train[test])

            fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.y_train[test],
                                                             probas[:, 1],
                                                             pos_label=1)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = sklearn.metrics.auc(fpr, tpr)
            plt.plot(fpr,
                     tpr,
                     label='ROC fold %d (area = %0.2f)'
                           % (i + 1, roc_auc))

        plt.plot([0, 1],
                 [0, 1],
                 linestyle='--',
                 color=(0.6, 0.6, 0.6),
                 label='random guessing')

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
        plt.plot([0, 0, 1],
                 [0, 1, 1],
                 linestyle=':',
                 color='black',
                 label='perfect performance')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.legend(loc="lower right")

        return [plt.tight_layout(), plt.show()]
