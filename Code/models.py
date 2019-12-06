from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold



"""CREATE DICTIONARY OF CLASSIFIERS"""
def classifer_dict():

        clfs = {'lr': LogisticRegression(random_state=0),
                'mlp': MLPClassifier(random_state=0),
                'dt': DecisionTreeClassifier(random_state=0),
                'rf': RandomForestClassifier(random_state=0),
                'xgb': XGBClassifier(seed=0)}

        return clfs

"""CREATE DICTIONARY OF PIPELINE"""
def pipeline_dict(clfs):
        pipe_clfs = {}

        for name, clf in clfs.items():
                pipe_clfs[name] = Pipeline([('StandardScaler', StandardScaler()), ('clf', clf)])


        return pipe_clfs

"""PARAMETER GRIDS"""
def create_param_grids():
        param_grids = {}

        """PARAMETER GRID FOR LOGISTIC REGRESSION"""
        C_range = [10 ** i for i in range(-4, 5)]
        param_grid_log_reg = [{'clf__multi_class': ['ovr'],
                        'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                        'clf__C': C_range},

                        {'clf__multi_class': ['multinomial'],
                        'clf__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                        'clf__C': C_range}]

        param_grids['lr'] = param_grid_log_reg


        """PARAMETER GRID FOR MULTILAYER PERCEPTRON"""
        param_grid_mlp = [{'clf__hidden_layer_sizes': [10, 100],
                       'clf__activation': ['identity', 'logistic', 'tanh', 'relu']}]

        param_grids['mlp'] = param_grid_mlp


        """PARAMETER GRID FOR DECISION TREE"""
        param_grid_dt = [{'clf__min_samples_split': [2, 10, 30],
                       'clf__min_samples_leaf': [1, 10, 30]}]

        param_grids['dt'] = param_grid_dt


        """PARAMETER GRID FOR RANDOM FOREST"""
        param_grid_rf = [{'clf__n_estimators': [10, 100, 1000],
                       'clf__min_samples_split': [2, 10, 30],
                       'clf__min_samples_leaf': [1, 10, 30]}]

        param_grids['rf'] = param_grid_rf


        """PARAMETER GRID FOR XGBOOST"""
        param_grid_xgb = [{'clf__eta': [10 ** i for i in range(-4, 1)],
                       'clf__gamma': [0, 10, 100],
                       'clf__lambda': [10 ** i for i in range(-4, 5)]}]

        param_grids['xgb'] = param_grid_xgb


        return param_grids



"""HYPERPARAMETER TUNING"""
class Hyperparameter_Tuning:
        def __init__(self, pipe_clfs, param_grids):
                self.pipe_clfs = pipe_clfs
                self.param_grids = param_grids

        def best_parameters_gs(self):
                # The list of [best_score_, best_params_, best_estimator_]
                best_score_param_estimators = []

                # For each classifier
                for name in pipe_clfs.keys():
                        # GridSearchCV
                        gs = GridSearchCV(estimator=pipe_clfs[name],
                                          param_grid=param_grids[name],
                                          scoring='accuracy',
                                          n_jobs=1,
                                          iid=False,
                                          cv=StratifiedKFold(n_splits=10,
                                                             shuffle=True,
                                                             random_state=0))

                        # Fit the pipeline
                        gs = gs.fit(X, y)

                        # Update best_score_param_estimators
                        best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

                        return best_score_param_estimators



class Model_Selection:
        def __init__(self, best_score_param_estimators):
                self.best_score_param_estimators = best_score_param_estimators


        def select_best(self):
                # Sort best_score_param_estimators in descending order of the best_score_
                best_score_param_estimators = sorted(self.best_score_param_estimators, key=lambda x: x[0], reverse=True)

                return best_score_param_estimators


















