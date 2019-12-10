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
                        'clf__C': C_range
                        # 'max_iter': [1000],
                        # 'random_state': 100
                               },

                        {'clf__multi_class': ['multinomial'],
                        'clf__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                        'clf__C': C_range
                        # 'max_iter': [1000],
                        #  'random_state': 100
                         }]
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
class HyperparameterTuning:
        def __init__(self, pipe_clfs, param_grids, X, y):
                self.pipe_clfs = pipe_clfs
                self.param_grids = param_grids
                self.X = X
                self.y = y

        def best_parameters_gs(self):
                best_score_param_estimators = []

                # For each classifier
                for name in self.pipe_clfs.keys():
                        gs = GridSearchCV(estimator=self.pipe_clfs[name],
                                          param_grid=self.param_grids[name],
                                          scoring='accuracy',
                                          n_jobs=1,
                                          iid=False,
                                          cv=StratifiedKFold(n_splits=10,
                                                             shuffle=True,
                                                             random_state=0
                                                             )
                                          )
                        gs = gs.fit(self.X, self.y)
                        best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])
                        return best_score_param_estimators


class HyperparameterOneModel:
        def __init__(self, pipe_clfs, param_grids, X, y, modelname):
                self.pipe_clfs = pipe_clfs
                self.param_grids = param_grids
                self.X = X
                self.y = y
                self.modelname = modelname

        def tune_one_model(self):
                best_score_param_estimators = []
                gs = GridSearchCV(estimator=self.pipe_clfs[self.modelname],
                                  param_grid=self.param_grids[self.modelname],
                                  scoring='accuracy',
                                  n_jobs=1,
                                  iid=False,
                                  cv=StratifiedKFold(n_splits=10,
                                                     shuffle=True,
                                                     # random_state=0
                                                     ))
                gs = gs.fit(self.X, self.y)
                best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])
                return best_score_param_estimators


class ModelSelection:
    def __init__(self, best_score_param_estimators):
        self.best_score_param_estimators = best_score_param_estimators

    def select_best(self):
        self.best_score_param_estimators = sorted(self.best_score_param_estimators, key=lambda x: x[0], reverse=True)
        return self.best_score_param_estimators

    def print_models_params(self):
        for best_score_param_estimator in self.best_score_param_estimators:
            print([best_score_param_estimator[0], best_score_param_estimator[1], type(best_score_param_estimator[2].named_steps['clf'])], end='\n\n')


















