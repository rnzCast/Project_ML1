from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler




def classifer_dict():

        clfs = {'lr': LogisticRegression(random_state=0),
                'mlp': MLPClassifier(random_state=0),
                'dt': DecisionTreeClassifier(random_state=0),
                'rf': RandomForestClassifier(random_state=0),
                'xgb': XGBClassifier(seed=0)}

        return clfs


def pipeline_dict():
        pipe_clfs = {}

        for name, clf in clfs.items():
                # Implement me
                pipe_clfs[name] = Pipeline([('StandardScaler', StandardScaler()), ('clf', clf)])


        return pipe_clfs








