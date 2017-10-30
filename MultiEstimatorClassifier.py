from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, f1_score, log_loss, recall_score, zero_one_loss

class MultiEstimatorClassifier(BaseEstimator):
    
    def __init__(self, ensemble=False, neuralNet=False):
        self._ensemble = ensemble
        self._neuralNet = neuralNet
        
        self._estimators = dict(
            logisticRegression = LogisticRegression(),
            kNearestNeighbors = KNeighborsClassifier(),
            linearSupportVectorClassifier = LinearSVC(),
            supportVectorClassifier = SVC(),
            decisionTree = DecisionTreeClassifier() 
        )
        
        if (ensemble):
            self._estimators.update(
                dict(
                randomForest = RandomForestClassifier(),
                adaBoost = AdaBoostClassifier(),
                gradientBoosting = GradientBoostingClassifier()
                )
            )
        
        if (neuralNet):
            self._estimators.update(dict(neuralNet = MLPClassifier()))
            
        
    def fit_transform(self, X, y=None):
        return self
    
    def fit(self, X, y):
        self._estimators = {key: estimator.fit(X, y) 
                            for key, estimator in self._estimators.items()}
        return self._estimators
    
    def predict(self, X):
        self._predictions = {key: estimator.predict(X) 
                             for key, estimator in self._estimators.items()}
        return self._predictions
    
    def score(self, X, y):
        self._scores = {key: estimator.score(X, y) 
                        for key, estimator in self._estimators.items()}
        return self._scores
    
    def evaluate(self, X, y):
        self.predict(self, X)
        self._evaluations = dict(
        accuracyScore = {key: accuracy_score(y, prediction) 
                            for key, prediction in self._predictions.items()},
        precisionScore = {key: precision_score(y, prediction) 
                            for key, prediction in self._predictions.items()},
        recallScore = {key: recall_score(y, prediction) 
                            for key, prediction in self._predictions.items()},
        f1Score = {key: f1_score(y, prediction) 
                            for key, prediction in self._predictions.items()},
        logLoss = {key: log_loss(y, prediction) 
                            for key, prediction in self._predictions.items()},
        zeroOneLoss = {key: zero_one_loss(y, prediction) 
                            for key, prediction in self._predictions.items()},
        confusionMatrix = {key: confusion_matrix(y, prediction) 
                            for key, prediction in self._predictions.items()},
        )
        return self._evaluations