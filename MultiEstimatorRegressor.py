from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

class MultiEstimatorRegressor(BaseEstimator):
    
    def __init__(self, ensemble=False, neuralNet=False):
        self._ensemble = ensemble
        self._neuralNet = neuralNet
        
        self._estimators = dict(
            linearRegressor = LinearRegression(),
            kNearestNeighbors = KNeighborsRegressor(),
            linearSupportVectorRegressor = LinearSVR(),
            supportVectorRegressor = SVR(),
            decisionTree = DecisionTreeRegressor() 
        )
        
        if (ensemble):
            self._estimators.update(
                dict(
                randomForest = RandomForestRegressor(),
                adaBoost = AdaBoostRegressor(),
                gradientBoosting = GradientBoostingRegressor()
                )
            )
        
        if (neuralNet):
            self._estimators.update(dict(neuralNet = MLPRegressor()))
            
        
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
        meanSquaredError = {key: mean_squared_error(y, prediction) 
                            for key, prediction in self._predictions.items()},
        explainedVarianceScore = {key: explained_variance_score(y, prediction) 
                            for key, prediction in self._predictions.items()},
        meanAbsoluteError = {key: mean_absolute_error(y, prediction) 
                            for key, prediction in self._predictions.items()},
        r2Score = {key: r2_score(y, prediction) 
                            for key, prediction in self._predictions.items()}
        )
        return self._evaluations