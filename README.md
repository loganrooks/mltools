# mltools
Tools to streamline the machine learning process.

### Currently contains:
######   - MultiEstimatorRegressor
######   - MultiEstimatorClassifier

#### MultiEstimatorRegressor
Utilizes sklearn and its estimator API to fit multiple regression models to the provided data. It also includes an evaluate method that provides a dictionary of metrics where you can access the particular score of a model. The following code would give me the *f1_score* of the *supportVectorRegressor* model.
```
modelEvaluations = MultiEstimatorRegressor.evaluate
print(modelEvaluations["supportVectorRegressor"]["f1_score"])
```
And if I wanted to output a dictionary containing all evaluation metrics of the *supportVectorRegressor* model, I would simply use the "supportVectorRegressor" key.
```
print(modelEvaluations["supportVectorRegressor"])
```
