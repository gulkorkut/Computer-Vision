import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

classification_data = pd.read_csv('heart.csv')
regression_data = pd.read_csv('cars.csv')

X_classification = classification_data.iloc[:, :-1]
y_classification = classification_data.iloc[:, -1]

X_regression = regression_data.iloc[:, :-1]
y_regression = regression_data.iloc[:, -1]

numerical_cols = X_regression.select_dtypes(include=['number']).columns
categorical_cols = X_regression.select_dtypes(include=['object']).columns

numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

classification_models = [
    LogisticRegression(max_iter=1000),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

regression_models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor()
]

cv_classification = StratifiedKFold(n_splits=5, shuffle=True)
cv_regression = KFold(n_splits=5, shuffle=True)

regression_pipelines = [(type(model).__name__, Pipeline(steps=[('preprocessor', preprocessor),
                                                               ('regressor', model)])) for model in regression_models]

y_regression_imputed = SimpleImputer(strategy='mean').fit_transform(y_regression.values.reshape(-1, 1)).ravel()

classification_results = {}
for model in classification_models:
    scores = cross_val_score(model, X_classification, y_classification, cv=cv_classification, scoring='accuracy')
    classification_results[type(model).__name__] = scores.mean()

regression_results = {}
for model_name, model_pipeline in regression_pipelines:
    try:
        scores = cross_val_score(model_pipeline, X_regression, y_regression_imputed, cv=cv_regression, scoring='r2')
        regression_results[model_name] = scores.mean()
    except Exception as e:
        print(f"Error for {model_name}: {e}")


print("Classification Results:")
print(classification_results)

print("\nRegression Results:")
print(regression_results)

classification_df = pd.DataFrame(list(classification_results.items()), columns=['Model', 'Accuracy'])
regression_df = pd.DataFrame(list(regression_results.items()), columns=['Model', 'R-square'])

classification_df.to_csv('classification_results.csv', index=False)
regression_df.to_csv('regression_results.csv', index=False)