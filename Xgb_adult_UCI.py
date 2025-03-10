import pandas as pd
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.datasets import fetch_openml
from scipy.stats import randint, uniform

# Load UCI Adult Dataset from sklearn
data = fetch_openml('adult', version=2, as_frame=True)

# Preprocess the data
df = data['data']
target = data['target']

# Label encoding for target
le = LabelEncoder()
y = le.fit_transform(target)

# Convert categorical columns into one-hot encoded columns
X = pd.get_dummies(df)

# Ensure all columns are numeric, converting any problematic columns
X = X.apply(pd.to_numeric, errors='coerce')
X = X.astype(int)

# Check for NaN values and fill them if any are present
if X.isnull().values.any():
    X = X.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='hist')
# xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='gpu_hist',gpu_id=0)     #GPU

# Define hyperparameters to tune random search
# param_dist = {
#     'n_estimators': randint(50, 200),
#     'max_depth': randint(3, 10),
#     'learning_rate': uniform(0.01, 0.3),
#     'subsample': uniform(0.7, 1.0),
#     'colsample_bytree': uniform(0.7, 1.0),
#     'gamma': uniform(0, 0.5),
#     'min_child_weight': randint(1, 10),
# }
# Define hyperparameters to tune grid search
param_dist = {
    'n_estimators': [50, 90, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7,10],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 10],
}


# Set up RandomizedSearchCV
# random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=50, scoring='accuracy',
#                                    cv=3, verbose=1, random_state=42, n_jobs=-1)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_dist, scoring='neg_mean_absolute_error', cv=5, refit=False, n_jobs=-1, pre_dispatch='100*n_jobs')


# Measure training time with hyperparameter optimization
start_train_time = time.time()
# random_search.fit(X_train, y_train)
grid_search.fit(X_train, y_train)
xgb_model.set_params(**grid_search.best_params_)
best_model = xgb_model.fit(X_train, y_train)
end_train_time = time.time()

# Best model from RandomizedSearchCV
# best_model = grid_search.best_params_

# Measure inference time
start_inference_time = time.time()
y_pred = best_model.predict(X_test)
end_inference_time = time.time()

# Evaluate performance (Accuracy)
accuracy = accuracy_score(y_test, y_pred)

# Print the results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Training Time with Hyperparameter Tuning: {end_train_time - start_train_time:.4f} seconds")
print(f"Inference Time: {end_inference_time - start_inference_time:.4f} seconds")
print(f"Accuracy: {accuracy:.4f}")
