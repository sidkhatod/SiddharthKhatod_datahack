import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load the datasets
train_features = pd.read_csv('training_set_features.csv')
train_labels = pd.read_csv('training_set_labels.csv')
test_features = pd.read_csv('test_set_features.csv')
submission_format = pd.read_csv('submission_format.csv')

# Identifying categorical and numerical columns
categorical_cols = train_features.select_dtypes(include=['object']).columns
numerical_cols = train_features.select_dtypes(include=['int64', 'float64']).columns

# Defining preprocessing for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combining preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Applying the transformations
X_train = preprocessor.fit_transform(train_features)
X_test = preprocessor.transform(test_features)

# Separate target variables
y_train_xyz = train_labels['xyz_vaccine']
y_train_seasonal = train_labels['seasonal_vaccine']

# Training model for xyz_vaccine
model_xyz = RandomForestClassifier()
model_xyz.fit(X_train, y_train_xyz)
scores_xyz = cross_val_score(model_xyz, X_train, y_train_xyz, cv=5, scoring='roc_auc')
print("ROC AUC for xyz_vaccine:", scores_xyz.mean())

# Training model for seasonal_vaccine
model_seasonal = RandomForestClassifier()
model_seasonal.fit(X_train, y_train_seasonal)
scores_seasonal = cross_val_score(model_seasonal, X_train, y_train_seasonal, cv=5, scoring='roc_auc')
print("ROC AUC for seasonal_vaccine:", scores_seasonal.mean())

# Predict probabilities for the test set
predictions_xyz = model_xyz.predict_proba(X_test)[:, 1]
predictions_seasonal = model_seasonal.predict_proba(X_test)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'xyz_vaccine': predictions_xyz,
    'seasonal_vaccine': predictions_seasonal
})

submission.to_csv('submission.csv', index=False)
