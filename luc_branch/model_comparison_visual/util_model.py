# Import Dependencies
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Apply Model Function
def apply_models(X, y):
    """
    Applies multiple machine learning models and returns the accuracy scores and predictions for each model.
    
    Args:
        X (pd.DataFrame): Features data
        y (pd.Series): Target labels
    
    Returns:
        dict: Dictionary containing accuracy and predictions for each model
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the models
    models = {
        'LogisticRegression': LogisticRegression(),
        'SVC': SVC(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'ExtraTrees': ExtraTreesClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'GradientBoost': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(),
        'CatBoost': CatBoostClassifier()
    }

    # Dictionary to store accuracy and predictions
    results = {}

    for model_name, model in models.items():
        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store the trained model and evaluation metrics
        results[model_name] = {
            'model': model,  # Store the trained model
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

    # Return results, X_test, and y_test
    return results, X_test, y_test

        # Store the results
        #results[model_name] = {
            #'accuracy': accuracy,
            #'confusion_matrix': confusion_matrix(y_test, y_pred),
            #'classification_report': classification_report(y_test, y_pred, output_dict=True)
        #}

    #return results