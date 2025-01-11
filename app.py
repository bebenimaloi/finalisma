import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt
import base64
import io

app = Flask(__name__)

def perform_decision_tree_analysis(X_train, X_test, y_train, y_test):
    # Initialize and train the model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

    # Calculate AUC-ROC
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)
    y_pred_binarized = lb.transform(y_pred)
    metrics['auc_roc'] = roc_auc_score(y_test_binarized, y_pred_binarized, 
                                     average='weighted', multi_class='ovr')

    # Create visualizations
    images = {}
    try:
        # Feature Importance Plot
        plt.figure(figsize=(10, 6))
        feature_importances = model.feature_importances_
        plt.barh(X_train.columns, feature_importances)
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title("Decision Tree Feature Importance")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images['feature_importance'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Decision Tree Plot
        plt.figure(figsize=(15, 10))
        plot_tree(model, filled=True, feature_names=X_train.columns, rounded=True)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images['tree_plot'] = base64.b64encode(buf.read()).decode('utf-8')
        
    finally:
        plt.close('all')

    return metrics, images, model.get_depth(), model.get_n_leaves()

def perform_random_forest_analysis(X_train, X_test, y_train, y_test):
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                             param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Calculate metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, average='weighted'),
        'recall': recall_score(y_train, y_train_pred, average='weighted'),
        'f1': f1_score(y_train, y_train_pred, average='weighted')
    }

    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, average='weighted'),
        'recall': recall_score(y_test, y_test_pred, average='weighted'),
        'f1': f1_score(y_test, y_test_pred, average='weighted')
    }

    # Calculate AUC-ROC
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)
    y_test_pred_binarized = lb.transform(y_test_pred)
    auc_roc = roc_auc_score(y_test_binarized, y_test_pred_binarized, 
                           average='weighted', multi_class='ovr')

    # Feature importance
    importances = best_model.feature_importances_
    feature_importance_dict = dict(zip(X_train.columns, importances))
    sorted_importance = dict(sorted(feature_importance_dict.items(), 
                                  key=lambda x: x[1], reverse=True))

    # Create visualization
    images = {}
    try:
        plt.figure(figsize=(10, 6))
        plt.barh(list(sorted_importance.keys()), 
                list(sorted_importance.values()))
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images['feature_importance'] = base64.b64encode(buf.read()).decode('utf-8')
        
    finally:
        plt.close('all')

    return (grid_search.best_params_, train_metrics, test_metrics, 
            auc_roc, sorted_importance, images)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'file' not in request.files:
            return "No file part in the request", 400

        file = request.files['file']
        model_choice = request.form.get('model')

        if file.filename == '':
            return "No selected file", 400

        if file:
            # Save and read the file
            upload_folder = "uploads"
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Read the CSV file
            data = pd.read_csv(file_path)

            # Prepare feature columns and target
            X = data[['Cognitive Abilities', 'Engagement Levels', 'Pre-Test Scores',
                     'Post-Test Scores', 'Time Spent on AR', 'Frequency of AR Use']]
            y = data['Performance']

            # Encode the target column
            le = LabelEncoder()
            y = le.fit_transform(y)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )

            if model_choice == 'decision_tree':
                metrics, images, tree_depth, tree_leaves = perform_decision_tree_analysis(
                    X_train, X_test, y_train, y_test
                )
                return render_template(
                    'decision_tree_results.html',
                    metrics=metrics,
                    images=images,
                    tree_depth=tree_depth,
                    tree_leaves=tree_leaves
                )
            
            elif model_choice == 'random_forest':
                (best_params, train_metrics, test_metrics, 
                 auc_roc, feature_importance, images) = perform_random_forest_analysis(
                    X_train, X_test, y_train, y_test
                )
                return render_template(
                    'random_forest_results.html',
                    best_params=best_params,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                    auc_roc=round(auc_roc, 3),
                    is_overfitting=train_metrics['accuracy'] - test_metrics['accuracy'] > 0.1,
                    feature_importance=feature_importance,
                    images=images
                )

            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        print(f"Error: {str(e)}")
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)