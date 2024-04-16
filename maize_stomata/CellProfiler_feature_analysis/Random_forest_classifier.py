from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
import matplotlib.pyplot as plt
import numpy as np

def run_random_forest_classifier (x, y, n_estimator, n) : 
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    rf_classifier = RandomForestClassifier(n_estimators = n_estimator , random_state = n)
    rf_classifier.fit(x, y_encoded)
    importances = rf_classifier.feature_importances_
    sorted_indices = importances.argsort()[::-1]
    sorted_importances = importances[sorted_indices]
    feature_names = x.columns[sorted_indices]
    return feature_names, sorted_importances
 

def visulize_feature_importance(a, b, feature_names, sorted_importances):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(a, b))
    # Define colors for SCO and GCO features
    colors = ['skyblue' if 'avg' in name else 'gray' for name in feature_names]
    plt.bar(range(len(feature_names)), sorted_importances, color=colors, edgecolor='black')
    plt.grid(False)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.xlabel('Feature', fontsize=22)
    plt.ylabel('Importance Score', fontsize=22)
    plt.title('Feature Importances', fontsize=22)
    plt.tight_layout()
    plt.show()



def evaluation_classifier(X, y, test_size, n_estimators, random_state):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

    # Initialize the RandomForestClassifier with a fixed random_state
    rf_classifier = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state)
    # Fit the classifier to your data
    rf_classifier.fit(X_train, y_train)

    # Predictions on the testing set
    y_pred = rf_classifier.predict(X_test)
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = rf_classifier.score(X_test, y_test)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)
    print(y_test)
    return y_pred, y_test



def visulize_confusion_matrix(y_ture, y_pred, columns, a, b):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
    # Compute the confusion matrix
    cm = confusion_matrix(y_ture, y_pred)
    # Visualize the confusion matrix
    plt.figure(figsize=(a, b))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels= columns, yticklabels=columns, annot_kws={"fontsize":14})
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Visualize the confusion matrix
    plt.figure(figsize=(a, b))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=columns, yticklabels=columns, annot_kws={"fontsize":14})
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    return y_ture, y_pred
