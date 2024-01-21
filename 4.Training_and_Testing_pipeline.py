"""
Basic ML models implementation includes
1.Logistic Regression
2.Random Forest
3.Naive Bayes Classifier
4.Support Vectors Machine
5.Gradient Boosting
6.Decision Tree
7.KNN
"""


# importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

# Loading split data
X_train = np.load('/content/drive/MyDrive/tomato_pest/X_train.npy')
y_train = np.load('/content/drive/MyDrive/tomato_pest/y_train.npy')
X_test = np.load('/content/drive/MyDrive/tomato_pest/X_test.npy')
y_test = np.load('/content/drive/MyDrive/tomato_pest/y_test.npy')


# Creating custom classifier pipeline and calling user declared models
class CustomClassifierPipeline:
    def __init__(self, classifier: str = 'Logistic Regression'):
        self.classifier = classifier
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'Naive Bayes': GaussianNB(),
            'Support Vector Machine': SVC(probability=True),
            'Gradient Boosting': GradientBoostingClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'K-Nearest Neighbors': KNeighborsClassifier()
        }
        # raising error if model is not present in the list
        if classifier not in self.models:
            raise ValueError(f"Invalid classifier '{classifier}'. Choose one from: {', '.join(self.models.keys())}")
        # calling pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', self.models[classifier])
        ])

    # train function for fitting the user called model
    def train(self, X_train_data, y_train_data):
        self.pipeline.fit(X_train_data, y_train_data)

    # evaluation/ testing
    def evaluate(self, X_test_data, y_test_data, plot_results=True):
        y_pred = self.pipeline.predict(X_test_data)
        y_prob = self.pipeline.predict_proba(X_test_data)

        # Evaluate performance
        accuracy = accuracy_score(y_test_data, y_pred)
        precision = precision_score(y_test_data, y_pred, average='macro')
        recall = recall_score(y_test_data, y_pred, average='macro')
        f1 = f1_score(y_test_data, y_pred, average='macro')
        cm = confusion_matrix(y_test_data, y_pred)
        classification_rep = classification_report(y_test_data, y_pred)
        average_auc = self.calculate_average_auc(y_test_data, y_prob)

        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 score: {f1:.2f}')
        print(f'Classification Report:\n{classification_rep}')
        print(f'Average AUC: {average_auc:.2f}')

        # Plotting results using confusion matrix and roc curve
        if plot_results:
            self.plot_confusion_matrix(cm)
            self.plot_roc_curve(y_test_data, y_prob)

    # Confusion matrix
    @staticmethod
    def plot_confusion_matrix(cm, cmap='Blues'):
        cm_df = pd.DataFrame(cm,
                             index=['BA', 'HA', 'SE', 'MP', 'TP', 'SL', 'ZC', 'TU'],
                             columns=['BA', 'HA', 'SE', 'MP', 'TP', 'SL', 'ZC', 'TU'])
        plt.figure(figsize=(14, 10))
        sns.heatmap(cm_df / np.sum(cm_df, axis=0), annot=True, fmt='.2%', cmap=cmap)
        plt.ylabel('Actual values')
        plt.xlabel('Predicted values')
        plt.show()

    # AUC score calculation
    @staticmethod
    def calculate_average_auc(y_test_data, y_prob):
        n_classes = len(np.unique(y_test_data))
        average_auc = 0.0
        y_test_binarized = label_binarize(y_test_data, classes=np.unique(y_test_data))

        for i in range(n_classes):
            average_auc += roc_auc_score(y_test_binarized[:, i], y_prob[:, i])

        average_auc /= n_classes
        return average_auc

    # ROC curve
    @staticmethod
    def plot_roc_curve(y_test_data, y_prob):
        class_names = ['BA', 'HA', 'SE', 'MP', 'TP', 'SL', 'ZC', 'TU']

        fpr = {}
        tpr = {}
        thresh = {}
        n_class = 8

        for i in range(n_class):
            fpr[i], tpr[i], thresh[i] = roc_curve(y_test_data, y_prob[:, i], pos_label=i)

        colors = ['orange', 'green', 'blue', 'red', 'purple', 'brown', 'yellow', 'cyan']

        for i in range(n_class):
            plt.plot(fpr[i], tpr[i], linestyle='--', color=colors[i], label='Class {}'.format(class_names[i]))

        # plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.show()


# Instantiate the class with a classifier
# We can change 'Random Forest' to any other classifier from the list
random_forest_pipeline = CustomClassifierPipeline(classifier='Random Forest')

# Training the pipeline
random_forest_pipeline.train(X_train, y_train)

# Evaluate and plot results
random_forest_pipeline.evaluate(X_test, y_test)
