# importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, f1_score, recall_score)

# Loading data
X_train = np.load('/content/drive/MyDrive/tomato_pest/X_train.npy')
y_train = np.load('/content/drive/MyDrive/tomato_pest/y_train.npy')
X_test = np.load('/content/drive/MyDrive/tomato_pest/X_test.npy')
y_test = np.load('/content/drive/MyDrive/tomato_pest/y_test.npy')

# Loading model and training
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Testing
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f'Accuracy:{accuracy:.2f}')
print(f'Precision:{precision:.2f}')
print(f'Recall:{recall:.2f}')
print(f'F1 score:{f1:.2f}')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                     index = ['BA', 'HA', 'SE', 'MP', 'TP', 'SL', 'ZC', 'TU'],
                     columns = ['BA', 'HA', 'SE', 'MP', 'TP', 'SL', 'ZC', 'TU'])
plt.figure(figsize=(14, 10))
sns.heatmap(cm_df / np.sum(cm_df, axis=0), annot=True, fmt='.2%', cmap='Blues')
# plt.title('Confusion matrix: Decison Tree')
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show()

print(classification_report(y_test, y_pred))

# Roc curve
from sklearn.metrics import roc_curve
class_names = ['BA', 'HA', 'SE', 'MP', 'TP', 'SL', 'ZC', 'TU']

pred_prob = model.predict_proba(X_test)
y_pred_lr = model.predict_proba(X_test)
random_probs = [0 for i in range(len(y_test))]

fpr = {}
tpr = {}
thresh = {}
n_class = 8

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_lr[:, i], pos_label=i)

colors = ['orange', 'green', 'blue', 'red', 'purple', 'brown', 'yellow', 'black', 'pink']

for i in range(n_class):
    plt.plot(fpr[i], tpr[i], linestyle='--', color=colors[i], label='Class {}'.format(class_names[i]))

# plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()

# AUC score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
n_classes = 8
average_auc = 0.0
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7])
for i in range(n_classes):
    average_auc += roc_auc_score(y_test_binarized[:, i], pred_prob[:, i])

average_auc /= n_classes

print("Average AUC:", average_auc)

# Precision vs recall plot
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision = dict()
recall = dict()
n_class = ['BA', 'HA', 'SE', 'MP', 'TP', 'SL', 'ZC', 'TU']

for i, class_name in enumerate(n_class):
    precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], pred_prob[:, i])

def print_recalls_precision(recall, precision, title):
    plt.figure(figsize=(6, 6))
    for i, class_name in enumerate(n_class):
        plt.plot(recall[i], precision[i], linestyle='--', label='Class {}'.format(class_name))

    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    # plt.title("Precision vs Recall plot - {0}".format(title), fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.legend(loc="best")
    plt.show()

print_recalls_precision(recall, precision, "DecisionTreeClassifier")
