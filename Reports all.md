Classification with LogisticRegression:
------------------------------
**Accuracy**:
 0.8297872340425532

**Confusion Matrix**:
 [[204  29]
 [ 35 108]]

**Classification Report**:
               precision    recall  f1-score   support

           0       0.85      0.88      0.86       233
           1       0.79      0.76      0.77       143

    accuracy                           0.83       376
   macro avg       0.82      0.82      0.82       376
weighted avg       0.83      0.83      0.83       376


Classification with SVC:
------------------------------
**Accuracy**:
 0.8563829787234043

**Confusion Matrix**:
 [[209  24]
 [ 30 113]]

**Classification Report**:
               precision    recall  f1-score   support

           0       0.87      0.90      0.89       233
           1       0.82      0.79      0.81       143

    accuracy                           0.86       376
   macro avg       0.85      0.84      0.85       376
weighted avg       0.86      0.86      0.86       376


Classification with DecisionTree:
------------------------------
**Accuracy**:
 0.8670212765957447

**Confusion Matrix**:
 [[214  19]
 [ 31 112]]

**Classification Report**:
               precision    recall  f1-score   support

           0       0.87      0.92      0.90       233
           1       0.85      0.78      0.82       143

    accuracy                           0.87       376
   macro avg       0.86      0.85      0.86       376
weighted avg       0.87      0.87      0.87       376


Classification with RandomForest:
------------------------------
**Accuracy**:
 0.9095744680851063

**Confusion Matrix**:
 [[227   6]
 [ 28 115]]

**Classification Report**:
               precision    recall  f1-score   support

           0       0.89      0.97      0.93       233
           1       0.95      0.80      0.87       143

    accuracy                           0.91       376
   macro avg       0.92      0.89      0.90       376
weighted avg       0.91      0.91      0.91       376


Classification with ExtraTrees:
------------------------------
**Accuracy**:
 0.8829787234042553

**Confusion Matrix**:
 [[221  12]
 [ 32 111]]

**Classification Report**:
               precision    recall  f1-score   support

           0       0.87      0.95      0.91       233
           1       0.90      0.78      0.83       143

    accuracy                           0.88       376
   macro avg       0.89      0.86      0.87       376
weighted avg       0.88      0.88      0.88       376


Classification with AdaBoost:
------------------------------
C:\Users\austi\AppData\Roaming\Python\Python310\site-packages\sklearn\ensemble\_weight_boosting.py:527: FutureWarning: The SAMME.R algorithm (the default) is deprecated and will be removed in 1.6. Use the SAMME algorithm to circumvent this warning.
  warnings.warn(
**Accuracy**:
 0.925531914893617

**Confusion Matrix**:
 [[221  12]
 [ 16 127]]

**Classification Report**:
               precision    recall  f1-score   support

           0       0.93      0.95      0.94       233
           1       0.91      0.89      0.90       143

    accuracy                           0.93       376
   macro avg       0.92      0.92      0.92       376
weighted avg       0.93      0.93      0.93       376


Classification with GradientBoost:
------------------------------
**Accuracy**:
 0.925531914893617

**Confusion Matrix**:
 [[226   7]
 [ 21 122]]

**Classification Report**:
               precision    recall  f1-score   support

           0       0.91      0.97      0.94       233
           1       0.95      0.85      0.90       143

    accuracy                           0.93       376
   macro avg       0.93      0.91      0.92       376
weighted avg       0.93      0.93      0.92       376


Classification with XGBoost:
------------------------------
**Accuracy**:
 0.9335106382978723

**Confusion Matrix**:
 [[227   6]
 [ 19 124]]

**Classification Report**:
               precision    recall  f1-score   support

           0       0.92      0.97      0.95       233
           1       0.95      0.87      0.91       143

    accuracy                           0.93       376
   macro avg       0.94      0.92      0.93       376
weighted avg       0.93      0.93      0.93       376


Classification with LightGBM:
------------------------------
[LightGBM] [Info] Number of positive: 609, number of negative: 894
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000474 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3843
[LightGBM] [Info] Number of data points in the train set: 1503, number of used features: 36
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.405190 -> initscore=-0.383888
[LightGBM] [Info] Start training from score -0.383888
**Accuracy**:
 0.9281914893617021

**Confusion Matrix**:
 [[226   7]
 [ 20 123]]

**Classification Report**:
               precision    recall  f1-score   support

           0       0.92      0.97      0.94       233
           1       0.95      0.86      0.90       143

    accuracy                           0.93       376
   macro avg       0.93      0.92      0.92       376
weighted avg       0.93      0.93      0.93       376



Catboost

**Accuracy**:
 0.9414893617021277

**Confusion Matrix**:
 [[226   7]
 [ 15 128]]

**Classification Report**:
               precision    recall  f1-score   support

           0       0.94      0.97      0.95       233
           1       0.95      0.90      0.92       143

    accuracy                           0.94       376
   macro avg       0.94      0.93      0.94       376
weighted avg       0.94      0.94      0.94       376