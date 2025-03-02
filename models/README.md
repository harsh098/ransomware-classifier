# Adapted from
This work is adaptation from the base research work done by  Willers and Phillipart  at
[TomasPhilippart/ebpfangel](https://github.com/TomasPhilippart/ebpfangel/blob/main/machinelearning/README.md). The goal of this pipeline is to extend the work done by them to include Voting Ensemble Techniques to enhance Ransomware Detection. 

# Dealing with Imbalanced Data

The dataset procured is extremely skewed towards the negative class (Benign Files) and Ransomware Samples (the positive class) are in minority (2.8%). Techniques such as [ADASYN](https://doi.org/10.1109/IJCNN.2008.4633969) and [SMOTE](https://arxiv.org/abs/1106.1813) were used to generate Synthetic Data to improve the overall MCC (Matthews Correlation Coefficient) of the models, during, training.

# Machine Learning

The machine learning pipeline consist of:
```mermaid
graph LR
  A(events from eBPF detector) --> B(data preparation <br/>& feature engineering)
  B --> C(model development <br/>& training)
  C --> D(model evaluation)
  D --> E(prediction)
```
## Instructions:

Prepare the data:
```shell
./dataprep.py
```

Sample output:
```rb
       C_max  C_sum  D_max  D_sum  E_max  E_sum  O_max  O_sum  P_max  P_sum   CDO  COC  COO   DOC  DOO   EEE   OCD  OCO  OOC  OOO
PID
1          0      0      0      0      0      0      4      5      0      0     0    0    0     0    0     0     0    0    0    3
221        0      0      0      0      0      0     20     20      0      0     0    0    0     0    0     0     0    0    0   18
626        0      0      0      0      0      0     16     16      0      0     0    0    0     0    0     0     0    0    0   14
714        0      0      0      0      0      0     12     12      0      0     0    0    0     0    0     0     0    0    0   10
838        0      0      0      0      0      0      3      9      0      0     0    0    0     0    0     0     0    0    0    7
1019       0      0      0      0     36   1501      0      0      0      0     0    0    0     0    0  1499     0    0    0    0
1195       0      0      0      0      0      0     11     11      0      0     0    0    0     0    0     0     0    0    0    9
1196       0      0      0      0      0      0      7      7      0      0     0    0    0     0    0     0     0    0    0    5
1197       0      0      0      0      0      0     10     10      0      0     0    0    0     0    0     0     0    0    0    8
1202       0      0      0      0      0      0      8      8      0      0     0    0    0     0    0     0     0    0    0    6
1230       0      0      0      0      0      0      1     60      0      0     0    0    0     0    0     0     0    0    0   58
1237       0      0      0      0      0      0      1     36      0      0     0    0    0     0    0     0     0    0    0   34
1238       0      0      0      0      0      0      1     27      0      0     0    0    0     0    0     0     0    0    0   25
1239       0      0      0      0      0      0      1     38      0      0     0    0    0     0    0     0     0    0    0   36
1240       0      0      0      0      0      0      1     30      0      0     0    0    0     0    0     0     0    0    0   28
1555       0      0      0      0    207   1978      0      0      0      0     0    0    0     0    0  1976     0    0    0    0
29952      0      0      0      0     63    326      0      0      0      0     0    0    0     0    0   324     0    0    0    0
30490     17   1690     17   1691      0      0     20   1750     17   1690  1689    0    0  1635   55     0  1690    0   55    5
30493      0      0      0      0      0      0     12     19      0      0     0    0    0     0    0     0     0    0    0   17
30495    233   1216     17    983      0      0    278   1458     17    983   983  228    5   949   34     0   983  233   39  201
30496      0      0      0      0      0      0      3      3      0      0     0    0    0     0    0     0     0    0    0    1
```

### Model development

Train the model and show predictions:
```shell
python $MODEL_FILE --train --labels file
```

Sample output:
```rb
Score: 1.000000
      PID  C_max  C_sum  D_max  D_sum  E_max  E_sum  O_max  O_sum  P_max  P_sum   CDO  COC  COO   DOC  DOO   EEE   OCD  OCO  OOC  OOO  PREDICTION
0       1      0      0      0      0      0      0      4      5      0      0     0    0    0     0    0     0     0    0    0    3           0
1     221      0      0      0      0      0      0     20     20      0      0     0    0    0     0    0     0     0    0    0   18           0
2     626      0      0      0      0      0      0     16     16      0      0     0    0    0     0    0     0     0    0    0   14           0
3     714      0      0      0      0      0      0     12     12      0      0     0    0    0     0    0     0     0    0    0   10           0
4     838      0      0      0      0      0      0      3      9      0      0     0    0    0     0    0     0     0    0    0    7           0
5    1019      0      0      0      0     36   1501      0      0      0      0     0    0    0     0    0  1499     0    0    0    0           0
6    1195      0      0      0      0      0      0     11     11      0      0     0    0    0     0    0     0     0    0    0    9           0
7    1196      0      0      0      0      0      0      7      7      0      0     0    0    0     0    0     0     0    0    0    5           0
8    1197      0      0      0      0      0      0     10     10      0      0     0    0    0     0    0     0     0    0    0    8           0
9    1202      0      0      0      0      0      0      8      8      0      0     0    0    0     0    0     0     0    0    0    6           0
10   1230      0      0      0      0      0      0      1     60      0      0     0    0    0     0    0     0     0    0    0   58           0
11   1237      0      0      0      0      0      0      1     36      0      0     0    0    0     0    0     0     0    0    0   34           0
12   1238      0      0      0      0      0      0      1     27      0      0     0    0    0     0    0     0     0    0    0   25           0
13   1239      0      0      0      0      0      0      1     38      0      0     0    0    0     0    0     0     0    0    0   36           0
14   1240      0      0      0      0      0      0      1     30      0      0     0    0    0     0    0     0     0    0    0   28           0
15   1555      0      0      0      0    207   1978      0      0      0      0     0    0    0     0    0  1976     0    0    0    0           0
16  29952      0      0      0      0     63    326      0      0      0      0     0    0    0     0    0   324     0    0    0    0           0
17  30490     17   1690     17   1691      0      0     20   1750     17   1690  1689    0    0  1635   55     0  1690    0   55    5           1
18  30493      0      0      0      0      0      0     12     19      0      0     0    0    0     0    0     0     0    0    0   17           0
19  30495    233   1216     17    983      0      0    278   1458     17    983   983  228    5   949   34     0   983  233   39  201           1
20  30496      0      0      0      0      0      0      3      3      0      0     0    0    0     0    0     0     0    0    0    1           0
```


## Files

### Standalone Models
|Model|File|
|--|--|
|SVM|[model_svm.py](./model_svm.py)|
|RandomForest|[model_rf.py](./model_rf.py)|
|kNN|[model_knn.py](./model_knn.py)|
|Adaboost|[model_adaboost.py](./model_adaboost.py)|
|GradientBoosting Classifier|[model_gboost.py](./model_gboost.py)|
|XGBoosted Trees|[model_xgboost.py](./model_xgboost.py)|
|Multi Layer Perceptron Classifier|[model_mlp.py](./model_mlp.py)|
|Logistic Regression|[model_lr.py](./model_lr.py)|

### Voting Ensemble Models (3-Model Soft Voting)
|Model 1| Model 2| Model 3|File|
|---|---|---|---|
|SVM|RF|kNN|[model_svm_knn_rf.py](./model_svm_knn_rf.py)|
|SVM|RF|Multi Layer Perceptron|[model_svm_mlp_rf.py](./model_svm_mlp_rf.py)|
|SVM|RF|Adaboost|[model_svm_adaboost_rf.py](./model_svm_adaboost_rf.py)|
|SVM|RF|XGBoosted Trees|[model_svm_xgboost_rf.py](./model_svm_xgboost_rf.py)|
|SVM|RF|Logistic Regression|[model_svm_lr_rf.py](./model_svm_lr_rf.py)|


## Model Benchmarks
 Model | TP |TN|FP|FN|MCC|Precision|Recall|Specificity|F1|Accuracy|AUC (Rounded to 2 decimal places)|Weight of Model 1|Weight of Model 2|Weight of Model 3|  
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|  
|SVM|21|1307|10|0|0.8199242038|0.6774193548|1|0.9924069856|0.8076923077|0.9925261584|1|NA|NA|NA|
|RF|12|1317|0|9|0.7533592085|1|0.5714285714|1|0.7272727273|0.9932735426|0.99|NA|NA|NA|
|KNN|15|1296|21|6|0.5364298664|0.4166666667|0.7142857143|0.9840546697|0.5263157895|0.9798206278|0.85|NA|NA|NA|
|Logistic Regression (LR)|20|1223|94|1|0.3922279792|0.1754385965|0.9523809524|0.9286256644|0.2962962963|0.9289985052|0.95|NA|NA|NA|
|MLP|21|1309|8|0|0.848374457|0.724137931|1|0.9939255885|0.84|0.9940209268|1|NA|NA|NA|
|Adaboost|11|1309|8|10|0.5438898523|0.5789473684|0.5238095238|0.9939255885|0.55|0.9865470852|0.99|NA|NA|NA|
|GradientBoosting|11|1316|1|10|0.6895782276|0.9166666667|0.5238095238|0.9992406986|0.6666666667|0.9917787743|0.95|NA|NA|NA|
|XGBoost|10|1311|11|6|0.5393026277|0.4761904762|0.625|0.9916792738|0.5405405405|0.9872944694|0.95|NA|NA|NA|
|SVM+RF+KNN|15|1315|2|6|0.7909960551|0.8823529412|0.7142857143|0.9984813971|0.7894736842|0.9940209268|1|1.2|0.99|0.78|
|SVM+RF+LR|21|1316|1|0|0.9766374285|0.9545454545|1|0.9992406986|0.976744186|0.9992526158|1|1.2|0.99|0.75|
|SVM+RF+MLP|21|1316|1|0|0.9766374285|0.9545454545|1|0.9992406986|0.976744186|0.9992526158|1|1|1|1|
|SVM+RF+Adaboost|19|1316|2|1|0.9259755363|0.9047619048|0.95|0.9984825493|0.9268292683|0.9977578475|1|1.2|0.99|0.78|
|SVM+RF+XGBoost|14|1316|1|7|0.7861235409|0.9333333333|0.6666666667|0.9992406986|0.7777777778|0.9940209268|1|1.2|0.99|0.7|