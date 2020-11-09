<!-- #region -->
# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The data is related with direct marketing campaigns of a Portuguese banking institution and it was made available by Kaggle [here](https://www.kaggle.com/yufengsui/portuguese-bank-marketing-data-set).  The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. The goal is to predict whether a product will be subscribed by the customer contacted.

The best performing model was a VotingEnsemble algorithm (accuracy of 0.91809) resulting from an Azure AutoML run.

The above mentioned model, even if only slightly, has proved to be more performing than the first one, trained through Azure HyperDrive on a LogisticRegression algorithm (accuracy of 0.91320).

## Scikit-learn Pipeline
The first model is not trained using an Azure ML Pipeline, but thanks to a simply *HyperDrive* run through an *HyperDriveConfig*, executed in order to train the model defined into the *train.py* script.

The train.py script implements the following points:

1. A *TabularDataset* is created using the *from_delimited_files()* method from the TabularDatasetFactory class and a web url.
2. The dataset is split in a dataframe containing all the features (x) and another one containing just the target variable after a cleaning phase that follows these steps:
    1. Drop each row containing missing values
    2. Convert the *job*, *contact* and *education* variables into dummy/indicator variables, removing then the original ones
    3. Convert the categorical variables *marital*, *default*, *housing*, *loan* and *poutcome* in a numeric (dummy) one (1 and 0)
    4. Convert the *month* and *day_of_week* variables into integers (ordinal encoding) using the map function
3. The dataset is split again this time in *train* and *test* dataframes (80%-20%) using a random sampling of the rows.
4. A Logistic Regression is fit using random combination of the parameters *C* and *max_iter* passed by HyperDrive
5. Parameters and performance metrics are logged into the context run
6. The fitted model is dumped into a file and persisted into the run

The chosen parameter sampler is the *RandomParameterSampling* one, because it is proven that randomly choosing trials is more efficient for hyper-parameter optimization than using all the trials of a grid.

The chosen early stopping policy is the *BanditPolicy* one. Its parameters are set so that it'll check the job every 2 iterations. If the primary metric (*accuracy* in our case) falls outside of the top 10% range, Azure ML terminate the job. This avoids having to explore hyperparameters that don't show promise of helping reach our target metric.

The best model found by HyperDrive has the following characteristics:
- Regularization Strength: 0.451
- Max iterations: 100
- Accuracy: 0.9129


## AutoML

The best model found by AutoML is a Voting Ensemble one made by the following models (with their respective hyper-parameters):

```
Pipeline(memory=None,  
	steps=[('maxabsscaler', MaxAbsScaler(copy=True)),  
		('lightgbmclassifier',  
		 LightGBMClassifier(boosting_type='gbdt', class_weight=None,  
							colsample_bytree=1.0,  
							importance_type='split', learning_rate=0.1,  
							max_depth=-1, min_child_samples=20,  
							min_child_weight=0.001, min_split_gain=0.0,  
							n_estimators=100, n_jobs=1, num_leaves=31,  
							objective=None, random_state=None,  
							reg_alpha=0.0, reg_lambda=0.0, silent=True,  
							subsample=1.0, subsample_for_bin=200000,  
							subsample_freq=0, verbose=-10))],  
	verbose=False), '1': Pipeline(memory=None,  
	steps=[('maxabsscaler', MaxAbsScaler(copy=True)),  
		('xgboostclassifier',  
		 XGBoostClassifier(base_score=0.5, booster='gbtree',  
						   colsample_bylevel=1, colsample_bynode=1,  
						   colsample_bytree=1, gamma=0,  
						   learning_rate=0.1, max_delta_step=0,  
						   max_depth=3, min_child_weight=1, missing=nan,  
						   n_estimators=100, n_jobs=1, nthread=None,  
						   objective='binary:logistic', random_state=0,  
						   reg_alpha=0, reg_lambda=1,  
						   scale_pos_weight=1, seed=None, silent=None,  
						   subsample=1, tree_method='auto', verbose=-10,  
						   verbosity=0))],  
	verbose=False), '31': Pipeline(memory=None,  
	steps=[('standardscalerwrapper',  
		 <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f40700323c8>),  
		('xgboostclassifier',  
		 XGBoostClassifier(base_score=0.5, booster='gbtree',  
						   colsample_bylevel=1, colsample_bynode=1,  
						   colsample_bytree=0.7, eta=0.2, gamma=1,  
						   learning_rate=0.1, max_delta_step=0,  
						   max_depth=5, max_leaves=0,  
						   min_child_weight=1, missing=nan,  
						   n_estimators=25, n_jobs=1, nthread=None,  
						   objective='reg:logistic', random_state=0,  
						   reg_alpha=0, reg_lambda=1.1458333333333335,  
						   scale_pos_weight=1, seed=None, silent=None,  
						   subsample=1, tree_method='auto', verbose=-10,  
						   verbosity=0))],  
	verbose=False), '35': Pipeline(memory=None,  
	steps=[('sparsenormalizer',  
		 <azureml.automl.runtime.shared.model_wrappers.SparseNormalizer object at 0x7f3ee5c989b0>),  
		('lightgbmclassifier',  
		 LightGBMClassifier(boosting_type='gbdt', class_weight=None,  
							colsample_bytree=0.99,  
							importance_type='split',  
							learning_rate=0.05789894736842106,  
							max_bin=240, max_depth=10,  
							min_child_samples=2727, min_child_weight=2,  
							min_split_gain=0.21052631578947367,  
							n_estimators=400, n_jobs=1, num_leaves=197,  
							objective=None, random_state=None,  
							reg_alpha=0.5789473684210527,  
							reg_lambda=0.21052631578947367, silent=True,  
							subsample=0.09947368421052633,  
							subsample_for_bin=200000, subsample_freq=0,  
							verbose=-10))],  
	verbose=False), '28': Pipeline(memory=None,  
	steps=[('sparsenormalizer',  
		 <azureml.automl.runtime.shared.model_wrappers.SparseNormalizer object at 0x7f3ee5d2f630>),  
		('xgboostclassifier',  
		 XGBoostClassifier(base_score=0.5, booster='gbtree',  
						   colsample_bylevel=0.8, colsample_bynode=1,  
						   colsample_bytree=1, eta=0.3, gamma=0,  
						   learning_rate=0.1, max_delta_step=0,  
						   max_depth=7, max_leaves=31,  
						   min_child_weight=1, missing=nan,  
						   n_estimators=100, n_jobs=1, nthread=None,  
						   objective='reg:logistic', random_state=0,  
						   reg_alpha=1.4583333333333335,  
						   reg_lambda=0.20833333333333334,  
						   scale_pos_weight=1, seed=None, silent=None,  
						   subsample=1, tree_method='auto', verbose=-10,  
						   verbosity=0))],  
	verbose=False), '4': Pipeline(memory=None,  
	steps=[('maxabsscaler', MaxAbsScaler(copy=True)),  
		('sgdclassifierwrapper',  
		 SGDClassifierWrapper(alpha=9.59184081632653,  
							  class_weight='balanced', eta0=0.01,  
							  fit_intercept=True,  
							  l1_ratio=0.3877551020408163,  
							  learning_rate='invscaling', loss='log',  
							  max_iter=1000, n_jobs=1, penalty='none',  
							  power_t=0, random_state=None,  
							  tol=0.01))],  
	verbose=False), '36': Pipeline(memory=None,  
	steps=[('sparsenormalizer',  
		 <azureml.automl.runtime.shared.model_wrappers.SparseNormalizer object at 0x7f3ef5beec50>),  
		('xgboostclassifier',  
		 XGBoostClassifier(base_score=0.5, booster='gbtree',  
						   colsample_bylevel=1, colsample_bynode=1,  
						   colsample_bytree=0.6, eta=0.2, gamma=0,  
						   learning_rate=0.1, max_delta_step=0,  
						   max_depth=2, max_leaves=0,  
						   min_child_weight=1, missing=nan,  
						   n_estimators=100, n_jobs=1, nthread=None,  
						   objective='reg:logistic', random_state=0,  
						   reg_alpha=0, reg_lambda=1.0416666666666667,  
						   scale_pos_weight=1, seed=None, silent=None,  
						   subsample=0.9, tree_method='auto',  
						   verbose=-10, verbosity=0))],  
	verbose=False), '40': Pipeline(memory=None,  
	steps=[('maxabsscaler', MaxAbsScaler(copy=True)),  
		('lightgbmclassifier',  
		 LightGBMClassifier(boosting_type='goss', class_weight=None,  
							colsample_bytree=0.6933333333333332,  
							importance_type='split',  
							learning_rate=0.07368684210526316,  
							max_bin=380, max_depth=3,  
							min_child_samples=1591, min_child_weight=2,  
							min_split_gain=0.2631578947368421,  
							n_estimators=800, n_jobs=1, num_leaves=32,  
							objective=None, random_state=None,  
							reg_alpha=0.5263157894736842,  
							reg_lambda=0.5789473684210527, silent=True,  
							subsample=1, subsample_for_bin=200000,  
							subsample_freq=0, verbose=-10))],  
	verbose=False)
```

Each model is weighted respectively according to the following values:

```
[0.09090909090909091, 0.09090909090909091, 0.18181818181818182, 0.18181818181818182, 0.09090909090909091, 0.09090909090909091, 0.18181818181818182, 0.09090909090909091]
```

## Pipeline comparison

The Voting Ensemble model outperform the Logistic Regression one with an accuracy of 0.9181 against 0.9129.

The differences in architecture are huge:
- **HyperDrive** starts multiple runs each of which trains the LogisticRegression model using different tuples of hyper-parameters
- **AutoML** starts multiple runs each of which executes complex pipelines with choices of hyper-parameters, models and configuration details

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

Here the possible improvements that can boost the models performance:
- Better data transformation / engineering
    - Variable "duration" is not known before a call is performed. Also, after the end of the call y is obviously known. So this variable should be removed to avoid data leaks
    - Variable "pdays" is the number of days that passed by after the client was last contacted from a previous campaign. As 999 means client was not previously contacted, a dummy variable "contacted_before" (0 if 999, 1 otherwise) could improve the model performance.
    - Numeric variables having a non-negligible number of unique values (like "duration", "cons.price.idx", etc.) should be binned
- Stratified strategies to be adopted as the target variable is imbalanced
    - Using of the *stratify* parameter in the function *train_test_split()* into the *train.py* script
    - Using the *cv_split_column_names* parameter in the AutoMLConfig settings in order to use calculated columns as placeholders for a custom stratified cross-validation
- Increase the *max_total_runs* parameter of HyperDriveConfig
- Increase the *experiment_timeout_minutes* parameter of the AutoMLConfig
- Using a primary metric that takes into account the unbalance of the classes (for both HyperDriveConfig and AutoMLConfig)
    - Otherwise an oversampling technique like SMOTE could be used to balance the classes distribution
    

## Proof of cluster clean up

```
compute_cluster.delete()
```
<!-- #endregion -->
