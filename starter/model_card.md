# Model Card

- Supervised learning classification model for predicting whether the annual salary of a person is below or above 50k USD. It models the US Census Income Data Set from https://archive.ics.uci.edu/ml/datasets/census+income

## Model Details

- It implements the training and inference of sklearn classification models
- Training implements model parameters optimization using GridSearchCV
- Model types:  DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, SVC are available
- Model date: Jan. 27th.  2022
- Model version: 1.0.0
- See https://www.kaggle.com/c/census-income for more information on this topic
- License: MIT
- Contact: esm2046@gmail.com

## Intended Use

- This is the implementation of project **Deploying a Machine Learning Model on Heroku with FastAPI**  from Udacity Machine Learning DevOps Engineer Nanodegree-Course.
- Used for evaluating Module 4 (Deploying a Scalable ML Pipeline in Production) of the above mentioned course

## The data

- US Census Income Data Set from https://archive.ics.uci.edu/ml/datasets/census+income
- Cleaned data with no invalid records or values
- Mixture of categorical and continuous data
- Categorical data encoded with OneHotEncoder
- Continuous data scaled with MinMaxScaler
- Training data: 80%, Evaluation data: 20%
- Evaluation data completely torn out of the training process
- Klearn's StratifiedShuffleSplit applied to split the data in order  to keep the proportion of the two classes in both training and testing sets

## Evaluation
- Performed with 20% of the Census Income Data Set.
- Models can be evaluated on the overall test data (model_performance(...)) or on data slices based on the categorical features (model_performance_on_slices(...))

## Metrics
- The models are evaluated by precision, recall, F-beta and accuracy.
- accuracy has been included in order to compare results with a kaggle competition on this data ( https://www.kaggle.com/c/census-income)
- The following models were tuned:
    - RandomForestClassifier: **rfc**
    - LogisticRegression: **lrc**
    - Support Vectors Classifier (SVC): **svc**
    - DecisionTreeClassifier: **dtc**
    - Best kaggle model (Karthi Kuppan): **bkm**
- Models performance were calculated with the script trained_model_metrics.py
- The scores are as follow:

    | Model/Metrics | rfc | lrc | svc | dtc | bkm |
    | ------- | ------ | ------ | ------ | ------ | ------ |
    | precision     | 0.7930 | 0.7262 | 0.7476 | 0.7930 |  |
    | recall        | 0.5196 | 0.6060 | 0.5763 | 0.5772 |  |
    | f-beta        | 0.6278 | 0.6607 | 0.6508 | 0.6599 |  |
    | accuracy      | 0.8480 | 0.8494 | 0.8488 | 0.8567 | 0.87584 |

## Ethical Considerations
- The data contains sensitive personal information which might be protected by law and hence it would require an express consent to be used. Therefore data anonymization should be a prerrequisite before distributing and modeling this data
- In certain scenarios, eg. credit assignment, the outcome of the model could have a high personal impact. Therefore, this prediction model should only be used as a support tool for decision-making. 

## Caveats and Recommendations

- Intuively, salary income is highly related with the square meters of the flat/house where a person lives. This kind of information is missing in the data . If provided, it would lead to better prediction results 
- Data only contains gender as male/not male. Data across a spectrum of genders might turn in better results. However, be aware of the ethical considerations mentioned above

## Additional Information

- Further information regarding model cards cab be found in:  https://arxiv.org/pdf/1810.03993.pdf