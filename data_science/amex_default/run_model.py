import pandas as pd
import numpy as np
import gc
from  matplotlib import pyplot as plt
import seaborn as sns
import sys,os
import sklearn
import xgboost as xgb
from xgboost import XGBClassifier,plot_importance

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(r"C:\Users\kaiyu\Desktop\miller")

from chenqian_tools.hdf_helper import *

def check_df_info(df):
    from humanize import naturalsize
    size = df.memory_usage(deep='True').sum()
    print(size)
    print(naturalsize(size))
    return df

def prepro_df(df):
    '''

    '''
    df['S_2'] = pd.to_datetime(df.S_2)
    features = [x for x in df.columns.values if x not in ['customer_ID', 'target']]
    df['n_missing'] = df[features].isna().sum(axis=1)
    df_out = df.groupby(['customer_ID']).nth(-1).reset_index(drop=True)
    del df
    _ = gc.collect()
    df_out = degrade_incuracy(df_out,degrage_level = 'low')
    df_out = check_df_info(df_out)
    return df_out



'''
binary_features: ['B_31','D_87']

P_2:是当前信用评级的模型
'''

data_path = '''D:\\amex-default-prediction'''

train = 'train_data.h5'
target_p = 'train_labels.h5'
test = 'test_data.h5'

# get_target
h5_file = h5_helper(f'''{data_path}\{target_p}''',)
target_ = h5_file.get_table('data')
# target_ = prepro_df(target_)

h5_file = h5_helper(f'''{data_path}\{train}''',)
train_ = h5_file.get_table('data')
train_ = train_.merge(target_,on='customer_ID')
train_ = prepro_df(train_)

h5_file = h5_helper(f'''{data_path}\{test}''',)
test_ = h5_file.get_table('data')
# test_ = test_.merge(target_,on='customer_ID')
test_ = prepro_df(test_)

# corr_matrix
corr = train_.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(1, 1, figsize=(20,14))
sns.heatmap(corr, mask=mask, center=0, linewidths=1, annot=True, fmt=".2f", ax=ax)
plt.show()

# Remove highly correlated features
corr_matrix = corr.abs()

# Create a boolean mask and apply it
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)

# List column names of highly correlated features (r > 0.7)
to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.7)]
print(f'Number of features: {len(to_drop)} \n {to_drop}'
)
# DataSet prepare for analysis
target = train_['target']
train_df = train_.drop(['target'], axis=1)
train_df.shape

# The numpy metric for evaluation has been taken from @rohanrao AMEX: Competition Metric Implementations
def amex_metric_numpy(y_true: np.array, y_pred: np.array) -> float:

    # count of positives and negatives
    n_pos = np.sum(y_true)
    n_neg = y_true.shape[0] - n_pos

    # sorting by describing prediction values
    indices = np.argsort(y_pred)[::-1]
    preds, target = y_pred[indices], y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = np.sum(target[four_pct_mask]) / n_pos

    # weighted gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)

num_features = train_df._get_numeric_data().columns
X, y = train_df._get_numeric_data(), target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state=100,
                                                    stratify=y)


# Instantiate the classifier. Can switch on parameter tree_method='gpu_hist' in the future
xg_cl = XGBClassifier(objective='binary:logistic',
                      n_estimators=10,
                      seed=123,
                      use_label_encoder=False,
                      eval_metric='aucpr', # updated to make use of the aucpr option
                      early_stopping_rounds=10,
                      tree_method='auto',
                      enable_categorical=False
                      )
eval_set = [(X_test, y_test)]
xg_cl.fit(X_train, y_train, eval_set=eval_set, verbose=True)


# Predict the labels of the test set
preds = xg_cl.predict(X_test)
preds_prob = xg_cl.predict_proba(X_test)[:,1]

# Compute accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, preds)
print(f'accuracy: {accuracy: .2%}')


# Review the important features
# print(xg_cl.feature_importances_)
def plot_features(booster, figsize, max_num_features=15):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax, max_num_features=max_num_features)
plot_features(xg_cl, (10,14))
plt.show()


# ----------------------- boosting and CV_methods --------------------------- #
# 现在进行交叉验证
# Understanding weighted class imbalance
from collections import Counter

counter = Counter(y)
print(counter)

# estimate scale_pos_weight value
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)

# Create the DMatrix from X and y: churn_dmatrix
d_train = xgb.DMatrix(data=X_train, label=y_train)
d_test = xgb.DMatrix(data=X_test, label=y_test)
xgd_test = xgb.DMatrix(data=test_._get_numeric_data())

# Create the parameter dictionary: params. NOTE: have to explicitly provide the objective param
params = {"objective":"binary:logistic",
          "max_depth": 6,
          "eval_metric":'aucpr', # updated to make use of the aucpr option
          "tree_method":'auto',
          "predictor": 'auto',
#           "scale_pos_weight": 30,
        }

# Reviewing the AUC metric
# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain=d_train, params=params,
                    nfold=5, num_boost_round=10,
                    metrics="aucpr", as_pandas=True, seed=123)
print(cv_results)
print((cv_results["test-aucpr-mean"]).iloc[-1])


# Review the train method
params = {
    "objective":"binary:logistic",
    "max_depth": 6,
    "eval_metric":'aucpr',
    "tree_method":'auto',
    "predictor": 'auto',
#     "scale_pos_weight": 30,
}

# train - verbose_eval option switches off the log outputs
xgb_clf = xgb.train(
                    params,
                    d_train,
                    num_boost_round=5000,
                    evals=[(d_train, 'train'), (d_test, 'test')],
                    early_stopping_rounds=10,
                    verbose_eval=0
                )

# predict
y_pred = xgb_clf.predict(d_test)

# Compute and print metrics
print('Metric Evaluation Values\n')
print(f'Numpy: {amex_metric_numpy(y_test.to_numpy().ravel(), y_pred)}')

# Rank Order table
rank_data = train_df._get_numeric_data()
xgd_rank = xgb.DMatrix(data=train_df._get_numeric_data())
rank_data['prob'] = xgb_clf.predict(xgd_rank)
rank_data['target'] = target
rank_data.head()

# First create the decile value by prob
rank = rank_data.loc[:, ['target', 'prob']]
rank["ranks"] = rank['prob'].rank(method="first")

# The notes displayed here had related to only using the X_test dataframe. With the train_df being used we can try using the probabilities again
# First method bunchs the final three buckets into one as there are a low of low probs
rank['decile'] = pd.qcut(rank.prob, 10, labels=False, duplicates='drop')
# Second method aims to use the rank method, however the nature of this rank is still random
# An alternative for this piece might be to put the 'prob' in order and sort by the target
# rank['decile'] = pd.qcut(rank.ranks, 10, labels=False)
# Reviewing the lowest probability
min_prob = np.min(rank.prob)
rank.loc[(rank.prob == min_prob)].head()

# Create a rank_order table
def rank_order(df: pd.DataFrame, y: str, target: str) -> pd.DataFrame:
    rank = df.groupby('decile').apply(lambda x: pd.Series([
        np.min(x[y]),
        np.max(x[y]),
        np.mean(x[y]),
        np.size(x[y]),
        np.sum(x[target]),
        np.size(x[target][x[target]==0]),
    ],
        index=(["min_prob","max_prob","avg_prob",
               "cnt_cust","cnt_def","cnt_non_def"])
    )).reset_index()
    rank = rank.sort_values(by='decile', ascending=False)
    rank["drate"] = round(rank["cnt_def"]*100/rank["cnt_cust"], 2)
    rank["cum_cust"] = np.cumsum(rank["cnt_cust"])
    rank["cum_def"] = np.cumsum(rank["cnt_def"])
    rank["cum_non_def"] = np.cumsum(rank["cnt_non_def"])
    rank["cum_cust_pct"] = round(rank["cum_cust"]*100/np.sum(rank["cnt_cust"]), 2)
    rank["cum_def_pct"] = round(rank["cum_def"]*100/np.sum(rank["cnt_def"]), 2)
    rank["cum_non_def_pct"] = round(rank["cum_non_def"]*100/np.sum(rank["cnt_non_def"]), 2)
    rank["KS"] = round(rank["cum_def_pct"] - rank["cum_non_def_pct"],2)
    rank["Lift"] = round(rank["cum_def_pct"] / rank["cum_non_def_pct"],2)
    return rank

rank_gains_table = rank_order(rank, "prob", "target")
test_preds = xgb_clf.predict(xgd_test)
test_preds.view()