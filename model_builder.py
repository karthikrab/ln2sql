import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.metrics.scorer import make_scorer
import dill as pickle
import os
import json

col_one_hot={}
col_classes={}

def my_custom_accuracy(y_true, y_pred):
    return float(sum(y_pred == y_true)) / len(y_true)


def train_tpot_classifier(dat, target,tname):
    # print(target)
    dat.to_csv('tpot_inp_'+tname+"_"+ str(target) + '.csv')
    df = dat.rename(columns={target: 'class'})
    X = df.drop(columns='class')
    y = df['class'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    my_custom_scorer = make_scorer(my_custom_accuracy, greater_is_better=True)
    # print('X_train)
    tpot = TPOTClassifier(generations=1, population_size=20, verbosity=2,
                          scoring=my_custom_scorer)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_pipeline_' +tname+"_"+ str(target) + '.py')
    os.system('/home/dev/T2DB/ln2sql/fixtpot.sh tpot_pipeline_' +tname+"_"+ str(target) + '.py '+str(target))
    os.system('python3 tpot_pipeline_' +tname+"_"+ str(target) + '.py ' 'tpot_inp_'+tname+"_"+ str(target) + '.csv')

def train_tpot_regressor(dat, target):
    # print(target)
    df = dat.rename(columns={target: 'class'})
    X = df.drop(columns='class')
    y = df['class'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    my_custom_scorer = make_scorer(my_custom_accuracy, greater_is_better=True)

    tpot = TPOTRegressor(generations=1, population_size=20, verbosity=2,
                         scoring=my_custom_scorer)
    tpot.fit(X_train, y_train)
    #tpot.export('tpot_mnist_pipeline_'+tname+"_"+ str(target) + '.py')
    with open('tpot_pkl_' +tname+"_"+ str(target) + '.pkl','wb') as f:
        pickle.dump(tpot,f)

def one_hot(dat, columns):
    final_dat = pd.DataFrame()
    for column_name in columns:
        dat_col = pd.get_dummies(dat[column_name], prefix=column_name, drop_first=True)
        final_dat = pd.concat([final_dat, dat_col], axis=1, sort=True)
        col_one_hot[column_name]=list(dat_col.columns)
    print(col_one_hot)
    return final_dat


def model_builder(filename,tname):
    data = pd.read_csv(filename)
    string_colums = []
    numeric_columns = []
    float_column = []
    column_list = []
    misc_columns = []
    numeric_list = []
    float_list = []
    for i, v in data.dtypes.items():
        if v == np.dtype(object):
            d_type = i, 'String'
            string_colums.append(d_type)
        elif np.issubdtype(v, int):
            d_type = i, 'Integer'
            numeric_columns.append(d_type)
        elif v == np.dtype(float):
            d_type = i, 'Float'
            float_column.append(d_type)
        else:
            d_type = i, v
            misc_columns.append(d_type)

    for i, v in string_colums:
        if data[i].nunique() < 20:
            column_list.append(i)
            col_classes[i]=list(data[i].unique())

    for i, v in numeric_columns:
        if data[i].nunique() <= 4:
            numeric_list.append(i)
            col_classes[i]=list(data[i].unique())
    with open('tpot_class_' +tname + '.json','w')as f: 
        json.dump(col_classes,f)
    for i, v in float_column:
        float_list.append(i)

    for items in column_list:
        iters = 0
        target = data[items].copy()
        one_hot_list = [x for x in column_list if x != items]
        depend_data = data.drop(columns=items)
        one_hot_data = one_hot(depend_data, one_hot_list)
        final_data = pd.concat([one_hot_data, target], axis=1, sort=True)
        final_data = pd.concat([final_data, data.drop(columns=column_list)], axis=1, sort=True)
        for i in data[items].unique():
            final_data[items] = final_data[items].mask(final_data[items] == i, iters)
            iters += 1
        if target.dtype == np.dtype(object):
            final_data[items] = final_data[items].astype('category')
        train_tpot_classifier(final_data, items,tname)

    for items in numeric_list:
        target = data[items].copy()
        one_hot_data = one_hot(data[column_list], column_list)
        final_data = pd.concat([one_hot_data, target], axis=1, sort=True)
        depend_data = data.drop(columns=column_list, axis=1)
        depend_data = depend_data.drop(columns=items, axis=1)
        final_data = pd.concat([final_data, depend_data], axis=1, sort=True)
        train_tpot_classifier(final_data, items,tname)

    for items in float_list:
        target = data[items].copy()
        one_hot_data = one_hot(data[column_list], column_list)
        final_data = pd.concat([one_hot_data, target], axis=1, sort=True)
        depend_data = data.drop(columns=column_list, axis=1)
        depend_data = depend_data.drop(columns=items, axis=1)
        final_data = pd.concat([final_data, depend_data], axis=1, sort=True)
        train_tpot_regressor(final_data, items,tname)


if __name__ == "__main__":
    model_builder('bank.csv')
