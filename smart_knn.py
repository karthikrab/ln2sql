import knn_impute
import numpy as np
import pandas as pd
from flask import Flask, request, Response, make_response, jsonify, render_template
import csv
import ast

# Initialize the Flask application
app = Flask(__name__)


@app.route('/api/smart_impute', methods=['POST'])
def smart_impute():
    data = pd.read_csv(request.files['file'])
    data_types = []
    types = ['Float', 'Integer']
    numeric_list = []
    drop_list = []
    for i, v in data.dtypes.items():
        if v == np.dtype(object):
            d_type = i, 'String'
        elif np.issubdtype(v, int):
            d_type = i, 'Integer'
        elif v == np.dtype(float):
            d_type = i, 'Float'
        else:
            d_type = i, v
        data_types.append(d_type)  # LIST CONTAINING THE INFO OF DATA TYPES

    for i, v in data_types:
        if v in types and data[i].isnull().sum(axis=0) > 0:
            numeric_list.append(i)  # LIST TO BE IMPUTED

    for items in numeric_list:
        column_name = items
        data_1 = data[column_name]
        outliers = []
        outliers.append(column_name)
        threshold = 5
        mean_1 = np.mean(data_1)
        std_1 = np.std(data_1)

        for y in data_1:
            z_score = (y - mean_1) / std_1
            if np.abs(z_score) > threshold:
                outliers.append(threshold)
            outliers.append(y)
        data[column_name] = outliers[1:]

    nulls = []
    for i, v in data.isnull().sum(axis=0).items():
        null_record = i, v
        nulls.append(null_record)
    for i, v in nulls:
        if not v/len(data[i]) < 0.5:
            drop_list.append(i)

    for items in numeric_list:
        column_name = items
        processed = data[column_name].copy()
        data_old = pd.DataFrame(columns=[column_name + '_OLD ']).copy()
        columns_to_overwrite = [column_name]
        data_old[column_name + '_OLD '] = data[column_name]
        # print(processed.count())
        while processed.isnull().values.any():
            processed = knn_impute.knn_impute(target=processed,
                                              attributes=data.drop(drop_list, 1),
                                              aggregation_method="mode", k_neighbors=5,
                                              numeric_distance='euclidean',
                                              categorical_distance='hamming',
                                              missing_neighbors_threshold=0.8)

        data.drop(labels=columns_to_overwrite, axis="columns", inplace=True)
        data[columns_to_overwrite] = processed[columns_to_overwrite]
        data.join(data_old)
        data = pd.concat([data, data_old], axis=1)
        filename = column_name + '.csv'
        resp = make_response(data.to_csv())
        resp.headers["Content-Disposition"] = ("attachment; filename=%s" % filename)
        resp.headers["Content-Type"] = "text/csv"

# start flask app
app.run()
