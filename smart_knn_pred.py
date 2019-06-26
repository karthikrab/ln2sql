import knn_impute
import numpy as np
import pandas as pd
from flask import Flask, request, Response, make_response, jsonify, render_template
import csv
import ast
from ln2sql.config import DATABASE_CONFIG as DB,sqldts
import psycopg2 as pg
import pandas.io.sql as psql
import json
import uuid
from ln2sql.ln2sql import Ln2sql
from flask_cors import CORS, cross_origin
from flask_jsonpify import jsonpify
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData
from model_builder import model_builder
from gensim.models import KeyedVectors
import spacy
from rephrasing import rephrase
from smalltalk import SmallTalk
from preprocess import chat
import joblib

filename = 'glove.6B.300d.txt.word2vec'
#model = KeyedVectors.load_word2vec_format(filename, binary=False)
#nlp = spacy.load('en_core_web_md')
#st = SmallTalk(nlp)
nlp=''
model=''
st = ''
# Initialize the Flask application
app = Flask(__name__)



@app.route('/api/login', methods=['POST'])
def login():
    user = request.form['username']
    password = request.form['password']
    connection = pg.connect("host='"+DB['host']+"' dbname="+DB['dbname']+" user="+DB['user']+" password='"+DB['password']+"'")
    df = pd.read_sql_query("select user_id from users where username='"+user+"' and password='"+password+"';",con=connection)
    resp = { 'success' : True, 'user_id':1} if len(df)==1 else { 'success' : False, 'message':'invalid credentials'}
    return Response(response=json.stringify(resp), status = 200)

@app.route('/api/testcon',methods=['POST'])
def testcon():
    resp = {'filename': request.json['filename']}
    return Response(response=json.dumps(resp), status = 200)


@app.route('/api/chat',methods=['POST'])
def budchat():
    resp=rephrase(request.json['sentence'],request.json['user_id'],model)
    print(resp)
    if( resp['Class'] == 'Prediction'):
        connection = pg.connect("host='"+DB['host']+"' dbname="+DB['dbname']+" user="+DB['user']+" password='"+DB['password']+"'")
        df = pd.read_sql_query("select column_name from column_meta where table_meta_id='"+request.json['table_meta_id']+"';",con=connection)
        resp['Columns'] = df['column_name'].values.tolist()
        return Response(response=json.dumps(resp), status = 200)
    else:
        try:
            ln2sql = Ln2sql(
                database_path='',
                language_path='lang_store/english.csv',
                userid=resp['user_id']
                ).get_query(resp['Mapped Query'])
            connection = pg.connect("host='"+DB['host']+"' dbname="+DB['dbname']+" user="+DB['user']+" password='"+DB['password']+"'")
            df = pd.read_sql_query(ln2sql,con=connection)
            res = {'data':df.head(20).values.tolist(),'Reply':'Here you go'}
            if(len(res['data'])==0):
                response={'Reply':'We are not able to find any relevant Data'}
                return Response(response=json.dumps(response), status = 200)
            res['data'].insert(0,df.columns.tolist())
            resp = jsonpify(res)
            return (resp)
        except Exception as e:
            print(e)
            response=chat(resp['Mapped Query'],resp['user_id'],st)
            return Response(response=json.dumps(response), status = 200)

@app.route('/api/predict',methods=['POST'])
def budpredict():
    connection = pg.connect("host='"+DB['host']+"' dbname="+DB['dbname']+" user="+DB['user']+" password='"+DB['password']+"'")
    dl = pd.read_sql_query("select table_name from table_meta where table_meta_id='"+request.json['table_meta_id']+"';",con=connection)
    tname=dl['table_name'][0]
    target=request.json['target'][0]
    ohl = {'poutcome': ['poutcome_other', 'poutcome_success', 'poutcome_unknown'], 'marital': ['marital_married', 'marital_single'], 'education': ['education_secondary', 'education_tertiary', 'education_unknown'], 'default': ['default_yes'], 'housing': ['housing_yes'], 'contact': ['contact_telephone', 'contact_unknown'], 'month': ['month_aug', 'month_dec', 'month_feb', 'month_jan', 'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep'], 'loan': ['loan_yes'], 'y': ['y_yes'],'job':['job_blue-collar','job_entrepreneur','job_housemaid','job_management','job_retired','job_self-employed','job_services','job_student','job_technician','job_unemployed','job_unknown']}
    jsondata = request.json['coldata']
    jsondata['']=0
    df=pd.DataFrame(jsondata,index=[0])    
    for cols in list(df.columns):
        if(cols in ohl.keys()):
            df_new = pd.DataFrame(columns=ohl[cols])
            for i,r in df.iterrows():
                df_new.loc[i] = [0]*len(ohl[cols])
                print(cols,df[cols][i])
                value = cols+'_'+df[cols][i]
                if (value in ohl[cols]):
                    df_new[value][i]=1
            df=df.drop(columns=cols)
            df=pd.concat([df, df_new], axis=1, sort=True)
    with open('tpot_class_'+tname+'.json','r') as fp:
        col=json.load(fp)
    mod=joblib.load('tpot_pipeline_'+tname+'_'+target+'.pkl')
    md=mod.predict(df)
    res = {'data':[[target],[col[target][int(md[0])]]],'Reply':'Here you go'}
    return Response(response=json.dumps(res), status = 200)
    
@app.route('/api/testt2db',methods=['POST'])
@cross_origin(origin='*')
def t2db():
    print('inside');
    sentence=request.form['sentence']
    ln2sql = Ln2sql(
        database_path='',
#	database_path='../people.sql',
        language_path='lang_store/english.csv',
        userid='fdf19c90-914d-11e9-8a3a-05a1d557cb54'
    ).get_query(sentence)
    connection = pg.connect("host='"+DB['host']+"' dbname="+DB['dbname']+" user="+DB['user']+" password='"+DB['password']+"'")
    df = pd.read_sql_query(ln2sql,con=connection)
    #print(df)   
    resp = jsonpify(df.values.tolist())
    return (resp)
#   return Response(response=json.dumps(resp), status = 200)
#    return Response(response='asd', status = 200)



@app.route('/api/smart_impute', methods=['POST'])
def smart_impute():
    data = pd.read_csv(request.json['filename'])
    data.columns = [x.lower() for x in data.columns]
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
    engine = create_engine('postgresql://'+DB['user']+':'+DB['password']+'@'+DB['host']+'/'+DB['dbname'],echo=False)    
    data.to_sql(request.json['table_name'],con=engine,if_exists = 'replace', index=False)
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
        print(outliers)
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
    
    data.to_sql(request.json['table_name']+"_im",con=engine,if_exists = 'replace', index=False)
    metadata = MetaData()
    col_meta = Table('column_meta', metadata,
        Column('column_meta_id', String, primary_key=True),
        Column('table_meta_id', String ),
        Column('column_name',String),
        Column('mappings', String),
	Column('type',String)  )
    for c,t in data_types:
        ins = col_meta.insert().values(column_meta_id=uuid.uuid4(),table_meta_id=request.json['table_meta_id'],column_name=c,mappings='',type=sqldts[t])
        engine.execute(ins)
    model_builder(request.json['filename'],request.json['table_name'])
    engine.execute("update table_meta set status = 'Processed' where table_meta_id='"+request.json['table_meta_id']+"';")
    return Response(response='Processed', status = 200)


# start flask app
app.run(host="0.0.0.0", port=1043)
