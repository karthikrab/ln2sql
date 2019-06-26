# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:06:45 2019

@author: Vishnu.Kumar1
"""

from nltk import pos_tag, word_tokenize
import pandas as pd
from ln2sql.config import DATABASE_CONFIG as DB
import psycopg2 as pg
import pandas.io.sql as psql

def getcoldetails(user_id):
    connection = pg.connect("host='"+DB['host']+"' dbname="+DB['dbname']+" user="+DB['user']+" password='"+DB['password']+"'")
    df = pd.read_sql_query("select table_meta_id,table_name from table_meta where user_id='"+user_id+"';",con=connection)
    tablemapping = {}
    for i,r in df.iterrows():
        rd=pd.read_sql_query("select * from column_meta where table_meta_id ='"+r['table_meta_id']+"';",con=connection)
        tablemapping[r['table_name']] =[{row['column_name']:row['mappings'].split(',')} for j,row in rd.iterrows()]
    return tablemapping

def SimilarGlove(word,model):
    try:
        result = model.most_similar(str(word), topn=5)
        result = [i[0] for i in result]
        return result
    except:
        return []

def POS(text):
    tagged = pos_tag(word_tokenize(text))
    verb = [i[0] for i in tagged if i[1]=='VB']
    noun = [i[0] for i in tagged if i[1]=='NN' or i[1]=='NNS']
    adj = [i[0] for i in tagged if i[1]=='JJ']
    num = [i[0] for i in tagged if i[1]=='CD']
    return verb, noun, adj, num

def ColumnNames(chatid,model):
    data = getcoldetails(chatid)
    l = []
    for key, value in data.items():
        for i in value:
            for k, v in i.items():
                di = {}
                di[k] = [SimilarGlove(k,model)] + Ontology(v,model) + [[k]]
                l.append(di)
    j = []
    for i in l:
        for k,v in i.items():
            for m in v:
                d = {}
                d[k] = m
                j.append(d)
    return j

def Ontology(array,model):
    mappings = []
    for i in array:
        mappings.append(SimilarGlove(i,model) + [i])
    return mappings

def chat(text,chat_id,st):
    response = st.get_reply(text)
    result ={}
    result['user_id'] = chat_id
    result['Reply'] = response
    return result
