# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 15:32:39 2019

@author: Vishnu.Kumar1
"""

from preprocess import ColumnNames, POS
from nltk import pos_tag, word_tokenize
import json

def rephrase(query, chatid,model):
    query = query.lower()
    query = query.replace('?', '')
    query = query.replace('.', '')
    tagged = pos_tag(word_tokenize(query))
    query = query.split()
    column_names = ColumnNames(chatid,model)
    column = []
    for j in column_names:
        for key, value in j.items():
            for i in value:
                if i in query:
                    ind = query.index(i)
                    column.append(key)
                    query.remove(i)
                    query.insert(ind, key)
    new_query = ' '.join(query)
    verb, noun, adj, num = POS(new_query)
    keywords = ['predict', 'forecast', 'after', 'post']
    clas = [i[0] for i in tagged if i[1]=='MD']
    keys = [i for i in keywords if i in new_query]
    dict_result = {}
    dict_result['Mapped Query'] = new_query
    dict_result['user_id'] = str(chatid)
    if len(clas) > 0 or len(keys) > 0:
        dict_result['Class'] = "Prediction"
        dict_result['Target Variable'] = list(set(noun)) + list(set(adj))
        dict_result['Reply'] = 'Please fill up this form'
    else:
        dict_result['Class'] = "Analytics"
        dict_result['Target Column'] = list(set(column))
    try:
        with open(str(chatid) + ".json", "r") as read_file:
            data = json.load(read_file)
    except:
        data = []
    with open(str(chatid) + ".json", 'w') as fout:
        json.dump(dict_result , fout)
    dict_result['Context'] = data
    return dict_result

