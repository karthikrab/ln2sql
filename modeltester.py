import pandas as pd
import joblib
import os
import json
ohl = {'poutcome': ['poutcome_other', 'poutcome_success', 'poutcome_unknown'], 'marital': ['marital_married', 'marital_single'], 'education': ['education_secondary', 'education_tertiary', 'education_unknown'], 'default': ['default_yes'], 'housing': ['housing_yes'], 'contact': ['contact_telephone', 'contact_unknown'], 'month': ['month_aug', 'month_dec', 'month_feb', 'month_jan', 'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep'], 'loan': ['loan_yes'], 'y': ['y_yes'],'job':['job_blue-collar','job_entrepreneur','job_housemaid','job_management','job_retired','job_self-employed','job_services','job_student','job_technician','job_unemployed','job_unknown']}
df=pd.read_csv('test2.csv')
print(df)
for cols in list(df.columns):
    if(cols in ohl.keys()):
         df_new = pd.DataFrame(columns=ohl[cols])
         for i,r in df.iterrows():
             df_new.loc[i] = [0]*len(ohl[cols])
             value = cols+'_'+df[cols][i]
             if (value in ohl[cols]):
                 df_new[value][i]=1
         df=df.drop(columns=cols)
         df=pd.concat([df, df_new], axis=1, sort=True)
print(list(df.columns))
print(len(list(df.columns)))
#df.to_csv('test3_new.csv')
#os.system("python3 tpot_pipeline_people_m2d07_job_test.py test3_new.csv job")
with open('tpot_class_people_nc0g1.json','r') as fp:
    col=json.load(fp)
mod=joblib.load('tpot_pipeline_people_nc0g1_job.pkl')
md=mod.predict(df)
print(md[0])
print(col['job'][int(md[0])])
print(mod.classes_)
print(mod.predict_proba(df))
