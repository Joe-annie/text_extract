import json
from lxml import etree
import os
import pandas as pd
import numpy as np
import re

mydic_list = []
with open("1008_ch.json","r") as load_f:
    dic_json = json.load(load_f)
num_list = list(dic_json.keys())
for index,num in enumerate(num_list):
    if index > 0: break
    atrr = dic_json[num]
    keywords = atrr['tag']
    dic_keywords = json.loads(keywords)#从字典中获得的keyword是str，用json换成dic，or pickle
    #guard 
    if 'disease_keywords' not in dic_keywords.keys():continue
    disease = dic_keywords['disease_keywords']
    treatment = dic_keywords['treat_methods']
    # with open('mydic.txt','a+') as f:
    #     for dic in disease:
    #         word = dic['keyword']
    #         if word not in mydic_list:
    #             f.write(word+'\n')
    #             mydic_list.append(word)
    #     for dic in treatment:
    #         word = dic['keyword']
    #         if word not in mydic_list:
    #             f.write(word+'\n')
    #             mydic_list.append(word)
