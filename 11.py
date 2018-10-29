import compare.topK as topk
import pandas as pd
import numpy as np
import json


# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer

# df = pd.read_csv('practice2.csv')
# #选前10篇文章测试特征提取
# df = df.head(3)

# size = 5
# tfidf = TfidfVectorizer(sublinear_tf=True,norm='l2', encoding='latin-1', 
#                 ngram_range=(1, 2))
# features = tfidf.fit_transform(df.aritcle).toarray()
# names = np.array(tfidf.get_feature_names())
# #names = tfidf.get_feature_names()
# # print(names)
# # print(tfidf.vocabulary_)
# # 输出一篇文章的feature，type list
# # if size > features.shape[1]: 
# #     return False
# for row in range(0,features.shape[0]):
#     onerow = features[row,]
#     # onerow_size = topk.mySelect(list(onerow),size)
#     idx = np.argpartition(onerow, -size)[-size:]
#     oneFeature = names[idx]
#     print(oneFeature)

# df = pd.read_csv('evaluation.csv')

# from decimal import *
# for time in df.time:
# #    Decimal(time).quantize(Decimal('0.00'))


# import re
# s1 = './tag0/129793'
# s2 = './tag10/129793'
# number = re.search('\d+',s2)
# print(number.group(0))


with open('better_trank.json','r') as f:
    better = json.load(f)
with open('worse_trank.json', 'r') as f:
    worse = json.load(f)
print(len(better.keys()))
print(len(worse.keys()))





# df = pd.read_csv('thucke_score_betterrank.csv')
# print(df.describe())
'''        rank score
count  1902.000000
mean      0.078319
std       0.108070
min       0.000000
25%       0.000000
50%       0.000000
75%       0.100000
max       0.666667

         tfi score
count  1902.000000
mean      0.079653
std       0.118696
min       0.000000
25%       0.000000
50%       0.000000
75%       0.100000
max       0.700000

       thucke score#sample from better_tfi.json
count     28.000000
mean       0.425000
std        0.132288
min        0.100000
25%        0.300000
50%        0.400000
75%        0.500000
max        0.700000

       thucke score#better textrank
count     13.000000
mean       0.437363
std        0.113620
min        0.285714
25%        0.400000
50%        0.400000
75%        0.500000
max        0.700000

'''



