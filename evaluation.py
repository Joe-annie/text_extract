from compare.keyword import keyword
import pandas as pd
import time
import json
# n = 20
# mykeyword = keyword('jieba','thucke',)
# mykeyword.getKeywords('1008_ch.json',10,10)
# print(mykeyword.TT)


# print(len(mykeyword.numlist))
# print(len(mykeyword.taglist))
# print(len(mykeyword.featurelist))

# dataframe = pd.DataFrame({'number':mykeyword.numlist,'tag':mykeyword.taglist,'feature':mykeyword.featurelist})
# dataframe.to_csv("jieba_rank.csv",index=False)



#evaluate the correction
def getPrecision(mykeyword):#after getKeywords(..)
    p = 0#alldoc
    true = mykeyword.taglist
    pred = mykeyword.featurelist
    for i in range(len(pred)):
        tp = 0#one article
        true_len = len(true[i])
        pred_len = len(pred[i])
        for feature in pred[i]:
            if feature in true[i]:
                tp += 1
        if pred_len == 0:
            tp = 1.0 if true_len == 0 else 0.0
        else:
            tp = tp / pred_len#一篇文章的平均准确度
            p += tp
    return p/len(pred) if len(pred) > 0 else 0


def getBetterpList(mykeyword):#undo
    p = 0#alldoc
    better = {}
    worse = {}
    tplist = []
    true = mykeyword.taglist
    pred = mykeyword.featurelist
    for i in range(len(pred)):
        tp = 0#one article
        true_len = len(true[i])
        pred_len = len(pred[i])
        for feature in pred[i]:
            if feature in true[i]:
                tp += 1
        if pred_len == 0:
            tp = 1.0 if true_len == 0 else 0
        else:
            tp = tp / pred_len
            tplist.append(tp)
            if tp >= 0.5: #better:50%的feature符合
                better[mykeyword.numlist[i]] = {'tags':list(true[i]), 'features':list(pred[i])}
            if tp <= 0.1:
                worse[mykeyword.numlist[i]] = {'tags':list(true[i]), 'features':list(pred[i])}
            p += tp
    dataframe = pd.DataFrame({'thucke score':tplist})
    dataframe.to_csv("thucke_score_betterrank.csv",index=False)
    with open('better_thucke_bettertrank.json','w') as json_f:
        json.dump(better,json_f,ensure_ascii = False) 
    with open('worse_thucke_bettertrank.json','w') as json_f:
        json.dump(worse,json_f,ensure_ascii = False)           
    return p/len(pred) if len(pred) > 0 else 0




def precision_n(segFn,extractFn, Ns = range(1,30,5)):
    nValues = []
    pValues = []
    num_doc = 13
    for n in Ns:
        nValues.append(n)
        mykeyword = keyword(segFn,extractFn)
        mykeyword.getKeywords('1008_ch.json',n,num_doc)
        pValues.append(getPrecision(mykeyword))
    return nValues, pValues


def precision_d(segFn,extractFn, Ns = range(1,50,5)):
    dValues = []
    pValues = []
    n = 10
    for d in Ns:
        dValues.append(d)
        mykeyword = keyword(segFn,extractFn)
        mykeyword.getKeywords('1008_ch.json', n, d)
        pValues.append(getPrecision(mykeyword))
    return dValues, pValues


# def precison_model(n = 10, d = 10):
#     extract_models = ['tfi','textrank','thucke']
#     segment_models = ['jieba','thulac']
#     mValues = []
#     pValues = []
#     for seg in segment_models:
#         for extract in extract_models:            
#             mykeyword = keyword(seg,extract)
#             mykeyword.getKeywords('1008_ch.json', n, d)
#             pValues.append(getPrecision(mykeyword))
#             mValues.append(seg + extract)
#     return mValues, pValues

def precison_time_model(n = 10, d = 5):
    extract_models = ['tfi','textrank']
    segment_models = ['jieba','thulac']
    # extract_models = ['thucke']
    # segment_models = ['thulac']
    mValues = []
    tValues = []
    pValues = []
    betterp = {}
    for seg in segment_models:
        for extract in extract_models:
            runtime = 0
            start = time.time()          
            mykeyword = keyword(seg,extract)
            mykeyword.getKeywords('1008_ch.json', n, d)
            end = time.time()
            pValues.append(getPrecision(mykeyword))
            # p, betterp[seg+extract]= getBetterpList(mykeyword)
            # pValues.append(p)
            runtime = (end - start) / len(mykeyword.numlist)
            tValues.append(runtime)
            mValues.append(seg + extract)
    with open('better.json','w') as json_f:
        json.dump(betterp,json_f,ensure_ascii = False)           
    return mValues, pValues, tValues



# import matplotlib.pyplot as plt

# Ns = range(1,30,5)
# nVals, pVals = precision_n('jieba','tfi',Ns)
# plt.plot( nVals, pVals, "--", color='red',label='model_jieba_tfi' )
# plt.xlabel('n')
# plt.ylabel('precision')
# plt.legend()
# plt.title("p-n")
# plt.show()


# mVals, pVals, tVals = precison_time_model()
# dataframe = pd.DataFrame({'model':mVals,'precison':pVals, 'time':tVals})
# dataframe.to_csv("evaluation_all.csv",index=False)

# df1 = pd.read_csv('evaluation_all.csv')
# df2 = pd.read_csv('evaluation_thucke.csv')
# df = df1.append(df2)
# xlist = df1.model
# ylist = df1.time
# plt.bar(xlist,ylist)
# plt.show()




mykeyword = keyword('jieba','thucke')
mykeyword.getKeywords('1008_ch.json', 10, 5)
print(getBetterpList(mykeyword))








