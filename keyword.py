import time
import os
import re
import pandas as pd
import numpy as np
import json
from lxml import etree
import jieba
import thulac
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import os
import subprocess
import compare.topK as topk



class keyword(object):
    def __init__(self,modelSegment,modelExtract):
        self.extract_models = ['tfi','textrank','thucke']
        self.segment_models = ['jieba','thulac']
        if modelExtract.lower() not in self.extract_models: return False
        if modelSegment.lower() not in self.segment_models: return False
        if modelExtract.lower() == 'thucke':
            modelSegment = 'thulac'
        self.modelExtract = modelExtract
        self.modelSegment = modelSegment
        self.numlist = []
        self.taglist = []
        self.featurelist = []       
        self.txtlist = []
        self.txtlist_cut = []
        #self.TT = 0
        

        
    # @property
    # def extract_models(self):
    #     return self.extract_models
    # @extract_models.setter 
    # def extract_models(self,value):
    #     self.extract_models = value

    # @property
    # def segment_models(self):
    #     return self.segment_models
    # @segment_models.setter
    # def segment_models(self,value):
    #     self.segment_models = value
    
    # @property
    # def taglist(self):
    #     return self.taglist
    # @taglist.setter
    # def taglist(self,value):
    #     self.taglist = value
    
    # @property
    # def featurelist(self):
    #     return self.featurelist
    # @featurelist.setter
    # def featurelist(self,value):
    #     self.featurelist = value

    # @property
    # def txtlist(self):
    #     return self.txtlist
    # @txtlist.setter
    # def txtlist(self,value):
    #     self.txtlist = value

    # @property
    # def txtlist_cut(self):
    #     return self.txtlist_cut
    # @txtlist_cut.setter
    # def txtlist_cut(self,value):
    #     self.txtlist_cut = value

    # @property
    # def TT(self):
    #     return self.TT
    # @TT.setter
    # def TT(self,svalue):
    #     self.TT = value



    def getKeywords(self,inputJson,wordLimit,n):
        start = time.clock()
        # self.readJson(inputJson,n)
        if self.modelSegment == 'thulac' and self.modelExtract == 'thucke':
            #直接提取
            self.sampling_better()
            #self.readJson(inputJson,n)
            self.thucke(wordLimit)             
        else:
            self.readJson(inputJson,n)
            #选择模型分词
            if self.modelSegment == 'jieba':
                self.seg_jieba()
            elif self.modelSegment == 'thulac':
                self.seg_thulac()
            else:
                return False 
            #选择模型提取关键词
            if self.modelExtract == 'tfi':
                self.tfi(wordLimit)
            elif self.modelExtract == 'textrank':
                self.textrank(wordLimit)
            else:
                return False  
        end = time.clock()
        #self.TT = (end - start) / len(self.txtlist) if self.TT == 0 else self.TT

    #model methods:

    def tfi(self,size):
        tfidf = TfidfVectorizer(sublinear_tf=True,norm='l2', encoding='latin-1', 
                        ngram_range=(1, 2))#,min_df=3
        features = tfidf.fit_transform(self.txtlist_cut).toarray()
        names = np.array(tfidf.get_feature_names())
        # print(tfidf.vocabulary_)
        # 输出一篇文章的feature，type list
        if size > features.shape[0]:return False
        for row in range(0,features.shape[0]):
            onerow = features[row,]
            # onerow_size = topk.mySelect(list(onerow),size)
            idx = np.argpartition(onerow, -size)[-size:]
            oneFeature = names[idx]
            self.featurelist.append(oneFeature)

    def textrank(self,size):
        tr4w = TextRank4Keyword()
        for txt in self.txtlist_cut:
            txtfeature = []
            tr4w.analyze(text=txt, lower=True, window=2)
            for item in tr4w.get_keywords(size, word_min_len=1):
                txtfeature.append(item.word)
            self.featurelist.append(txtfeature)

    def thucke(self,size):
        cur_dir = os.getcwd()
        for txt in self.txtlist:
            os.chdir('/Users/zhenganni/Downloads/THUCKE')
            with open('article.txt', 'w') as f:
                f.write(txt)
            cmd = './thucke -i ./article.txt -n 10 -m ./res'#size = 10 
            try:
                output = subprocess.check_output(cmd,shell=True)
            except subprocess.CalledProcessError as err:
                print(err)
            #print(output.decode())
            if output.decode()[0] != '{':
                self.featurelist.append('')
            else:
                output = json.loads(output.decode())#loads的时候格式出问题
                #self.TT  += output['timeuse']
                txtf = [dic['keyword'] for dic in  output['result'] ]
                self.featurelist.append(txtf)
        os.chdir(cur_dir)
        #self.TT = self.TT / len(self.txtlist)
    
    
    
    
    def seg_jieba(self):
        jieba.load_userdict('mydic.txt')
        stop = [line.strip()  for line in open('stopwords.txt').readlines() ]
        for txt in self.txtlist:
            onecut = self.cutWords(txt, stop)
            self.txtlist_cut.append(onecut)
    def cutWords(self,msg, stopWords):
        seg_list = jieba.cut(msg,cut_all=False)    
        #use list to store words
        #txtWords = [] 
        #use string to store words
        txtWords = ""    
        for i in seg_list:
            #过滤stop words       
            if (i not in stopWords):            
                #txtWords.append(i) 
                txtWords = txtWords + i + ' '          
        return txtWords


    def seg_thulac(self):
        thu1 = thulac.thulac(user_dict='mydic.txt',seg_only=True,filt=True)
        for txt in self.txtlist:
            #onecut = [word[0] for word in thu1.cut(txt)]#list
            onecut_str = ''
            for word in thu1.cut(txt):
                onecut_str = onecut_str + str(word[0]) + ' '
            self.txtlist_cut.append(onecut_str)


    
    def readJson(self, inputJson, n):
        with open(inputJson,"r") as load_f:#"1008_ch.json"
            dic_json = json.load(load_f)
        num_list = list(dic_json.keys())
        for index,num in enumerate(num_list):
            # if index > 5: break#test line
            #解析json里保存的官方tag信息#用xpath改进?
            atrr = dic_json[num]
            path = dic_json[num].get('path')
            #忽略找不到的html
            if not os.path.exists(path):
                print(path+"  file can't find")
                continue
            keywords = atrr['tag']
            dic_keywords = json.loads(keywords)#从字典中获得的keyword是str，用json换成dic，or pickle
            if 'disease_keywords' not in dic_keywords.keys():continue
            self.numlist.append(str(num))
            disease = dic_keywords['disease_keywords']
            treatment = dic_keywords['treat_methods']
            tags =[]
            for dic in disease:
                tags.append(dic['keyword'])
            for dic in treatment:
                tags.append(dic['keyword'])
            self.taglist.append(tags)

            #从查询到的路径打开html文件并解析文字内容
            with open(path,'r') as article_f:
                html_file = article_f.read()
                html = etree.HTML(html_file)
                h1 = html.xpath('//h1[@class="news-detail-title"]/text()')
                h2 = html.xpath('//p[@class="desc"]/text()')
                content = html.xpath('//div[@class="news-content"]//text()')
                content_str = ""
                for x in content[:-1]: content_str+=x
                #txt_str = h1[0] + '\n' + h2[0] + '\n' + content_str
                txt_str = h1[0]  + h2[0]  + content_str#不添加换行符
                txt_str = ''.join(list(filter(lambda x: x.isalpha(), txt_str)))#过滤数字#有英文关键词
                self.txtlist.append(txt_str)
    
    def sampling(self):
        df = pd.read_csv('thucke_sample.csv')#input
        series = df.groupby('tag number').article.count()
        count = 0
        num = 100#一次测试文章数，单篇19秒
        for val in series.values:
            count += val
        for idx,val in enumerate(series):
            series[idx] = num * (val/count)

        piece = dict(list(df.groupby('tag number')))
        for tag in piece.keys():
            if tag == 1008 :break
            df_onetag = piece[tag]
            series_sample = df_onetag.sample(n = int(series[tag]),axis=0)
            for val in series_sample.values:
                self.taglist.append(val[1])
                self.txtlist.append(val[3])
    
    def sampling_better(self):
        df = pd.read_csv('thucke_sample.csv')#input
        with open('better_trank.json','r') as f:#input 
            better = json.load(f)
        for idx,num in enumerate(df['article number']):
            if str(num) in better.keys():
                self.taglist.append(df['origin tags'][idx])
                self.txtlist.append(df['article'][idx])
                self.numlist.append(num)
        



if __name__ == '__main__':
    pass




