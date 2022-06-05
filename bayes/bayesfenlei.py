#!/usr/bin/python
# -*- coding:utf-8 -*-
import csv

import jieba.analyse
import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df_news = pd.read_table(filepath_or_buffer='./bayes/data.txt',names=['category','theme','URL','content'],encoding='UTF-8')
df_news.dropna()
df_news.tail()

stopwords = pd.read_table(filepath_or_buffer='./bayes/stopwords.txt',names=['stopword'],sep='\t',quoting=csv.QUOTE_NONE,encoding='UTF-8')
stopwords = stopwords['stopword'].values.tolist()


content_arr = df_news['content'].values.tolist()
content_arr[0]
print(content_arr[0])
content_words = []
for line in content_arr:
    current_segment = jieba.lcut(line) #分词
    if len(current_segment) > 1 and current_segment != '\t\r':
        content_words.append(current_segment)
content_words[0]
print(content_words[0])

def drop_stopwords(content_words,stopwords):
    content_words_clean = []
    for line_words in content_words:
        line_clean = []
        for word in line_words:
            if word in stopwords:
                continue
            line_clean.append(word)
        content_words_clean.append(line_clean)
    return content_words_clean
content_words_clean = drop_stopwords(content_words,stopwords)
print(content_words_clean[0])
train_data = pd.DataFrame({"content_clean":content_words_clean,"label":df_news['category']})

train_data.head()
print(train_data.head())
label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育":5, "教育": 6,"文化": 7,"军事": 8,"娱乐": 9,"时尚": 0}
train_data['label'].unique()

train_data['label'] = train_data['label'].map(label_mapping)
train_data.head()
print(train_data.head())
# x_t是训练文本，xtest 是训练文本标签，y_train是测试文本，ytest 是测试文本标签
x_train,x_test,y_train,y_test = train_test_split(train_data['content_clean'].values,train_data['label'].values,random_state=1,test_size=0.15)
train_words = []
for line_index in range(len(x_train)):
    train_words.append(' '.join(x_train[line_index]))
train_words[0]
print(train_words[0])
test_words = []
for line_index in range(len(x_test)):
    test_words.append(' '.join(x_test[line_index]))
test_words[0]
# cv = CountVectorizer(analyzer='word',max_features=5000,lowercase=False)
# feature  = cv.fit_transform(train_words)
# # 贝叶斯
# classifier = MultinomialNB()
# # 模型
# classifier.fit(feature,y_train)
# classifier.score(cv.transform(test_words),y_test)
# print("cv正确率：",classifier.score(cv.transform(test_words),y_test))
tv = TfidfVectorizer(analyzer='word',max_features=9900,lowercase=False)
feature = tv.fit_transform(train_words)
classifier = MultinomialNB()#分类器
classifier.fit(feature,y_train)#模型构建
classifier.score(tv.transform(test_words),y_test)#测试
print("tv正确率：",classifier.score(tv.transform(test_words),y_test))

test_content  = "昨日，沪指收盘击穿钻石底，报２１２６点，创２００９年３月以来新低，深指破位９１００点关口。　钻石底沦陷，两市昨日交投不足千亿，Ａ股持仓账户比例下滑至３３．９６％创新低，股民投资意愿降至“冰点”。Ａ头地产股大跌Ａ绞凶蛉找蝗缂韧低开，盘初窄幅震荡，沪指一度突破５日均线，升至日内高点２１４７．６６点，深指冲上９２００点。Ｎ绾螅受困基本面表现乏力、利好消息缺失，两市成交持续低迷。地产板块午后大幅下挫，四大龙头地产股“招保万金”放量大跌，加重场内担忧，权重股纷纷翻绿，导致两市最后半小时放量跳水。Ｗ钪眨沪指报收２１２６点，下跌０．４８％，创２００９年３月９日以来收盘新低。深指下跌０．８０％，收报９０８１．９０点，失守９１００点关口。＃冻烧嘶Э詹铸＠醋灾械枪司的最新数据显示，７月１６日至７月２０日当周，新增Ａ股开户数为８．４１万户，较上周增加０．６７万户，增幅８．５６％，已连续两周增加。但上周市场参与度仅为５．０１％，已连降两周；截至上周末，Ａ股持仓账户数为５６４５万户，较前一周减少７．１７万户，比例下滑至３３．９６％，续创历史新低。Ｍ庾士始唱多Ｓ肷⒒У摹白山观虎斗”迥异的是，外资机构唱多声音此起彼伏，更有机构“冰川期”满仓操作。ＤΩ士丹利在其最新研报中称，中国Ａ股和Ｈ股的估值都显著低于历史水平，年内应有较好表现；高盛高华则认为，上证指数有望到年底达到２７５０点。Ｍ庾驶构“看多”同时也做多。２００６年获得Ａ股ＱＦＩＩ资格的爱德蒙得洛希尔资产管理公司总经理汤熠近日透露，该公司在Ａ股市场的７亿多美元投资目前已满仓操作。"
test_current_segment = jieba.lcut(test_content)
test_contents_clean = drop_stopwords(content_words = [test_current_segment],stopwords=stopwords)
t_words = [' '.join(test_contents_clean[0])]
print("测试结果：",classifier.predict(tv.transform(t_words)))


