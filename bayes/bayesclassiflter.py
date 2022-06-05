#!usr/bin/python
# -*- coding:utf-8 -*-

import os, sys
import math
import csv

import jieba.analyse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import jieba

jieba.set_dictionary("dict.txt")
jieba.initialize()

# 导入tkinter模块
from tkinter import *
from PIL import ImageTk, Image
import os
from tkinter import filedialog

root = Tk()
root.title('查看文件')
root.geometry('1000x750')  # tkinter基本设置

ABSPATH = os.path.abspath(sys.argv[0])
ABSPATH = os.path.dirname(ABSPATH) + '/'
print(ABSPATH)


# def getPofClass(index, word_list):
#     # 输入类index的贝叶斯训练结果文件
#     index_training_path = ABSPATH + 'bayes_training_outcome/' + str(index) + '_bayestraining.txt'
#     file_index_training = open(index_training_path, 'r', encoding='UTF-8')
#     dic_training = {}  # 存储 index_bayestraining.txt 中的 (单词：P)
#     training_word_p_list = file_index_training.readlines()
#     allwords_fre_allwords_num = training_word_p_list[0].strip()  # index_bayestraining.txt的第一行
#     allwords_fre = int(allwords_fre_allwords_num[1])  # 所有样本的所有单词的词频
#     allwords_num = int(allwords_fre_allwords_num[0])  # 所有样本的所有单词个数
#     for i in range(1, len(training_word_p_list)):
#         word_p = training_word_p_list[i].strip().split(',')
#         dic_training[word_p[0]] = float(word_p[1])
#
#     # 遍历测试样本的wordlist，求每个Word的p
#     p_list = []
#     for word in word_list:
#         word = word.strip()
#         if word in dic_training:
#             p_list.append(str(dic_training[word]))
#         else:
#             p_list.append(str(1.0 / (allwords_fre + allwords_num)))
#     # 计算P
#     p_index = 0
#     for p in p_list:
#         p = math.log(float(p), 2)
#         p_index = p_index + p
#     return -p_index


def bayes(text):
    df_news = pd.read_table(filepath_or_buffer='./bayes/data.txt', names=['category', 'theme', 'URL', 'content'],
                            encoding='UTF-8')
    df_news.dropna()
    df_news.tail()

    stopwords = pd.read_table(filepath_or_buffer='./bayes/stopwords.txt', names=['stopword'], sep='\t',
                              quoting=csv.QUOTE_NONE, encoding='UTF-8')
    stopwords = stopwords['stopword'].values.tolist()

    content_arr = df_news['content'].values.tolist()

    content_words = []
    for line in content_arr:
        current_segment = jieba.lcut(line)
        if len(current_segment) > 1 and current_segment != '\t\r':
            content_words.append(current_segment)

    def drop_stopwords(content_words, stopwords):
        content_words_clean = []
        for line_words in content_words:
            line_clean = []
            for word in line_words:
                if word in stopwords:
                    continue
                line_clean.append(word)
            content_words_clean.append(line_clean)
        return content_words_clean

    content_words_clean = drop_stopwords(content_words, stopwords)

    train_data = pd.DataFrame({"content_clean": content_words_clean, "label": df_news['category']})

    train_data.head()

    label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育": 5, "教育": 6, "文化": 7, "军事": 8, "娱乐": 9, "时尚": 0}
    train_data['label'].unique()

    train_data['label'] = train_data['label'].map(label_mapping)
    train_data.head()

    x_train, x_test, y_train, y_test = train_test_split(train_data['content_clean'].values, train_data['label'].values,
                                                        random_state=1)
    train_words = []
    for line_index in range(len(x_train)):
        train_words.append(' '.join(x_train[line_index]))

    test_words = []
    for line_index in range(len(x_test)):
        test_words.append(' '.join(x_test[line_index]))

    # cv = CountVectorizer(analyzer='word', max_features=5000, lowercase=False)
    # feature = cv.fit_transform(train_words)
    # classifier = MultinomialNB()
    # classifier.fit(feature, y_train)
    # classifier.score(cv.transform(test_words), y_test)

    tv = TfidfVectorizer(analyzer='word', max_features=5000, lowercase=False)
    feature = tv.fit_transform(train_words)
    classifier = MultinomialNB()
    classifier.fit(feature, y_train)
    classifier.score(tv.transform(test_words), y_test)
    test_content = text
    # test_content = "昨日，沪指收盘击穿钻石底，报２１２６点，创２００９年３月以来新低，深指破位９１００点关口。　钻石底沦陷，两市昨日交投不足千亿，Ａ股持仓账户比例下滑至３３．９６％创新低，股民投资意愿降至“冰点”。Ａ头地产股大跌Ａ绞凶蛉找蝗缂韧低开，盘初窄幅震荡，沪指一度突破５日均线，升至日内高点２１４７．６６点，深指冲上９２００点。Ｎ绾螅受困基本面表现乏力、利好消息缺失，两市成交持续低迷。地产板块午后大幅下挫，四大龙头地产股“招保万金”放量大跌，加重场内担忧，权重股纷纷翻绿，导致两市最后半小时放量跳水。Ｗ钪眨沪指报收２１２６点，下跌０．４８％，创２００９年３月９日以来收盘新低。深指下跌０．８０％，收报９０８１．９０点，失守９１００点关口。＃冻烧嘶Э詹铸＠醋灾械枪司的最新数据显示，７月１６日至７月２０日当周，新增Ａ股开户数为８．４１万户，较上周增加０．６７万户，增幅８．５６％，已连续两周增加。但上周市场参与度仅为５．０１％，已连降两周；截至上周末，Ａ股持仓账户数为５６４５万户，较前一周减少７．１７万户，比例下滑至３３．９６％，续创历史新低。Ｍ庾士始唱多Ｓ肷⒒У摹白山观虎斗”迥异的是，外资机构唱多声音此起彼伏，更有机构“冰川期”满仓操作。ＤΩ士丹利在其最新研报中称，中国Ａ股和Ｈ股的估值都显著低于历史水平，年内应有较好表现；高盛高华则认为，上证指数有望到年底达到２７５０点。Ｍ庾驶构“看多”同时也做多。２００６年获得Ａ股ＱＦＩＩ资格的爱德蒙得洛希尔资产管理公司总经理汤熠近日透露，该公司在Ａ股市场的７亿多美元投资目前已满仓操作。"
    test_current_segment = jieba.lcut(test_content)
    test_contents_clean = drop_stopwords(content_words=[test_current_segment], stopwords=stopwords)
    t_words = [' '.join(test_contents_clean[0])]
    classifier.predict(tv.transform(t_words))
    return classifier.predict(tv.transform(t_words))



# 从本地文件获取文本内容
def getTextFromNative(filepath):
    # 输入测试样本
    test_file_path = filepath
    file_test = open(test_file_path, 'r', encoding='UTF-8')
    text = file_test.read()
    return text


# 测试本地文件
def nativeTest():
    dir_path = 'data/test/'
    # dir_path = os.path.join(os.path.dirname())
    file_list = os.listdir(dir_path)
    all_count = 0
    right_count = 0
    index_all_count = 0
    index_right_count = 0
    file_outcome = open(ABSPATH + 'outcome/outcome_native.txt', 'w', encoding='UTF-8')
    for filename in file_list:
        # 去除隐藏文件
        a = filename.split('.')
        if a[1] == 'txt':
            all_count = all_count + 1
            index_all_count = index_all_count + 1
            b = filename.split('_')
            rightIndex = int(b[0])
            text = getTextFromNative(filename)
            getIndex = bayes(text)
            if getIndex == rightIndex:
                print(filename + '----' + str(getIndex) + ' : right')
                right_count = right_count + 1
                index_right_count = index_right_count + 1
            else:
                print(filename + '----' + str(getIndex) + ' : error')

            if index_all_count == 100:
                string = str(index_right_count) + ' / ' + str(index_all_count) + ' = ' + str(
                    float(index_right_count) / index_all_count)
                print(string)
                file_outcome.write(string + '\n')
                index_all_count = 0
                index_right_count = 0

    string = str(right_count) + ' / ' + str(all_count) + ' = ' + str(float(right_count) / all_count)
    print(string)
    file_outcome.write(string)
    file_outcome.close()


# tkinter框架
fm1 = Frame(root)
fm2 = Frame(root)
fm3 = Frame(root)
fm4 = Frame(root)
fm5 = Frame(root)
# 框架放置
fm1.pack()
fm2.pack(anchor='w', pady=20)
fm3.pack(anchor='w', pady=20)
fm4.pack(anchor='w', pady=20)
fm5.pack(anchor='w', pady=20)
# 标签图片1
img = Image.open('BGP.png')
photo = ImageTk.PhotoImage(img)
thelabel = Label(fm1, image=photo)
thelabel.pack()


# 浏览文件 查看路径
def file_select():
    global file_path
    file_path = filedialog.askopenfilename()
    thetext1.delete(1.0, 'end')
    thetext1.insert(1.0, file_path)


b1 = Button(fm2, text='选择文件', width=20, height=2, command=file_select, bg='lightgreen', fg='black')
b1.pack(side='left', padx=100)
thetext1 = Text(fm2, font=('Fixedsys', 15), width=40, height=3, bg='lightblue')
thetext1.pack(side='left', padx=50)


# 分类
def detect_oneTime(filepath):
    return bayes(getTextFromNative(filepath))


# 分类结果
def classify_res():
    num1 = detect_oneTime(file_path)
    if num1 == 1:
        return '汽车'
    elif num1 == 2:
        return '财经'
    elif num1 == 3:
        return '科技'
    elif num1 == 4:
        return '健康'
    elif num1 == 5:
        return '体育'
    elif num1 == 6:
        return '教育'
    elif num1 == 7:
        return '文化'
    elif num1 == 8:
        return '军事'
    elif num1 == 9:
        return '娱乐'
    elif num1 == 0:
        return '时尚'
    else:
        return '其他'


var1 = StringVar()  # 分类结果
var2 = StringVar()  # 保存结果
var3 = StringVar()  # 删除结果
var1.set('  ')
var2.set('  ')
var3.set('  ')


def classify_insert():
    str = classify_res()
    var1.set('该新闻的分类为: ' + str + '类')


b2 = Button(fm3, text='分类结果', width=20, height=2, command=classify_insert, bg='lightgreen', fg='black')
b2.pack(side='left', padx=100)
thelabel2 = Label(fm3, textvariable=var1, width=20, height=2, bg='lightblue')
thelabel2.pack(side='left', padx=50)


# 记录结果
def file_check():
    f = open('information_path.txt', 'r', encoding='UTF-8')
    text = f.readlines()
    s = []
    for line in text:
        s.append(line[0:-8])
    # print(s)
    if file_path in s:
        # print('存在')
        return False
    else:
        # print('不存在')
        return True
    f.close()


def file_save():
    num2 = detect_oneTime(file_path)
    res = file_check()
    # print(res)
    if res == False:
        var2.set('结果已存在')
    elif res == True:
        f = open('information_path.txt', 'a', encoding='UTF-8')
        f.write(file_path + ' :  ' + str(num2) + '\n')
        var2.set('完成')
        f.close()


b3 = Button(fm4, text='保存结果', width=20, height=2, command=file_save, bg='lightgreen', fg='black')
b3.pack(side='left', padx=100)
thelabel3 = Label(fm4, textvariable=var2, width=20, height=2, bg='lightblue')
thelabel3.pack(side='left', padx=50)


# 显示分类结果

# 删除
def file_del():
    res = file_check()
    # print(res)
    if res == True:
        var3.set('结果不存在')
    elif res == False:
        f = open('information_path.txt', 'r', encoding='UTF-8')
        text = f.readlines()
        # print(text)
        for i in range(len(text)):
            if text[i][0:-8] == file_path:
                del text[i]
                break
        f.close()
        f = open('information_path.txt', 'w', encoding='UTF-8')
        for i in range(len(text)):
            f.write(text[i])
        f.close()
        var3.set('完成')


b4 = Button(fm5, text='删除结果', width=20, height=2, command=file_del, bg='lightgreen', fg='black')
b4.pack(side='left', padx=100)
thelabel4 = Label(fm5, textvariable=var3, width=20, height=2, bg='lightblue')
thelabel4.pack(side='left', padx=50)

mainloop()




