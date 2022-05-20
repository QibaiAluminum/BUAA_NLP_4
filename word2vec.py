# -*- coding: utf-8 -*-
import jieba
import re
import numpy as np
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle as pkl
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
def dispose(data1, data2):  # 读取语料内容
    content = []
    sw = ["的", "了", "在", "是", "我", "有", "和", "就",
          "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
          "会", "着", "没有", "看", "好", "自己", "这", "罢", "这", '在', '又', '在', '得', '那', '他', '她', '不', '而', '道', '与', '之',
          '见', '却', '问', '可', '但'
        , '没', '啦', '给', '来', '既', '叫', '只', '中', '么', '便'
        , '听', '为', '跟', '个', '甚', '下', '还', '过', '向', '如此'
        , '已', '位', '对', '如何', '将', '岂', '哪', '似', '以免', '均'
        , '虽然', '即', '由', '再', '使', '从', '麽', '其实', '阿', '被','当','里','时','虽','远','轻','凑','兄']

    for line in open(data1, 'r', encoding='UTF-8',errors='ignore'):
        line.strip('\n')
        line = re.sub('\s', '', line)
        line = re.sub('[\u0000-\u4DFF]', '', line)
        line = re.sub('[\u9FA6-\uFFFF]', '', line)
        line = re.sub('[\u9FA6-\uFFFF]', '', line)
        for a in sw:
            line = re.sub(a, '', line)
        if len(line) == 0:
            continue
        seg_list = list(jieba.cut(line, cut_all=False))
        line_seg = ""
        for term in seg_list:
            line_seg += term + " "
        con = jieba.cut(line, cut_all=False) # 结巴分词
            # content.append(con)
        content.append(" ".join(con))
    with open(data2, "w", encoding='UTF-8',errors='ignore') as f:
        f.writelines(content)
    return content

def cluster(vector):
    with open('./data2/射雕vec.txt', 'rb') as f:
        vec_dist = pkl.load(f)
    vec = []
    for d in vector:
        vec.append(vec_dist[d])
    center, label, inertia = k_means(vec, n_clusters=3)
    vec = PCA(n_components=2).fit_transform(vec)

    #plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['font.family'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(vec[:, 0], vec[:, 1], cmap='viridis', c=label,alpha=0.5)
    for i, w in enumerate(vector):
        plt.annotate(text=w, xy=(vec[:, 0][i], vec[:, 1][i]),
                     xytext=(vec[:, 0][i] + 0.01, vec[:, 1][i] + 0.01))
    plt.colorbar()
    plt.show()


if __name__ == '__main__':   ##
    book='./data2/射雕英雄传.txt'
    dispose("./data1/射雕英雄传.txt", './data2/射雕英雄传.txt')
    # model = Word2Vec(data_txt, vector_size=400, window=5, min_count=5, epochs=200, workers=multiprocessing.cpu_count())

    word_name = ['郭靖','黄蓉','杨康']
    word_Kungfu = ['降龙十八掌','九阴真经','蛤蟆功']
    word_group= ['丐帮','桃花岛','蒙古']

    print(book)
    model = Word2Vec(sentences=LineSentence(book), hs=1, min_count=10, window=5, vector_size=200, sg=0, epochs=200)
    model.wv.vectors = model.wv.vectors / (np.linalg.norm(model.wv.vectors, axis=1).reshape(-1, 1))
    vec_dist = dict(zip(model.wv.index_to_key, model.wv.vectors))
    with open('./data2/射雕vec.txt', 'wb') as f:
        pkl.dump(vec_dist, f)

    for i in range(0,3):
        print()
        print("人物名字："+ word_name[i])
        for result in model.wv.similar_by_word(word_name[i], topn=10):
            print(result[0], result[1])
    for i in range(0, 3):
        print()
        print("武功："+ word_Kungfu[i])
        for result in model.wv.similar_by_word(word_Kungfu[i], topn=10):
            print(result[0], result[1])
    for i in range(0, 3):
        print()
        print("特征："+ word_group[i])
        for result in model.wv.similar_by_word(word_group[i], topn=10):
            print(result[0], result[1])
    cluster(['郭靖', '黄蓉', '杨康','洪七公', '周伯通', '欧阳锋', '降龙十八掌','九阴真经','蛤蟆功', '丐帮','桃花岛','蒙古'])
