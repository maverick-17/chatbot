# -*- coding: utf-8 -*-
# @Time : 2020/2/10 10:45 PM 
# @Author : qiujiafa
# @File : match.py

"""
question 召回模块
"""

import json
import logging
import pickle
import re
from collections import Counter
from functools import reduce
from operator import and_

import jieba
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from chatbot.config import chatbot_config

qa_data_path = './chatbot/data/qa_corpus.csv'
wiki_wordvec_model_path = './chatbot/data/sgns.wiki.word'
word_vocab_path = './chatbot/data/vocab.json'
qa_processed_data_path = './chatbot/data/qa_corpus_processed.txt'
km_centers_path = './chatbot/data/km_centers.txt'
qid_to_answer_path = './chatbot/data/qid_to_answer.txt'

regrex = re.compile('^[\d\w]+$')

logging.basicConfig(level=logging.INFO)


class Retriever:
    def __init__(self, first_time):
        # self.embedding = SentenceEmbedding()
        self.first_time = first_time
        self.logger = logging.getLogger(__name__)
        if first_time:
            self.df_qa = pd.read_csv(qa_data_path)
        else:
            with open(qa_processed_data_path, 'rb') as f:
                self.df_qa = pickle.load(f)

            with open(km_centers_path, 'rb') as cf:
                self.km_centers = pickle.load(cf)
        self.qa_pairs = self.df_qa[['qid', 'question', 'answer']].to_dict(orient='record')

    def qa_process(self):
        if self.first_time:
            df_qa = self.df_qa
            df_qa.dropna(inplace=True)
            df_qa['qid'] = df_qa['qid'].astype(int)
            df_qa['question'] = df_qa['question'].astype(str)
            df_qa['question_cutted'] = df_qa['question'].apply(lambda x: ' '.join(jieba.cut(x)))
            df_qa['question_emb'] = df_qa['question_cutted'].apply(
                lambda x: self.embedding.get_sentence_embedding_average(x.split()))
            self.df_qa = df_qa
        else:
            return None

    def question_cluster(self, first_time):
        # self.df_qa.drop('km_label', axis=1, inplace=True)

        data = self.df_qa
        print(len(data))
        # 去掉矩阵范数为 0 的无用数据
        data = data[data['question_emb'].apply(
            lambda x: True if np.linalg.norm(x) > 0 else False)]
        print(len(data))

        skm = KMeans(n_clusters=100, max_iter=300)
        skm.fit(data['question_emb'].to_list())

        data['km_label'] = skm.labels_
        data = data[['qid', 'km_label']]

        for idx, group in data.groupby('km_label'):
            print(idx, ' : ', len(group))

        self.df_qa = pd.merge(self.df_qa, data, how='left', on='qid')

        self.df_qa['km_label'].fillna(-1, inplace=True)

        self.km_centers = skm.cluster_centers_

        if first_time:
            with open(qa_processed_data_path, 'wb') as f:
                pickle.dump(self.df_qa, f, -1)
            with open(km_centers_path, 'wb') as fr:
                pickle.dump(self.km_centers, fr, -1)
            self.logger.info(f"Successfully Load qa data into {qa_processed_data_path}")

    def gen_vocab(self):
        word_counter = Counter()
        for sentence in self.df_qa['question']:
            word_counter.update([w for w in jieba.cut(sentence) if regrex.search(sentence)])
        with open(word_vocab_path, 'w') as f:
            json.dump(word_counter, f)
        self.logger.info(f'Successfuly dump questions word vocabulary into {word_vocab_path}')

    def init_process(self):
        # self.gen_vocab()
        self.qa_process()
        self.question_cluster(self.first_time)

    def _init_search(self):
        self.vectorizer = TfidfVectorizer(max_features=12000)
        self.tfidf = self.vectorizer.fit_transform(self.df_qa['question_cutted'].tolist())

        self.word_to_id = self.vectorizer.vocabulary_
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        # 转置 得到每个单词对应问题文本的向量表示
        self.tfidf_trans = self.tfidf.transpose().toarray()

    def bool_search(self, words):
        """
        布尔搜索, 返回含有关键词的排名前 10 的 qid
        :param words: list 用户提问的问题 切词得到的 list
        :return: top 10 qids
        """
        word_index = []
        for word in words:
            if word not in self.word_to_id:
                continue
            # 拿到 每个单词对应出现的问题文本
            word_index.append(set(np.where(self.tfidf_trans[self.word_to_id[word]])[0]))
        # 对query 中每个单词出现的qa_pair 问题文本做交集 得到出现 query 中出现单词的 question (index)
        result_index = reduce(and_, word_index)

        query_vector = self.vectorizer.transform([' '.join(words)]).toarray()[0]

        sorted_result = sorted(result_index,
                               key=lambda i: self.consine_distince(query_vector, self.tfidf[i].toarray()[0]))

        search_result = [self.qa_pairs[idx] for idx in sorted_result]

        top_result = [row['qid'] for row in search_result][:chatbot_config['top']]

        self.logger.info(f"bool_search top10 question: {[self.qid_to_question(i) for i in top_result]}")
        return top_result

    def cluster_match(self, words):
        """
        从已分类的问题文本中， 找到输入问题是否归属某个类别， 是的话返回这个类别中匹配度排名前十的问题
        :return: top 10 qids
        """
        query_vector = self.embedding.get_sentence_embedding(words)
        if np.linalg.norm(query_vector) == 0:
            return None
        min_distance = float('inf')
        center_id = -1
        for cid, center in enumerate(self.km_centers):
            distance = self.consine_distince(query_vector, center)
            if distance < 0.5 and distance < min_distance:
                min_distance = distance
                center_id = cid
        if center_id == -1:
            self.logger.info(f"question {''.join(words)} do not match any cluster")
            return None

        # 找到归属 center_id 这个类之后 再挑出在这个类别中最接近的 10 个 question
        qa_candidate = self.df_qa[self.df_qa['km_label'] == center_id]
        qa_candidate['distance'] = qa_candidate['question_emb'].apply(lambda x: self.consine_distince(query_vector, x))

        qa_candidate.sort_values(by='distance', ascending=True)
        self.logger.info(
            f"center_id: {center_id}, \n top 10 question: {qa_candidate[['qid', 'question', 'km_label', 'distance', 'answer']][:10]}")

        return qa_candidate['qid'].tolist()[:chatbot_config['top']]

    @staticmethod
    def consine_distince(v1, v2):
        """ 计算量个向量的余弦距离 """
        return cosine(v1, v2)

    def qid_to_question(self, qid):
        return self.df_qa[self.df_qa['qid'] == qid]['question'].tolist()[0]


class SentenceEmbedding(object):
    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format(wiki_wordvec_model_path, binary=False)
        self.vocab = self.get_vocab()

    def get_sentence_embedding(self, sentence: list):
        """
        利用预训练好的 wiki 词向量(300维) 生成 句向量 简单加权取平均的方法
        TO_DO: TF_IDF 权值求句向量
        :param sentence: word 组成的 list
        :return: sentenct vector
        """
        print(self.model['李白'].shape)
        sentence_emb = np.zeros(shape=(300,))
        count = 0
        a = 0.001
        for word in sentence:
            if word in self.model:
                sentence_emb += a / (a + self.vocab.get(word, 0)) * self.model[word]
                count += 1
        return sentence_emb / max(count, 1)

    def get_sentence_embedding_average(self, sentence: list):
        """
        return the average of all word_vector in a sentence
        :return:
        """
        count = 0
        sentence_emb = np.zeros(shape=(300,))
        for word in sentence:
            if word in self.model:
                sentence_emb += self.model[word]
                count += 1
        return sentence_emb / max(count, 1)

    def get_vocab(self):
        with open(word_vocab_path, 'r') as f:
            return json.load(f)


class TextCluster(object):
    def __init__(self):
        self.data = pd.read_csv(qa_processed_data_path)

    def kmeans_cluster(self):
        # s_kmeans = SphericalKMeans(n_clusters=100, init='k-means++', max_iter=300)
        # s_kmeans.fit(self.data['questin_emb'].to_list())
        data = self.data
        print(data.head())

        import pdb
        pdb.set_trace()
        data = data[data['questin_emb'].apply(
            lambda x: True if np.linalg.norm(x) > 0 else False)]
        print(data.head())


if __name__ == '__main__':
    retrive_test = Retriever(first_time=False)
    retrive_test._init_search()
    # retrive_test.question_cluster(first_time=True)
    # retrive_test.cluster_match(['如何', '查询', '账单'])
    # retrive_test.cluster_match('现金 取款'.split())
    # retrive_test.cluster_match('保险 退保 材料'.split())

    # retrive_test.init_process()
    # retrive_test._init_search()
    qids = retrive_test.bool_search(['代发', '工资'])
    print(qids)

    # def consine_distince(v1, v2):
    #     """ 计算量个向量的余弦距离 """
    #     return cosine(v1, v2)
    # test = SentenceEmbedding()
    # a = test.get_sentence_embedding('你 今天 真 好看'.split())
    # b = test.get_sentence_embedding('你 今天 非常 好看'.split())
    # c = test.get_sentence_embedding('我 明天 出发 广州'.split())
    #
    # print('a:b', consine_distince(a, b))
    # print('a:c', consine_distince(a, c))
