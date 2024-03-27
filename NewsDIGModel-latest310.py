# 结果还行！！！！！初版完成！！！！
# 现在我有一个训练好的分类模型bert-base-multilingual-cased_classification_undersampled_new_epoch_20.pth，我需要加载这个分类模型对文本进行分类操作，具体描述为：有一个新闻语料文件./Data231202-231211/Data231202.csv，语料中body列为新闻文本，category1列为新闻类别，但语料中有部分语料的
# 类别在['খেলাধুলা','রাজনীতি','বিনোদন','অর্থনীতি','আইন','শিক্ষা','বিজ্ঞান','লাইফস্টাইল','অন্যান্য']以外，我现在需要根据语料中category1类别不属于这些类别的语料进行处理，根据其新闻文本对语料的新闻类别进行预测，并将预测后的结果替换原来的类别，并将结果保存到新的csv文件中，请给出完整详细的代码
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载训练好的模型
model_path = '../NewsAthm/bert-base-multilingual-cased'
modelNew_load_path = './classificationModel/bert-base-multilingual-cased_classification_undersampled_new_epoch_20.pth'

model = BertForSequenceClassification.from_pretrained(model_path, num_labels=9)

model.load_state_dict(torch.load(modelNew_load_path))
model.eval()

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# 定义类别列表
categories = ['খেলাধুলা', 'রাজনীতি', 'বিনোদন', 'অর্থনীতি', 'আইন', 'শিক্ষা', 'বিজ্ঞান', 'লাইফস্টাইল', 'অন্যান্য']

# 读取csv文件
data = pd.read_csv('./Data231202-231211/Data231202.csv')

# 定义预测函数
def predict_category(text):
    # 对文本进行编码
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True
    )

    # 进行预测
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)

    # 返回预测的类别
    return categories[predictions.item()]

# 对数据进行处理
def process_data(data):
    # 找出category1不在指定类别列表中的数据
    mask = ~data['category1'].isin(categories)
    data_to_predict = data[mask]

    # 对需要预测的数据进行预测
    data_to_predict['category1'] = data_to_predict['body'].apply(predict_category)

    # 将预测后的数据与原数据合并
    data[mask] = data_to_predict

    return data

# 处理数据
processed_data = process_data(data)

# 保存处理后的数据到新的csv文件
processed_data.to_csv('./Data231202-231211/Data231202_processed.csv', index=False)

print("FINISH!!")


# conda angle https://github.com/SeanLee97/AnglE/tree/main
# pip install nltk
# pip install --upgrade pip
# pip install spacy==2.3.5
# pip install bn_core_news_sm-0.1.0.tar.gz
# pip install matplotlib
import pandas as pd
# from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModel, AutoTokenizer
from angle_emb import AnglE

# yes! 聚类评估！！！可跑 TP, FP, TN, FN 得到RI、Precision、Recall、F1，ARI
# update:单个成簇的处理
from itertools import combinations
from math import comb

from sklearn.preprocessing import MinMaxScaler

import networkx as nx
from collections import defaultdict
from nltk.tokenize import word_tokenize # 使用NLTK进行分词，根据需要替换为适合孟加拉语的分词方法

import spacy
# from gensim.summarization import keywords
from collections import defaultdict
import bn_core_news_sm
from sklearn.preprocessing import MinMaxScaler # 归一化
import matplotlib.pyplot as plt
# import pytextrank
# =======
# 去除停用词
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import string
# ====================

# data_ORI = pd.read_csv('./Data231202-231211/Data231202.csv') # 所有子任务都是使用这个
data_ORI = processed_data

# 使用angle加载
model_id = '../NewsAthmTask2/models/angle-bert-base-uncased-nli-en-v1'
angle = AnglE.from_pretrained(model_id, pooling_strategy='cls_avg').cuda()

# 加载数据
data = data_ORI

# 将日期转换为日期时间格式
data['pub_time'] = pd.to_datetime(data['pub_time'])

# 获取唯一日期列表
dates = data['pub_time'].dt.date.unique()


# 定义聚类中心更新函数
def update_cluster_center(cluster):
    cluster_embeddings = angle.encode(cluster, to_numpy=True) # 使用angle加载
     
    return np.mean(cluster_embeddings, axis=0)

def get_predicted_clusters(data,threshold):
    # 对于每个日期
    cluster_results = []
    cnt = 0
    for date in dates:
        print(cnt)
        cnt+=1
        # 获取该日期的新闻标题
        news_data = data[data['pub_time'].dt.date == date]['title'].tolist()
        # 获取该日期的新闻正文
        # news_data = data[data['pub_time'].dt.date == date]['body'].tolist() # ByBody

        embeddings = angle.encode(news_data, to_numpy=True) # 使用angle加载

        # 定义当天的簇列表
        daily_clusters = []

        # 对于每个新闻数据
        for i, embedding in enumerate(embeddings):
            # 如果簇列表为空，则新开一个簇
            if not daily_clusters:
                # daily_clusters.append({'center': embedding, 'members': [news_data[i]]})
                daily_clusters.append({'center': embedding, 'members': [i],'news':[news_data[i]]}) # 改为存index
                continue

            # 计算当前数据点与各个簇中心的相似度
            similarities = [cosine_similarity([embedding], [cluster['center']])[0][0] for cluster in daily_clusters]

            # 找到最大相似度及其对应的簇索引
            max_similarity = max(similarities)
            max_index = similarities.index(max_similarity)

            # 如果最大相似度大于阈值，则将当前数据点加入对应簇，并更新簇中心
            if max_similarity > threshold:
                daily_clusters[max_index]['members'].append(i) # 改为存index
                daily_clusters[max_index]['news'].append(news_data[i]) # 改为存index
                daily_clusters[max_index]['center'] = update_cluster_center(daily_clusters[max_index]['news'])
            # 否则新开一个簇
            else:
                daily_clusters.append({'center': embedding, 'members': [i],'news':[news_data[i]]}) # 改为存index

        # 将当天的簇信息添加到结果列表中
        cluster_results.append({'date': date, 'clusters': daily_clusters})

    predicted_clusters = []
    for cluster in cluster_results[0]['clusters']: # 2023-12-02的簇s
        clus_index = []
        for i in cluster['members']:
            clus_index.append(i)
        predicted_clusters.append(clus_index)
    print(predicted_clusters)
    
    return predicted_clusters

# 设置阈值
threshold = 0.972
clusters = get_predicted_clusters(data,threshold)

# 创建一个字典，键是语料索引，值是对应的簇大小
index_to_cluster_size = {index: len(cluster) for cluster in clusters for index in cluster}

# 读取语料文件
df = data_ORI

# 新增列clus_news_num，记录每个语料对应的簇的大小
df['T1_clus_news_num'] = df.index.map(index_to_cluster_size)

# 根据簇大小进行排序，并添加排名，相同大小的排名相同
df = df.sort_values(by='T1_clus_news_num', ascending=False)
df['T1_rank'] = df['T1_clus_news_num'].rank(method='min', ascending=False)

# 新增列S_scale，为簇大小的归一化结果
scaler = MinMaxScaler()
df['T1_S_scale'] = scaler.fit_transform(df[['T1_clus_news_num']])

# 新增列S_score，为S_scale的值乘以20
df['T1_S_score'] = df['T1_S_scale'] * 20

# 新增列index，表示语料原始的坐标
df['T1_ori_indexFrom0'] = df.index

# 只保留需要的列，并保存到新的CSV文件
T1_final_df = df[['id','T1_ori_indexFrom0', 'title', 'body', 'T1_clus_news_num', 'T1_rank','T1_S_scale', 'T1_S_score']]
T1_final_df.to_csv('./T1ClusterScore/final_result_new.csv', index=False)
print("FINISH!")


# 40个网站的排名以及赋分结果在./T2WebsiteRank/website_Rank_new.csv
# Data231202-231211/Data231202.csv
# 读取Data231202-231211/Data231202.csv，其中的website_id为网站id，现在读取./T2WebsiteRank/website_Rank_new.csv，该文件存有website_id对应的S_task_web，现在需要将Data231202.csv中的每个语料对应的website_id对应的S_task_web新增一列进行存储，然后根据S_task_web进行排序，允许并列，新增rank列，将结果中website_id,title,S_task_web,rank存到新的csv文件

# 读取两个csv文件
data_df = data_ORI
rank_df = pd.read_csv('./T2WebsiteRank/website_Rank_new.csv')

# 将两个DataFrame合并
merged_df = pd.merge(data_df, rank_df, on='website_id')

# 根据S_task_web列进行排序，并添加排名，相同权重的排名相同
merged_df = merged_df.sort_values(by='T2_S_score', ascending=False)
merged_df['T2_rank'] = merged_df['T2_S_score'].rank(method='min', ascending=False)

# 只保留需要的列，并保存到新的CSV文件
T2_final_df = merged_df[['id','website_id', 'title', 'T2_S_score', 'T2_rank']]
T2_final_df.to_csv('./T2WebsiteRank/Data231202_scoreResult.csv', index=False)


# 读取CSV文件并计算正文长度
df = data_ORI
df['body_len'] = df['body'].apply(lambda x: len(str(x).split()))  # 假设每个单词之间用空格分隔

# 按正文长度进行排序
df = df.sort_values(by='body_len', ascending=False)

# 添加排名列
df['T3_rank'] = df['body_len'].rank(method='min', ascending=False)

# 计算S_scale并添加列
max_len = df['body_len'].max()
min_len = df['body_len'].min()
df['T3_S_scale'] = (df['body_len'] - min_len) / (max_len - min_len)

# 计算body_len_score并添加列
df['T3_S_score'] = 20 * df['T3_S_scale']

# 保存结果到新的CSV文件
output_file = './T3BodyLenRank/Data231202_newDATA_rank_Score_new.csv'  # 替换为你的输出文件路径
df.to_csv(output_file, index=False)

# 只保留需要的列，并保存到新的CSV文件
T3_final_df = df[['id','title', 'body_len', 'T3_rank','T3_S_scale', 'T3_S_score']]
T3_final_df.to_csv('./T3BodyLenRank/Data231202_T3scoreResult.csv', index=False)
print("处理完成，并将结果保存到新的CSV文件中。")




# 加载孟加拉语模型
nlp = bn_core_news_sm.load()
# # textrank算法计算权重
# update 3.9：改进版！！
def textrank_weighted_word_graph(merged_titles):
    tokens = nlp(merged_titles) # 分词
    print(len(tokens))
    # print(tokens)
    
    graph = nx.Graph()
    window_size = 80  # 根据需要调整窗口大小
    
    for i, token in enumerate(tokens):
        for j in range(i+1, min(i+window_size+1, len(tokens))):
            if token != tokens[j]:  # 添加边,避免自环
                if graph.has_edge(token, tokens[j]):
                    graph[token][tokens[j]]['weight'] += 1 #在添加边时,先检查边是否已经存在。如果边已经存在,则将权重加1;否则,添加一个新边,权重为1。这样可以避免重复添加边。
                else:
                    graph.add_edge(token, tokens[j], weight=1)
    
    # 使用NetworkX的PageRank算法计算每个节点（词）的权重
    pagerank_scores = nx.pagerank(graph, weight='weight')

    return pagerank_scores,graph

# 读取CSV文件并合并所有标题
df = data_ORI

merged_titles = ' '.join(title.strip() for title in df['title'])

# ====================================
# 获取孟加拉语的停用词列表
stop_words = set(stopwords.words('bengali'))
# print(stop_words)

# 自定义标点符号列表
custom_punctuation = ['‘', '’']

# 合并 NLTK 提供的标点符号列表和自定义标点符号列表
all_punctuation = string.punctuation + ''.join(custom_punctuation)

print(all_punctuation)
# 分词# word_tokens = word_tokenize(merged_titles)

word_tokens = nlp(merged_titles) # 分词
# word_tokens = merged_titles.split() # 根据空格分词
token_texts = [token.text.strip() for token in word_tokens] # 去除多余空格

# print(token_texts)
print(type(token_texts))



# 去除停用词
# filtered_titles = [w for w in word_tokens if not w in stop_words]
filtered_titles = [w for w in token_texts if not w in stop_words] # 去除停用词
filtered_titles = [word for word in filtered_titles if word not in all_punctuation] # 去除标点符号

print("filtered_titles len\n",len(filtered_titles)) # 字符串数量！

# 将去除停用词后的词重新组合成字符串
filtered_titles_text = ' '.join(filtered_titles)

print(len(filtered_titles_text)) # 字符串长度！别被误导（所少个字符）
# ====================================

# 计算词权重
word_weights,graph = textrank_weighted_word_graph(filtered_titles_text)

# 保存pagerank算法后的词关系权重 可视化
# 根据PageRank值更新边的权重
# 记录权重关系 字典形式存储
pagerank_weighted_graph = nx.Graph()
for node, score in word_weights.items():
    pagerank_weighted_graph.add_node(node)

for u, v, data in graph.edges(data=True):
    weight = data['weight'] * word_weights[u] * word_weights[v]
    pagerank_weighted_graph.add_edge(u, v, weight=weight)

with open('./T4TitleTextRank/graph_content.txt', 'w') as file:
    file.write(str(nx.to_dict_of_dicts(pagerank_weighted_graph)))

sorted_words = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)

# 保存到新的CSV文件
# word_weights_df = pd.DataFrame(word_weights.items(), columns=['word', 'weight'])
word_weights_df = pd.DataFrame(sorted_words, columns=['word', 'weight'])


# word_weights_df.to_csv('./T4TitleTextRank/word_weight.csv', index=False)
word_weights_df.to_csv('./T4TitleTextRank/word_weight_new.csv', index=False)

# 接下来，计算每个标题的权重
# 读取词权重文件
# word_weights_df = pd.read_csv('./T4TitleTextRank/word_weight.csv')
word_weights_df = pd.read_csv('./T4TitleTextRank/word_weight_new.csv')

# 将词权重转换为字典，方便查找
word_weights = pd.Series(word_weights_df.weight.values, index=word_weights_df.word).to_dict()

# print(word_weights)
# 读取新闻标题文件
titles_df = data_ORI
# titles_df = pd.read_csv('./Data231202-231211/Data231202.csv')
# titles_df = titles_df['title']



# 定义一个函数，用于计算标题的权重
def calculate_title_weight(title):
    doc = nlp(title)
    # 对标题进行分词并计算总权重
    return sum(word_weights.get(token.text, 0) for token in doc)  # 如果词不在word_weights中，则默认权重为0
    # return sum(word_weights.get(token.text, 0) for token in doc if token.text not in stop_words and token.text not in all_punctuation)  # 如果词不在word_weights中，则默认权重为0
    # return sum(word_weights.get(token.text, 0) for token in doc if token.text not in stop_words and token.text not in string.punctuation)  # 如果词不在word_weights中，则默认权重为0


# 计算每个标题的权重
titles_df['T4_title_weight'] = titles_df['title'].apply(calculate_title_weight)
# print(titles_df['T4_title_weight'])

# 根据权重排序并添加排名，相同权重的排名相同
titles_df = titles_df.sort_values(by='T4_title_weight', ascending=False)
titles_df['T4_rank'] = titles_df['T4_title_weight'].rank(method='min', ascending=False)

# 对权重进行归一化处理，并存储结果到"S_scale"列
scaler = MinMaxScaler()
titles_df['T4_S_scale'] = scaler.fit_transform(titles_df[['T4_title_weight']])  # 归一化映射到分数！

# 创建"S_score"列
titles_df['T4_S_score'] = titles_df['T4_S_scale'] * 20

# 只保留需要的列
T4_final_df = titles_df[['id','title', 'T4_title_weight', 'T4_rank', 'T4_S_scale', 'T4_S_score']]


# 保存到新的csv文件
# final_df.to_csv('./T4TitleTextRank/titles_weight.csv', index=False)
T4_final_df.to_csv('./T4TitleTextRank/titles_weight_new.csv', index=False)

# 提取新闻的category1进行类别评分

category_df = pd.read_csv('./T5CateforyScore/category_score.csv')

# Load the CSV file with news data
# news_df = pd.read_csv('./Data231202-231211_FIX/Data231202_newDATA.csv')
news_df = data_ORI


# Merge the two DataFrames based on the "category1" column
merged_df = pd.merge(news_df, category_df, how='left', left_on='category1', right_on='category')

# Sort the merged DataFrame based on the "rank" column
sorted_df = merged_df.sort_values(by='T5_rank')

# Select the desired columns
selected_columns = ['id','title', 'category1', 'T5_rank', 'T5_S_scale', 'T5_S_score']
T5_final_df = sorted_df[selected_columns]

# Save the result to a new CSV file
T5_final_df.to_csv('./T5CateforyScore/Data231202_categoryScore_new.csv', index=False)


# T1_final_df :'id','T1_ori_indexFrom0', 'title', 'body', 'T1_clus_news_num', 'T1_rank','T1_S_scale', 'T1_S_score'
# T2_final_df:'id','website_id', 'title', 'T2_S_score', 'T2_rank'
# T3_final_df:'id','title', 'body_len', 'T3_rank','T3_S_scale', 'T3_S_score'
# T4_final_df: 'id','title', 'T4_title_weight', 'T4_rank', 'T4_S_scale', 'T4_S_score'
# T5_final_df:'id','title', 'category1', 'T5_rank', 'T5_S_scale', 'T5_S_score'
# 合并5个dataframe：
# 第一步:将T1_final_df和T2_final_df合并
merged_df = pd.merge(T1_final_df, T2_final_df, on=['id', 'title'], how='outer')

# 第二步:将第一步合并后的DataFrame与T3_final_df合并
merged_df = pd.merge(merged_df, T3_final_df, on=['id', 'title'], how='outer')

# 第三步:将第二步合并后的DataFrame与T4_final_df合并
merged_df = pd.merge(merged_df, T4_final_df, on=['id', 'title'], how='outer')

# 第四步:将第三步合并后的DataFrame与T5_final_df合并
merged_df = pd.merge(merged_df, T5_final_df, on=['id', 'title'], how='outer')

# 打印合并后的 DataFrame
merged_df.to_csv('./MergeFiveDScore/FiveDScore_Merge.csv', index=False)
# print(merged_df)


# 假设权重 
w1, w2, w3, w4, w5 = 0.5,0.05,0.05,0.3,0.1
# 权重设置思路：
# ①层次分析法 根据各任务的重要性赋权
# ②迭代 需要一个评估指标（正确个数？）来进行迭代找出模型最优权重！

# 计算总分数
merged_df['total_S_score'] = w1 * merged_df['T1_S_score'] + w2 * merged_df['T2_S_score'] + w3 * merged_df['T3_S_score'] + w4 * merged_df['T4_S_score'] + w5 * merged_df['T5_S_score']

# 生成排名
merged_df['total_rank'] = merged_df['total_S_score'].rank(method='min', ascending=False)

# 根据总分数降序排序
merged_df = merged_df.sort_values('total_S_score', ascending=False)

# 将结果保存到csv文件
merged_df.to_csv('./MergeFiveDScore/total_result.csv', index=False)

selected_columns = ['id','T1_ori_indexFrom0', 'category1','title','body','total_S_score','total_rank']
merged_df_pure =  merged_df[selected_columns]

# Save the result to a new CSV file
merged_df_pure.to_csv('./MergeFiveDScore/total_result_pure.csv', index=False)


