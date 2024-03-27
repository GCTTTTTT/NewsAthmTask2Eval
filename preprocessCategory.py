# 结果还行！！！！！初版完成！！！！
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification


# 加载训练好的模型
model_path = '../NewsAthm/bert-base-multilingual-cased'  ## 可更换
# modelNew_load_path = './classificationModel/bert-base-multilingual-cased_classification_undersampled_new_epoch_20.pth'  ## 可更换
modelNew_load_path = '../NewsAthmTask2Score/classificationModel/bert-base-multilingual-cased_classification_undersampled_new_epoch_20.pth'  ## 可更换

model = BertForSequenceClassification.from_pretrained(model_path, num_labels=9)

model.load_state_dict(torch.load(modelNew_load_path))
model.eval()

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# 定义类别列表
categories = ['খেলাধুলা', 'রাজনীতি', 'বিনোদন', 'অর্থনীতি', 'আইন', 'শিক্ষা', 'বিজ্ঞান', 'লাইফস্টাইল', 'অন্যান্য']

# 读取csv文件
# data = pd.read_csv('./Data231202-231211/Data231202.csv')  ## 
data = pd.read_csv('./datasets/news_20240302_20240311.csv')  ## 对0302-0311这10天进行评估  从数据库爬取（定时任务）--->拿到数据--->根据date筛选

data['pub_time'] = pd.to_datetime(data['pub_time'])

date_UNI = '2024-03-02'

from datetime import datetime, timedelta

# 初始日期
date_UNI = '2024-03-02'
# 将字符串日期转换为datetime对象
start_date = datetime.strptime(date_UNI, '%Y-%m-%d')
# 结束日期
end_date = start_date + timedelta(days=9)  # 加上9天，因为要遍历10天（包括开始日期）

# 创建日期列表
date_list = []
current_date = start_date
while current_date <= end_date:
    date_list.append(current_date.strftime('%Y-%m-%d'))
    current_date += timedelta(days=1)

# 遍历日期列表并执行操作
for date in date_list:
    date_UNI = date

    # 筛选 pub_time 为 '2024-03-02' 的数据
    filtered_data = data[data['pub_time'] == date_UNI]  ## 这个日期是参数！系统端传过来后进行处理 系统传日期---》查询数据库（看是否有缓存。没有的话就现查）---》筛选


    nan_check = filtered_data['body'].isna().sum()
    nan_check_c = filtered_data['category1'].isna().sum()
    print(nan_check)
    print(nan_check_c)

    filtered_data = filtered_data.dropna(subset=['category1','body'])
    nan_check = filtered_data['body'].isna().sum()
    nan_check_c = filtered_data['category1'].isna().sum()
    print(nan_check)
    print(nan_check_c)

    # FIX!!
    import os

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


    processed_data_file_name = f"./datasets/news_{date_UNI}_processed.csv"
    # FIX:缓存操作 若已有文件则直接读取 否则才进行预测
    # 判断文件是否存在
    if os.path.exists(processed_data_file_name):
        # 如果文件存在，则直接读取数据
        processed_data = pd.read_csv(processed_data_file_name)
    else:
        # 如果文件不存在，则执行处理数据的函数
        # processed_data = process_data(data)
        processed_data = process_data(filtered_data)


        # 将处理后的数据保存到文件中
        processed_data.to_csv(processed_data_file_name, index=False)


    print("FINISH!!")
