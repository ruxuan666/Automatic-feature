##手动特征工程

import pandas as pd
# Machine learning
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

csv_s3 = "s3://featurelabs-static/online-retail-logs.csv"
data = pd.read_csv(csv_s3, parse_dates=["order_date"])
data['price'] = data['price'] * 1.65
data['total'] = data['price'] * data['quantity']
# Restrict data to 2011
data = data[data['order_date'].dt.year == 2011]
# drop the duplicates
data = data.drop_duplicates()
# 删除有nan值的行
data = data.dropna(axis=0)
#显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
print(data.head())
#载入人工标签
labels = pd.read_csv('../input1/labels.csv', index_col=0)
"""
There are 375250 observations in the final data with 4244 unique customers.
There are 28133 labels."""
print(f'There are {len(data)} observations in the final data with {data["customer_id"].nunique()} unique customers.')
print(f'There are {len(labels)} labels.')

##添加月份属性
data['month'] = data['order_date'].dt.month
labels['cutoff_time'] = pd.to_datetime(labels['cutoff_time'])
labels['month'] = labels['cutoff_time'].dt.month #添加月份列

#划分训练样本测试样本
train = data[data['month'] < 11]
test = data[data['month'] == 11] #使用11月份的特征预测12月份的标签，这个相当于12月份的特征
train_labels = labels.loc[labels['month'] < 12, ['customer_id', 'label', 'month']]
test_labels = labels.loc[labels['month'] == 12, ['customer_id', 'label', 'month']]

#根据用户编号，月份分组，求每一组数据各个数值属性的平均值，最大值，最小值，和，计数
train_agg = train.groupby(['customer_id', 'month']).agg(['mean', 'max', 'min', 'sum','count'])
# 为列重命名
new_cols = []
for col in train_agg.columns.levels[0]:#第一行（原属性名）
    for stat in train_agg.columns.levels[1]:#第二行(统计值)
        new_cols.append(f'{col}-{stat}') #哪个属性的哪个统计量
train_agg.columns = new_cols
print(train_agg.head())
train_agg.reset_index(inplace = True)
train_agg['month'] = train_agg['month'] + 1 #本月的值其实是下月的特征，修改月份索引
train_agg.set_index(['customer_id', 'month'], inplace=True)#以用户编号与月份为索引
#根据索引在左边拼接属性
train_data = train_labels.merge(train_agg, on=['customer_id', 'month'], how='left')

test_agg = test.groupby(['customer_id', 'month']).agg(['mean', 'max', 'min','sum','count'])
new_cols = []
for col in test_agg.columns.levels[0]:
    for stat in test_agg.columns.levels[1]:
        new_cols.append(f'{col}-{stat}')
test_agg.columns = new_cols
test_agg.reset_index(inplace = True)
test_agg['month'] = test_agg['month'] + 1
test_agg.set_index(['customer_id', 'month'], inplace=True)
test_data = test_labels.merge(test_agg, on=['customer_id', 'month'], how='left')
print(test_data.head())

#用来做训练的特征名称
feature_names = train_data.drop(columns = ['customer_id', 'label', 'month']).columns
print('特征：',feature_names)

##处理缺失值，归一化处理
pipeline = Pipeline([('imputer', SimpleImputer(strategy = 'median')),
                      ('scaler', MinMaxScaler())])
train_ready = pipeline.fit_transform(train_data.drop(columns = ['customer_id', 'label', 'month']))
test_ready = pipeline.transform(test_data.drop(columns = ['customer_id', 'label', 'month']))

##———模型———##
model = RandomForestClassifier(n_estimators = 1000, random_state = 50, n_jobs = -1)
model.fit(train_ready, train_data['label'])#训练

predictions = model.predict(test_ready) #预测值
probs = model.predict_proba(test_ready)[:, 1] #预测为1的概率
p = precision_score(test_data['label'], predictions)
r = recall_score(test_data['label'], predictions)
f = f1_score(test_data['label'], predictions)
auc = roc_auc_score(test_data['label'], probs)
print(f'Precision: {round(p, 5)}')
print(f'Recall: {round(r, 5)}')
print(f'F1 Score: {round(f, 5)}')
print(f'ROC AUC: {round(auc, 5)}')