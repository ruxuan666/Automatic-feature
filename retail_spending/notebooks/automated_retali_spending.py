import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import featuretools as ft #自动特征工程
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier #随机森林分类器
import os
from utils1 import plot_feature_importances

###在线数据集###
csv_s3 = "s3://featurelabs-static/online-retail-logs.csv"
data = pd.read_csv(csv_s3, parse_dates=["order_date"]) #用于后面解析具体日期
data['price'] = data['price'] * 1.65 #将价格转换为美元
data['total'] = data['price'] * data['quantity'] #得到总数目属性
data = data[data['order_date'].dt.year == 2011] #筛选年份为2011年的数据
print(data.columns) #属性列
#订单编号，商品编号
#Index(['order_id', 'product_id', 'description', 'quantity', 'order_date', 'price', 'customer_id', 'country', 'total'],dtype='object')
print(data.head(5))
data = data.drop_duplicates() #删除重复数据行
data = data.dropna(axis=0)# 删除带有空值的行

###数据统计####
data['cancelled'] = data['order_id'].str.startswith('C')#匹配以C开始的字符（表示取消订单），匹配上为True
data['cancelled'].value_counts().plot.bar(figsize = (6, 5)) #绘制频数柱状图
plt.title('Cancelled Purchases Breakdown')
print(data.describe())#计算各列的最大值，平均值等
#分国家绘制箱线图
plt.figure(figsize = (20, 6))
sns.boxplot(x = 'country', y = 'total', data = data[(data['total'] > 0) & (data['total'] < 1000)])
plt.title("Total Purchase Amount by Country")
plt.xticks(rotation = 90)
plt.figure(figsize = (20, 6))
sns.boxplot(x = 'country', y = 'quantity', data = data[(data['total'] > 0) & (data['total'] < 1000)]);
plt.title("Purchased Quantity by Country");
plt.xticks(rotation = 90)
#绘制比例图
def ecdf(data):
    x = np.sort(data)#从小到大排序
    y = np.arange(1, len(x) + 1) / len(x)#分数排名
    return x, y
plt.figure(figsize = (14, 4))
# Total
plt.subplot(121)
x, y = ecdf(data.loc[data['total'] > 0, 'total'])
plt.plot(x, y, marker = '.')
plt.xlabel('Total')
plt.ylabel('Percentile')
plt.title('ECDF of Purchase Total')
# Quantity
plt.subplot(122)
x, y = ecdf(data.loc[data['total'] > 0, 'quantity'])
plt.plot(x, y, marker = '.')
plt.xlabel('Quantity'); plt.ylabel('Percentile'); plt.title('ECDF of Purchase Quantity')
plt.show()
#分月份统计
def make_retail_cutoffs_total(start_date, end_date, threshold=500):
    # 以start_date之前的用户id为用户池
    customer_pool = data[data['order_date'] < start_date]['customer_id'].unique()
    tmp = pd.DataFrame({'customer_id': customer_pool})#用户id
    # 对于用户池中的每个用户，分别求total和
    totals = data[data['customer_id'].isin(customer_pool) &
                  (data['order_date'] > start_date) &
                  (data['order_date'] < end_date)
                  ].groupby('customer_id')['total'].sum().reset_index()#以customer_id为索引
    # 将用户id合并进来
    totals = totals.merge(tmp, on='customer_id', how='right')
    # 将Nan数据以0填充（因为有些池中用户在本月无消费）
    totals['total'] = totals['total'].fillna(0)
    # 基于阈值设置标签
    totals['label'] = (totals['total'] > threshold).astype(int)
    # 统一设置切片开始时间
    totals['cutoff_time'] = pd.to_datetime(start_date)
    #排列属性顺序
    totals = totals[['customer_id', 'cutoff_time', 'total', 'label']]
    return totals
may_spending = make_retail_cutoffs_total(pd.datetime(2011, 5, 1), pd.datetime(2011, 6, 1))
print(may_spending.head(10))
may_spending['label'].value_counts().plot.bar()
plt.title('Label Distribution for May')
march_spending = make_retail_cutoffs_total('2011-03-01', '2011-04-01', 500)
april_spending = make_retail_cutoffs_total('2011-04-01', '2011-05-01', 500)
june_spending = make_retail_cutoffs_total('2011-06-01', '2011-07-01', 500)
july_spending = make_retail_cutoffs_total('2011-07-01', '2011-08-01', 500)
august_spending = make_retail_cutoffs_total('2011-08-01', '2011-09-01', 500)
september_spending = make_retail_cutoffs_total('2011-09-01', '2011-10-01', 500)
october_spending = make_retail_cutoffs_total('2011-10-01', '2011-11-01', 500)
november_spending = make_retail_cutoffs_total('2011-11-01', '2011-12-01', 500)
december_spending = make_retail_cutoffs_total('2011-12-01', '2012-01-01', 500)
#按行拼接，仍然各自用各自索引
labels = pd.concat([march_spending, april_spending, may_spending, june_spending, july_spending, august_spending,
                    september_spending, october_spending, november_spending, december_spending], axis = 0)
if not os.path.exists('../input1'):
    os.makedirs('../input1')
labels.to_csv('../input1/labels.csv')#保存数据
print(labels.describe())
plot_labels = labels.copy()
plot_labels['month'] = plot_labels['cutoff_time'].dt.month#添加了月属性
#按照月份绘制直方图
plt.figure(figsize = (12, 6))
sns.boxplot(x = 'month', y = 'total',
            data = plot_labels[(plot_labels['total'] > 0) & (plot_labels['total'] < 1000)])
plt.title('Customer Spending Distribution by Month')
#绘制某位用户的花费情况
print(labels.loc[labels['customer_id'] == 12347])#筛选出该用户的行数据
#重设索引并取total列
labels.loc[labels['customer_id'] == 12347].set_index('cutoff_time')['total'].plot(figsize = (6, 4), linewidth = 3)
plt.xlabel('Date', size = 16)
plt.ylabel('Spending', size = 16)
plt.title('Monthly Spending for Customer', size = 20)
plt.xticks(size = 16)
plt.yticks(size = 16)

##———————自动特征工程———————##
###建立4个实体###
es = ft.EntitySet(id="Online Retail Logs")#建立一个空实体
# 添加data数据表作为实体
es.entity_from_dataframe("purchases",
                         dataframe=data,
                         index="purchases_index",#自动创建新的属性
                         time_index = 'order_date',#用于指示第一次出现时间
                         variable_types = {'description': ft.variable_types.Text})#指定属性类型
print(es['purchases']) #输出这个实体信息
print(es['purchases'].df.head(5)) #输出实体
# 在purchases实体的基础上创建一个products实体
es.normalize_entity(new_entity_id="products",
                    base_entity_id="purchases",
                    index="product_id",#以商品编号为索引
                    additional_variables=["description"])
print(es['products'])
print(es['products'].df.head(5))#输出实体.（ product_id,description,first_purchases_time）
# create a new "customers" entity based on the 'purchase' entity
es.normalize_entity(new_entity_id="customers",
                    base_entity_id="purchases",
                    index="customer_id")
# create a new "orders" entity
es.normalize_entity(new_entity_id="orders",
                    base_entity_id="purchases",
                    index="order_id",
                    additional_variables=["country", 'cancelled'])
print(es['orders'].df.head())#(order_id,country,cancelled first_purchases_time)

def feture():#涉及进程这部分需要放到main函数中运行
    ##建立特征矩阵，对顾客实体进行深度特征生成##
    print('------自动特征工程------')
    feature_matrix, feature_names = ft.dfs(entityset=es, target_entity='customers',
                                           cutoff_time=labels, verbose=2,#labels属性（customers_id,'cutoff_time', 'total', 'label'）
                                           cutoff_time_in_index=True,#双索引
                                           chunk_size=len(labels), n_jobs=-1,
                                           max_depth=1)#max_depth越大生成的特征数目越多
    print(feature_matrix.columns)#每个用户每个时间段29个特征
    # 丢弃部分特征
    feature_matrix = feature_matrix.drop(columns=['MODE(purchases.order_id)', 'MODE(purchases.product_id)'])
    print('特征矩阵型号', feature_matrix.shape)  # (28133, 27)
    print(feature_matrix.loc[12347, :].sample(10, axis=1))  # 给定用户id索引，按列采样,10个月索引*10个属性

    feature_matrix.groupby('time')['COUNT(purchases)'].mean().plot();
    plt.title('Average Monthly Count of Purchases');  # 每月各用户的平均购买数
    plt.ylabel('Purchases Per Customer');
    feature_matrix.groupby('time')['SUM(purchases.quantity)'].mean().plot();
    plt.title('Average Monthly Sum of Purchased Products');  #购买量
    plt.ylabel('Total Purchased Products Per Customer');

    feature_matrix = pd.get_dummies(feature_matrix).reset_index()#加入了customer_id与time索引属性
    print(feature_matrix.columns)
    corrs = feature_matrix.corr().sort_values('total')#计算各属性对间的相关性，竖着按照与total的相关性降序排列
    print(corrs)#28*28(无time)
    print(corrs['total'].head())#输出各属性与total的相关性
    print(corrs['total'].dropna().tail())#删除nan

    g = sns.FacetGrid(
        feature_matrix[(feature_matrix['SUM(purchases.total)'] > 0) & (feature_matrix['SUM(purchases.total)'] < 1000)],
        hue='label', size=4, aspect=3)
    g.map(sns.kdeplot, 'SUM(purchases.total)')
    g.add_legend();
    plt.title('Distribution of Purchases Total by Label');#分标签概率密度函数

    feature_matrix['month'] = feature_matrix['time'].dt.month
    sns.violinplot(x='month', y='NUM_UNIQUE(purchases.order_id)', hue='label', figsize=(24, 6),
                   data=feature_matrix[(feature_matrix['SUM(purchases.total)'] > 0) & (feature_matrix['SUM(purchases.total)'] < 1000)])
    plt.title('Number of Unique Purchases by Label');#垂直线形图
    return feature_matrix

#第二种自动特征工程
def feature11(labels=labels):
    print('----自动特征工程1------')
    labels['month'] = labels['cutoff_time'].dt.month
    labels = labels.reset_index(drop=True)#重新设置索引
    #利用特征基元扩充
    feature_matrix, feature_names = ft.dfs(entityset=es, target_entity='customers',
                                           agg_primitives=['std', 'max', 'min', 'mode',
                                                           'mean', 'skew', 'last', 'avg_time_between'],
                                           trans_primitives=['cum_sum', 'cum_mean', 'day',
                                                             'month', 'hour', 'weekday'],
                                           n_jobs=-1, chunk_size=100, max_depth=2,
                                           cutoff_time=labels, cutoff_time_in_index=True,
                                           verbose=1)
    feature_matrix.drop(columns=['MODE(purchases.order_id)', 'MODE(purchases.product_id)',
                                 'LAST(purchases.order_id)', 'LAST(purchases.product_id)'], inplace=True)
    feature_matrix = pd.get_dummies(feature_matrix).reset_index()#添加customer_id与time属性
    print(feature_matrix.shape) #28133*190
    print('feature1属性：',feature_matrix.columns)
    return feature_matrix


#——————模型——————#
model = RandomForestClassifier(n_estimators = 1000, random_state = 50, n_jobs = -1)
#分月预测标签
def predict_month(month, feature_matrix, return_probs=True):
    # labels
    test_labels = feature_matrix.loc[feature_matrix['month'] == month, 'label']#得到标签
    train_labels = feature_matrix.loc[feature_matrix['month'] < month, 'label'] #用小于该月份的做训练
    # Features,total与SUM(total)是重复的
    X_train = feature_matrix[feature_matrix['month'] < month].drop(columns=['customer_id', 'time',
                                                                                    'month', 'label', 'total'])
    X_test = feature_matrix[feature_matrix['month'] == month].drop(columns=['customer_id', 'time',
                                                                                    'month', 'label', 'total'])
    feature_names = list(X_train.columns)
    # 处理缺失值，以及归一化特征
    pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('scaler', MinMaxScaler())])
    # Fit and transform training data
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    # Labels
    y_train = np.array(train_labels).reshape((-1,))#横向平铺
    y_test = np.array(test_labels).reshape((-1,))
    print('Training on {} observations.'.format(len(X_train)))
    print('Testing on {} observations.\n'.format(len(X_test)))

    # 利用特征以及标签进行训练
    model.fit(X_train, y_train)
    # Make predictions
    predictions = model.predict(X_test)#预测标签
    probs = model.predict_proba(X_test)[:, 1]#预测为1的概率

    # Calculate metrics
    p = precision_score(y_test, predictions)#精度
    r = recall_score(y_test, predictions)#召回
    f = f1_score(y_test, predictions)#F1
    auc = roc_auc_score(y_test, probs)#ROC
    print('Precision:',round(p, 5))#保留5位小数
    print('Recall:', round(r, 5))#两label平均
    print('F1 Score:',round(f, 5))
    print('ROC AUC:',round(auc, 5))

    # Feature importances
    fi = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})#两列
    if return_probs:
        return fi, probs
    return fi


if __name__=='__main__':
    """
    feature_matrix=feture()
    june_fi,june_probs = predict_month(6, feature_matrix=feature_matrix)
    print(june_fi)
    print(june_probs)
    norm_june_fi = plot_feature_importances(june_fi,n=15)"""

    """
    #显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    print(ft.primitives.list_primitives())#输出全部特征基元，有aggregation型的和transform型的"""

    feature_matrix=feature11()
    december_fi = predict_month(month=12, feature_matrix=feature_matrix,return_probs=False)
    norm_fi = plot_feature_importances(december_fi)
