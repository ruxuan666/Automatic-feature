# Boruta 进行特征选择
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# load X（特征） and y（标签）
# NOTE BorutaPy accepts numpy arrays only, hence the values attribute
X = pd.read_csv('examples/test_X.csv', index_col=0).values #将第一列作为索引。
print(X[:5])
y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
y = y.ravel() #视图化，以行展开成1维
print(y[:5])


# define random forest classifier
# n_jobs=-1表示并行进程数等于核数
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
# find all relevant features --5 features should be selected
feat_selector.fit(X, y)

# check selected features -- first 5 features are selected
print('feat_selector.support_ ：', feat_selector.support_)#True，False形成的列表
# check ranking of features
print('feat_selector.ranking_ :', feat_selector.ranking_)#得到每一个属性的排名
print('n_features_ :',feat_selector.n_features_)#选出的特征数目
# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X) #得到筛选属性下的特征值
print(X_filtered[:5])
