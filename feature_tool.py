#featuretools工具使用
import featuretools as ft

#data包含三张表，即3个实体
data = ft.demo.load_mock_customer()#载入demo数据
customers_df = data["customers"]#一个客户可以有多个session
#print(customers_df) #有customer_id
sessions_df=data['sessions']#一个session可以对应多个交易
#print(sessions_df.sample(5))#有session_id,customer_id
transactions_df = data["transactions"]
#print(transactions_df[:5])#有transactions_id,session_id


#为3个实体指定一个字典，id是必需的
entities = {
 "customers" : (customers_df, "customer_id"),
"sessions" : (sessions_df, "session_id", 'session_start'),
 "transactions" : (transactions_df, "transaction_id", 'transaction_time')
}
#指定实体间的关联方式
#当两个实体存在一对多的关系(即父子实体关系时)。根据关键字进行对应
relationships = [("sessions", "session_id", "transactions", "session_id"),
 ("customers", "customer_id", "sessions", "customer_id")]
#运行深度特征合成。修改target_entity可以得到描述其它实体的特征值
feature_matrix_customers,features_defs=ft.dfs(entities=entities,
                                              relationships=relationships,target_entity='customers')
print(feature_matrix_customers)#得到描述客户的十几个特征
