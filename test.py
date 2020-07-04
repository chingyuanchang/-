import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.ar_model import AR
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
warnings.filterwarnings('ignore')

#数据导入
df1=pd.read_csv('LMEAluminium3M_train.csv',header=None
                ,names=['ID','time','open_price','high_price','low_price','close_price','volume'])
df2=pd.read_csv('LMEAluminium_OI_train.csv',header=None
                ,names=['ID','time','open interest']) 
df3=pd.read_csv('Label_LMEAluminium_train_1d.csv',header=None
                ,names=['ID','time','label'])      
sub1=pd.read_csv('LMEAluminium3M_validation.csv',header=None
                ,names=['ID','time','open_price','high_price','low_price','close_price','volume'])                      
sub2=pd.read_csv('LMEAluminium_OI_validation.csv',header=None
                ,names=['ID','time','open interest'])  

#处理df1
train_data=df1.copy()
train_data.drop(['ID','high_price','low_price'],axis=1,inplace=True)
#处理df2
train_data=pd.merge(train_data,df2,on=['time'])
train_data.drop(['ID'],axis=1,inplace=True)
#处理df3
train_data=pd.merge(train_data,df3,on=['time'])
train_data.drop(['ID'],axis=1,inplace=True)

train_data['open_price'] = train_data['open_price'].astype(float)
train_data['close_price'] = train_data['close_price'].astype(float)
train_data['volume'] = train_data['volume'].astype(float)
train_data['open interest'] = train_data['open interest'].astype(float)
train_data['label'] = train_data['label'].astype(float)

temp = pd.to_datetime(train_data['time'])
train_data['day']=temp.dt.day
train_data['month']=temp.dt.month
train_data['year']=temp.dt.year
train_data.drop(['time'],axis=1,inplace=True)

#测试数据
test_data=sub1.copy()
test_data.drop(['ID','high_price','low_price'],axis=1,inplace=True)
test_data=pd.merge(test_data,sub2,on=['time'])
test_data.drop(['ID'],axis=1,inplace=True)
test_data['label']=0
test_data['open_price'] = test_data['open_price'].astype(float)
test_data['close_price'] = test_data['close_price'].astype(float)
test_data['volume'] = test_data['volume'].astype(float)
test_data['open interest'] = test_data['open interest'].astype(float)


temp2 = pd.to_datetime(test_data['time'])
test_data['day']=temp2.dt.day
test_data['month']=temp2.dt.month
test_data['year']=temp2.dt.year
test_data.drop(['time'],axis=1,inplace=True)

test_new=pd.DataFrame([test_data['open_price']
                     ,test_data['close_price']
                     ,test_data['volume']
                     ,test_data['open interest']]).T

target=train_data['label']
X_data = train_data.drop(['label','day','month','year'],axis = 1)	#删除LABEL及其它非特征列
#模型
model_lgb=lgb.LGBMRegressor(num_leaves=20
                            ,max_depth=6
                            ,learning_rate=0.1
                            ,n_estimators=1100#最大生成树数量
                            ,n_jobs=-1)
keys=[]
sk=KFold(n_splits=10,shuffle=True)
for train,test in sk.split(X_data,target):
    x_train=X_data.iloc[train]
    y_train=target.iloc[train]
    x_test=X_data.iloc[test]
    y_test=target.iloc[test]
    model_lgb.fit(x_train,y_train)
    y_hat=model_lgb.predict(x_test)
    y_hat[y_hat<0]=0
    print(1/(mean_squared_error(y_test,y_hat)**0.5+1))
    ypred=model_lgb.predict(test_new)
    y_pred = (ypred >= 0.5)*1
    keys.append(y_pred)
    # 设置阈值, 输出一些评价指标
keys=np.array(keys)
km=keys.mean(axis=0)
km[km<0]=0
test=pd.DataFrame(km)
test[test>=0.5]=1
test[test<0.5]=0
test.to_csv('result.csv',encoding='utf-8',header=None,index=None)