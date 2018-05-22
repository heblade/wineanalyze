import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

def startjob():
    data = pd.read_csv('./data/wine.data', header=None)
    x, y = data.iloc[:, 1:], data[0]
    #将属性缩放到一个指定的最大和最小值（通常是1-0）之间
    # 使用这种方法的目的包括：
    # 1、对于方差非常小的属性可以增强其稳定性。
    # 2、维持稀疏矩阵中为0的条目。
    x  = MinMaxScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size= 0.3)

    lr = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=3)
    lr.fit(x_train, y_train.ravel())
    print('参数alpha=%.2f' % lr.alpha_)
    y_train_pred = lr.predict(x_train)
    y_test_pred = lr.predict(x_test)
    print('Logistic回归训练集准确率: ', accuracy_score(y_train, y_train_pred))
    print('Logistic回归测试集准确率: ', accuracy_score(y_test, y_test_pred))

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, oob_score=True)
    rf.fit(x_train, y_train.ravel())
    print('OOB Score=%.5f' % rf.oob_score_)
    y_train_pred = rf.predict(x_train)
    y_test_pred = rf.predict(x_test)
    print('随机森林回归训练集准确率: ', accuracy_score(y_train, y_train_pred))
    print('随机森林回归测试集准确率: ', accuracy_score(y_test, y_test_pred))

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2)
    gb.fit(x_train, y_train.ravel())
    y_train_pred = gb.predict(x_train)
    y_test_pred = gb.predict(x_test)
    print('GBDT训练集准确率: ', accuracy_score(y_train, y_train_pred))
    print('GBDT测试集准确率: ', accuracy_score(y_test, y_test_pred))

    y_train[y_train == 3] = 0
    y_test[y_test == 3] = 0
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    params = {# 构建树的深度，越大越容易过拟合
              'max_depth': 1,
              #eta, 如同学习率
              'eta': 0.9,
              # silent, 设置成1则没有运行信息输出
              'silent': 1,
              'num_class':3, #类别数，与 multisoftmax 并用
              # objective, 设置为multi:softmax，用于处理多分类的问题
              'objective': 'multi:softmax'}
    bst = xgb.train(params, data_train, num_boost_round=5, evals=watch_list)
    y_train_pred = bst.predict(data_train)
    y_test_pred = bst.predict(data_test)
    print('XGBoost训练集准确率: ', accuracy_score(y_train, y_train_pred))
    print('XGBoost测试集准确率: ', accuracy_score(y_test, y_test_pred))

if __name__ == '__main__':
    startjob()