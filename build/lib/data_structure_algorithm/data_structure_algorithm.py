# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:36:56 2017

@author: xx

名称：分类算法封装
部门：xx
版本：
基本思想：三个主要功能类
         1.模型数据准备类
         2.模型训练类
         3.结果分析类
数据要求：分类特征在前，连续性的数值特征在后，最后一个字段为目标列
数据读取：把excel或者文本文件读取进来，产生包含表头的dataframe数据
         file()
数据检查：查看数据的一些特性，如字段缺失值，字段相关性
         数据特性检查：file.check(self)
数据预处理：
         缺失值处理：file.deal_null(self)
         哑变量处理：file.deal_get_dummies(self)
         采样处理：
         拆分处理：file.split(self,test_size=0.25)
         归一化处理：file.scaler(self,columns=[],scaler_type='Min-Max')
         类型转换：
         特征编码：
特征分析：用画图和一些相关性比较来体现
         file.__Column_Plot__(self,column_name=None,kind='bar')
         column_name传入list类型
         kind选值如下：
         bar:柱状图（单值分类字段）
         kde:密度曲线（单值连续字段）
         scatter:散点图（目标字段和连续字段,column_name[0]为目标字段，column_name[1]为连续字段）
         stacked:堆积图（目标字段和分类字段,column_name[0]为目标字段，column_name[1]为连续字段）
特征筛选：


模型训练：逻辑回归、GBDT、KNN算法
         逻辑回归：
结果分析：
模型融合：
"""

import pandas as pd
import numpy as np
######众数
from scipy.stats import mode
######预处理
from sklearn.preprocessing import Imputer,OneHotEncoder,LabelEncoder,binarize
from sklearn.preprocessing.data import MinMaxScaler,StandardScaler
######画图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
######算法
#GBDT
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
#逻辑回归
from sklearn.linear_model import LogisticRegression
#朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
#k-最近邻
from sklearn.neighbors import KNeighborsClassifier
#决策树
from sklearn.tree import DecisionTreeClassifier
#支持向量机
from sklearn.svm import SVC
#随机森林
from sklearn.ensemble import RandomForestRegressor
#算法评估
from sklearn.metrics import roc_curve, auc, precision_recall_curve,accuracy_score,precision_score,recall_score,classification_report,confusion_matrix

######交叉验证
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_predict


######模型保存
from sklearn.externals import joblib

class file():
    '''
    描述：此类用于记录文件属性并读取数据、检查数据、数据预处理操作等
    Parameters
    -----------
    filename : str
        Original data file path (with file name). 
        文件全路径地址
        
    sheet_name : str
        *if excel
        如果是excel默认打开的是第一个sheet，也可以指定打开的sheet名称
        
    sep : str
        if csv or txt
        文本文件必须指定分隔符
        
    header : int
        indicate if there is a row containing column names
        有表头可以不用指定，无表头则指定为None,并指定表头字段名称columns
    
    columns : []
        字段名：如['性别','学历']

    #举例
    #有表头表格文件
    a = file("xx.xlsx",sheet_name='xx')
    a.read_data()
    
    #有表头文本文件
    b = file("xx.txt",sep='  ')
    b.read_data()
    
    #无表头表格文件
    columns = ['xx','xxx']
    c = file(filename="xx.xlsx",sheet_name='xx',header=None,columns=columns)
    c.read_data()
    
    #无表头文本文件
    columns = ['xx','xxx']
    d = file(filename="xx.txt",sep='  ',header=None,columns=columns)
    d.read_data()

     
    '''
    #初始化对象信息，包括文件全路径、表头指定等
    def __init__(self,filename='',sheet_name=None,sep='\t',header=0,columns=[],n_classify=0,encoding='utf8'):
        self.filename,self.sheet_name,self.sep,self.header,self.columns,self.n_classify,self.encoding = filename,sheet_name,sep,header,columns,n_classify,encoding
        self.data=pd.DataFrame()
        self.data_base=pd.DataFrame()
        
    ######################数据读取#######################
    #读取文件数据
    def read_data(self):
        '''
        读取数据
        '''
        #获取文件类型
        file_type = lambda x : x[x.rindex('.')+1:]
        #表格文件
        if file_type(self.filename) in ['xls','xlt','xlsx','xlsm','xlsb']:
            if self.header==0:
                self.data=pd.read_excel(self.filename,sheetname=self.sheet_name,sep=self.sep,header=self.header)
            else:
                self.data=pd.read_excel(self.filename,sheetname=self.sheet_name,sep=self.sep,header=None,names=columns)
        #文本文件
        if file_type(self.filename) in ['csv','txt']:
            if self.header==0:
                self.data=pd.read_csv(self.filename,sep=self.sep,header=self.header,encoding=self.encoding)
            else:
                self.data=pd.read_csv(self.filename,sep=self.sep,header=None,names=columns,encoding=self.encoding)
        self.data_base = self.data
    ######################数据读取#######################
    
    ######################数据检查#########################
    #检查dataframe对象信息或者检查某个字段信息
    #a.check()   #检查dataframe数据对象
    #a.check(['刷新简历次数','是否在职'])   #检查字段信息
    def check(self,column_name=[]):
        '''
        检查dataframe对象信息或者检查某个字段信息
        file.check()   #检查dataframe数据对象
        file.check(['刷新简历次数','是否在职'])   #检查字段信息
        '''
        if column_name==[]:
            print('*******************head*********************')
            print(self.data.head())
            print('*******************info*********************')
            print(self.data.info())
            print('*******************describe*********************')
            print(self.data.describe())
        else:
            print('*******************head*********************')
            print(self.data[column_name].head())
            print('*******************info*********************')
            print(self.data[column_name].info())
            print('*******************describe*********************')
            print(self.data[column_name].describe())
            
    #检查相关系数,其中以column_base为基数，column_corr的相关性
    def check_corr(self,column_base=None,column_corr=None):
        '''
        检查相关系数,其中以column_base为基数，column_corr的相关性
        '''
        if column_corr==None and column_base==None:
            print('[error]please input column_base!')
        elif column_corr==None:
            print('相关系数：%s'%(column_base))
            print(self.data.corr()[column_base])
        else:
            print('相关系数：%s'%(column_base))
            print(self.data[column_base].corr(self.data[column_corr]))

    def get_column_property(self):
        pass
    
    ######################数据检查#########################
    
    ######################数据预处理#######################
    # 对于连续型特征，分别用中位数、均值、和众数填充缺失值
    # 对所有分类特征，分别用'缺失'、-1、和-1.0填充缺失值
    def deal_null(self):
        '''
        对于连续型特征，分别用中位数、均值、和众数填充缺失值
        对所有分类特征，分别用'缺失'、-1、和-1.0填充缺失值
        '''
        for column in self.data[0:self.n_classify]:
            if self.data.dtypes[column] == 'int64':
                self.data.fillna(-1, inplace=True) # 对整型值用中位数
            elif self.data.dtypes[column] == 'float64':
                self.data[column].fillna(-1.0, inplace=True)   # 对浮点值用均值
            elif self.data.dtypes[column] == 'object':
                if isinstance(self.data[column][0], str):
                    self.data[column].fillna('缺失', inplace=True)   # 对文本用众数
                else:
                    print("Error when fill NaN in column " , column)
                    
        for column in self.data.columns[self.n_classify:-1]:
            if self.data.dtypes[column] == 'int64':
                self.data[column].fillna(int(self.data[column].median()), inplace=True) # 对整型值用中位数
            elif self.data.dtypes[column] == 'float64':
                self.data[column].fillna(self.data[column].mean(), inplace=True)   # 对浮点值用均值
            elif self.data.dtypes[column] == 'object':
                if isinstance(self.data[column][0], str):
                    self.data[column].fillna(mode(self.data[column]).mode[0], inplace=True)   # 对文本用众数
                else:
                    print("Error when fill NaN in column " , column)
    
    
    #对分类字段做标签化处理                
    def deal_label(self):
        '''
        对分类字段做标签化处理   
        '''
        le = LabelEncoder()
        for i in range(self.n_classify):
            self.data[[i]]=le.fit_transform(self.data[[i]])
            print(le.transform(le.classes_))
    
    #对分类字段做哑变量处理
#    def deal_onehotencoder(self):
#        enc = OneHotEncoder()
#        for i in range(self.n_classify):
#            t_data=enc.fit(alldata[:,0:self.files[tablename].N_OHE])
            
    def deal_get_dummies(self):
        '''
        对分类字段做哑变量处理
        '''
        r_data = pd.DataFrame()
        l_data = self.data[self.data.columns[self.n_classify:]]
        for column in self.data.columns[:self.n_classify]:
            r_data = pd.concat([r_data,pd.get_dummies(self.data[[column]], prefix = column)],axis=1)
        self.data = pd.concat([r_data,l_data],axis=1)
        
    #数据拆分
    #将数据集按比例切分，默认切分比例为3:1
    def split(self,test_size=0.25):
        '''
        描述：对数据集做切分处理，默认切分比例为0.25即3:1
        Parameters
        -----------
        test_size : float64
            值范围：[0,1)
        '''
        self.train,self.test = train_test_split(self.data,test_size=test_size)

    #归一化处理
    #线性函数归一化(Min-Max scaling):线性函数将原始数据线性化的方法转换到[0 1]的范围
    #0均值标准化(Z-score standardization):0均值归一化方法将原始数据集归一化为均值为0、方差1的数据集
    def scaler(self,columns=[],scaler_type='Min-Max'):
        '''
        线性函数归一化(Min-Max scaling):线性函数将原始数据线性化的方法转换到[0 1]的范围
        0均值标准化(Z-score standardization):0均值归一化方法将原始数据集归一化为均值为0、方差1的数据集
        '''
        if scaler_type == 'Min-Max':
            Scaler = MinMaxScaler()
            self.data[columns] = Scaler.fit_transform(self.data[columns])
        else:
            Scaler = StandardScaler()
            self.data[columns] = Scaler.fit_transform(self.data[columns])
     ######################数据预处理####################### 
      
     ######################特征分析####################### 
     ##画图
     
     #bar:柱状图（单值分类字段）
     #kde:密度曲线（单值连续字段）
     #scatter:散点图（目标字段和连续字段）
     #stacked:柱状堆积图（目标字段和根据目标字段分类统计某一分类字段）
    def __Column_Plot__(self,column_name=None,kind='bar'):  
        '''
        描述：画图
        Parameters
        -----------
        column_name : list
                      用于展示的字段值
        kind ： str
                指定画图类型
                bar:柱状图（单值分类字段）
                kde:密度曲线（单值连续字段）
                scatter:散点图（目标字段和连续字段）
                stacked:柱状堆积图（目标字段和根据目标字段分类统计某一分类字段）
        '''
        if column_name==None:
            print('[error]请输入要分析的字段')
        #####一个字段的分布情况
        #柱状图
        elif len(column_name)==1 and isinstance(column_name,list) and kind=='bar':
            self.data[column_name[0]].value_counts().plot(kind=kind)
            plt.ylabel('数量')  
            plt.title(column_name[0]+'分布情况')
            plt.show()
        #密度曲线
        elif len(column_name)==1 and isinstance(column_name,list) and kind=='kde':
            self.data[column_name[0]].plot(kind=kind)
            plt.ylabel('密度') 
            plt.title(column_name[0]+'分布情况')
            plt.show()
        #####两个字段的分布情况
        #散点图
        elif len(column_name)==2 and isinstance(column_name,list) and kind=='scatter' and self.data.dtypes[column_name[1]]=='float64':
            plt.scatter(self.data[column_name[0]],self.data[column_name[1]])
            plt.xlabel(column_name[0])
            plt.ylabel(column_name[1])
            plt.title(column_name[0]+'_'+column_name[1]+'分布情况')
            plt.show()   
        #堆积图
        elif len(column_name)==2 and isinstance(column_name,list) and kind=='stacked' and (self.data.dtypes[column_name[1]]=='int64' or self.data.dtypes[column_name[1]]=='O'):
            stacked_dict={}
            for i in list(self.data[column_name[0]].drop_duplicates()):
                stacked_dict.update({i:self.data[column_name[1]][self.data[column_name[0]]==i].value_counts()})
            pd.DataFrame(stacked_dict).plot(kind='bar', stacked=True)  
            plt.xlabel(column_name[1])
            plt.ylabel(column_name[0]+'数量')
            plt.title(column_name[1]+'_'+column_name[0]+'堆积图')
        else:
            print('[error]错误的输入')
    

    ######################特征分析#######################

    
class model():
    '''
    描述：此类用于算法调用
    Parameters
    -----------
    data : dataframe
        传递的数据必须都为数值类型（float或者int），最后一列为目标列
    algorithm_type ：str
        算法选择：
        逻辑回归：LogisticRegression
        梯度渐进分类树：GradientBoostingClassifier
        K-最近邻：KNeighborsClassifier
        决策树：DecisionTreeClassifier
        向量机：SVC
        朴素贝叶斯：GaussianNB
        随机森林：RandomForestRegressor
    '''
    def __init__(self,data=pd.DataFrame(),algorithm_type='LogisticRegression'):
        self.data,self.algorithm_type = data,algorithm_type
        self.X = np.asarray(self.data[self.data.columns[:-1]])
        self.y = np.asarray(self.data[self.data.columns[-1]])
        
       
        self.model='no_train'
        
    #数据拆分
    #将数据集按比例切分，默认切分比例为3:1
    def split(self,test_size=0.25):
        '''
        描述：对数据集做切分处理，默认切分比例为0.25即3:1
        Parameters
        -----------
        test_size : float64
            值范围：[0,1)
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,test_size=test_size)
        
        
    ######################模型训练#######################
    def algorithm(self,**kw):
        '''
        描述：用指定算法训练模型
        
        '''
        
    # 大多数情况下被用来解决分类问题（二元分类），但多类的分类（所谓的一对多方法）也适用。
    # 这个算法的优点是对于每一个输出的对象都有一个对应类别的概率。
        if self.algorithm_type == 'LogisticRegression': 
            self.estimator = LogisticRegression(**kw)
            self.model = self.estimator.fit(self.X,self.y)
    # GBDT是一个应用很广泛的算法，可以用来做分类、回归
        elif self.algorithm_type == 'GradientBoostingClassifier':
            self.estimator = GradientBoostingClassifier(**kw)
            self.model = self.estimator.fit(self.X,self.y)
    # kNN（k-最近邻）方法通常用于一个更复杂分类算法的一部分。
    # 例如，我们可以用它的估计值做为一个对象的特征。
    # 有时候，一个简单的kNN算法在良好选择的特征上会有很出色的表现。
    # 当参数（主要是metrics）被设置得当，这个算法在回归问题中通常表现出最好的质量      
        elif self.algorithm_type == 'KNeighborsClassifier':
            self.estimator = KNeighborsClassifier(**kw)
            self.model = self.estimator.fit(self.X,self.y)
    # 分类和回归树（CART）经常被用于这么一类问题，在这类问题中对象有可分类的特征且被用于回归和分类问题。
    # 决策树很适用于多类分类
        elif self.algorithm_type == 'DecisionTreeClassifier':
            self.estimator = DecisionTreeClassifier(**kw)
            self.model = self.estimator.fit(self.X,self.y)
    # SVM（支持向量机）是最流行的机器学习算法之一，它主要用于分类问题。
    # 同样也用于逻辑回归，SVM在一对多方法的帮助下可以实现多类分类。
        elif self.algorithm_type == 'SVC':    
            self.estimator = SVC(**kw)
            self.model=self.estimator.fit(self.X,self.y)
    # 它也是最有名的机器学习的算法之一，它的主要任务是恢复训练样本的数据分布密度。
    # 这个方法通常在多类的分类问题上表现的很好。
        elif self.algorithm_type == 'GaussianNB':
            self.estimator = GaussianNB(**kw)
            self.model = self.estimator.fit(self.X,self.y)
    # 在数据集上表现良好
    # 在当前的很多数据集上，相对其他算法有着很大的优势
    # 它能够处理很高维度（feature很多）的数据，并且不用做特征选择
    # 在训练完后，它能够给出哪些feature比较重要
    # 在创建随机森林的时候，对generlization error使用的是无偏估计
    # 训练速度快
    # 在训练过程中，能够检测到feature间的互相影响
    # 容易做成并行化方法
    # 实现比较简单
        elif self.algorithm_type == 'RandomForestRegressor':
            self.estimator = RandomForestRegressor(**kw)
            self.model = self.estimator.fit(self.X,self.y)
        else:
            pass
    
    ######################模型训练#######################
    
    ######################交叉验证#######################
    def cross_validation(self,**kw):
        '''
        描述：交叉验证
        
        '''
        
    # 大多数情况下被用来解决分类问题（二元分类），但多类的分类（所谓的一对多方法）也适用。
    # 这个算法的优点是对于每一个输出的对象都有一个对应类别的概率。
        if self.algorithm_type == 'LogisticRegression': 
            self.estimator = LogisticRegression(**kw)
            predicted = cross_val_predict(self.estimator, self.X, self.y, cv=10)
    # GBDT是一个应用很广泛的算法，可以用来做分类、回归
        elif self.algorithm_type == 'GradientBoostingClassifier':
            self.estimator = GradientBoostingClassifier(**kw)
            predicted = cross_val_predict(self.estimator, self.X, self.y, cv=10)
    # kNN（k-最近邻）方法通常用于一个更复杂分类算法的一部分。
    # 例如，我们可以用它的估计值做为一个对象的特征。
    # 有时候，一个简单的kNN算法在良好选择的特征上会有很出色的表现。
    # 当参数（主要是metrics）被设置得当，这个算法在回归问题中通常表现出最好的质量      
        elif self.algorithm_type == 'KNeighborsClassifier':
            self.estimator = KNeighborsClassifier(**kw)
            predicted = cross_val_predict(self.estimator, self.X, self.y, cv=10)
    # 分类和回归树（CART）经常被用于这么一类问题，在这类问题中对象有可分类的特征且被用于回归和分类问题。
    # 决策树很适用于多类分类
        elif self.algorithm_type == 'DecisionTreeClassifier':
            self.estimator = DecisionTreeClassifier(**kw)
            predicted = cross_val_predict(self.estimator, self.X, self.y, cv=10)
    # SVM（支持向量机）是最流行的机器学习算法之一，它主要用于分类问题。
    # 同样也用于逻辑回归，SVM在一对多方法的帮助下可以实现多类分类。
        elif self.algorithm_type == 'SVC':    
            self.estimator = SVC(**kw)
            predicted = cross_val_predict(self.estimator, self.X, self.y, cv=10)
    # 它也是最有名的机器学习的算法之一，它的主要任务是恢复训练样本的数据分布密度。
    # 这个方法通常在多类的分类问题上表现的很好。
        elif self.algorithm_type == 'GaussianNB':
            self.estimator = GaussianNB(**kw)
            predicted = cross_val_predict(self.estimator, self.X, self.y, cv=10)
    # 在数据集上表现良好
    # 在当前的很多数据集上，相对其他算法有着很大的优势
    # 它能够处理很高维度（feature很多）的数据，并且不用做特征选择
    # 在训练完后，它能够给出哪些feature比较重要
    # 在创建随机森林的时候，对generlization error使用的是无偏估计
    # 训练速度快
    # 在训练过程中，能够检测到feature间的互相影响
    # 容易做成并行化方法
    # 实现比较简单
        elif self.algorithm_type == 'RandomForestRegressor':
            self.estimator = RandomForestRegressor(**kw)
            predicted = cross_val_predict(self.estimator, self.X, self.y, cv=10)
        else:
            pass
        acc=cross_val_score(self.estimator, self.X, self.y, cv=10, scoring='accuracy')
        print('十折交叉验证(accuracy)：%r\n平均值：%f'%(acc,acc.mean()))
        rec=cross_val_score(self.estimator, self.X, self.y, cv=10, scoring='recall')
        print('十折交叉验证(accuracy)：%r\n平均值：%f'%(rec,rec.mean()))
        f1=cross_val_score(self.estimator, self.X, self.y, cv=10, scoring='f1')
        print('十折交叉验证(accuracy)：%r\n平均值：%f'%(f1,f1.mean()))
        plt.scatter(self.y, predicted)
    ######################交叉验证#######################
    
    ######################数据输出#######################
    # 预测数据输出
    # 用类中训练好的模型去预测数据,数据必须经过相同处理
    def predict(self,data_X,data_y):
        '''
        描述：用训练好的模型预测
        '''
        self.predict_result = self.model.predict(data_X)
        # summarize the fit of the model
        print('*******************predict*********************')
        print('************classification_report**************')
        print(classification_report(data_y, self.predict_result))
        self.cr=classification_report(data_y, self.predict_result)
        print('***************confusion_matrix****************')
        print(confusion_matrix(data_y, self.predict_result))
        self.cm=confusion_matrix(data_y, self.predict_result)
        print('*******************predict*********************')
        
        
    # 输出预测概率，针对有预测概率输出的模型
    def proba(self,data_X):
        '''
        描述：训练好的模型预测得出概率分布，针对可以输出概率分布的模型
        '''
        self.proba_result = self.model.predict_proba(data_X)
        
    def threshold(self,data_X,data_y,l_precision=0.50,m_recall=0.70):
        '''
        描述：通过指定最低阈值和召回率来确定阈值，阈值是用来对概率进行控制的
        '''
        self.proba(data_X)
        max_values=0
        th=0
        precision,recall,thresholds = precision_recall_curve(data_y, self.proba_result[:,1])
        for (l,m,n) in zip(precision,recall,np.insert(thresholds,0,values=0)):
            if  (l > 0.50 or m > 0.70) and m*l > max_values:
                max_values = m*l
                th = n
        if max == 0:
            print("error")
        else:
            pred = binarize(self.proba_result[:,1],threshold=th).reshape((self.proba_result[:,1].size,))
            print('*******************threshold*********************')
            print('*******************Threshold*********************')
            print('Threshold : ', th)
            self.thresholds=th
            print('************classification_report****************')
            print(classification_report(data_y, pred))
            self.cr=classification_report(data_y, pred)
            print('***************confusion_matrix******************')
            print(confusion_matrix(data_y, pred))
            self.cm=confusion_matrix(data_y, pred)
            print('*******************threshold*********************')
            
    def job_dump_model(self,filename='model.sav'):
        '''
        描述：训练好的模型进行保存
        '''
        # save the model to disk
        joblib.dump(self.model, filename)
    
    def job_load_model(self,filename):
        '''
        描述：加载模型文件给model.model
        '''
        self.model = joblib.load(filename)
        print(self.model)
        
                
    def statistics_analy_report(self,data_X,data_y):
        '''
        描述：输出model的相关统计信息
        '''
        print('算法：%s'%(self.algorithm_type))
        print('训练模型:')
        print(self.model)
        print('阈值：%s'%(self.thresholds))
        print('二分类报告：')
        print(self.cr)
        print('混淆矩阵')
        print(self.cm)
        self.proba = self.model.predict_proba(data_X)
        prc=precision_recall_curve(data_y, self.proba_result[:,1])
        plt.plot(prc[1],prc[0])
        plt.xlabel('Recall')  
        plt.ylabel('Precision') 
        plt.title('P/R图')
        plt.show()
       
    ######################数据输出#######################

    

    ######################模型评估#######################
    
    
    ######################模型评估#######################

    

