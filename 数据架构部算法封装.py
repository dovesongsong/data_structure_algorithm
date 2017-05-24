# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:36:56 2017

@author: 248948

名称：分类算法封装
部门：数据架构部
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
         拆分处理：
         归一化处理：
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
from sklearn.preprocessing import Imputer,OneHotEncoder,LabelEncoder
from sklearn.preprocessing.data import MinMaxScaler
######画图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
######算法
from sklearn import metrics
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

class file():
    '''
    描述：此类用于记录文件属性并读取数据和检查数据
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
    a = file("d:\\248948\\Desktop\\模型算法\\算法封装\\9月.xlsx",sheet_name='数据验证')
    a.read_data()
    
    #有表头文本文件
    b = file("d:\\248948\\Desktop\\模型算法\\算法封装\\9月.txt",sep='  ')
    b.read_data()
    
    #无表头表格文件
    columns = ['性别','学历','岗位等级','离职暂存','是否有配偶','配偶是否在公司','籍贯省份','籍贯城市','序列','社保所在地','16年是否迁移社保','绩效得分','补考勤次数','请假次数','请假天数','打卡心情','打卡时长','上班时间','加班天数','异动次数','认证次数','储备次数','交通费','收入证明开具次数','晋升次数','投递简历网站个数','刷新简历次数','工龄（月）','是否在职']
    c = file(filename="d:\\248948\\Desktop\\模型算法\\算法封装\\9月v1.xlsx",sheet_name='数据验证',header=None,columns=columns)
    c.read_data()
    
    #无表头文本文件
    columns = ['性别','学历','岗位等级','离职暂存','是否有配偶','配偶是否在公司','籍贯省份','籍贯城市','序列','社保所在地','16年是否迁移社保','绩效得分','补考勤次数','请假次数','请假天数','打卡心情','打卡时长','上班时间','加班天数','异动次数','认证次数','储备次数','交通费','收入证明开具次数','晋升次数','投递简历网站个数','刷新简历次数','工龄（月）','是否在职']
    d = file(filename="d:\\248948\\Desktop\\模型算法\\算法封装\\9月v1.txt",sep='  ',header=None,columns=columns)
    d.read_data()

     
    '''
    #初始化对象信息，包括文件全路径、表头指定等
    def __init__(self,filename='',sheet_name=None,sep='\t',header=0,columns=[],n_classify=0):
        self.filename,self.sheet_name,self.sep,self.header,self.columns,self.n_classify = filename,sheet_name,sep,header,columns,n_classify
        self.data=pd.DataFrame()
        self.data_base=pd.DataFrame()
        
    ######################数据读取#######################
    #读取文件数据
    def read_data(self):
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
                self.data=pd.read_csv(self.filename,sep=self.sep,header=self.header)
            else:
                self.data=pd.read_csv(self.filename,sep=self.sep,header=None,names=columns)
        self.data_base = self.data
    ######################数据读取#######################
    
    ######################数据检查#########################
    #检查dataframe对象信息或者检查某个字段信息
    #a.check()   #检查dataframe数据对象
    #a.check(['刷新简历次数','是否在职'])   #检查字段信息
    def check(self,column_name=[]):
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
        r_data = pd.DataFrame()
        l_data = self.data[self.data.columns[self.n_classify:]]
        for column in self.data.columns[:self.n_classify]:
            r_data = pd.concat([r_data,pd.get_dummies(self.data[[column]], prefix = column)],axis=1)
        self.data = pd.concat([r_data,l_data],axis=1)
     ######################数据预处理####################### 
      
     ######################特征分析####################### 
     ##画图
     
     #bar:柱状图（单值分类字段）
     #kde:密度曲线（单值连续字段）
     #scatter:散点图（目标字段和连续字段）
     #stacked:柱状堆积图（目标字段和根据目标字段分类统计某一分类字段）
    def __Column_Plot__(self,column_name=None,kind='bar'):  
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

        
class model():
    '''
    描述：此类用于算法调用
    Parameters
    -----------
    filename : str
    '''
    def __init__(self,data=pd.DataFrame(),algorithm_type='LogisticRegression'):
        self.data,self.algorithm_type = data,algorithm_type
        self.X = np.asarray(self.data[self.data.columns[:-1]])
        self.y = np.asarray(self.data[self.data.columns[-1]])
        self.model='no_train'
    def algorithm(self):
        #主要用于分类问题
#        self.estimator = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
#        self.model=self.estimator.fit(self.X,self.y)
        if self.algorithm_type == 'LogisticRegression': 
            self.estimator = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
            self.model=self.estimator.fit(self.X,self.y)
        elif self.algorithm_type == 'GradientBoostingClassifier':
            self.estimator = GradientBoostingClassifier()
            self.model=self.estimator.fit(self.X,self.y)
        elif self.algorithm_type == 'KNeighborsClassifier':
            self.estimator = KNeighborsClassifier()
            self.model=self.estimator.fit(self.X,self.y)
        elif self.algorithm_type == 'DecisionTreeClassifier':
            self.estimator = DecisionTreeClassifier()
            self.model=self.estimator.fit(self.X,self.y)
        elif self.algorithm_type == 'SVC':    
            self.estimator = SVC()
            self.model=self.estimator.fit(self.X,self.y)
        #主要用于回归为
        elif self.algorithm_type == 'GaussianNB':
            self.estimator = GaussianNB()
            self.model=self.estimator.fit(self.X,self.y)
        else:
            pass
    def mode(self):
         pass
         

#stacked_dict={}
#for i in list(a.data['学历'].drop_duplicates()):
#    stacked_dict.update({i:a.data['是否在职'][a.data['学历']==i].value_counts()})
#
#pd.DataFrame(stacked_dict).plot(kind='bar', stacked=True)    

a = file("d:\\248948\\Desktop\\模型算法\\算法封装\\9月.xlsx",sheet_name='数据验证',n_classify=12)
a.read_data()
a.deal_null()
a.check()
#a.deal_get_dummies()
b=model(a.data)



#a.deal_label()
#a.check_corr(column_base='是否在职')
#a.check_corr(column_base='是否在职',column_corr='刷新简历次数')