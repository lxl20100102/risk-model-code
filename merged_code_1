# Auto-merged batch 1/4
# Total files in this batch: 59



#==============================================================================
# File: 01提现全渠道不限成本子分融合模型FPD30_report.html.py
#==============================================================================

#!/usr/bin/env python
# coding: utf-8

# # 一、模型简介

# 本模型为贷中实时提现融合风险模型，可用于授信审批和提现审批场景。模型的Y标签为fpd30，特征变量来源于提现行为模型子分，催收模型子分，征信模型子分，百融模型子分，三方评分，其中，提现行为模型子分、催收模型子分为离线数据，征信模型子分、百融模型子分、三方评分为实时数据，应用lgb算法进行开发。
# 训练样本：选取2024年7月21日至2024年9月20日，api全渠道提现通过的客群。
# oot样本：选取2024年9月21日至2024年10月15日，api全渠道提现通过的客群。
# 
# 

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import toad
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
import joblib
import pickle
import time
from datetime import datetime
import os 
import gc
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_row',None)
pd.set_option('display.width',None)
pd.set_option('display.precision', 6)


# # 二、 样本概况

# In[2]:


def load_model_from_pkl(path):
    """
    从路径path加载模型
    :param path: 保存的目标路径
    """
    
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


# In[ ]:





# In[4]:


result_path = '../result/全渠道实时提现行为模型fpd30/'


# In[5]:


# 最终模型打分
lgb_model= load_model_from_pkl(result_path + '全渠道实时提现行为模型fpd30_v3_20241218154942.pkl')


# In[6]:


varsname = lgb_model.feature_name()


# In[14]:


# usecols= ['order_no','channel_id', 'lending_time','lending_month', 'mob',\
#           'maxdpd', 'fpd', 'fpd10', 'fpd30', 'mob4dpd30', 'diff_days','data_set'] + varsname
# df_sample = pd.read_csv(result_path+'全渠道实时提现行为模型_建模数据集_241218.csv',usecols=usecols)
# df_sample.info(show_counts=True)
# df_sample.head()


# In[63]:


# 获取数据
def get_data(sql):
    from odps import ODPS
    import time
    from datetime import datetime
    # 输入账号密码
    conn= ODPS(username='liaoxilin', password='j02vYCxx')

    print('开始跑数' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = time.time()
    # 执行脚本
    instance = conn.execute_sql(sql)
    # 输出执行结果
    with instance.open_reader() as reader:
        print('===================')
        data = reader.to_pandas()

    end = time.time()
    print('结束跑数' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("运行事件：{}秒".format(end-start))   

    return data


# 插入数据
def execute_sql(sql):
    from odps import ODPS
    import time
    from datetime import datetime
    # 输入账号密码
    conn= ODPS(username='liaoxilin', password='j02vYCxx')
    
    print('开始跑数' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = time.time()
    # 执行脚本
    conn.execute_sql(sql)
    end = time.time()
    print('结束跑数' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("运行事件：{}秒".format(end-start))   


# In[64]:


df_sample_dict = {}


# In[65]:



# 计算今天的时间
from datetime import datetime, timedelta, date

today = datetime.now().strftime('%Y-%m-%d')
print(today)

this_day =datetime.strptime('2024-10-31', '%Y-%m-%d')
end_day = datetime.strptime('2024-07-21', '%Y-%m-%d')

while this_day >= end_day:
    run_day = this_day.strftime('%Y-%m-%d')
    sql = f'''
select 
 t.order_no
,t.id_no_des
,t.channel_id
,t.lending_time
,substr(t.lending_time, 1, 7) as lending_month
,t.mob
,t.maxdpd
,t.fpd
,t.fpd10
,t.fpd30
,t.mob4dpd30

--行为分数
,t0.bad_score

from 
    (
    select * 
    from znzz_fintech_ads.dm_f_lxl_test_order_Y_target as t 
    where dt=date_sub(current_date(), 1) 
      and lending_time='{run_day}'
    ) as t 
--行为模型分数 from znzz_fintech_ads.dm_f_lxl_test_behave_model_merge_fpd30_score
left join 
    (
    select * 
    from znzz_fintech_ads.dm_f_lxl_test_r_behave_model_merge_fpd30_score as t 
    where dt='{run_day}'
    ) as t0 on t.id_no_des=t0.id_no_des
;
'''
    print(f'=========================={run_day}=============================')
    df_sample_dict[run_day] = get_data(sql)
    this_day = this_day - timedelta(days=1)


# In[83]:


df_sample_ = pd.concat(df_sample_dict.values(), ignore_index=True)
df_sample_.info(show_counts=True)


# In[88]:


df_sample = df_sample_.query("channel_id!='1'").reset_index(drop=True)
df_sample['lending_month_new'] = df_sample['lending_month']
df_sample.loc[df_sample.query("lending_time>='2024-10-01' & lending_time<='2024-10-15'").index, 'lending_month_new']='2024-10_oot1'
df_sample.loc[df_sample.query("lending_time>='2024-10-16' & lending_time<='2024-10-31'").index, 'lending_month_new']='2024-10_oot2'


# In[84]:


df_sample = df_sample_.query("channel_id!='1'")


# In[89]:


# df_sample.reset_index(drop=True,inplace=True)
df_sample.info(show_counts=True)


# In[80]:


# print(df_sample['lending_time'].max(), df_sample['lending_time'].min())
# print(df_sample['data_set'].value_counts())
# df_sample.groupby(['lending_time','fpd30'])['order_no'].count().unstack()


# In[16]:


df_sample = df_sample.replace(-1, np.nan)


# In[17]:


df_sample['y_prob'] = lgb_model.predict(df_sample[varsname], num_iteration=lgb_model.best_iteration)


# In[18]:


# 设置标签和分数列
target = 'fpd30'
score = 'y_prob'


# In[19]:


def get_target_summary(df, target, groupby_col):
    """
    对 DataFrame 进行分组聚合，并添加一个汇总行。
    
    参数:
    - df: 待处理的 DataFrame
    - groupby_col: 用于分组的列名
    - agg_cols: 字典，键是列名，值是聚合函数名称（如 'count', 'sum', 'mean'）
    - new_col_name: 字典,键是旧列的名称，值是新列的名称
    
    返回:
    - 包含分组聚合结果和汇总行的新 DataFrame
    """
    # 使用 groupby 和 agg 进行分组和聚合
    grouped = df.groupby(groupby_col)[target].agg(total=lambda x: len(x), 
            bad=lambda x: x.sum(), 
            good=lambda x: (x== 0).sum(), 
            bad_rate=lambda x: x.mean()).reset_index()
    
    # 计算整个 DataFrame 的聚合统计量
    total_summary = df[target].agg(total=lambda x: len(x), 
            bad=lambda x: x.sum(), 
            good=lambda x: (x== 0).sum(), 
            bad_rate=lambda x: x.mean()).to_frame().T
    total_summary[groupby_col] = 'Total'
    
    # 将汇总行添加到分组结果中
    result = pd.concat([grouped, total_summary], ignore_index=True)
    result.rename(columns={groupby_col: 'bins'}, inplace=True)
    
    # 返回结果
    return result


# In[81]:


df_target_summary_month = get_target_summary(df_sample, target, 'lending_month_new')
df_target_summary_month


# In[90]:


df_target_summary_month = get_target_summary(df_sample, target, 'lending_month_new')
df_target_summary_month


# In[85]:


df_target_summary_month = get_target_summary(df_sample, target, 'lending_month')
df_target_summary_month


# In[20]:


df_target_summary_month = get_target_summary(df_sample, target, 'lending_month')
df_target_summary_month


# In[23]:


# # 使用的数据，训练模型
# X_train, X_test, y_train, y_test = train_test_split(df_sample.query("data_set!='3_oot'")[varsname],
#                                                     df_sample.query("data_set!='3_oot'")[target],
#                                                     test_size=0.2, 
#                                                     random_state=22, 
#                                                     stratify=df_sample.query("data_set!='3_oot'")[target])
# print(X_train.shape, X_test.shape)
# df_sample.loc[X_train.index, 'data_set']='1_train'
# df_sample.loc[X_test.index, 'data_set']='2_test'


# In[24]:


# df_sample['data_set'].value_counts()


# In[25]:


df_target_summary_set = get_target_summary(df_sample, target, 'data_set')
df_target_summary_set


# # 三、变量重要性

# In[26]:



def cal_psi_by_month(df_actual, df_expect, cols, month_col, combiner, return_frame = True):
    """
    计算每个月每的psi。
    
    参数:
    - df_actual: 测试集
    - df_expect: 训练集
    - cols: 需要计算稳定性的列名列表
    - month_col: 分组的列名

    返回:
    - 包含每个月的新 DataFrame
    """
    bins_df_list = []
    psi_list = []
    for month_, df_actual_group in df_actual.groupby(month_col):
        if return_frame:
            psi_, bins_df = toad.metrics.PSI(df_actual_group[cols], df_expect[cols], 
                                            combiner = combiner, return_frame = return_frame)
            psi_ = pd.DataFrame({month_: psi_}, index=cols)
            psi_list.append(psi_)
            bins_df['month'] = month_
            bins_df_list.append(bins_df)
        else:
            psi_ = toad.metrics.PSI(df_actual_group[cols], df_expect[cols], 
                                            combiner = combiner, return_frame = return_frame)
            psi_ = pd.DataFrame({month_: psi_}, index=cols)
            psi_list.append(psi_)
        
    # 合并所有结果 DataFrame
    if return_frame:
        psi_df = pd.concat(psi_list, axis=1)
        bins_df = pd.concat(bins_df_list, axis=0)
        
        return (psi_df, bins_df)
    else:
        psi_df = pd.concat(psi_list, axis=1)
        
        return psi_df


def cal_iv_by_month(df, cols, target, month_col, combiner):
    """
    计算每个变量每个月的iv。
    
    参数:
    - df: 待处理的 DataFrame
    - target: Y标签
    - cols: 需要计算iv的列名列表
    - month_col：月份列名
    
    返回:
    - 包含每个月每列iv的新 DataFrame
    """
    df_ = combiner.transform(df[cols+[target, month_col]], labels=True)
    result = pd.DataFrame(columns=sorted(list(df_[month_col].unique())), index=cols)
    for col in cols:
        for month in sorted(list(df_[month_col].unique())):
            data = df_[df_[month_col] == month]
            regroup = data.groupby(col)[target].agg(total=lambda x: x.count(), bad=lambda x: x.sum())
            regroup['good'] = regroup['total'] - regroup['bad']
            regroup['bad_pct'] = regroup['bad']/regroup['bad'].sum()
            regroup['good_pct'] = regroup['good']/regroup['good'].sum()
            regroup['woe'] = np.log(regroup['bad_pct']/regroup['good_pct'])
            regroup['iv'] = (regroup['bad_pct']-regroup['good_pct'])*regroup['woe']
            result.loc[col, month] = regroup['iv'].sum()     
      
    return result


def calculate_vars_distribute(df, cols, target, group_col):    
    """
    参数:
    - df: 待处理的 DataFrame
    - target: Y标签
    - cols: 需要分箱的列名列表
    - group_col：分组列名，如月份、渠道、数据类型
    
    返回:
    - 包含每个月每列iv的新 DataFrame
    """
    result = pd.DataFrame()
    vars = sorted(list(df[group_col].unique()))
    for col in cols:
        for var in vars:
            data = df[df[group_col] == var]
            regroup = data.groupby(col)[target].agg(total=lambda x: x.count(), bad=lambda x: x.sum())
            regroup['good'] = regroup['total'] - regroup['bad']
            regroup['bad_rate'] = regroup['bad']/regroup['total']
            regroup['bad_rate_cum'] = regroup['bad'].cumsum()/regroup['total'].cumsum()
            regroup['total_pct'] = regroup['total']/regroup['total'].sum()
            regroup['bad_pct'] = regroup['bad']/regroup['bad'].sum()
            regroup['good_pct'] = regroup['good']/regroup['good'].sum()
            regroup['bad_pct_cum'] = regroup['bad_pct'].cumsum()
            regroup['good_pct_cum'] = regroup['good_pct'].cumsum()
            regroup['total_pct_cum'] = regroup['total_pct'].cumsum()
            regroup['ks_bin'] = regroup['bad_pct_cum'] - regroup['good_pct_cum']
            regroup['ks'] = regroup['ks_bin'].max()
            regroup['lift_cum'] = regroup['bad_rate_cum']/data[target].mean()
            regroup['lift'] = regroup['bad_rate']/data[target].mean()
            regroup['woe'] = np.log(regroup['bad_pct']/regroup['good_pct'])
            regroup['iv_bins'] = (regroup['bad_pct']-regroup['good_pct'])*regroup['woe']
            regroup['iv'] = regroup['iv_bins'].sum()
            regroup['bins'] = regroup.index
                
            total_summary = data[target].agg(total=lambda x: x.count(), bad=lambda x: x.sum()).to_frame().T
            total_summary['good'] = regroup['total'] - regroup['bad']
            total_summary['bad_rate'] = total_summary['bad']/total_summary['total']
            total_summary['iv'] = regroup['iv_bins'].sum()
            total_summary['ks'] = regroup['ks_bin'].max()
            total_summary['bins'] = 'Total'
            
            regroup = pd.concat([regroup, total_summary], axis=0, ignore_index=True)
            regroup['varsname'] = col
            regroup['groupvars'] = var
            
            usecols = ['groupvars', 'varsname', 'bins', 'total', 'bad', 'good', 'bad_rate', 'bad_rate_cum', 'woe', 'iv', 'iv_bins', 
                       'ks', 'ks_bin', 'lift', 'lift_cum', 'total_pct', 'total_pct_cum', 'bad_pct', 'bad_pct_cum', 'good_pct','good_pct_cum']
            regroup = regroup[usecols]
            result = pd.concat([result, regroup], axis=0, ignore_index=True)

    return result


def feature_importance(model):
    if isinstance(model, lgb.Booster):
        # print("这是原生接口的模型 (Booster)")
        # 获取特征重要性
        feature_importance_gain = model.feature_importance(importance_type='gain')
        feature_importance_split = model.feature_importance(importance_type='split')
        # 获取特征名称
        feature_names = model.feature_name()
        # 将特征重要性转换为数据框
        df_importance = pd.DataFrame({'gain': feature_importance_gain,
                                      'split': feature_importance_split}, 
                                     index=feature_names)
        df_importance = df_importance.sort_values('gain', ascending=False)
        df_importance.index.name = 'feature'
    elif isinstance(model, (LGBMClassifier, LGBMRegressor)):
        # print("这是 sklearn 接口的模型")
        df1_dict = model.get_booster().get_score(importance_type='weight')
        importance_type_split = pd.DataFrame.from_dict(df1_dict, orient='index')
        importance_type_split.columns = ['split']
        importance_type_split = importance_type_split.sort_values('split', ascending=False)
        importance_type_split['split_pct'] = importance_type_split['split'] / importance_type_split['split'].sum()

        df2_dict = model.get_booster().get_score(importance_type='gain')
        importance_type_gain = pd.DataFrame.from_dict(df2_dict, orient='index')
        importance_type_gain.columns = ['gain']
        importance_type_gain = importance_type_gain.sort_values('gain', ascending=False)
        importance_type_gain['gain_pct'] = importance_type_gain['gain'] / importance_type_gain['gain'].sum()

        df_importance = pd.concat([importance_type_gain, importance_type_split], axis=1)
        df_importance = df_importance.sort_values('gain', ascending=False)
        df_importance.index.name = 'feature'
    else:
        print("未知模型类型")
        df_importance = None
    
    return df_importance


# In[27]:


# # 读取模型

# def load_model_from_pkl(path):
#     """
#     从路径path加载模型
#     :param path: 保存的目标路径
#     """
    
#     with open(path, 'rb') as f:
#         model = pickle.load(f)
#     return model


# In[28]:


# # 读取模型结果
# lgb_model = load_model_from_pkl('./result/227渠道实时提现行为模型fpd30/全渠道实时提现行为模型fpd30_v3_20241216162000.pkl')


# In[29]:


# # 入模的变量
# varsname = lgb_model.feature_name()


# In[30]:


# 模型变量重要性
df_importance = feature_importance(lgb_model) 


# In[31]:


# 计算分布前先变量分箱
combiner = toad.transform.Combiner()
# combiner.fit(df_sample[varsname+[target]], y=target, 
#              method='dt', n_bins=6, min_samples = 0.05, empty_separate=True) 


# In[32]:


# result_path = './result/227渠道实时提现行为模型fpd30/'


# In[34]:


# with open(result_path + '变量分箱字典_{timestamp}.pkl', 'rb') as f:
#     new_bins_dict = pickle.load(f)
# print(new_bins_dict)


# In[36]:


# combiner.load(new_bins_dict)


# In[37]:


df_bins = combiner.transform(df_sample, labels=True)
selected_cols = ['groupvars', 'varsname', 'bins', 'total','bad', 'good', 'total_pct', 'bad_pct','good_pct', 'bad_rate', 'iv']


# In[38]:


df_group_set = calculate_vars_distribute(df_bins, varsname, target, 'data_set')[selected_cols]  
df_group_month = calculate_vars_distribute(df_bins, varsname, target, 'lending_month')[selected_cols] 
 


# In[39]:


# 计算psi
df_psi_by_month = cal_psi_by_month(df_sample, df_sample.query("lending_month=='2024-08'"), varsname, 'lending_month', combiner, return_frame = False)
# 计算iv
df_iv_by_month = cal_iv_by_month(df_sample, varsname, target, 'lending_month', combiner)

df_psi_iv_by_month = pd.merge(df_psi_by_month, df_iv_by_month, how='inner', left_index=True,right_index=True, suffixes=('_psi', '_iv'))


# In[40]:


# 计算psi
df_psi_by_set = cal_psi_by_month(df_sample, df_sample.query("data_set=='1_train'"), varsname, 'data_set', combiner, return_frame = False)
# 计算iv
df_iv_by_set = cal_iv_by_month(df_sample, varsname, target, 'data_set', combiner)

df_psi_iv_by_set = pd.merge(df_psi_by_set, df_iv_by_set, how='inner', left_index=True,right_index=True, suffixes=('_psi', '_iv'))


# In[41]:


# df_psi_iv_by_month.head()


# In[42]:


# df_psi_iv_by_set.head()


# In[44]:


# varsname


# In[45]:


comment = {'behave_fpd30_score': '提现行为fpd30模型子分'
            ,'pudao_34': '避雷针定制分v1'
            ,'prob_fpd30_v1': '人行征信fpd30v1版模型子分'
            ,'br_v3_fpd30_score': '百融多头v3版模型子分'
            ,'ruizhi_6': 'Fico/FICO联合建模定制分2(上海)'
            ,'dpd30_4m_bad_prob': '贷中提现风险dpd30_4m模型子分'
            ,'hengpu_4': '恒普反欺诈分M3'
            ,'aliyun_5': '阿里申请反欺诈V5'
            ,'pudao_68': '朴道/朴道-银商银杏定制分'
            ,'hengpu_5': '恒普定制信用分Y'
            ,'cpd1_v1_prob': '入催cpd1模型子分'
            ,'duxiaoman_6': '度小满-欺诈因子V4'
            ,'pudao_20': '朴道-腾讯天御反欺诈V7通用版'
            ,'dpd30_6m_bad_prob': '贷中截面风险dpd30_6m模型子分'
            ,'score_fpd6_v1': '人行征信fpd6v1版模型子分'
            ,'score_fpd10_v2': '人行征信fpd10v2版模型子分'
            ,'score_fpd10_v1': '人行征信fpd10v1版模型子分'
          }


# In[46]:


df_importance_set = pd.merge(df_importance, df_psi_iv_by_set, how='inner', left_index=True,right_index=True)
df_importance_set.drop(columns=['1_train_psi','2_test_psi'],inplace=True)
df_importance_set = df_importance_set.reset_index()
df_importance_set = df_importance_set.rename(columns={'index':'varsname','3_oot_psi':'vars_psi'})
df_importance_set['comment'] = df_importance_set['varsname'].map(comment)
df_importance_set = df_importance_set[['varsname','comment','gain','split','vars_psi','1_train_iv','2_test_iv','3_oot_iv']]
df_importance_set


# In[47]:


df_importance_month = pd.merge(df_importance, df_psi_iv_by_month, how='inner', left_index=True,right_index=True)
df_importance_month.drop(columns=['2024-07_psi'],inplace=True)
df_importance_month = df_importance_month.reset_index()
df_importance_month = df_importance_month.rename(columns={'index':'varsname'})
df_importance_month['comment'] = df_importance_month['varsname'].map(comment)
df_importance_month = df_importance_month[['varsname','comment','gain','split','2024-08_psi','2024-09_psi','2024-10_psi','2024-07_iv','2024-08_iv','2024-09_iv','2024-10_iv']]
df_importance_month


# # 四、模型效果评估

# In[48]:



def model_ks_auc(df, target, y_pred, group_col):
    """
    Args:
        df (dataframe): 含有Y标签和预测分数的数据集
        target (string): Y标签列名
        y_pred (string): 坏概率分数列名
        group_col (string): 分组列名如月份，数据集

    Returns:
        dataframe: AUC和KS值的数据框
    """
    df_ks_auc = pd.DataFrame(index=['KS', 'AUC'])
    for col, group_df in df.groupby(group_col):  
        # 计算 AUC
        auc_ = roc_auc_score(group_df[target], group_df[y_pred])      
        fpr, tpr, _ = roc_curve(group_df[target], group_df[y_pred], pos_label=1)
        ks_ = max(tpr-fpr)
        df_ks_auc.loc['KS', col] = ks_
        df_ks_auc.loc['AUC', col] = auc_
    df_ks_auc = df_ks_auc.T
    df_ks_auc['bins'] = df_ks_auc.index
    
    return df_ks_auc


# In[82]:


# 评估模型效果
# df_ks_auc_month = model_ks_auc(df_sample, target, score, 'lending_month')
# df_ks_auc_month = pd.merge(df_target_summary_month, df_ks_auc_month, how='inner', on='bins')
# df_ks_auc_month
model_ks_auc(df_sample, target, 'bad_score', 'lending_month_new')


# In[50]:


df_ks_auc_set = model_ks_auc(df_sample, target, score, 'data_set')
df_ks_auc_set = pd.merge(df_target_summary_set, df_ks_auc_set, how='inner', on='bins')
df_ks_auc_set


# # 五、评分分布

# In[51]:


c = toad.transform.Combiner()
c.fit(df_sample.query("data_set=='1_train'")[[score, target]], y=target, method='quantile', n_bins=20) 
df_sample['score_bins'] = c.transform(df_sample[score], labels=True)


# In[52]:


# 小数转换百分数
def to_percentage(x):
    if isinstance(x, (float)) and pd.notnull(x):
        return f"{x * 100:.2f}%"
    return x

def float_format(x):
    if isinstance(x, (float)) and pd.notnull(x):
        return '%.3f' %x
    return x


# In[53]:


score_group_by_month = calculate_vars_distribute(df_sample, ['score_bins'], target, 'lending_month')
df_score_group_by_month = score_group_by_month.pivot_table(index='bins', columns='groupvars', values='total_pct')
df_score_group_by_month = df_score_group_by_month.applymap(to_percentage)
df_score_group_by_month


# In[54]:


score_group_by_dataset = calculate_vars_distribute(df_sample, ['score_bins'], target, 'data_set') 
score_group_by_dataset_1 = score_group_by_dataset.query("groupvars=='3_oot' & bins!='Total'").reset_index(drop=True)
score_group_by_dataset_1 = score_group_by_dataset_1[['groupvars', 'bins', 'total', 'bad', 'good', 'bad_rate', 'bad_rate_cum','total_pct_cum', 'lift', 'lift_cum']]
score_group_by_dataset_1[['bad_rate', 'bad_rate_cum','total_pct_cum']] = score_group_by_dataset_1[['bad_rate', 'bad_rate_cum','total_pct_cum']].applymap(to_percentage) 
score_group_by_dataset_1[['lift', 'lift_cum']] = score_group_by_dataset_1[['lift', 'lift_cum']].applymap(float_format) 
score_group_by_dataset_1


# # 6.变量分布

# In[58]:



# 调用函数绘制堆叠柱状图
def plot_stacked_bar(df, var, month_col, bins, values, filename=None):
    # 假设df是一个DataFrame，包含您的数据
    # month_col 是月份列
    # bins 是分箱列
    # values 是要绘制的值列
    # 创建一个透视表
    pivot_df = df.pivot_table(index=month_col, columns=bins, values=values, fill_value=0)

    # 初始化图形
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 获取所有的分箱类别
    bins_list = pivot_df.columns.tolist()
    
    # 计算每个柱子的宽度
    bar_width = 0.8
    
    # 计算x轴上的位置
    x_pos = range(len(pivot_df.index))
    
    # 初始化底部
    bottom = [0] * len(x_pos)
    
    # 遍历每个分箱，并绘制柱状图
    for bin in bins_list:
        # 使用fillna(0)处理NaN值
        pivot_df[bin] = pivot_df[bin].fillna(0)
        
        ax.bar(x_pos,
               pivot_df[bin],
               width=bar_width,
               label=bin,
               bottom=bottom,
               align='center',
               alpha=0.8)
        
        # 更新底部的值
        bottom = [b + v for b, v in zip(bottom, pivot_df[bin])]
    
    # 设置图形属性
    ax.set_title(f'{values}——{var}')
    # ax.set_xlabel('Month')
    ax.set_ylabel(f'{values}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pivot_df.index)
    ax.legend()

    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # 如果提供了文件名，则保存图表
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show() 
    
# 调用函数绘制时间序列图
def draw_time_series(df, var, month_col, bins, values, filename=None):

    pivot_df = df.pivot_table(index=month_col, columns=bins, values=values)
    
    # 绘制时间序列折线图
    plt.figure(figsize=(14, 7))
    for bin in pivot_df.columns:
        i = list(pivot_df.index)
        j = list(pivot_df[bin])
        plt.plot(i, j, label=bin, marker='o')

    plt.title(f'{values}——{var}')
    # plt.xlabel('Month')
    plt.ylabel(f'{values}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # 如果提供了文件名，则保存图表
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


def draw_line_bar(df, col, bin, var1, var2, var3, filename=None):
    # 处理数据
    bins = df[bin]
    varsname = df[col].unique()[0]
    values1 = list(df[var1].fillna(0))
    values2 = list(df[var2].fillna(0))
    values3 = df[var3].unique()[0]

    # 创建图形和主轴
    fig, ax1 = plt.subplots()

    # 主纵轴 - 柱状图 (分箱占比)
    color = 'tab:blue'
    ax1.set_xlabel(f'{varsname}')
    ax1.set_ylabel(f'{var1}', color=color)
    bars = ax1.bar(bins, values1, color=color)
    # 在柱状图上添加数值标签
    for bar in bars:
        yval = bar.get_height()
        # va='bottom' to place label below the bar
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{round(yval, 3)}', ha='center', va='top') 
    
    ax1.tick_params(axis='y', labelcolor=color, rotation=45)

    # 副纵轴 - 折线图 (坏占比)
    ax2 = ax1.twinx()  # 创建第二个纵坐标轴
    color = 'tab:red'
    ax2.set_ylabel(f'{var2}', color=color)  # 设置标签颜色
    _ = ax2.plot(values2, color=color, marker='o')  # 绘制折线图
    # 在折线图上添加数值标签
    for xtick, txt in zip(ax1.get_xticks(), values2):
        ax2.text(xtick, txt, f'{round(txt, 3)}', ha='center', va='top', rotation=45)
    ax2.tick_params(axis='y', labelcolor=color)

    # 获取x轴的刻度位置
    xtick_positions = range(len(bins))
    # 设置X轴标签自动调整
    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels(bins, rotation=45, ha='right')
    
    # 设置标题和网格
    ax1.set_title(f'{varsname}')
    ax1.grid(False)

    # 在左上角添加文本
    ax1.text(0.05, 0.95, f'IV value:{round(values3,3)}', transform=ax1.transAxes, verticalalignment='top')
    # 调整布局
    # fig.tight_layout()
    # 如果提供了文件名，则保存图表
    if filename:
        plt.savefig(filename, dpi=300)
    # 显示图表
    plt.show()

    
def plot_combined_chart(df,varsname,var_des,bins_col,totalpct_train,                        totalpct_oot,badrate_train, badrate_oot,filename="../SourceHanSansSC-Bold.otf"):
    import matplotlib
    # fname 为 你下载的字体库路径，注意 SourceHanSansSC-Bold.otf 字体的路径
    zhfont1 = matplotlib.font_manager.FontProperties(fname=filename) 
    fig, ax1 = plt.subplots(figsize=(14, 7))

    bar_width = 0.35
    index = np.arange(len(df))

    # 使用更深的对色盲友好的颜色
    color_train = '#004494'  # 深蓝色
    color_oot = '#D66100'    # 深橙色

    # 绘制柱状图
    bars1 = ax1.bar(index, df[totalpct_train], bar_width, label=f'Total Pct Train',
                    color=color_train, alpha=0.6)
    bars2 = ax1.bar(index + bar_width, df[totalpct_oot], bar_width, label=f'Total Pct OOT',
                    color=color_oot, alpha=0.6)

    # 柱状图数据标签，字体颜色设为黑色
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2%}', ha='center', va='bottom', fontsize=9, color='black')
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2%}', ha='center', va='bottom', fontsize=9, color='black')

    ax1.set_ylabel('Percentage')
    ax1.set_title(f'Distribution and Bad Rate of {varsname}  {var_des}',fontproperties=zhfont1)
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(df[bins_col], rotation=45, ha='right')

    ax2 = ax1.twinx()
    
    # 折线图，使用更深的颜色和标记
    data_train = df[badrate_train].to_numpy()
    line1, = ax2.plot(index + bar_width / 2, data_train, color=color_train, marker='o',
                      linestyle='-', label='Bad Rate Train')
    
    data_oot = df[badrate_oot].to_numpy()
    line2, = ax2.plot(index + bar_width / 2, data_oot, color=color_oot, marker='s',
                      linestyle='--', label='Bad Rate OOT')  # 使用方形标记
    ax2.set_ylabel('Bad Rate')

    # 折线图数据标签，字体颜色设为黑色
    for x, y in zip(index + bar_width / 2, df[badrate_train]):
        ax2.text(x, y, f'{y:.2%}', ha='center', va='bottom', fontsize=9, color='black')
    for x, y in zip(index + bar_width / 2, df[badrate_oot]):
        ax2.text(x, y, f'{y:.2%}', ha='center', va='bottom', fontsize=9, color='black')

    # 调整图例位置
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整图表布局，给顶部图例留出空间
    plt.show()



# In[59]:


importance_dict = df_importance_set.set_index('varsname')['comment'].to_dict()


# In[60]:



for col in importance_dict.keys():
    df_train_tmp = df_group_set.query("varsname==@col & bins!='Total' & groupvars=='1_train'")
    df_train_tmp = df_train_tmp[['varsname','bins','total_pct','bad_rate']]
    
    df_oot_tmp = df_group_set.query("varsname==@col & bins!='Total' & groupvars=='3_oot'")
    df_oot_tmp = df_oot_tmp[['varsname','bins','total_pct','bad_rate']]
    
    df_pct_bad = pd.merge(df_train_tmp,df_oot_tmp,how='inner',on=['varsname','bins'],suffixes=('_train','_oot'))
    df_pct_bad = df_pct_bad[['varsname','bins','total_pct_train','total_pct_oot','bad_rate_train','bad_rate_oot']]
    var_des = importance_dict[col]
    # 调用函数
    plot_combined_chart(df_pct_bad,col,var_des,'bins','total_pct_train','total_pct_oot','bad_rate_train','bad_rate_oot')


# In[ ]:


# 输出模型报告
jupyter nbconvert --to html --no-input ./模型开发/行为模型/模型开发报告html/全渠道实时提现行为模型开发报告.ipynb




#==============================================================================
# File: 01提现全渠道不限成本子分融合模型FPD30标签.py
#==============================================================================

#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import toad
import lightgbm as lgb
# import shap
import hyperopt 
from hyperopt import fmin, hp, Trials, tpe, rand, anneal, STATUS_OK, partial, space_eval
from hyperopt.early_stop import no_progress_loss
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
import joblib
import pickle
import time
from datetime import datetime
import os 
import gc
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_row',None)
pd.set_option('display.width',None)
pd.set_option('display.precision', 6)


# In[4]:


# 设置数据存储
task_name = '全渠道实时提现行为模型fpd30'
timestamp = datetime.now().strftime('%Y%m%d')
directory = f'./result/{task_name}'
if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
result_path = f'{directory}/'


# # 函数定义

# In[3]:


# 获取数据
def get_data(sql):
    from odps import ODPS
    import time
    from datetime import datetime
    # 输入账号密码
    conn= ODPS(username='liaoxilin', password='j02vYCxx')

    print('开始跑数' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = time.time()
    # 执行脚本
    instance = conn.execute_sql(sql)
    # 输出执行结果
    with instance.open_reader() as reader:
        print('===================')
        data = reader.to_pandas()

    end = time.time()
    print('结束跑数' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("运行事件：{}秒".format(end-start))   

    return data


# 插入数据
def execute_sql(sql):
    from odps import ODPS
    import time
    from datetime import datetime
    # 输入账号密码
    conn= ODPS(username='liaoxilin', password='j02vYCxx')
    
    print('开始跑数' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = time.time()
    # 执行脚本
    conn.execute_sql(sql)
    end = time.time()
    print('结束跑数' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("运行事件：{}秒".format(end-start))   


# # 0. 数据读取

# In[4]:


df_sample_dict = {}


# In[393]:



# 计算今天的时间
from datetime import datetime, timedelta, date

today = datetime.now().strftime('%Y-%m-%d')
print(today)

this_day =datetime.strptime('2024-11-04', '%Y-%m-%d')
end_day = datetime.strptime('2024-10-14', '%Y-%m-%d')

while this_day >= end_day:
    run_day = this_day.strftime('%Y-%m-%d')
    sql = f'''
select 
 t.order_no
,t.id_no_des
,t.channel_id
,t.lending_time
,substr(t.lending_time, 1, 7) as lending_month
,t.mob
,t.maxdpd
,t.fpd
,t.fpd10
,t.fpd30
,t.mob4dpd30
,t.diff_days
,t.order_no_auth
--行为分数
,t0.standard_score as behave_fpd30_score
,t0.bad_score as behave_fpd30_bad_prob
,t1.standard_score as dpd30_6m_score
,t1.bad_score as dpd30_6m_bad_prob
,t2.standard_score as dpd30_4m_score
,t2.bad_score as dpd30_4m_bad_prob
--催收分数
,t3.score as cpd10_v1_score
,t3.proba as cpd10_v1_prob
,t4.standard_score as cpd1_v1_score
,t4.prob as cpd1_v1_prob
--征信分数
,t5.score_fpd0_v1
,t5.score_fpd6_v1
,t5.score_fpd10_v1
,t5.score_fpd10_v2
,t5.score_fpd30_v1
,t5.score_dpd30_3c_v1
,t5.prob_fpd0_v1
,t5.prob_fpd6_v1
,t5.prob_fpd10_v1
,t5.prob_fpd10_v2
,t5.prob_fpd30_v1
,t5.prob_dpd30_3c_v1
--百融分数
,br_v3_fpd30_score
--三方分数
,aliyun_5    
,bileizhenv1    
,duxiaoman_6    
,hengpu_4    
,hengpu_5    
,rong360_4    
,tengxun_1    
,tianchuang_7 
,baihang_28  
,hengpu_7   
,pudao_68   
,pudao_91 
,pudao_20 
,pudao_34 
,wanxiangfen
,feicuifen  
,ruizhi_6   
,ali_fraud_score3   
,ali_fraud_score9  
,umeng_score_v5  
,jd_fraud_v2    
,jd_xyd    
,tengxun_credit_score 
,tengxun_cash_score 
from 
    (
    select * 
    from znzz_fintech_ads.dm_f_lxl_test_order_Y_target as t 
    where dt=date_sub(current_date(), 1) 
      and lending_time='{run_day}'
    ) as t 
--行为模型分数
left join 
(select * from znzz_fintech_ads.dm_f_lxl_test_behave_model_fpd30_score as t where dt=date_sub('{run_day}',1)) as t0 on t.id_no_des=t0.id_no_des
left join 
(select * from znzz_fintech_ads.clm_model_dpd30_6m_score_v1 as t where dt=date_sub('{run_day}',1)) as t1 on t.id_no_des=t1.id_no_des
left join 
(select * from znzz_fintech_ads.clm_model_dpd30_4m_score_v1 as t where dt=date_sub('{run_day}',1)) as t2 on t.id_no_des=t2.id_no_des
--催收模型分数
left join
    (
    select t.*, row_number() over(partition by id_no_des order by dt desc) as rk 
    from znzz_fintech_ads.fkmodel_cscore_cpd10_v1_score as t 
    where dt<=date_sub('{run_day}',1)
      and dt>=date_sub('{run_day}',100)
    ) as t3 on t.id_no_des=t3.id_no_des and t3.rk=1
left join
    (
    select t.*, row_number() over(partition by id_no_des order by dt desc) as rk 
    from znzz_fintech_ads.ly_cpd1_model_v1_score as t 
    where dt<=date_sub('{run_day}',1)
      and dt>=date_sub('{run_day}',100)
    ) as t4 on t.id_no_des=t4.id_no_des and t4.rk=1
--征信模型分数
left join
    (
    select t.*, row_number() over(partition by id_no_des order by dt desc) as rk 
    from znzz_fintech_dwd.llji_yhx_ascore_model_all_score as t 
    where dt<=date_sub('{run_day}',0 )
      and dt>=date_sub('{run_day}',99)
    ) as t5 on t.id_no_des=t5.id_no_des and t5.rk=1
--百融模型分数
left join 
    (
    select t.*, row_number() over(partition by id_no_des order by dt desc) as rk 
    from znzz_fintech_ads.bairong_loan_data_info_id_fd_modelv3_score as t 
    where dt<='{run_day}'
      and dt>=date_sub('{run_day}',29)
    ) as t7 on t.id_no_des=t7.id_no_des and t7.rk=1
--三方评分分数
left join 
(select * from znzz_fintech_ads.dm_f_lxl_test_three_score_data_realtime as t where dt='{run_day}') as t8 on t.id_no_des=t8.id_no_des
;
'''
    print(f'=========================={run_day}=============================')
    df_sample_dict[run_day] = get_data(sql)
    this_day = this_day - timedelta(days=1)


# In[391]:


tmp_dict = {}


# In[392]:


tmp_dict = {}
# 计算今天的时间
from datetime import datetime, timedelta, date

today = datetime.now().strftime('%Y-%m-%d')
print(today)

this_day =datetime.strptime('2024-11-06', '%Y-%m-%d')
end_day = datetime.strptime('2024-07-21', '%Y-%m-%d')

while this_day >= end_day:
    run_day = this_day.strftime('%Y-%m-%d')
    sql = f'''
select 
 t.order_no
,t.id_no_des
,t.channel_id
,t.lending_time
,substr(t.lending_time, 1, 7) as lending_month
,t.mob
,t.maxdpd
,t.fpd
,t.fpd10
,t.fpd30
,t.mob4dpd30
,t.diff_days

--征信分数
,t5.score_fpd0_v1
,t5.score_fpd6_v1
,t5.score_fpd10_v1
,t5.score_fpd10_v2
,t5.score_fpd30_v1
,t5.score_dpd30_3c_v1
,t5.prob_fpd0_v1
,t5.prob_fpd6_v1
,t5.prob_fpd10_v1
,t5.prob_fpd10_v2
,t5.prob_fpd30_v1
,t5.prob_dpd30_3c_v1

from 
    (
    select * 
    from znzz_fintech_ads.dm_f_lxl_test_order_Y_target as t 
    where dt=date_sub(current_date(), 1) 
      and lending_time='{run_day}'
    ) as t 

--征信模型分数
left join
    (
    select t.*, row_number() over(partition by id_no_des order by dt desc) as rk 
    from znzz_fintech_dwd.llji_yhx_ascore_model_all_score as t 
    where dt<=date_sub('{run_day}',0)
      and dt>=date_sub('{run_day}',99)
    ) as t5 on t.id_no_des=t5.id_no_des and t5.rk=1

;
'''
    print(f'=========================={run_day}=============================')
    tmp_dict[run_day] = get_data(sql)
    this_day = this_day - timedelta(days=1)


# In[ ]:


# 计算今天的时间
from datetime import datetime, timedelta, date

today = datetime.now().strftime('%Y-%m-%d')
print(today)

this_day =datetime.strptime('2024-12-06', '%Y-%m-%d')
end_day = datetime.strptime('2024-07-21', '%Y-%m-%d')

while this_day >= end_day:
    run_day = this_day.strftime('%Y-%m-%d')
    sql = f'''
select 
 t.order_no
,t.id_no_des
,t.channel_id
,t.lending_time
,substr(t.lending_time, 1, 7) as lending_month
,t.mob
,t.maxdpd
,t.fpd
,t.fpd10
,t.fpd30
,t.mob4dpd30
,t.diff_days

--征信分数
,t5.score_fpd0_v1
,t5.score_fpd6_v1
,t5.score_fpd10_v1
,t5.score_fpd10_v2
,t5.score_fpd30_v1
,t5.score_dpd30_3c_v1
,t5.prob_fpd0_v1
,t5.prob_fpd6_v1
,t5.prob_fpd10_v1
,t5.prob_fpd10_v2
,t5.prob_fpd30_v1
,t5.prob_dpd30_3c_v1

from 
    (
    select * 
    from znzz_fintech_ads.dm_f_lxl_test_order_Y_target as t 
    where dt=date_sub(current_date(), 1) 
      and lending_time='{run_day}'
    ) as t 
inner join
    (
    select t.*, row_number() over(partition by id_no_des order by dt desc) as rk 
    from znzz_fintech_ads.dm_f_lxl_test_order_model_merge_fpd30_score as t 
    where dt=date_sub('{run_day}',0)
    ) as t5 on t.id_no_des=t5.id_no_des and t5.rk=1
select * 
from znzz_fintech_ads.dm_f_lxl_test_order_model_merge_fpd30_score
where dt>='2025-02-04'
;
'''
    print(f'=========================={run_day}=============================')
    tmp_dict[run_day] = get_data(sql)
    this_day = this_day - timedelta(days=1)


# In[396]:


data_time = pd.DataFrame({'run_day':list(df_sample_dict.keys())})
data_time['months'] = data_time['run_day'].str[0:7]
data_time.groupby(['months'])['run_day'].count()


# In[397]:


df_sample_ = pd.concat(df_sample_dict.values(), keys=df_sample_dict.keys())
df_sample_ = df_sample_.reset_index(drop=True)
df_sample_.info(show_counts=True)
df_sample_.head()


# In[398]:


print(df_sample_.shape, df_sample_['order_no'].nunique(), 
      df_sample_['id_no_des'].nunique(), df_sample_['order_no_auth'].nunique())


# In[399]:


df_sample_.drop_duplicates(inplace=True)
df_sample_ = df_sample_.reset_index(drop=True)
print(df_sample_.shape)


# In[400]:


df_sample_.dropna(how='all', axis=1, inplace=True)
print(df_sample_.shape)


# In[401]:


df_tmp_dict_ = pd.concat(tmp_dict.values(), keys=tmp_dict.keys())
df_tmp_dict_ = df_tmp_dict_.reset_index(drop=True)
df_tmp_dict_.info(show_counts=True)
df_tmp_dict_.head()


# In[404]:


print(df_tmp_dict_.shape, df_tmp_dict_['order_no'].nunique())


# In[403]:


df_tmp_dict_.drop_duplicates(inplace=True)
df_tmp_dict_ = df_tmp_dict_.reset_index(drop=True)


# In[405]:


df_tmp_dict_.dropna(how='all', axis=1, inplace=True)
print(df_tmp_dict_.shape)


# In[423]:


cols1 = ['order_no']+[col for col in df_sample_.columns if col not in df_tmp_dict_.columns]
print(cols1)


# In[424]:


df_sample_=pd.merge(df_tmp_dict_, df_sample_[cols1], how='inner', on=['order_no'])
df_sample_.info(show_counts=True)
df_sample_.head()


# In[425]:


print(df_sample_.shape, df_sample_['order_no'].nunique())


# In[426]:


df_sample_.drop(columns=['order_no_auth'],inplace=True)


# In[427]:


df_sample_.columns[:12]


# In[428]:


varsname = [col for col in df_sample_.columns.to_list()[12:]]

print(varsname[:10], varsname[-10:])
print("初始特征变量个数：",len(varsname))


# In[429]:


print(result_path)


# In[430]:


for i, col in enumerate(varsname):
    if df_sample_[col].dtype=='object':
        print(f"======第{i}个变量：{col}========")
        df_sample_[col] = pd.to_numeric(df_sample_[col])


# In[431]:


df_sample_.to_csv(result_path+'全渠道实时提现行为模型_原始建模数据集_241218.csv',index=False)


# In[432]:


df_sample = df_sample_.query("channel_id!='1'").reset_index(drop=True)
df_sample.info(show_counts=True)
df_sample.head()


# In[433]:


print(df_sample['lending_time'].min(), df_sample['lending_time'].max())


# In[434]:


df_sample.select_dtypes('object').columns


# In[ ]:


# df_sample['fpd30'] = pd.to_numeric(df_sample['fpd30'])
# df_sample['diff_days'] = pd.to_numeric(df_sample['diff_days'])


# In[435]:


pd.set_option('display.max_row',None)
df_sample.groupby(['lending_time','fpd30'])['order_no'].count().unstack()


# In[436]:


df_sample = df_sample.drop(index=df_sample.query("lending_time>='2024-10-16'").index)
df_sample = df_sample.reset_index(drop=True)
print(df_sample.shape)


# In[437]:


df_sample.loc[df_sample.query("lending_time>='2024-07-21' & lending_time<='2024-09-20'").index, 'data_set']='1_train'
df_sample.loc[df_sample.query("lending_time>='2024-09-21' & lending_time<='2024-10-15'").index, 'data_set']='3_oot'


# In[438]:


target = 'fpd30'


# In[439]:


df_sample[[target]+varsname].info(show_counts=True)
df_sample[[target]+varsname].head()


# In[440]:


df_sample.to_csv(result_path + '全渠道实时提现行为模型_建模数据集_241218.csv',index=False)


# # 1. 样本概况

# In[40]:


def get_target_summary(df, target, groupby_col):
    """
    对 DataFrame 进行分组聚合，并添加一个汇总行。
    
    参数:
    - df: 待处理的 DataFrame
    - groupby_col: 用于分组的列名
    - agg_cols: 字典，键是列名，值是聚合函数名称（如 'count', 'sum', 'mean'）
    - new_col_name: 字典,键是旧列的名称，值是新列的名称
    
    返回:
    - 包含分组聚合结果和汇总行的新 DataFrame
    """
    # 使用 groupby 和 agg 进行分组和聚合
    grouped = df.groupby(groupby_col)[target].agg(total=lambda x: len(x), 
            bad=lambda x: x.sum(), 
            good=lambda x: (x== 0).sum(), 
            bad_rate=lambda x: x.mean()).reset_index()
    
    # 计算整个 DataFrame 的聚合统计量
    total_summary = df[target].agg(total=lambda x: len(x), 
            bad=lambda x: x.sum(), 
            good=lambda x: (x== 0).sum(), 
            bad_rate=lambda x: x.mean()).to_frame().T
    total_summary[groupby_col] = 'Total'
    
    # 将汇总行添加到分组结果中
    result = pd.concat([grouped, total_summary], ignore_index=True)
    result.rename(columns={groupby_col: 'bins'}, inplace=True)
    
    # 返回结果
    return result


# In[442]:


print(df_sample[target].value_counts())


# In[443]:


df_target_summary_month = get_target_summary(df_sample, target, 'lending_month')
print(df_target_summary_month)


# In[31]:


df_target_summary_month = get_target_summary(df_sample, target, 'lending_month')
print(df_target_summary_month)


# In[444]:


df_target_summary_set = get_target_summary(df_sample, target, 'data_set')
print(df_target_summary_set)


# In[445]:


df_target_summary = pd.concat([df_target_summary_month, df_target_summary_set], axis=0, ignore_index=True)
df_target_summary


# In[446]:


task_name


# In[447]:



timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

with pd.ExcelWriter(result_path + f"1_样本概况_{task_name}_{timestamp}.xlsx") as writer:
    df_target_summary_month.to_excel(writer, sheet_name='df_target_summary_month')
    df_target_summary_set.to_excel(writer, sheet_name='df_target_summary_set')
    df_target_summary.to_excel(writer, sheet_name='df_target_summary')
    
print(f"数据存储完成: {timestamp}")
print(result_path + f"1_样本概况_{task_name}_{timestamp}.xlsx")


# # 2.数据探索性分析

# In[448]:


# 2.1 变量分布
df_explor = toad.detect(df_sample[varsname])


# In[449]:


# 2.2 添加最高占比
for i, col in enumerate(varsname):
    if i>=100 and i%500==0:
        print(i)
    df_explor.loc[col, 'mod_null'] = df_sample[col].value_counts(normalize=True, ascending=False, dropna=False).max()
    df_explor.loc[col, 'mod_notna'] = df_sample[col].value_counts(normalize=True, ascending=False).max()


# In[450]:



def calculate_missing_rate_by_month(df, columns, groupby_col):
    """
    计算每个月每列的缺失率。
    
    参数:
    - df: 待处理的 DataFrame
    - groupby_col: 分组的列名
    - columns: 需要计算缺失率的列名列表
    
    返回:
    - 包含每个月每列缺失率的新 DataFrame
    """
    # 提取月份信息
    # df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    # df['month'] = df[timestamp_col].dt.to_period('M')

    # 分组并计算缺失率
    missing_rates = df.groupby(groupby_col)[columns].apply(lambda x: x.isnull().sum()/len(x)).T
#     missing_rates['mean'] = missing_rates.mean(axis=1)
#     missing_rates['std'] = missing_rates.std(axis=1)
#     missing_rates['cv'] = missing_rates['std'] / missing_rates['mean']

    return missing_rates


# ## 2.1缺失值处理

# In[468]:


df_sample = df_sample.replace(-1, np.nan)
gc.collect()


# In[469]:


# 2.2 缺失率按月分布
columns = varsname
groupby_col = 'lending_month'
df_miss_month = calculate_missing_rate_by_month(df_sample, columns, groupby_col)
df_miss_month.index.name = 'variable'
print(df_miss_month.head())


# In[470]:


# 2.2 缺失率按数据集分布
columns = varsname
groupby_col = 'data_set'
df_miss_set = calculate_missing_rate_by_month(df_sample, columns, groupby_col)
df_miss_set.index.name = 'variable'
print(df_miss_set.head())


# In[471]:


# 2.3 快速查看特征重要性
df_iv = toad.quality(df_sample[varsname+[target]], target, iv_only=True, 
                     method='dt', min_samples=0.05, n_bins=6)
df_iv.index.name = 'variable'
print(df_iv.head())


# In[472]:


timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

with pd.ExcelWriter(result_path + f'2_数据探索性分析_{task_name}_{timestamp}.xlsx') as writer:
        df_explor.to_excel(writer, sheet_name='df_explor')
        df_miss_month.to_excel(writer, sheet_name='df_miss_month')
        df_miss_set.to_excel(writer, sheet_name='df_miss_set')
        df_iv.to_excel(writer, sheet_name='iv') 
print(f"数据存储完成时间：{timestamp}")
print(result_path + f'2_数据探索性分析_{task_name}_{timestamp}.xlsx')


# # 3.特征粗筛选

# ## 3.1 基于自身属性删除变量

# In[473]:


# 删除近期不可使用的特征(最近月份的缺失率大于等于0.95)
# to_drop_recent = list(df_miss_month[(df_miss_month>=0.90).any(axis=1)].index)
to_drop_recent = []
print("to_drop_recent:", len(to_drop_recent))

# 删除缺失率大于0.95/删除枚举值只有一个/删除方差等于0/删除集中度大于0.95
to_drop_missing = list(df_explor[df_explor.missing.str[:-1].astype(float)/100>=0.90].index)
print("to_drop_missing:", len(to_drop_missing))

to_drop_unique = list(df_explor[df_explor.unique==1].index)
print("to_drop_unique:", len(to_drop_unique))

to_drop_std = list(df_explor[df_explor.std_or_top2==0].index)
print("to_drop_std:", len(to_drop_std))

to_drop_mode = list(df_explor[df_explor.mod_notna>=0.90].index)
print("to_drop_mode:", len(to_drop_mode))

to_drop_iv = list(df_iv[df_iv.iv<0.01].index)
print("to_drop_iv:", len(to_drop_iv))

to_drop1 = list(set(to_drop_recent + to_drop_missing +  to_drop_unique +  to_drop_std + to_drop_mode + to_drop_iv))
print(f"删除的变量有{len(to_drop1)}个")


# In[474]:


df_iv.loc[to_drop_iv,:]


# In[475]:


varsname_v1 = [col for col in varsname if col not in to_drop1]

print(f"保留的变量有{len(varsname_v1)}个")
print(varsname_v1)


# ## 3.2 基于相关性删除变量
# 

# In[476]:


train_selected, dropped = toad.selection.select(df_sample[varsname_v1+[target]], target=target, 
                                                empty=0.90, iv=0.01, corr=0.70, 
                                                return_drop=True, exclude=None)
train_selected.shape


# In[478]:


to_drop2 = []
for k, v in dropped.items():
    print(k, ":", len(v))
    to_drop2.extend(list(v))
print(len(set(to_drop2)))


# In[479]:


df_iv.loc[to_drop2,:]


# In[480]:


to_drop2 = []
varsname_v2 = [col for col in varsname_v1 if col not in to_drop2]

print(f"保留的变量有{len(varsname_v2)}个")


# # 4.特征细筛选

# ## 4.1 基于变量稳定性筛选

# In[481]:



def cal_psi_by_month(df_actual, df_expect, cols, month_col, combiner, return_frame = True):
    """
    计算每个月每的psi。
    
    参数:
    - df_actual: 测试集
    - df_expect: 训练集
    - cols: 需要计算稳定性的列名列表
    - month_col: 分组的列名

    返回:
    - 包含每个月的新 DataFrame
    """
    bins_df_list = []
    psi_list = []
    for month_, df_actual_group in df_actual.groupby(month_col):
        if return_frame:
            psi_, bins_df = toad.metrics.PSI(df_actual_group[cols], df_expect[cols], 
                                            combiner = combiner, return_frame = return_frame)
            psi_ = pd.DataFrame({month_: psi_}, index=cols)
            psi_list.append(psi_)
            bins_df['month'] = month_
            bins_df_list.append(bins_df)
        else:
            psi_ = toad.metrics.PSI(df_actual_group[cols], df_expect[cols], 
                                            combiner = combiner, return_frame = return_frame)
            psi_ = pd.DataFrame({month_: psi_}, index=cols)
            psi_list.append(psi_)
        
    # 合并所有结果 DataFrame
    if return_frame:
        psi_df = pd.concat(psi_list, axis=1)
        bins_df = pd.concat(bins_df_list, axis=0)
        
        return (psi_df, bins_df)
    else:
        psi_df = pd.concat(psi_list, axis=1)
        
        return psi_df


def cal_iv_by_month(df, cols, target, month_col, combiner):
    """
    计算每个变量每个月的iv。
    
    参数:
    - df: 待处理的 DataFrame
    - target: Y标签
    - cols: 需要计算iv的列名列表
    - month_col：月份列名
    
    返回:
    - 包含每个月每列iv的新 DataFrame
    """
    df_ = combiner.transform(df[cols+[target, month_col]], labels=True)
    result = pd.DataFrame(columns=sorted(list(df_[month_col].unique())), index=cols)
    for col in cols:
        for month in sorted(list(df_[month_col].unique())):
            data = df_[df_[month_col] == month]
            regroup = data.groupby(col)[target].agg(total=lambda x: x.count(), bad=lambda x: x.sum())
            regroup['good'] = regroup['total'] - regroup['bad']
            regroup['bad_pct'] = regroup['bad']/regroup['bad'].sum()
            regroup['good_pct'] = regroup['good']/regroup['good'].sum()
            regroup['woe'] = np.log(regroup['bad_pct']/regroup['good_pct'])
            regroup['iv'] = (regroup['bad_pct']-regroup['good_pct'])*regroup['woe']
            result.loc[col, month] = regroup['iv'].sum()    
      
    return result


def calculate_vars_distribute(df, cols, target, group_col):    
    """
    参数:
    - df: 待处理的 DataFrame
    - target: Y标签
    - cols: 需要分箱的列名列表
    - group_col：分组列名，如月份、渠道、数据类型
    
    返回:
    - 包含每个月每列iv的新 DataFrame
    """
    result = pd.DataFrame()
    vars = sorted(list(df[group_col].unique()))
    for col in cols:
        for var in vars:
            data = df[df[group_col] == var]
            regroup = data.groupby(col)[target].agg(total=lambda x: x.count(), bad=lambda x: x.sum())
            regroup['good'] = regroup['total'] - regroup['bad']
            regroup['bad_rate'] = regroup['bad']/regroup['total']
            regroup['bad_rate_cum'] = regroup['bad'].cumsum()/regroup['total'].cumsum()
            regroup['total_pct'] = regroup['total']/regroup['total'].sum()
            regroup['bad_pct'] = regroup['bad']/regroup['bad'].sum()
            regroup['good_pct'] = regroup['good']/regroup['good'].sum()
            regroup['bad_pct_cum'] = regroup['bad_pct'].cumsum()
            regroup['good_pct_cum'] = regroup['good_pct'].cumsum()
            regroup['total_pct_cum'] = regroup['total_pct'].cumsum()
            regroup['ks_bin'] = regroup['bad_pct_cum'] - regroup['good_pct_cum']
            regroup['ks'] = regroup['ks_bin'].max()
            regroup['lift_cum'] = regroup['bad_rate_cum']/data[target].mean()
            regroup['lift'] = regroup['bad_rate']/data[target].mean()
            regroup['woe'] = np.log(regroup['bad_pct']/regroup['good_pct'])
            regroup['iv_bins'] = (regroup['bad_pct']-regroup['good_pct'])*regroup['woe']
            regroup['iv'] = regroup['iv_bins'].sum()
            regroup['bins'] = regroup.index
                
            total_summary = data[target].agg(total=lambda x: x.count(), bad=lambda x: x.sum()).to_frame().T
            total_summary['good'] = regroup['total'] - regroup['bad']
            total_summary['bad_rate'] = total_summary['bad']/total_summary['total']
            total_summary['iv'] = regroup['iv_bins'].sum()
            total_summary['ks'] = regroup['ks_bin'].max()
            total_summary['bins'] = 'Total'
            
            regroup = pd.concat([regroup, total_summary], axis=0, ignore_index=True)
            regroup['varsname'] = col
            regroup['groupvars'] = var
            
            usecols = ['groupvars', 'varsname', 'bins', 'total', 'bad', 'good', 'bad_rate', 'bad_rate_cum', 'woe', 'iv', 'iv_bins', 
                       'ks', 'ks_bin', 'lift', 'lift_cum', 'total_pct', 'total_pct_cum', 'bad_pct', 'bad_pct_cum', 'good_pct','good_pct_cum']
            regroup = regroup[usecols]
            result = pd.concat([result, regroup], axis=0, ignore_index=True)

    return result


# In[482]:



# 删除当前索引值所在行的后一行(按从小到大排序,合并都是保留较小值)，配合左闭右开
def DelIndexPlus1(np_regroup, index_value):
np_regroup[index_value,1] = np_regroup[index_value,1] + np_regroup[index_value+1,1]#坏客户
np_regroup[index_value,2] = np_regroup[index_value,2] + np_regroup[index_value+1,2]#好客户
np_regroup = np.delete(np_regroup, index_value+1, axis=0)

return np_regroup
 
# 删除当前索引值所在行(按从小到大排序,合并都是保留较小值)，配合左闭右开
def DelIndex(np_regroup, index_value):
np_regroup[index_value-1,1] = np_regroup[index_value,1] + np_regroup[index_value-1,1]#坏客户
np_regroup[index_value-1,2] = np_regroup[index_value,2] + np_regroup[index_value-1,2]#好客户
np_regroup = np.delete(np_regroup, index_value, axis=0)

return np_regroup 

# 删除/合并客户数为0的箱子
def MergeZero(np_regroup):
#合并好坏客户数连续都为0的箱子
i = 0
while i<=np_regroup.shape[0]-2:
    if (np_regroup[i,1]==0 and np_regroup[i+1,1]==0) or (np_regroup[i,2]==0 and np_regroup[i+1,2]==0):
        np_regroup = DelIndexPlus1(np_regroup,i)
    i = i+1
    
#合并坏客户数为0的箱子
while True:
    if all(np_regroup[:,1]>0) or np_regroup.shape[0]==2:
        break
    bad_zero_index = np.argwhere(np_regroup[:,1]==0)[0][0]
    if bad_zero_index==0:
        np_regroup = DelIndexPlus1(np_regroup, bad_zero_index)
    elif bad_zero_index==np_regroup.shape[0]-1:
        np_regroup = DelIndex(np_regroup, bad_zero_index)
    else:
        bad_1 = np_regroup[bad_zero_index-1,1]
        good_1 = np_regroup[bad_zero_index-1,2]
        badplus1 = np_regroup[bad_zero_index+1,1]
        goodplus1 = np_regroup[bad_zero_index+1,2]           
        if (bad_1/(bad_1+good_1)) <= (badplus1/(badplus1 + goodplus1)):
            np_regroup = DelIndex(np_regroup, bad_zero_index)
        else:
            np_regroup = DelIndexPlus1(np_regroup, bad_zero_index)
#合并好客户数为0的箱子
while True:
    if all(np_regroup[:,2]>0) or np_regroup.shape[0]==2:
        break
    good_zero_index = np.argwhere(np_regroup[:,2]==0)[0][0]
    if good_zero_index==0:
        np_regroup = DelIndexPlus1(np_regroup, good_zero_index)
    elif good_zero_index==np_regroup.shape[0]-1:
        np_regroup = DelIndex(np_regroup, good_zero_index)
    else:
        bad_1 = np_regroup[good_zero_index-1,1]
        good_1 = np_regroup[good_zero_index-1,2]
        badplus1 = np_regroup[good_zero_index+1,1]
        goodplus1 = np_regroup[good_zero_index+1,2]                
        if (bad_1/(bad_1 + good_1)) <= (badplus1/(badplus1 + goodplus1)):
            np_regroup = DelIndexPlus1(np_regroup, good_zero_index)
        else:
            np_regroup = DelIndex(np_regroup, good_zero_index)
            
return np_regroup

#箱子最小占比
def MinPct(np_regroup, threshold=0.05):
while True:
    bins_pct = [(np_regroup[i,1]+np_regroup[i,2])/np_regroup.sum() for i in range(np_regroup.shape[0])]
    min_pct = min(bins_pct)
    if np_regroup.shape[0]==2:
        print(f"箱子最小占比：箱子数达到最小值2个,最小箱子占比{min_pct}")
        break
    if min_pct>=threshold:
        print(f"箱子最小占比：各箱子的样本占比最小值: {threshold}，已满足要求")
        break
    else:
        min_pct_index = bins_pct.index(min(bins_pct))
        if min_pct_index==0:
            np_regroup = DelIndexPlus1(np_regroup, min_pct_index)
        elif min_pct_index == np_regroup.shape[0]-1:
            np_regroup = DelIndex(np_regroup, min_pct_index)
        else:
            BadRate = [np_regroup[i,1]/(np_regroup[i,1]+np_regroup[i,2]) for i in range(np_regroup.shape[0])]
            BadRateDiffMin = [abs(BadRate[i]-BadRate[i+1]) for i in range(np_regroup.shape[0]-1)]
            if BadRateDiffMin[min_pct_index-1]>=BadRateDiffMin[min_pct_index]:
                np_regroup = DelIndexPlus1(np_regroup, min_pct_index)
            else:
                np_regroup = DelIndex(np_regroup, min_pct_index)
return np_regroup


# 箱子的单调性
def MonTone(np_regroup):
while True:
    if np_regroup.shape[0]==2:
        print("箱子单调性：箱子数达到最小值2个")
        break
    BadRate = [np_regroup[i,1]/(np_regroup[i,1]+np_regroup[i,2]) for i in range(np_regroup.shape[0])]
    BadRateMonetone = [BadRate[i]<BadRate[i+1] for i in range(np_regroup.shape[0]-1)]
    #确定是否单调
    if_Montone = len(set(BadRateMonetone))
    #判断跳出循环
    if if_Montone==1:
        print("箱子单调性：各箱子的坏样本率单调")
        break
    else:
        BadRateDiffMin = [abs(BadRate[i]-BadRate[i+1]) for i in range(np_regroup.shape[0]-1)]
        Montone_index = BadRateDiffMin.index(min(BadRateDiffMin))
        np_regroup = DelIndexPlus1(np_regroup, Montone_index)
        
return np_regroup


# 变量分箱，返回分割点，特殊值不参与分箱
def Vars_Bins(data, target, col, cutbins=[]):
df = data[data[target]>=0][[target, col]]
df = df[df[col].notnull()].reset_index(drop=True)
#区间左闭右开
df['bins'] = pd.cut(df[col], cutbins, duplicates='drop', right=False, precision=4, labels=False)
regroup = pd.DataFrame()
regroup['bins'] = df.groupby(['bins'])[col].min()
regroup['total'] = df.groupby(['bins'])[target].count()
regroup['bad'] = df.groupby(['bins'])[target].sum()
regroup['good'] = regroup['total'] - regroup['bad']
regroup.drop(['total'], axis=1, inplace=True)
np_regroup = np.array(regroup)
np_regroup = MergeZero(np_regroup)
np_regroup = MinPct(np_regroup)
np_regroup = MonTone(np_regroup)
cutoffpoints = list(np_regroup[:,0])
# 判断重新分箱后最高集中度占比
mode = [(np_regroup[i,1] + np_regroup[i,2])/np_regroup.sum()>0.95 for i in range(np_regroup.shape[0])]
is_drop_mode = any(mode)
# 判断第一个分割点是否最小值
if df[col].min()==cutoffpoints[0]:
    print(f"变量{col}：最小值所在箱子没有被合并过")
    cutoffpoints=cutoffpoints[1:]

return (cutoffpoints, is_drop_mode)


# In[487]:


# 计算分布前先变量分箱
combiner = toad.transform.Combiner()
combiner.fit(df_sample[varsname_v2+[target]], y=target, 
             method='dt', n_bins=10, min_samples = 0.05, empty_separate=True) 


# In[488]:


existing_bins_dict = combiner.export()
existing_bins_dict


# In[489]:


new_bins_dict = {}
to_drop_mode = []
for i, col in enumerate(varsname_v2):
    print(f"======第{i+1}个变量：{col}=========")
    empty = [x for x in existing_bins_dict[col] if pd.isnull(x)]
    not_empty = [x for x in existing_bins_dict[col] if pd.notnull(x)]
    
    cutbins = [float('-inf')] + not_empty + [float('inf')]
    # 确保分箱无0值，单调，最小占比符合要求
    cutbins,  is_drop_mode = Vars_Bins(df_sample, target, col, cutbins=cutbins)
    # 新的分箱分割点，符合toad包要求
    new_bins_dict[col] = cutbins + empty
    # 删除重新分箱后，高度集中的变量
    if is_drop_mode:
        print(f"{col}重新分箱后，集中度占比超95%")
        to_drop_mode.append(col)


# In[490]:


new_bins_dict


# In[491]:


combiner.load(new_bins_dict)


# In[492]:


timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
with open(result_path + '变量分箱字典_{timestamp}.pkl', 'wb') as f:
    pickle.dump(new_bins_dict, f)
print(result_path + f'变量分箱字典_{timestamp}.pkl')


# In[493]:


to_drop_mode


# In[56]:


# varsname_v2 = [x for x in varsname_v2 if x not in to_drop_mode]


# In[494]:


# 计算psi
df_psi_by_month = cal_psi_by_month(df_sample, df_sample.query("lending_month=='2024-07'"), varsname_v2,                                    'lending_month', combiner, return_frame = False)
print(df_psi_by_month.head(10))

df_psi_by_set = cal_psi_by_month(df_sample, df_sample.query("data_set=='1_train'"), varsname_v2,                                  'data_set', combiner, return_frame = False)
print(df_psi_by_set.head(10))


# In[495]:


# 计算iv
df_iv_by_month = cal_iv_by_month(df_sample, varsname_v2, target, 'lending_month', combiner)
print(df_iv_by_month.head(10))

df_iv_by_set = cal_iv_by_month(df_sample, varsname_v2, target, 'data_set', combiner)
print(df_iv_by_set.head(10)) 


# In[496]:


df_bins = combiner.transform(df_sample, labels=True)
selected_cols = ['groupvars', 'varsname', 'bins', 'total', 
                 'bad', 'good', 'total_pct', 'bad_pct', 
                 'good_pct', 'bad_rate', 'iv']


# In[497]:


df_group_month = calculate_vars_distribute(df_bins, varsname_v2, target, 'lending_month')[selected_cols] 
print(df_group_month.head())

df_group_set = calculate_vars_distribute(df_bins, varsname_v2, target, 'data_set')[selected_cols]   
print(df_group_set.head())


# In[498]:


# 计算total_pct 和 # bad_rate 以及iv的时间分布
df_total_bad = pd.DataFrame()
pivot_df_iv = pd.DataFrame()
    
for i in varsname_v2:      
    df_tmp = df_group_month.query("varsname==@i & bins!='Total'")
    pivot_df_totalpct = df_tmp.pivot_table(index=['varsname','bins'], columns='groupvars', 
                                           values='total_pct', fill_value=0).reset_index()
    pivot_df_badrate = df_tmp.pivot_table(index=['varsname','bins'], columns='groupvars', 
                                          values='bad_rate', fill_value=0).reset_index()
    pivot_df = pd.merge(pivot_df_totalpct, pivot_df_badrate, how='inner', 
                        on=['varsname','bins'], suffixes=('_total', '_bad'))
    df_total_bad = pd.concat([df_total_bad, pivot_df], axis=0) 

    df_tmp = df_group_month.query("varsname==@i & bins=='Total'")
    df_tmp_iv = df_tmp.pivot_table(index='varsname', columns='groupvars', values='iv').reset_index()
    pivot_df_iv = pd.concat([pivot_df_iv, df_tmp_iv], axis=0)
    
print(df_total_bad.head())
print(pivot_df_iv.head())


# In[499]:


# 计算total_pct 和 # bad_rate 以及iv的时间分布
df_total_bad_set = pd.DataFrame()
pivot_df_iv_set = pd.DataFrame()
    
for i in varsname_v2:      
    df_tmp = df_group_set.query("varsname==@i & bins!='Total'")
    pivot_df_totalpct = df_tmp.pivot_table(index=['varsname','bins'], columns='groupvars', 
                                           values='total_pct', fill_value=0).reset_index()
    pivot_df_badrate = df_tmp.pivot_table(index=['varsname','bins'], columns='groupvars', 
                                          values='bad_rate', fill_value=0).reset_index()
    pivot_df = pd.merge(pivot_df_totalpct, pivot_df_badrate, how='inner', 
                        on=['varsname','bins'], suffixes=('_total', '_bad'))
    df_total_bad_set = pd.concat([df_total_bad_set, pivot_df], axis=0)

    df_tmp = df_group_set.query("varsname==@i & bins=='Total'")
    df_tmp_iv = df_tmp.pivot_table(index='varsname', columns='groupvars', values='iv').reset_index()
    pivot_df_iv_set = pd.concat([pivot_df_iv_set, df_tmp_iv], axis=0)
    
    pivot_df_badrate_iv = pd.merge(pivot_df_badrate, pivot_df_iv_set, how='inner', 
                        on=['varsname'], suffixes=('_bad', '_iv'))
        
print(df_total_bad_set.head() )
print(pivot_df_iv_set.head() )


# ### 删除不稳定特征

# In[508]:


# drop_by_psi_month = list(df_psi_by_month[df_psi_by_month>=0.10].dropna(how='all').index) 
drop_by_psi_set = list(df_psi_by_set[df_psi_by_set>=0.10].dropna(how='all').index)
# drop_by_psi = drop_by_psi_month + drop_by_psi_set
drop_by_psi = drop_by_psi_set
print("drop_by_psi: ", len(drop_by_psi))

# df_iv_by_set.drop(columns=['mean','std','cv'], inplace=True)
# drop_by_iv_month = list(df_iv_by_month[df_iv_by_month<0.01].dropna(how='all').index)
drop_by_iv_set = list(df_iv_by_set[df_iv_by_set<0.01].dropna(how='all').index)
# drop_by_iv = drop_by_iv_month + drop_by_iv_set
drop_by_iv = drop_by_iv_set
print("drop_by_iv: ", len(drop_by_iv))

to_drop3 = list(set(drop_by_psi + drop_by_iv))
print("剔除的变量有: ", len(to_drop3))


# In[509]:


df_iv_by_set.loc[drop_by_iv_set,:]


# In[510]:


df_psi_by_set.loc[drop_by_psi_set,:]


# In[511]:


varsname_v3 = [ col for col in varsname_v2 if col not in to_drop3]
print(f"保留的变量有{len(varsname_v3)}个: ")


# In[515]:


print(varsname_v3)


# ## 4.2 Y标签相关性删除

# In[512]:


# 计算相关性
exclude = [col for col in df_bins.columns if col not in varsname_v3]

transer = toad.transform.WOETransformer()
df_sample_woe = transer.fit_transform(df_bins, df_bins[target], exclude=exclude)
print(df_sample_woe.shape) 


# In[513]:


df_sample_woe.head(2)


# In[516]:


def find_high_correlation_pairs(df, iv_series, method='kendall', threshold=0.85):
    """
    找出相关系数大于指定阈值的变量对，并排除对角线。保留IV值较大的变量。

    :param df: 输入的DataFrame
    :param iv_series: 包含每个变量的IV值的Series，变量名为行索引
    :param method: 计算相关系数的方法，可以是'pearson', 'kendall', 'spearman'
    :param threshold: 相关系数的阈值，默认为0.85
    :return: 包含高相关性变量对及其相关系数的DataFrame，以及保留的变量
    """
    # 计算相关系数矩阵
    corr_matrix = df.corr(method=method)
    # 初始化一个空列表来存储高相关性变量对
    high_corr_pairs = []
    # 遍历相关系数矩阵
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                high_corr_pairs.append((var1, var2, corr_value))
    
    # 将结果转换为DataFrame
    high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Var1', 'Var2', 'Correlation'])
    # 初始化一个空列表来存储需要删除的变量
    to_remove = set()
    # 遍历高相关性变量对，保留IV值较大的变量，删除IV值较小的变量
    for _, row in high_corr_df.iterrows():
        var1 = row['Var1']
        var2 = row['Var2']
        iv1 = iv_series[var1]
        iv2 = iv_series[var2]
        
        if iv1 >= iv2:
            to_remove.add(var2)
        else:
            to_remove.add(var1)
    
    # 返回高相关性变量对及其相关系数，以及保留的变量
    return high_corr_df, list(to_remove)


# In[517]:


gc.collect()


# In[518]:


df_iv_by_set


# In[523]:


# 调用函数

df_high_corr, to_drop4 = find_high_correlation_pairs(df_sample_woe[varsname_v3],
                                                     df_iv_by_set['3_oot'],
                                                     method='kendall',
                                                     threshold=0.85)

# 查看结果
print("删除的变量有：", len(to_drop4))


# In[524]:


df_high_corr


# In[520]:


df_high_corr


# In[525]:


to_drop4


# In[526]:


varsname_v4 = [col for col in varsname_v3 if col not in to_drop4]
print(f"保留变量{len(varsname_v4)}个")
print(varsname_v4)


# In[527]:


df_iv_by_set.loc[df_high_corr['Var1'],:]


# In[528]:


df_iv_by_set.loc[df_high_corr['Var2'],:]


# In[529]:


def calculate_correlations(df, group_col, varsname, target, method='pointbiserialr'):
    from scipy.stats import pointbiserialr, pearsonr, spearmanr, kendalltau
    
    # 按指定的分组列分组
    grouped = df.groupby(group_col)
    # 初始化一个空的DataFrame来存储所有分组的相关系数
    all_corrs = pd.DataFrame()
    all_pvalue = pd.DataFrame()
    # 遍历每个分组
    for name, group in grouped:      
        # 计算每个变量与目标变量的相关系数
        # 初始化一个空的字典来存储结果
        corr_series = pd.Series(index=varsname)
        corr_series.name = name
        result_pvalue = pd.Series(index=varsname)
        result_pvalue.name = name
        
        binary_var = group[target]
        # 计算每个连续变量与二分类变量之间的点二列相关系数
        for column in varsname:
            if method=='pointbiserialr':
                corr_series[column] =  pointbiserialr(binary_var, group[column])[0]
                result_pvalue[column] =  pointbiserialr(binary_var, group[column])[1]
            elif method=='pearsonr':
                corr_series[column] =  pearsonr(binary_var, group[column])[0]
                result_pvalue[column] =  pearsonr(binary_var, group[column])[1]
            elif method=='spearmanr':
                corr_series[column] =  spearmanr(binary_var, group[column])[0]
                result_pvalue[column] =  spearmanr(binary_var, group[column])[1]
            elif method=='kendalltau':
                corr_series[column] =  kendalltau(binary_var, group[column])[0]
                result_pvalue[column] =  kendalltau(binary_var, group[column])[1]
            else:
                raise ValueError("Invalid method. Choose from 'pointbiserialr','pearson', 'spearman', or 'kendall'.")

        # corr_series = group[varsname].corrwith(group[target], method=method)
        # 将结果添加到总的DataFrame中，并添加分组标识
        all_corrs = pd.concat([all_corrs, corr_series], axis=1)
        all_pvalue = pd.concat([all_pvalue, result_pvalue], axis=1)
    
    # 返回包含所有分组相关系数的DataFrame
    return (all_corrs, all_pvalue)


# In[533]:


# 调用函数
df_corr_vars_target, df_pvalue_vars_target = calculate_correlations(df_sample_woe,
                                                                    'lending_month',
                                                                    varsname_v4,
                                                                    target,
                                                                    method='pointbiserialr'
                                                                   )

# 查看前几行
df_corr_vars_target


# In[534]:


to_drop5 = list(df_corr_vars_target[df_corr_vars_target.apply(lambda row: (row > 0).any() and (row < 0).any(), axis=1)].index)
print("删除的变量有：", len(to_drop5))


# In[535]:


varsname_v5 = [ col for col in varsname_v4 if col not in to_drop5]
print(f"保留的变量{len(varsname_v5)}个")


# ## 4.3 逐步回归筛选

# In[78]:


# # 将woe转化后的数据做逐步回归
# train_woe = df_sample_woe.query("data_set=='1_train'")[varsname_v3+[target]]
# final_data, to_drop6 = toad.selection.stepwise(train_woe, target=target, estimator='ols', direction = 'both', \
#                                      criterion = 'aic', exclude = None, return_drop=True)

# print(final_data.shape) # 逐步回归从31个变量中选出了10个


# In[536]:


timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

with pd.ExcelWriter(result_path + f'3_变量分析_dis_iv_psi_{task_name}_{timestamp}.xlsx') as writer:
        df_group_month.to_excel(writer, sheet_name='df_group_month') 
        df_group_set.to_excel(writer, sheet_name='df_group_set') 
        df_iv_by_month.to_excel(writer, sheet_name='df_iv_by_month')
        df_iv_by_set.to_excel(writer, sheet_name='df_iv_by_set')
        df_psi_by_month.to_excel(writer, sheet_name='df_psi_by_month')
        df_psi_by_set.to_excel(writer, sheet_name='df_psi_by_set')
        df_pvalue_vars_target.to_excel(writer, sheet_name='df_pvalue_vars_target')
        df_corr_vars_target.to_excel(writer, sheet_name='df_corr_vars_target')
        df_total_bad.to_excel(writer, sheet_name='df_total_bad')
        df_total_bad_set.to_excel(writer, sheet_name='df_total_bad_set')
        pivot_df_iv.to_excel(writer, sheet_name='pivot_df_iv')
        pivot_df_iv_set.to_excel(writer, sheet_name='pivot_df_iv_set') 
        pivot_df_badrate_iv.to_excel(writer, sheet_name='pivot_df_badrate_iv')
print(f"数据存储完成时间：{timestamp}！")        
print(result_path + f'3_变量分析_dis_iv_psi_{task_name}_{timestamp}.xlsx')


# In[537]:


gc.collect()


# # 5.模型训练

# ## 5.1 模型训练

# In[5]:



def model_ks_auc(df, target, y_pred, group_col):
    """
    Args:
        df (dataframe): 含有Y标签和预测分数的数据集
        target (string): Y标签列名
        y_pred (string): 坏概率分数列名
        group_col (string): 分组列名如月份，数据集

    Returns:
        dataframe: AUC和KS值的数据框
    """
    df_ks_auc = pd.DataFrame(index=['KS', 'AUC'])
    for col, group_df in df.groupby(group_col):  
        # 计算 AUC
        group_df = group_df[(group_df[target].notna())&(group_df[y_pred].notna())]
        auc_ = roc_auc_score(group_df[target], group_df[y_pred])      
        fpr, tpr, _ = roc_curve(group_df[target], group_df[y_pred], pos_label=1)
        ks_ = max(tpr-fpr)
        df_ks_auc.loc['KS', col] = ks_
        df_ks_auc.loc['AUC', col] = auc_
#         print(f"{col}：KS值{ks_}，AUC值{auc_}")
    df_ks_auc = df_ks_auc.T
    
    return df_ks_auc



def feature_importance(model):
    if isinstance(model, lgb.Booster):
        print("这是原生接口的模型 (Booster)")
        # 获取特征重要性
        feature_importance_gain = model.feature_importance(importance_type='gain')
        feature_importance_split = model.feature_importance(importance_type='split')
        # 获取特征名称
        feature_names = model.feature_name()
        # 将特征重要性转换为数据框
        df_importance = pd.DataFrame({'gain': feature_importance_gain,
                                      'split': feature_importance_split}, 
                                     index=feature_names)
        df_importance = df_importance.sort_values('gain', ascending=False)
        df_importance.index.name = 'feature'
    elif isinstance(model, (LGBMClassifier, LGBMRegressor)):
        print("这是 sklearn 接口的模型")
        df1_dict = model.get_booster().get_score(importance_type='weight')
        importance_type_split = pd.DataFrame.from_dict(df1_dict, orient='index')
        importance_type_split.columns = ['split']
        importance_type_split = importance_type_split.sort_values('split', ascending=False)
        importance_type_split['split_pct'] = importance_type_split['split'] / importance_type_split['split'].sum()

        df2_dict = model.get_booster().get_score(importance_type='gain')
        importance_type_gain = pd.DataFrame.from_dict(df2_dict, orient='index')
        importance_type_gain.columns = ['gain']
        importance_type_gain = importance_type_gain.sort_values('gain', ascending=False)
        importance_type_gain['gain_pct'] = importance_type_gain['gain'] / importance_type_gain['gain'].sum()

        df_importance = pd.concat([importance_type_gain, importance_type_split], axis=1)
        df_importance = df_importance.sort_values('gain', ascending=False)
        df_importance.index.name = 'feature'
    else:
        print("未知模型类型")
        df_importance = None
    
    return df_importance

# Pickle方式保存和读取模型
def save_model_as_pkl(model, path):
    """
    保存模型到路径path
    :param model: 训练完成的模型
    :param path: 保存的目标路径
    """
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(model, f, protocol=2)
        
# xgb模型保存.bin 格式
def save_model_as_bin(model, save_file_path):
    #保存lgb模型为bin格式
    model.save_model(save_file_path)
    

def load_model_from_pkl(path):
    """
    从路径path加载模型
    :param path: 保存的目标路径
    """
    
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


# In[540]:


# 2 定义超参空间
# hp.quniform("参数名称",下界,上界,步长)-适用于离散均匀分布的浮点点数
# hp.uniform("参数名称",下界, 下界)-适用于连续随机分布的浮点数
# hp.randint("参数名称",上界)-适用于[0,上界)的整数,区间为左闭右开
# hp.choice("参数名称",["字符串1","字符串2",...])-适用于字符串类型,最优参数由索引表示
# hp.loguniform: continuous log uniform (floats spaced evenly on a log scale)
# choice : categorical variables
# quniform : discrete uniform (integers spaced evenly)
# uniform: continuous uniform (floats spaced evenly)
# loguniform: continuous log uniform (floats spaced evenly on a log scale)
# 可以根据需要，注释掉偏后的一些不太重要的超参

spaces = {
          # general parameters
#           "learning_rate":hp.loguniform("learning_rate",np.log(0.001), np.log(0.2)),
          "learning_rate":0.1,
          # tuning parameters
          "num_leaves":hp.quniform("num_leaves",21,200,1),
          "max_depth":3,
          "min_data_in_leaf":hp.quniform("min_data_in_leaf",30,200,1),
          "feature_fraction":hp.uniform("feature_fraction",0.6,1.0),
          "bagging_fraction":hp.uniform("bagging_fraction",0.6,1.0),
#           "feature_fraction":1.0,
#           "bagging_fraction":1.0,
#           "min_gain_to_split":10,
          "min_gain_to_split":hp.uniform("min_gain_to_split",0.1, 1.0),
          "lambda_l1": 0,
#           "lambda_l1": hp.randint("lambda_l1", 1),
#           "lambda_l2": hp.uniform("lambda_l2", 100, 1000),
          "lambda_l2": 3,
#           "early_stopping_rounds": hp.quniform("early_stopping_rounds", 50, 60, 10)
          "early_stopping_rounds": 50
          }
spaces


# In[541]:


# 3，执行超参搜索
# 有了目标函数和参数空间,接下来要进行优化,需要了解以下参数:
# fmin:自定义使用的代理模型(参数algo),hyperopt支持如下搜索算法：
#       随机搜索(hyperopt.rand.suggest)
#       模拟退火(hyperopt.anneal.suggest)
#       TPE算法（hyperopt.tpe.suggest，算法全称为Tree-structured Parzen Estimator Approach）
# partial:修改算法涉及到的具体参数,包括模型具体使用了多少少个初始观测值(参数n_start_jobs),
#         以及在计算采集函数值时究竟考虑多少个样本(参数n_EI_candidates)
# trials:记录整个迭代过程,从hyperopt库中导入的方法Trials(),优化完成之后,
#        可以从保存好的trials中查看损失、参数等各种中间信息
# early_stop_fn:提前停止参数,从hyperopt库导入的方法no_progresss_loss(),可以输入具体的数字n,
#               表示当损失连续n次没有下降时,让算法提前停止
def param_hyperopt(param_spaces, X_train, y_train, X_test=None, y_test=None, 
                   num_boost_round=10000, nfolds=5, max_evals=20):
    """
    贝叶斯调参, 确定其他参数
    """
    
    # 1 定义目标函数
    def lgb_hyperopt_object(params, num_boost_round=num_boost_round, n_folds=nfolds, init_model=None):

        """定义目标函数"""
        param = {
                # general parameters
                'objective': 'binary',
                'boosting': 'gbdt',
                'metric': 'auc',
                'learning_rate': params['learning_rate'],
                # tuning parameters
                'num_leaves': int(params['num_leaves']),
                'min_data_in_leaf': int(params['min_data_in_leaf']),
                'max_depth': int(params['max_depth']),
                'bagging_freq': 1,
                'bagging_fraction': params['bagging_fraction'],
                'feature_fraction': params['feature_fraction'],
                'lambda_l1': params['lambda_l1'],
                'lambda_l2': params['lambda_l2'],
                'min_gain_to_split':params['min_gain_to_split'],
                'early_stopping_rounds': int(params['early_stopping_rounds']),
                'scale_pos_weight': 1,
                'seed': 1,
                'num_threads': -1
                }
        train_set = lgb.Dataset(X_train, label=y_train)
        if X_test is None:
            cv_results = lgb.cv(param, 
                                train_set, 
                                num_boost_round=num_boost_round,
                                nfold = n_folds, 
                                stratified=True, 
                                shuffle=True, 
                                metrics='auc',
                                init_model=init_model,
                                seed=1
                                )
            best_score = max(cv_results['auc-mean'])
            loss = 1 - best_score
            
        else:
            valid_set = lgb.Dataset(X_test, label=y_test)
            clf_obj = lgb.train(param, train_set, valid_sets=valid_set,                                 num_boost_round=num_boost_round, init_model=init_model)
            loss = 1 - roc_auc_score(y_test, clf_obj.predict(X_test, num_iteration=clf_obj.best_iteration))
        
        return loss
    
    #保存迭代过程
    trials = Trials()
    #设置提前停止
    early_stop_fn = no_progress_loss(30)
    #定义代理模型
    #algo = partial(tpe.suggest, n_startup_jobs=20, r_EI_candidates=50)
    
    best_params = fmin(lgb_hyperopt_object #目标函数
                      ,space=param_spaces  #参数空间
                      ,algo = tpe.suggest  #代理模型
                      ,max_evals=max_evals #允许的迭代次数
                      ,verbose=True
                      ,trials = trials
                      ,early_stop_fn = early_stop_fn
                       )
    
    return (best_params, trials)


# In[542]:


# 查看训练数据集
df_sample['data_set'].value_counts()


# In[543]:


# 查看训练数据集
df_sample.loc[df_sample.query("data_set!='3_oot'").index, 'data_set']='1_train'
df_sample['data_set'].value_counts()


# In[544]:


varsname_v5


# In[128]:


# 训练数据集
X_train = df_sample.query("data_set!='3_oot'")[varsname_v5]
y_train = df_sample.query("data_set!='3_oot'")[target]
print(X_train.shape)


# In[129]:


# 4，获取最优参数，调参过程
# 确定一个较高的学习率
# 对决策树基本参数调参
# 正则化参数调参
# 降低学习率
best_params, trials = param_hyperopt(spaces, X_train, y_train, X_test=None, y_test=None, max_evals=10)


# In[130]:


# 5，绘制搜索过程
losses = [x["result"]["loss"] for x in trials.trials]
minlosses = [np.min(losses[0:i+1]) for i in range(len(losses))] 
steps = range(len(losses))

fig,ax = plt.subplots(figsize=(6,3.7),dpi=144)
ax.scatter(x = steps, y = losses, alpha = 0.3)
ax.plot(steps,minlosses,color = "red",axes = ax)
plt.xlabel("step")
plt.ylabel("loss")


# In[131]:


print("最优参数best_params: ", best_params)


# In[132]:


# best_params={'bagging_fraction': 0.9428977631473267, 'feature_fraction': 0.9921325170745515, 'min_data_in_leaf': 44.0, 'min_gain_to_split': 0.26088155837498217, 'num_leaves': 200.0}


# In[133]:


# print("最优参数best_params: ", best_params)


# In[545]:


### 添加无需调参的通用参数
bst_params = {}
bst_params['boosting'] = 'gbdt'
bst_params['objective'] = 'binary'
bst_params['metric'] = 'auc'
bst_params['bagging_freq'] = 1
bst_params['scale_pos_weight'] = 1 
bst_params['seed'] = 1 
bst_params['num_threads'] = -1 
# 调参时设置成不用调参的参数
bst_params['learning_rate'] = spaces['learning_rate']
## 正则参数，防止过拟合
bst_params['bagging_fraction'] = best_params['bagging_fraction']    
bst_params['feature_fraction'] = best_params['feature_fraction'] 
bst_params['lambda_l1'] = spaces['lambda_l1']
bst_params['lambda_l2'] = spaces['lambda_l2']
bst_params['early_stopping_rounds'] = spaces['early_stopping_rounds']

# 调参后的参数需要变成整数型
bst_params['num_leaves'] = int(best_params['num_leaves'] )
bst_params['min_data_in_leaf'] = int(best_params['min_data_in_leaf'] )
bst_params['max_depth'] = spaces['max_depth']
# 调参后的其他参
bst_params['min_gain_to_split'] = best_params['min_gain_to_split']


# In[546]:


print("最优参数bst_params: ", bst_params)


# In[547]:


# 确定参数后，确定训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df_sample.query("data_set!='3_oot'")[varsname_v5],
                                                    df_sample.query("data_set!='3_oot'")[target],
                                                    test_size=0.2, 
                                                    random_state=22, 
                                                    stratify=df_sample.query("data_set!='3_oot'")[target])
print(X_train.shape, X_test.shape)

df_sample.loc[X_train.index, 'data_set']='1_train'
df_sample.loc[X_test.index, 'data_set']='2_test'
print(df_sample['data_set'].value_counts())


# In[548]:


# 6，训练/保存/评估模型
# 最初训练模型
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
lgb_model = lgb.train(bst_params, train_set, valid_sets=valid_set, num_boost_round=10000, init_model=None)


# In[549]:


# 最初评估模型效果
df_sample['y_prob'] = lgb_model.predict(df_sample[X_train.columns], num_iteration=lgb_model.best_iteration)


# In[550]:


# 最初评估模型效果
df_ks_auc_month = model_ks_auc(df_sample, target, 'y_prob', 'lending_month')
tmp = get_target_summary(df_sample, target, 'lending_month').set_index('bins')
df_ks_auc_month = pd.concat([tmp, df_ks_auc_month], axis=1)
print(df_ks_auc_month)


df_ks_auc_set = model_ks_auc(df_sample, target, 'y_prob', 'data_set')
tmp = get_target_summary(df_sample, target, 'data_set').set_index('bins')
df_ks_auc_set = pd.concat([tmp, df_ks_auc_set], axis=1)
print(df_ks_auc_set)


# In[551]:


# 最初评估模型效果-30+客群
df_ks_auc_month_30 = model_ks_auc(df_sample[df_sample['diff_days']>30], target, 'y_prob', 'lending_month')
tmp = get_target_summary(df_sample[df_sample['diff_days']>30], target, 'lending_month').set_index('bins')
df_ks_auc_month_30 = pd.concat([tmp, df_ks_auc_month_30], axis=1)
print(df_ks_auc_month_30)


df_ks_auc_set_30 = model_ks_auc(df_sample[df_sample['diff_days']>30], target, 'y_prob', 'data_set')
tmp = get_target_summary(df_sample[df_sample['diff_days']>30], target, 'data_set').set_index('bins')
df_ks_auc_set_30 = pd.concat([tmp, df_ks_auc_set_30], axis=1)
print(df_ks_auc_set_30)


# In[552]:


# 227
df_ksauc227_month = model_ks_auc(df_sample.query("channel_id=='227'"), target, 'y_prob', 'lending_month')
tmp = get_target_summary(df_sample.query("channel_id=='227'"), target, 'lending_month').set_index('bins')
df_ksauc227_month = pd.concat([tmp, df_ksauc227_month], axis=1)
print(df_ksauc227_month)

df_ksauc227_set = model_ks_auc(df_sample.query("channel_id=='227'"), target, 'y_prob', 'data_set')
tmp = get_target_summary(df_sample.query("channel_id=='227'"), target, 'data_set').set_index('bins')
df_ksauc227_set = pd.concat([tmp, df_ksauc227_set], axis=1)
print(df_ksauc227_set)


# In[553]:


# 227
df_ksauc227_month_30 = model_ks_auc(df_sample.query("channel_id=='227'&diff_days>30"), target, 'y_prob', 'lending_month')
tmp = get_target_summary(df_sample.query("channel_id=='227'&diff_days>30"), target, 'lending_month').set_index('bins')
df_ksauc227_month_30 = pd.concat([tmp, df_ksauc227_month_30], axis=1)
print(df_ksauc227_month_30)

df_ksauc227_set_30 = model_ks_auc(df_sample.query("channel_id=='227'&diff_days>30"), target, 'y_prob', 'data_set')
tmp = get_target_summary(df_sample.query("channel_id=='227'&diff_days>30"), target, 'data_set').set_index('bins')
df_ksauc227_set_30 = pd.concat([tmp, df_ksauc227_set_30], axis=1)
print(df_ksauc227_set_30)


# In[555]:


# 模型变量重要性
# df_iv_by_month.drop(columns=['mean', 'std', 'cv'], inplace=True)
df_importance_month = feature_importance(lgb_model) 
df_importance_month = pd.merge(df_importance_month, df_iv_by_month, how='inner', left_index=True,right_index=True)
df_importance_month = df_importance_month.reset_index()
df_importance_month = df_importance_month.rename(columns={'index':'varsname'})
df_importance_month


# In[556]:



# 效果评估后模型变量重要性
df_importance_set = feature_importance(lgb_model) 
df_psi_iv = pd.merge(df_psi_by_set, df_iv_by_set, how='inner', left_index=True,right_index=True, suffixes=('_psi', '_iv'))
df_importance_set = pd.merge(df_importance_set, df_psi_iv, how='inner', left_index=True,right_index=True)
df_importance_set['iv的变化幅度'] = df_importance_set['3_oot_iv']/df_importance_set['1_train_iv'] - 1
df_importance_set.drop(columns=['1_train_psi'], inplace=True)
df_importance_set = df_importance_set.reset_index()
df_importance_set = df_importance_set.rename(columns={'index':'varsname'})
df_importance_set


# In[557]:


result_path


# In[558]:


# 效果评估后保存模型
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
save_model_as_pkl(lgb_model, result_path + f'{task_name}_{timestamp}.pkl')
save_model_as_bin(lgb_model, result_path + f'{task_name}_{timestamp}.bin')
print(f"模型保存完成！：{timestamp}")
print(result_path + f'{task_name}_{timestamp}.pkl')
print(result_path + f'{task_name}_{timestamp}.bin')


# In[559]:


timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
with pd.ExcelWriter(result_path + f'4_模型结果分析_{task_name}_{timestamp}.xlsx') as writer:
    df_importance_month.to_excel(writer, sheet_name='df_importance_month')
    df_importance_set.to_excel(writer, sheet_name='df_importance_set')
    df_ks_auc_month.to_excel(writer, sheet_name='df_ks_auc_month')
    df_ks_auc_set.to_excel(writer, sheet_name='df_ks_auc_set')
    df_ks_auc_month_30.to_excel(writer, sheet_name='df_ks_auc_month_30')
    df_ks_auc_set_30.to_excel(writer, sheet_name='df_ks_auc_set_30')
    df_ksauc227_month.to_excel(writer, sheet_name='df_ksauc227_month')
    df_ksauc227_month_30.to_excel(writer, sheet_name='df_ksauc227_month_30')
    df_ksauc227_set.to_excel(writer, sheet_name='df_ksauc227_set')
    df_ksauc227_set_30.to_excel(writer, sheet_name='df_ksauc227_set_30') 
print("数据存储完成！{timestamp}")
print(result_path + f'4_模型结果分析_{task_name}_{timestamp}.xlsx')


# ## 5.2 模型优化

# ### 5.2.1参数优化

# In[ ]:


bst_params:  {'boosting': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'bagging_freq': 1, 'scale_pos_weight': 1, 'seed': 1, 'num_threads': -1, 'learning_rate': 0.1, 'bagging_fraction': 0.8628008772208227, 'feature_fraction': 0.6177619614753441, 'lambda_l1': 0, 'lambda_l2': 3, 'early_stopping_rounds': 50, 'num_leaves': 107, 'min_data_in_leaf': 169, 'max_depth': 3, 'min_gain_to_split': 0.6152625859854175}


# In[588]:


### 优化调参1
opt_params = {}
opt_params['boosting'] = 'gbdt'
opt_params['objective'] = 'binary'
opt_params['metric'] = 'auc'
opt_params['bagging_freq'] = 1
opt_params['scale_pos_weight'] = 1 
opt_params['seed'] = 1 
opt_params['num_threads'] = -1 
# 调参时设置成不用调参的参数
opt_params['learning_rate'] = 0.1
## 正则参数，防止过拟合
opt_params['bagging_fraction'] = 0.8628008772208227     
opt_params['feature_fraction'] = 0.6177619614753441
opt_params['lambda_l1'] = 0
opt_params['lambda_l2'] = 300
opt_params['early_stopping_rounds'] = 600

# 调参后的参数需要变成整数型
opt_params['num_leaves'] = 107
opt_params['min_data_in_leaf'] = 169
opt_params['max_depth'] = 3
# 调参后的其他参
opt_params['min_gain_to_split'] = 10


# In[589]:


print("最优参数opt_params: ", opt_params)


# In[590]:


# 查看训练数据集
df_sample['data_set'].value_counts()


# In[591]:


# 查看训练数据集
df_sample.loc[df_sample.query("data_set!='3_oot'").index, 'data_set']='1_train'
df_sample['data_set'].value_counts()


# In[592]:


# 确定数据集参数后，训练模型
X_train, X_test, y_train, y_test = train_test_split(df_sample.query("data_set!='3_oot'")[varsname_v5],
                                                    df_sample.query("data_set!='3_oot'")[target],
                                                    test_size=0.2, 
                                                    random_state=22, 
                                                    stratify=df_sample.query("data_set!='3_oot'")[target])
print(X_train.shape, X_test.shape)

df_sample.loc[X_train.index, 'data_set']='1_train'
df_sample.loc[X_test.index, 'data_set']='2_test'
print(df_sample['data_set'].value_counts())


# In[593]:


# 6，训练/保存/评估模型
# 优化训练模型
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
lgb_model = lgb.train(opt_params, train_set, valid_sets=valid_set, num_boost_round=10000)


# In[594]:


# 优化后评估模型效果
df_sample['y_prob_v1'] = lgb_model.predict(df_sample[X_train.columns], num_iteration=lgb_model.best_iteration)


# In[595]:


# 优化后评估模型效果
df_ks_auc_month_v2 = model_ks_auc(df_sample, target, 'y_prob_v1', 'lending_month')
tmp = get_target_summary(df_sample, target, 'lending_month').set_index('bins')
df_ks_auc_month_v2 = pd.concat([tmp, df_ks_auc_month_v2], axis=1)
print(df_ks_auc_month_v2)


df_ks_auc_set_v2 = model_ks_auc(df_sample, target, 'y_prob_v1', 'data_set')
tmp = get_target_summary(df_sample, target, 'data_set').set_index('bins')
df_ks_auc_set_v2 = pd.concat([tmp, df_ks_auc_set_v2], axis=1)
print(df_ks_auc_set_v2)


# In[596]:


# 优化后评估模型效果-30+客群
df_ks_auc_month_30_v2 = model_ks_auc(df_sample.query("diff_days>30"), target, 'y_prob_v1', 'lending_month')
tmp = get_target_summary(df_sample.query("diff_days>30"), target, 'lending_month').set_index('bins')
df_ks_auc_month_30_v2 = pd.concat([tmp, df_ks_auc_month_30_v2], axis=1)
print(df_ks_auc_month_30_v2)


df_ks_auc_set_30_v2 = model_ks_auc(df_sample.query("diff_days>30"), target, 'y_prob_v1', 'data_set')
tmp = get_target_summary(df_sample.query("diff_days>30"), target, 'data_set').set_index('bins')
df_ks_auc_set_30_v2 = pd.concat([tmp, df_ks_auc_set_30_v2], axis=1)
print(df_ks_auc_set_30_v2)


# In[ ]:





# In[599]:


# 227渠道
df_ksauc227_month_v2 = model_ks_auc(df_sample.query("channel_id=='227'"), target, 'y_prob_v1', 'lending_month')
tmp = get_target_summary(df_sample.query("channel_id=='227'"), target, 'lending_month').set_index('bins')
df_ksauc227_month_v2 = pd.concat([tmp, df_ksauc227_month_v2], axis=1)
print(df_ksauc227_month_v2)

df_ksauc227_set_v2 = model_ks_auc(df_sample.query("channel_id=='227'"), target, 'y_prob_v1', 'data_set')
tmp = get_target_summary(df_sample.query("channel_id=='227'"), target, 'data_set').set_index('bins')
df_ksauc227_set_v2 = pd.concat([tmp, df_ksauc227_set_v2], axis=1)
print(df_ksauc227_set_v2)


# In[600]:


# 227渠道
df_ksauc227_month_30_v2 = model_ks_auc(df_sample.query("channel_id=='227'&diff_days>30"), target, 'y_prob_v1', 'lending_month')
tmp = get_target_summary(df_sample.query("channel_id=='227'&diff_days>30"), target, 'lending_month').set_index('bins')
df_ksauc227_month_30_v2 = pd.concat([tmp, df_ksauc227_month_30_v2], axis=1)
print(df_ksauc227_month_30_v2)

df_ksauc227_set_30_v2 = model_ks_auc(df_sample.query("channel_id=='227'&diff_days>30"), target, 'y_prob_v1', 'data_set')
tmp = get_target_summary(df_sample.query("channel_id=='227'&diff_days>30"), target, 'data_set').set_index('bins')
df_ksauc227_set_30_v2 = pd.concat([tmp, df_ksauc227_set_30_v2], axis=1)
print(df_ksauc227_set_30_v2)


# In[601]:


# 模型变量重要性
# df_iv_by_month.drop(columns=['mean', 'std', 'cv'], inplace=True)
df_importance_v2 = feature_importance(lgb_model) 
df_importance_v2 = pd.merge(df_importance_v2, df_iv_by_month, how='inner', left_index=True,right_index=True)
df_importance_v2 = df_importance_v2.reset_index()
df_importance_v2 = df_importance_v2.rename(columns={'index':'varsname'})
df_importance_v2.head(20)


# In[602]:


task_name


# In[604]:


# 效果评估后保存模型
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
save_model_as_pkl(lgb_model, result_path + f'{task_name}_{timestamp}.pkl')
save_model_as_bin(lgb_model, result_path + f'{task_name}_{timestamp}.bin')
print(f"模型保存完成！：{timestamp}")
print(result_path + f'{task_name}_opt_{timestamp}.pkl')
print(result_path + f'{task_name}_opt_{timestamp}.bin')


# In[605]:


timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
with pd.ExcelWriter(result_path + f'5_模型优化_{task_name}_opt_{timestamp}.xlsx') as writer:
    df_importance_v2.to_excel(writer, sheet_name='df_importance_v2')
    df_ks_auc_month_v2.to_excel(writer, sheet_name='df_ks_auc_month_v2')
    df_ks_auc_set_v2.to_excel(writer, sheet_name='df_ks_auc_set_v2')
    df_ks_auc_month_30_v2.to_excel(writer, sheet_name='df_ks_auc_month_30_v2')
    df_ks_auc_set_30_v2.to_excel(writer, sheet_name='df_ks_auc_set_30_v2')
    df_ksauc227_month_v2.to_excel(writer, sheet_name='df_ksauc227_month_v2')
    df_ksauc227_month_30_v2.to_excel(writer, sheet_name='df_ksauc227_month_30_v2')
    df_ksauc227_set_v2.to_excel(writer, sheet_name='df_ksauc227_set_v2')
    df_ksauc227_set_30_v2.to_excel(writer, sheet_name='df_ksauc227_set_30_v2')        
print("数据存储完成！{timestamp}")
print(result_path + f'5_模型优化_{task_name}_opt_{timestamp}.xlsx')


# ### 5.2.2 特征优化

# #### 5.2.2.1 特征woe

# In[606]:


# 查看训练数据集
df_sample_woe['data_set'].value_counts()


# In[ ]:


# 查看训练数据集
df_sample_woe.loc[df_sample_woe.query("data_set!='3_oot'").index, 'data_set']='1_train'
df_sample_woe['data_set'].value_counts()


# In[235]:


# 使用woe分箱后的数据，训练模型
X_train, X_test, y_train, y_test = train_test_split(df_sample_woe.query("data_set!='3_oot'")[varsname_v5],
                                                    df_sample_woe.query("data_set!='3_oot'")[target],
                                                    test_size=0.2, 
                                                    random_state=22, 
                                                    stratify=df_sample_woe.query("data_set!='3_oot'")[target])
print(X_train.shape, X_test.shape)
df_sample_woe.loc[X_train.index, 'data_set']='1_train'
df_sample_woe.loc[X_test.index, 'data_set']='2_test'
df_sample_woe['data_set'].value_counts()


# In[242]:


opt_params


# In[243]:


# 6，训练/保存/评估模型
# 最初训练模型
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
lgb_model = lgb.train(opt_params, train_set, valid_sets=valid_set, num_boost_round=10000, init_model=None)


# In[244]:


# 最初评估模型效果
df_sample_woe['y_prob_woe'] = lgb_model.predict(df_sample_woe[X_train.columns], num_iteration=lgb_model.best_iteration)


# In[245]:


# 最初评估模型效果
df_ks_auc_month_woe = model_ks_auc(df_sample_woe, target, 'y_prob_woe', 'lending_month')
tmp = get_target_summary(df_sample_woe, target, 'lending_month').set_index('bins')
df_ks_auc_month_woe = pd.concat([tmp, df_ks_auc_month_woe], axis=1)
print(df_ks_auc_month_woe)


df_ks_auc_set_woe = model_ks_auc(df_sample_woe, target, 'y_prob_woe', 'data_set')
tmp = get_target_summary(df_sample_woe, target, 'data_set').set_index('bins')
df_ks_auc_set_woe = pd.concat([tmp, df_ks_auc_set_woe], axis=1)
print(df_ks_auc_set_woe)


# In[246]:



# 最初评估模型效果-30+客群
df_ks_auc_month_woe_30 = model_ks_auc(df_sample_woe.query("diff_days>30"), target, 'y_prob_woe', 'lending_month')
tmp = get_target_summary(df_sample_woe.query("diff_days>30"), target, 'lending_month').set_index('bins')
df_ks_auc_month_woe_30 = pd.concat([tmp, df_ks_auc_month_woe_30], axis=1)
print(df_ks_auc_month_woe_30)


df_ks_auc_set_woe_30 = model_ks_auc(df_sample_woe.query("diff_days>30"), target, 'y_prob_woe', 'data_set')
tmp = get_target_summary(df_sample_woe.query("diff_days>30"), target, 'data_set').set_index('bins')
df_ks_auc_set_woe_30 = pd.concat([tmp, df_ks_auc_set_woe_30], axis=1)
print(df_ks_auc_set_woe_30)


# In[248]:


# 227
df_ksauc227_month_woe = model_ks_auc(df_sample_woe.query("channel_id=='227'"), target, 'y_prob_woe', 'lending_month')
tmp = get_target_summary(df_sample_woe.query("channel_id=='227'"), target, 'lending_month').set_index('bins')
df_ksauc227_month_woe = pd.concat([tmp, df_ksauc227_month_woe], axis=1)
print(df_ksauc227_month_woe)

df_ksauc227_set_woe = model_ks_auc(df_sample_woe.query("channel_id=='227'"), target, 'y_prob_woe', 'data_set')
tmp = get_target_summary(df_sample_woe.query("channel_id=='227'"), target, 'data_set').set_index('bins')
df_ksauc227_set_woe = pd.concat([tmp, df_ksauc227_set_woe], axis=1)
print(df_ksauc227_set_woe)


# In[249]:


# 227
df_ksauc227_month_woe_30 = model_ks_auc(df_sample_woe.query("channel_id=='227'&diff_days>30"), target, 'y_prob_woe', 'lending_month')
tmp = get_target_summary(df_sample_woe.query("channel_id=='227'&diff_days>30"), target, 'lending_month').set_index('bins')
df_ksauc227_month_woe_30 = pd.concat([tmp, df_ksauc227_month_woe_30], axis=1)
print(df_ksauc227_month_woe_30)

df_ksauc227_set_woe_30 = model_ks_auc(df_sample_woe.query("channel_id=='227'&diff_days>30"), target, 'y_prob_woe', 'data_set')
tmp = get_target_summary(df_sample_woe.query("channel_id=='227'&diff_days>30"), target, 'data_set').set_index('bins')
df_ksauc227_set_woe_30 = pd.concat([tmp, df_ksauc227_set_woe_30], axis=1)
print(df_ksauc227_set_woe_30)


# In[254]:


# 模型变量重要性
# df_iv_by_month.drop(columns=['mean', 'std', 'cv'], inplace=True)
df_importance = feature_importance(lgb_model) 
df_importance = pd.merge(df_importance, df_iv_by_month, how='inner', left_index=True,right_index=True)
df_importance = df_importance.reset_index()
df_importance = df_importance.rename(columns={'index':'varsname'})
df_importance.head(20)


# In[251]:


# 效果评估后保存模型
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
save_model_as_pkl(lgb_model, result_path + f'{task_name}_woe_{timestamp}.pkl')
save_model_as_bin(lgb_model, result_path + f'{task_name}_woe_{timestamp}.bin')
print(f"模型保存完成！：{timestamp}")
print(f"result_path + f'{task_name}_woe_{timestamp}.pkl")
print(f"result_path + f'{task_name}_woe_{timestamp}.bin")


# In[256]:


timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
with pd.ExcelWriter(result_path + f'5_模型优化_{task_name}_woe_{timestamp}.xlsx') as writer:
    df_importance.to_excel(writer, sheet_name='df_importance')
    df_ks_auc_month_woe.to_excel(writer, sheet_name='df_ks_auc_month_woe')
    df_ks_auc_set_woe.to_excel(writer, sheet_name='df_ks_auc_set_woe')
    df_ks_auc_month_woe_30.to_excel(writer, sheet_name='df_ks_auc_month_woe_30')
    df_ks_auc_set_woe_30.to_excel(writer, sheet_name='df_ks_auc_set_woe_30')
    df_ksauc227_month_woe.to_excel(writer, sheet_name='df_ksauc227_month_woe')
    df_ksauc227_month_woe_30.to_excel(writer, sheet_name='df_ksauc227_month_woe_30')
    df_ksauc227_set_woe.to_excel(writer, sheet_name='df_ksauc227_set_woe')
    df_ksauc227_set_woe_30.to_excel(writer, sheet_name='df_ksauc227_set_woe_30')        
print("数据存储完成！{timestamp}")
print(result_path + f'5_模型优化_{task_name}_woe_{timestamp}.xlsx')


# #### 5.2.2.2 增减特征

# In[607]:


# 查看训练数据集
df_sample['data_set'].value_counts()


# In[608]:


# 查看训练数据集
df_sample.loc[df_sample.query("data_set!='3_oot'").index, 'data_set']='1_train'
df_sample['data_set'].value_counts()


# In[609]:


to_drop6 = ['behave_fpd30_bad_prob', 'prob_fpd10_v1', 'prob_fpd10_v2', 'wanxiangfen']
varsname_v6 = [col for col in varsname_v5 if col not in to_drop6]
# varsname_v6 = ['behave_fpd30_score','pudao_34','br_v3_fpd30_score','ruizhi_6','hengpu_4','pudao_68','aliyun_5','hengpu_5','pudao_20','duxiaoman_6','feicuifen']
print(varsname_v6) 


# In[610]:


# 使用的数据，训练模型
X_train, X_test, y_train, y_test = train_test_split(df_sample.query("data_set!='3_oot'")[varsname_v6],
                                                    df_sample.query("data_set!='3_oot'")[target],
                                                    test_size=0.2, 
                                                    random_state=22, 
                                                    stratify=df_sample.query("data_set!='3_oot'")[target])
print(X_train.shape, X_test.shape)
df_sample.loc[X_train.index, 'data_set']='1_train'
df_sample.loc[X_test.index, 'data_set']='2_test'
df_sample['data_set'].value_counts()


# In[307]:


# ### 添加无需调参的通用参数
# opt_params = {}
# opt_params['boosting'] = 'gbdt'
# opt_params['objective'] = 'binary'
# opt_params['metric'] = 'auc'
# opt_params['bagging_freq'] = 1
# opt_params['scale_pos_weight'] = 1 
# opt_params['seed'] = 1 
# opt_params['num_threads'] = -1 
# # 调参时设置成不用调参的参数
# opt_params['learning_rate'] = 0.1
# ## 正则参数，防止过拟合
# opt_params['bagging_fraction'] = 0.9428977631473267    
# opt_params['feature_fraction'] = 0.9921325170745515
# opt_params['lambda_l1'] = 0
# opt_params['lambda_l2'] = 300
# opt_params['early_stopping_rounds'] = 50

# # 调参后的参数需要变成整数型
# opt_params['num_leaves'] = 200
# opt_params['min_data_in_leaf'] = 44
# opt_params['max_depth'] = 3
# # 调参后的其他参
# opt_params['min_gain_to_split'] = 5
# print("最优参数opt_params: ", opt_params)


# In[611]:


# opt_params = {'boosting': 'gbdt',
#  'objective': 'binary',
#  'metric': 'auc',
#  'bagging_freq': 1,
#  'scale_pos_weight': 1,
#  'seed': 1,
#  'num_threads': -1,
#  'learning_rate': 0.1,
#  'bagging_fraction': 0.8628008772208227,
#  'feature_fraction': 0.6177619614753441,
#  'lambda_l1': 100,
#  'lambda_l2': 300,
#  'early_stopping_rounds': 50,
#  'num_leaves': 107,
#  'min_data_in_leaf': 169,
#  'max_depth': 3,
#  'min_gain_to_split': 10}


# In[612]:


# 6，训练/保存/评估模型
# 最初训练模型
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
lgb_model = lgb.train(opt_params, train_set, valid_sets=valid_set, num_boost_round=10000, init_model=None)


# In[613]:


# 最初评估模型效果
df_sample['y_prob_v3'] = lgb_model.predict(df_sample[X_train.columns], num_iteration=lgb_model.best_iteration)


# In[614]:



# 最初评估模型效果
df_ks_auc_month_v3 = model_ks_auc(df_sample, target, 'y_prob_v3', 'lending_month')
tmp = get_target_summary(df_sample, target, 'lending_month').set_index('bins')
df_ks_auc_month_v3 = pd.concat([tmp, df_ks_auc_month_v3], axis=1)
print(df_ks_auc_month_v3)


df_ks_auc_set_v3 = model_ks_auc(df_sample, target, 'y_prob_v3', 'data_set')
tmp = get_target_summary(df_sample, target, 'data_set').set_index('bins')
df_ks_auc_set_v3 = pd.concat([tmp, df_ks_auc_set_v3], axis=1)
print(df_ks_auc_set_v3)


# In[615]:



# 最初评估模型效果-30+客群
df_ks_auc_month_v3_30 = model_ks_auc(df_sample.query("diff_days>30"), target, 'y_prob_v3', 'lending_month')
tmp = get_target_summary(df_sample.query("diff_days>30"), target, 'lending_month').set_index('bins')
df_ks_auc_month_v3_30 = pd.concat([tmp, df_ks_auc_month_v3_30], axis=1)
print(df_ks_auc_month_v3_30)


df_ks_auc_set_v3_30 = model_ks_auc(df_sample.query("diff_days>30"), target, 'y_prob_v3', 'data_set')
tmp = get_target_summary(df_sample.query("diff_days>30"), target, 'data_set').set_index('bins')
df_ks_auc_set_v3_30 = pd.concat([tmp, df_ks_auc_set_v3_30], axis=1)
print(df_ks_auc_set_v3_30)


# In[616]:



# 227
df_ksauc227_month_v3 = model_ks_auc(df_sample.query("channel_id=='227'"), target, 'y_prob_v3', 'lending_month')
tmp = get_target_summary(df_sample.query("channel_id=='227'"), target, 'lending_month').set_index('bins')
df_ksauc227_month_v3 = pd.concat([tmp, df_ksauc227_month_v3], axis=1)
print(df_ksauc227_month_v3)

df_ksauc227_set_v3 = model_ks_auc(df_sample.query("channel_id=='227'"), target, 'y_prob_v3', 'data_set')
tmp = get_target_summary(df_sample.query("channel_id=='227'"), target, 'data_set').set_index('bins')
df_ksauc227_set_v3 = pd.concat([tmp, df_ksauc227_set_v3], axis=1)
print(df_ksauc227_set_v3)


# In[617]:



# 227 30+客群
df_ksauc227_month_v3_30 = model_ks_auc(df_sample.query("channel_id=='227'&diff_days>30"), target, 'y_prob_v3', 'lending_month')
tmp = get_target_summary(df_sample.query("channel_id=='227'&diff_days>30"), target, 'lending_month').set_index('bins')
df_ksauc227_month_v3_30 = pd.concat([tmp, df_ksauc227_month_v3_30], axis=1)
print(df_ksauc227_month_v3_30)

df_ksauc227_set_v3_30 = model_ks_auc(df_sample.query("channel_id=='227'&diff_days>30"), target, 'y_prob_v3', 'data_set')
tmp = get_target_summary(df_sample.query("channel_id=='227'&diff_days>30"), target, 'data_set').set_index('bins')
df_ksauc227_set_v3_30 = pd.concat([tmp, df_ksauc227_set_v3_30], axis=1)
print(df_ksauc227_set_v3_30)


# In[618]:


# 模型变量重要性
# df_iv_by_month.drop(columns=['mean', 'std', 'cv'], inplace=True)
df_importance_month_v3 = feature_importance(lgb_model) 
df_importance_month_v3 = pd.merge(df_importance_month_v3, df_iv_by_month, how='inner', left_index=True,right_index=True)
df_importance_month_v3 = df_importance_month_v3.reset_index()
df_importance_month_v3 = df_importance_month_v3.rename(columns={'index':'varsname'})
df_importance_month_v3


# In[619]:


# 效果评估后模型变量重要性
df_importance_set_v3 = feature_importance(lgb_model) 
df_psi_iv = pd.merge(df_psi_by_set, df_iv_by_set, how='inner', left_index=True,right_index=True, suffixes=('_psi', '_iv'))
df_importance_set_v3 = pd.merge(df_importance_set_v3, df_psi_iv, how='inner', left_index=True,right_index=True)
df_importance_set_v3['iv的变化幅度'] = df_importance_set_v3['3_oot_iv']/df_importance_set_v3['1_train_iv'] - 1
df_importance_set_v3.drop(columns=['1_train_psi'], inplace=True)
df_importance_set_v3 = df_importance_set_v3.reset_index()
df_importance_set_v3 = df_importance_set_v3.rename(columns={'index':'varsname'})
df_importance_set_v3


# In[620]:



# 效果评估后保存模型
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
save_model_as_pkl(lgb_model, result_path + f'{task_name}_v3_{timestamp}.pkl')
save_model_as_bin(lgb_model, result_path + f'{task_name}_v3_{timestamp}.bin')
print(f"模型保存完成！：{timestamp}")
print(result_path + f'{task_name}_v3_{timestamp}.pkl')
print(result_path + f'{task_name}_v3_{timestamp}.bin')

with pd.ExcelWriter(result_path + f'5_模型优化_{task_name}_v3_{timestamp}.xlsx') as writer:
    df_importance_month_v3.to_excel(writer, sheet_name='df_importance_month_v3')
    df_importance_set_v3.to_excel(writer, sheet_name='df_importance_set_v3')
    df_ks_auc_month_v3.to_excel(writer, sheet_name='df_ks_auc_month_v3')
    df_ks_auc_set_v3.to_excel(writer, sheet_name='df_ks_auc_set_v3')
    df_ks_auc_month_v3_30.to_excel(writer, sheet_name='df_ks_auc_month_v3_30')
    df_ks_auc_set_v3_30.to_excel(writer, sheet_name='df_ks_auc_set_v3_30')
    df_ksauc227_month_v3.to_excel(writer, sheet_name='df_ksauc227_month_v3')
    df_ksauc227_month_v3_30.to_excel(writer, sheet_name='df_ksauc227_month_v3_30')
    df_ksauc227_set_v3.to_excel(writer, sheet_name='df_ksauc227_set_v3')
    df_ksauc227_set_v3_30.to_excel(writer, sheet_name='df_ksauc227_set_v3_30')        
print("数据存储完成！{timestamp}")
print(result_path + f'5_模型优化_{task_name}_v3_{timestamp}.xlsx')


# ### 5.2.3增量学习

# In[317]:


# ### 优化调参2
# opt_params = {}
# opt_params['boosting'] = 'gbdt'
# opt_params['objective'] = 'binary'
# opt_params['metric'] = 'auc'
# opt_params['bagging_freq'] = 1
# opt_params['scale_pos_weight'] = 1 
# opt_params['seed'] = 1 
# opt_params['num_threads'] = -1 
# # 调参时设置成不用调参的参数
# opt_params['learning_rate'] = 0.1
# ## 正则参数，防止过拟合
# opt_params['bagging_fraction'] = 0.5587482897523352  
# opt_params['feature_fraction'] = 0.6029505806953829
# opt_params['lambda_l1'] = 0
# opt_params['lambda_l2'] = 300
# opt_params['early_stopping_rounds'] = 30

# # 调参后的参数需要变成整数型
# opt_params['num_leaves'] = 21
# opt_params['min_data_in_leaf'] = 69
# opt_params['max_depth'] = 3
# # 调参后的其他参
# opt_params['min_gain_to_split'] = 5.0


# In[318]:


# # 确定数据集参数后，训练模型
# X_train, X_test, y_train, y_test = train_test_split(df_sample.query("data_set!='3_oot'")[init_model_vars],
#                                                     df_sample.query("data_set!='3_oot'")[target],
#                                                     test_size=0.2, 
#                                                     random_state=22, 
#                                                     stratify=df_sample.query("data_set!='3_oot'")[target])
# print(X_train.shape, X_test.shape)

# df_sample.loc[X_train.index, 'data_set']='1_train'
# df_sample.loc[X_test.index, 'data_set']='2_test'


# In[319]:


# # 6，训练/保存/评估模型
# # 优化训练模型
# train_set = lgb.Dataset(X_train, label=y_train)
# valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
# lgb_model = lgb.train(opt_params, train_set, valid_sets=valid_set, num_boost_round=10000, init_model=init_model)


# In[320]:


# # 优化后评估模型效果
# df_sample['y_prob_v3'] = lgb_model.predict(df_sample[X_train.columns], num_iteration=lgb_model.best_iteration)


# In[321]:


# # 优化后评估模型效果
# df_ks_auc_month_v3 = model_ks_auc(df_sample, target, 'y_prob_v3', 'lending_month')
# tmp = get_target_summary(df_sample, target, 'lending_month').set_index('bins')
# df_ks_auc_month_v3 = pd.concat([tmp, df_ks_auc_month_v3], axis=1)
# print(df_ks_auc_month_v3)


# df_ks_auc_set_v3 = model_ks_auc(df_sample, target, 'y_prob_v3', 'data_set')
# tmp = get_target_summary(df_sample, target, 'data_set').set_index('bins')
# df_ks_auc_set_v3 = pd.concat([tmp, df_ks_auc_set_v3], axis=1)
# print(df_ks_auc_set_v3)


# In[322]:


# # 优化后评估模型效果-30+客群
# df_ks_auc_month_30_v3 = model_ks_auc(df_sample.query("diff_days>30"), target, 'y_prob_v3', 'lending_month')
# tmp = get_target_summary(df_sample.query("diff_days>30"), target, 'lending_month').set_index('bins')
# df_ks_auc_month_30_v3 = pd.concat([tmp, df_ks_auc_month_30_v3], axis=1)
# print(df_ks_auc_month_30_v3)


# df_ks_auc_set_30_v3 = model_ks_auc(df_sample.query("diff_days>30"), target, 'y_prob_v3', 'data_set')
# tmp = get_target_summary(df_sample.query("diff_days>30"), target, 'data_set').set_index('bins')
# df_ks_auc_set_30_v3 = pd.concat([tmp, df_ks_auc_set_30_v3], axis=1)
# print(df_ks_auc_set_30_v3)


# In[323]:


# # 模型变量重要性
# df_importance_v3 = feature_importance(lgb_model) 
# df_psi_iv = pd.merge(df_psi_by_set, df_iv_by_set, how='inner', left_index=True,right_index=True, suffixes=('_psi', '_iv'))
# df_importance_v3 = pd.merge(df_importance_v3, df_psi_iv, how='inner', left_index=True,right_index=True)
# df_importance_v3['iv的变化幅度'] = df_importance_v3['3_oot_iv']/df_importance_v3['1_train_iv'] - 1
# df_importance_v3.drop(columns=['1_train_psi'], inplace=True)
# df_importance_v3 = df_importance_v3.reset_index()
# df_importance_v3 = df_importance_v3.rename(columns={'index':'varsname'})
# df_importance_v3.head(10)


# In[324]:


# # 效果评估后保存模型
# result_path = f'{directory}/'
# timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
# save_model_as_pkl(lgb_model, result_path + f'{task_name}_{timestamp}.pkl')
# save_model_as_bin(lgb_model, result_path + f'{task_name}_{timestamp}.bin')
# print(f"模型保存完成！：{timestamp}")


# ## 5.3 模型效果对比

# ### 5.3.1数据处理

# In[621]:


print(df_sample['lending_time'].min(), df_sample['lending_time'].max())


# In[9]:


varsname_v5


# In[11]:


usecols= ['order_no','channel_id', 'lending_time','lending_month', 'mob',          'maxdpd', 'fpd', 'fpd10', 'fpd30', 'mob4dpd30', 'diff_days'] + varsname_v5
df_model_vars = pd.read_csv(result_path+'全渠道实时提现行为模型_原始建模数据集_241218.csv',usecols=usecols)
df_model_vars.info(show_counts=True)
df_model_vars.head()


# In[12]:


df_model_vars[varsname_v5].describe().T


# In[13]:


df_model_vars[varsname_v5] = df_model_vars[varsname_v5].replace(-1, np.nan)
gc.collect()


# In[14]:


df_model_vars[varsname_v5].describe().T


# In[15]:


# 衍生Y标签
print(df_model_vars['fpd'].min(), df_model_vars['fpd'].max())
print(df_model_vars['fpd30'].min(), df_model_vars['fpd30'].max())


# In[16]:


# 衍生Y标签
df_model_vars['fpd10'] = df_model_vars['fpd'].apply(lambda x: 1 if x>10 else 0)
df_model_vars['fpd20'] = df_model_vars['fpd'].apply(lambda x: 1 if x>20 else 0)


# In[19]:


df_model_vars.groupby(["lending_time",'fpd10'])['order_no'].count().unstack()


# In[20]:


df_model_vars.loc[df_model_vars.query("lending_time>='2024-10-16'").index, 'fpd30'] = -1
df_model_vars.loc[df_model_vars.query("lending_time>='2024-10-26'").index, 'fpd20'] = -1


# In[21]:


df_model_vars.drop(index=df_model_vars.query("lending_time>='2024-11-06'").index, inplace=True)


# In[22]:


df_model_vars = df_model_vars.reset_index(drop=True)


# In[23]:


# 添加客群标签
def diff_days_(x):
    if x<=30:
        days = 'T30-'
    elif x>30:
        days = 'T30+'
    else:
        days = np.nan
    return days

df_model_vars['客群'] = df_model_vars['diff_days'].apply(diff_days_)


# In[24]:


result_path


# In[25]:


df_model_vars.info(show_counts=True)


# In[10]:


lgb_model= load_model_from_pkl(result_path + '全渠道实时提现行为模型fpd30_20241218142433.pkl')
varsname_v5 = lgb_model.feature_name()


# In[26]:


# 第一次模型打分 
lgb_model= load_model_from_pkl(result_path + '全渠道实时提现行为模型fpd30_20241218142433.pkl')
# varsname_v5 = lgb_model.feature_name
df_model_vars['y_prob_1'] = lgb_model.predict(df_model_vars[varsname_v5],
                                               num_iteration=lgb_model.best_iteration)


# In[720]:


result_path


# In[27]:


# 第二次模型打分  ./result/全渠道实时提现行为模型fpd30/全渠道实时提现行为模型fpd30_opt_20241218153521.pkl
lgb_model= load_model_from_pkl('./result/全渠道实时提现行为模型fpd30/全渠道实时提现行为模型fpd30_20241218153521.pkl')
df_model_vars['y_prob_2'] = lgb_model.predict(df_model_vars[varsname_v5],
                                               num_iteration=lgb_model.best_iteration)


# In[28]:


# 最终模型打分
lgb_model= load_model_from_pkl(result_path + '全渠道实时提现行为模型fpd30_v3_20241218154942.pkl')
varsname_v6 = lgb_model.feature_name()
df_model_vars['y_prob'] = lgb_model.predict(df_model_vars[varsname_v6],
                                               num_iteration=lgb_model.best_iteration)


# In[ ]:





# In[35]:


def realtime_or_offline(x):
    if x in (213,227,231,240,241,245):
        model_type = 'realtime'
    elif x in (209, 229, 233, 235, 236, 244, 226, 234, 246, 247):
        model_type = 'offline'
    else :
        model_type = 'other'
    return model_type


# In[36]:


# df_model = df_model_vars.query("diff_days>30").reset_index(drop=True)
df_model['model_type'] = df_model['channel_id'].apply(realtime_or_offline)


# In[42]:


tmp1 = model_ks_auc(df_model.query("model_type=='realtime'&fpd30>=0"), 'fpd30', 'y_prob', 'lending_month')
tmp2 = get_target_summary(df_model.query("model_type=='realtime'&fpd30>=0"), 'fpd30', 'lending_month').set_index('bins')
pd.concat([tmp2, tmp1], axis=1)


# In[41]:


tmp1 = model_ks_auc(df_model.query("model_type=='offline'&fpd30>=0"), 'fpd30', 'y_prob', 'lending_month')
tmp2 = get_target_summary(df_model.query("model_type=='offline'&fpd30>=0"), 'fpd30', 'lending_month').set_index('bins')
pd.concat([tmp2, tmp1], axis=1)


# In[636]:


# 其他提现模型数据 
sql="""
select t2.*
from
(
    select order_no 
    from znzz_fintech_ads.dm_f_lxl_test_order_Y_target as t 
    where dt=date_sub(current_date(), 2) 
      and lending_time>='2024-07-21'
      and lending_time<='2024-11-05'
) as t1 
inner join znzz_fintech_ads.dm_f_cnn_test_tx_allchan_mxf_model as t2 on t1.order_no=t2.order_no
;
"""
df_tx_bj = get_data(sql)
df_tx_bj.info(show_counts=True)
df_tx_bj.head() 


# In[637]:


df_tx_bj['t_beha3_fpd'] = pd.to_numeric(df_tx_bj['t_beha3_fpd'])
df_tx_bj['t_beha3_mob4'] = pd.to_numeric(df_tx_bj['t_beha3_mob4'])


# In[638]:


selected_cols = df_tx_bj.columns.to_list()[3:]
df_tx_bj[selected_cols].describe().T


# In[639]:


df_tx_bj_copy = df_tx_bj.copy()


# In[640]:


result_path


# In[641]:


df_tx_bj.to_csv(result_path + '全渠道其他提现模型分数_241218.csv')
print(result_path + '全渠道其他提现模型分数_241218.csv')


# In[642]:


# 好分数转为坏分数
for i, col in enumerate(selected_cols):
    print(f'第{i}个变量：{col}')
    df_tx_bj[col] = 1 - df_tx_bj[col]


# In[725]:


print(df_model_vars.shape, df_tx_bj.shape)


# In[726]:


df_evalue = pd.merge(df_model_vars, df_tx_bj, how='left',on=['order_no'])
print(df_evalue.shape, df_evalue['order_no'].nunique())


# In[727]:


df_evalue.info(show_counts=True)


# In[728]:


df_evalue.drop(columns=['apply_date','channel_id_y'], inplace=True)
df_evalue.rename(columns={'channel_id_x':'channel_id'},inplace=True)


# ### 5.3.2 不同标签下对比其他提现模型的效果

# In[740]:


# 小数转换百分数
def to_percentage(x):
    if isinstance(x, (float)) and pd.notnull(x):
        return f"{x * 100:.2f}%"
    return x

def float_format(x):
    if isinstance(x, (float)) and pd.notnull(x):
        return '%.3f' %x
    return x

def cal_data_item(df, label_col, score_col, percentile=0.95):
    from sklearn.metrics import auc
    fpr, tpr, _ = roc_curve(df[label_col], df[score_col], pos_label=1)
    auc_value = auc(fpr, tpr)
    ks_value = max(abs(tpr - fpr))
    badrate = df[label_col].mean()
    
    if percentile>=0.90:#概率分数是坏分数，计算最坏5%客群的lift
        pct_n = df[score_col].quantile(percentile)
        pct_n_badrate = df[df[score_col]>pct_n][label_col].mean()
    elif percentile<=0.10:#概率分数是好分数，计算最坏5%客群的lift
        pct_n = df[score_col].quantile(percentile)
        pct_n_badrate = df[df[score_col]<pct_n][label_col].mean()
    else:
        print("请根据概率分数是好分数还是坏分数，决定分位数的位置")
    
    if badrate>0 and pct_n_badrate>0:
        lift_n = pct_n_badrate/badrate
    else:
        lift_n = np.nan
    return pd.Series({'KS': ks_value, 'AUC': auc_value, 'top5lift':lift_n})


# 计算KS
def cal_ks_auc(df, groupkeys, model_score_label_dict):
    # groupkeys: 分组字段
    # model_score_label_dict: value: score_list: 得分字段列表, key: label_list: 标签字段列表
    # df: 有标签和得分的数据框
    # 输出KS、AUC
    
    ks_auc_result = pd.DataFrame()
    if not isinstance(groupkeys, list):
        groupkeys = [groupkeys]
    for label_, score_list in model_score_label_dict.items():
        data1 = df[df[label_]>=0]
        total_bad = data1.groupby(groupkeys)[label_].agg(total=lambda x: len(x), 
                                                        bad=lambda x: x.sum(), 
                                                        badrate=lambda x: x.mean())
        total_bad['badrate'] = total_bad['badrate'].apply(to_percentage)
        total_bad.insert(loc=0, column='target_type', value=label_,                          allow_duplicates=False)
        
        ks_auc_list = []
        for score_ in score_list:
            data = df[(df[label_]>=0) & (df[score_].notnull())]
            tmp_ks_auc = data.groupby(groupkeys).apply(cal_data_item,                                                        label_col=label_,                                                        score_col=score_)
            tmp_ks_auc = tmp_ks_auc.rename(columns={'KS':f'KS_{score_}',
                                                    'AUC':f'AUC_{score_}',
                                                    'top5lift':f'top5lift_{score_}'})
            ks_auc_list.append(tmp_ks_auc)
        df_ks_auc = pd.concat(ks_auc_list, axis=1)
        ks_columns = [col for col in df_ks_auc.columns if 'KS' in col]
        AUC_columns = [col for col in df_ks_auc.columns if 'AUC' in col]
        lift_columns = [col for col in df_ks_auc.columns if 'top5lift' in col]
        df_ks_auc[ks_columns] = df_ks_auc[ks_columns].applymap(float_format)
        df_ks_auc[AUC_columns] = df_ks_auc[AUC_columns].applymap(float_format)
        df_ks_auc[lift_columns] = df_ks_auc[lift_columns].applymap(float_format)        
        df_ks_auc = df_ks_auc[ks_columns + AUC_columns + lift_columns]
        
        df_ks_auc = pd.concat([total_bad, df_ks_auc], axis=1)
        df_ks_auc = df_ks_auc.reset_index()
        
        ks_auc_result = pd.concat([ks_auc_result, df_ks_auc], axis=0, ignore_index=True)
        print(f'==============完成标签：{label_}===============')
    
    return ks_auc_result


# In[731]:


colsname = df_evalue.columns[34:].to_list() + varsname_v5

print(colsname)
target_list = ['fpd10', 'fpd20', 'fpd30']
labels_models_dict = {target: colsname for target in target_list}
print(labels_models_dict)


# In[752]:


colsname.append('order_no')
print(colsname)


# In[753]:


df_evalue[colsname].to_csv(result_path + '全渠道其他提现模型分数_241220_evalue.csv')
print(result_path + '全渠道其他提现模型分数_241220_evalue.csv')


# In[732]:


# groupkeys1 = ['lending_month_new']
# df_ksauc_all_v1 = cal_ks_auc(df_evalue, groupkeys1, labels_models_dict)
# df_ksauc_all_v1.insert(loc=(len(groupkeys1)), column='渠道', value='全渠道', allow_duplicates=False)
# df_ksauc_all_v1.insert(loc=(len(groupkeys1)+1), column='客群', value='全体', allow_duplicates=False)
# df_ksauc_all_v1.head()


# In[733]:


def is_channel(x):
    if x==227:
        channel='227渠道'
    elif x in (209, 213, 226, 229, 231, 233, 234, 235, 236):
        channel='金科其他渠道'
    elif x==1:
        channel='桔子商城'
    else:
        channel='非金科渠道'
    return channel
df_evalue['渠道'] = df_evalue['channel_id'].apply(is_channel)
df_evalue['渠道'].value_counts()


# In[735]:


df_evalue['lending_month_new'] = df_evalue['lending_month']
df_evalue.loc[df_evalue.query("lending_time>='2024-09-01' & lending_time<='2024-09-20'").index, 'lending_month_new']='2024-09_1train'
df_evalue.loc[df_evalue.query("lending_time>='2024-09-21' & lending_time<='2024-09-30'").index, 'lending_month_new']='2024-09_3oot'


# In[741]:



groupkeys2 = ['lending_month_new', '渠道']
df_ksauc_all_v2 = cal_ks_auc(df_evalue, groupkeys2, labels_models_dict)
df_ksauc_all_v2.insert(loc=(len(groupkeys2)), column='客群', value='全体', allow_duplicates=False)
df_ksauc_all_v2.head()


# In[749]:



# groupkeys3 = ['lending_month_new', '客群']
# df_ksauc_all_v3 = cal_ks_auc(df_evalue, groupkeys3, labels_models_dict)
# df_ksauc_all_v3.insert(loc=(len(groupkeys3)-1), column='渠道', value='全渠道', allow_duplicates=False)
# df_ksauc_all_v3.head()


# In[750]:



groupkeys4 = ['lending_month_new', '渠道', '客群']
df_ksauc_all_v4 = cal_ks_auc(df_evalue, groupkeys4, labels_models_dict)
df_ksauc_all_v4.head()


# In[1]:





# In[767]:


category_cols = ['pudao_34','prob_fpd30_v1','br_v3_fpd30_score','ruizhi_6','hengpu_4','aliyun_5','pudao_68',
                 'hengpu_5','duxiaoman_6','pudao_20','score_fpd6_v1','score_fpd10_v2','score_fpd10_v1']


# In[768]:


# 判断每行是否所有字段都不为空
not_null_mask = df_evalue[category_cols].notnull().all(axis=1)

# 判断每行是否所有字段都为空
all_null_mask = df_evalue[category_cols].isnull().all(axis=1)

# 初始化分类字段，默认先设置为2（对应其余情况）
df_evalue['category'] = 2

# 将所有字段都不为空的行对应的分类字段设置为1
df_evalue.loc[not_null_mask, 'category'] = 1

# 将所有字段都为空的行对应的分类字段设置为0
df_evalue.loc[all_null_mask, 'category'] = 0


# In[769]:


target_list = ['fpd10', 'fpd20', 'fpd30']
labels_models_dict_2 = {target: ['y_prob'] for target in target_list}
print(labels_models_dict_2)


# In[770]:


groupkeys5 = ['lending_month_new', '渠道', '客群', 'category']
df_ksauc_all_v5 = cal_ks_auc(df_evalue, groupkeys5, labels_models_dict_2)
df_ksauc_all_v5.head()


# In[771]:



groupkeys6 = ['lending_month_new', '渠道', 'category']
df_ksauc_all_v6 = cal_ks_auc(df_evalue, groupkeys6, labels_models_dict_2)
df_ksauc_all_v6.insert(loc=(len(groupkeys6)), column='客群', value='全体', allow_duplicates=False)
df_ksauc_all_v6.head()


# In[772]:


df_ksauc_all_1 = pd.concat([df_ksauc_all_v2,df_ksauc_all_v4], axis=0)
df_ksauc_all_2 = pd.concat([df_ksauc_all_v5,df_ksauc_all_v6], axis=0)


# In[773]:


result_path


# In[774]:



timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

with pd.ExcelWriter(result_path + f'6_模型对比分析_{task_name}_{timestamp}.xlsx') as writer:
    df_ksauc_all_1.to_excel(writer, sheet_name='df_ksauc_all_1')
    df_ksauc_all_2.to_excel(writer, sheet_name='df_ksauc_all_2')
print(f"数据存储完成！{timestamp}")
print(result_path + f'6_模型对比分析_{task_name}_{timestamp}.xlsx')


# In[ ]:


target_list = ['fpd10', 'fpd20', 'fpd30']
labels_models_dict_3 = {target: ['y_prob'] for target in target_list}
print(labels_models_dict_3)


# In[ ]:


cal_ks_auc(df_evalue, groupkeys6, labels_models_dict_2)


# ### 腾讯反欺诈

# In[656]:


usecols = ['order_no','tengxun_cash_score','tengxun_credit_score']
df_tengxun = pd.read_csv(result_path+'全渠道实时提现行为模型_原始建模数据集_241218.csv',usecols=usecols)


# In[657]:


df_tengxun = pd.merge(df_model_vars, df_tengxun, how='left',on=['order_no'])
df_tengxun.info(show_counts=True)


# In[658]:


df_tengxun = df_tengxun.query('channel_id!=1')
df_tengxun = df_tengxun.reset_index(drop=True)


# In[659]:


df_tengxun['渠道'] = df_tengxun['channel_id'].apply(is_channel)
df_tengxun['渠道'].value_counts()


# In[661]:


df_tengxun['客群'] = df_tengxun['diff_days'].apply(lambda x: 'T30+' if x>30  else 'T0' if x==0 else 'T30-')
df_tengxun['客群'].value_counts()


# In[663]:


colsname = ['tengxun_cash_score', 'tengxun_credit_score']
print(colsname)
target_list = ['fpd10', 'fpd30']
labels_models_dict1 = {target: colsname for target in target_list}
print(labels_models_dict1)


# In[667]:


df_ksauc_tx1 = cal_ks_auc(df_tengxun, ['lending_month'], labels_models_dict1)
df_ksauc_tx2 = cal_ks_auc(df_tengxun, ['lending_month', '渠道'], labels_models_dict1)


# In[671]:


df_tengxun.query("tengxun_credit_score==tengxun_credit_score & fpd10>=0").groupby(['lending_time','fpd10'])['order_no'].count().unstack()


# In[668]:



timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

with pd.ExcelWriter(result_path + f'腾讯_{task_name}_{timestamp}.xlsx') as writer:
    df_ksauc_tx1.to_excel(writer, sheet_name='df_ksauc_tx1')
    df_ksauc_tx2.to_excel(writer, sheet_name='df_ksauc_tx2')
print(f"数据存储完成！{timestamp}")
print(result_path + f'腾讯_{task_name}_{timestamp}.xlsx')


# In[672]:


colsname = ['y_prob']
print(colsname)
target_list = ['fpd10', 'fpd20', 'fpd30']
labels_models_dict2 = {target: ['y_prob'] for target in target_list}
print(labels_models_dict2)


# In[673]:


df_evalue['lending_month_new'] = df_evalue['lending_month']
df_evalue.loc[df_evalue.query("lending_time>='2024-09-01' & lending_time<='2024-09-20'").index, 'lending_month_new']='2024-09_train'
df_evalue.loc[df_evalue.query("lending_time>='2024-09-21' & lending_time<='2024-09-30'").index, 'lending_month_new']='2024-09_oot'


# In[679]:


# # groupkeys2 = ['lending_month_new', '渠道']
# cal_ks_auc(df_evalue, ['lending_month_new', '渠道1'], labels_models_dict2)


# In[ ]:



groupkeys2 = ['lending_month_new', '渠道']
df_ksauc_all_v2 = cal_ks_auc(df_evalue, groupkeys2, labels_models_dict)
df_ksauc_all_v2.insert(loc=(len(groupkeys2)), column='客群', value='全体', allow_duplicates=False)
df_ksauc_all_v2.head()


# # 6. 评分分布

# In[374]:


result_path


# In[375]:


df_sample.to_csv(result_path + r'全渠道实时提现行为模型fpd30样本.csv',index=False)


# In[376]:


score = 'y_prob_v3'


# In[377]:


df_sample['lending_month'].value_counts()


# In[378]:


c = toad.transform.Combiner()
c.fit(df_sample.query("lending_month=='2024-07'")[[score, target]], y=target, method='quantile', n_bins=20) 
df_sample['score_bins'] = c.transform(df_sample[score], labels=True)


# In[379]:


df_sample['score_bins'].head()


# In[380]:


score_psi_by_month = cal_psi_by_month(df_sample, df_sample.query("lending_month=='2024-08'"), 
                                                [score], 'lending_month_new', c, return_frame = False)
print(score_psi_by_month)

# score_psi_by_dataset = cal_psi_by_month(df_sample, df_sample.query("lending_month=='2024-07'"), 
#                                                 [score], 'data_set', c, return_frame = False)
# print(score_psi_by_dataset)


# In[381]:


def get_model_psi(df, cols, month_col, combiner):
    # 获取所有唯一的月份
    months = sorted(list(set(df[month_col])))
    # 初始化一个空的 DataFrame 来存储 PSI 值
    psi_matrix = pd.DataFrame(index=months, columns=months, dtype=float)
    # 循环计算每个月份与其他月份之间的PSI
    for i, month_i in enumerate(months):
        for j, month_j in enumerate(months):
            if i != j:
                # 从原始数据集中提取特定月份的数据
                df_actual_i = df[df[month_col] == month_i]
                df_expect_j = df[df[month_col] == month_j]
                # 调用函数计算PSI
                psi_ = toad.metrics.PSI(df_actual_i[cols], df_expect_j[cols], 
                                        combiner = combiner, return_frame = False)
                # 将结果存入矩阵
                psi_matrix.loc[month_i, month_j] = psi_
            else:
                # 对角线上的值设为 NaN 或 0，表示同一月份的 PSI
                psi_matrix.loc[month_i, month_j] = 0.0
    
    return psi_matrix


# In[382]:


df_psi_matrix = get_model_psi(df_sample, score, 'lending_month', c)

# 打印最终的 PSI 矩阵
print(df_psi_matrix)


# In[383]:


score_group_by_dataset = calculate_vars_distribute(df_sample, ['score_bins'], target, 'data_set') 
score_group_by_dataset = score_group_by_dataset[['groupvars', 'bins', 'total', 'bad',
                                                 'good', 'bad_rate', 'bad_rate_cum',
                                                 'total_pct_cum', 'ks_bin', 'lift', 'lift_cum']]


# In[384]:


timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

with pd.ExcelWriter(result_path + f'6_评分分布_{task_name}_{timestamp}.xlsx') as writer:
    score_psi_by_month.to_excel(writer, sheet_name='score_psi_by_month')
#     score_psi_by_dataset.to_excel(writer, sheet_name='score_psi_by_dataset')
#     df_score_group_by_month.to_excel(writer, sheet_name='df_score_group_by_month')
#     score_group_by_month.to_excel(writer, sheet_name='score_group_by_month')
#     df_score_group_by_dataset.to_excel(writer, sheet_name='df_score_group_by_dataset')
    score_group_by_dataset.to_excel(writer, sheet_name='score_group_by_dataset')
#     score_group_by_dataset_1.to_excel(writer, sheet_name='score_group_by_dataset_1')
print(f"数据存储完成！:{timestamp}")
print(result_path + f'6_评分分布_{task_name}_{timestamp}.xlsx')




#==============================================================================
# File: 03提现全渠道无成本子分融合模型fpd30标签_2409_2411.py
#==============================================================================

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import toad
import lightgbm as lgb
# import shap
import hyperopt 
from hyperopt import fmin, hp, Trials, tpe, rand, anneal, STATUS_OK, partial, space_eval
from hyperopt.early_stop import no_progress_loss
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
import joblib
import pickle
import time
from datetime import datetime
import os 
import gc
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_row',None)
pd.set_option('display.width',None)
pd.set_option('display.precision', 6)


# In[2]:


# 设置数据存储
task_name = '06_提现全渠道无成本子分融合模型fpd30标签_2410_2411'
timestamp = datetime.now().strftime('%Y%m%d')
directory = f'./result'
if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
result_path = f'{directory}/'


# # 函数定义

# In[3]:


# 获取数据
def get_data(sql):
    from odps import ODPS
    import time
    from datetime import datetime
    import multiprocessing
    
    # 获取cpu核的数量
    n_process = multiprocessing.cpu_count()
    # 输入账号密码
    conn= ODPS(username='liaoxilin', password='j02vYCxx')

    print('开始跑数：' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = time.time()
    # 执行脚本
    instance = conn.execute_sql(sql)
    # 输出执行结果
    with instance.open_reader() as reader:
        print('-------数据开始转换为DataFrame--------')
        data = reader.to_pandas(n_process=n_process) # 多核处理，避免单核处理

    end = time.time()
    print('结束跑数：' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("运行事件：{}秒".format(end-start))   

    return data

# 插入数据
def execute_sql(sql):
    from odps import ODPS
    import time
    from datetime import datetime
    # 输入账号密码
    conn= ODPS(username='liaoxilin', password='j02vYCxx')
    
    print('开始跑数' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = time.time()
    # 执行脚本
    conn.execute_sql(sql)
    end = time.time()
    print('结束跑数' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("运行事件：{}秒".format(end-start))   


# # 0. 数据读取

# In[4]:


df_sample_dict = {}


# In[5]:



# 计算今天的时间
from datetime import datetime, timedelta, date

today = datetime.now().strftime('%Y-%m-%d')
print(today)

this_day =datetime.strptime('2024-12-27', '%Y-%m-%d')
end_day = datetime.strptime('2024-08-01', '%Y-%m-%d')

while this_day >= end_day:
    run_day = this_day.strftime('%Y-%m-%d')
    sql = f'''
select 
t.order_no,
t.user_id,
t.id_no_des,
t.channel_id,
t.apply_date,
t.lending_time,
t.order_no_auth,
t.apply_date_auth,
t.diff_days,
t.fpd,
t.spd,
t.tpd,
t.fpd0,
t.fpd1,
t.fpd3,
t.fpd7,
t.fpd10,
t.fpd15,
t.fpd20,
t.fpd30,
t.mob4dpd30
,all_a_app_free_fpd30_202502_s
,all_a_bhdj_fpd10_v1_p
,all_a_br_derived_fpd30_202408_g_p
,all_a_br_derived_v1_mob4dpd30_202502_st_p
,all_a_br_derived_v2_fpd30_202411_g_p
,all_a_br_derived_v3_fpd30_202412_g_p
,all_a_dz_derived_v1_fpd30_202502_g_p
,all_a_dz_derived_v2_fpd30_202502_g_p
,HLV_D_HOLO_certNo_variableCode_dpd30_4m_BD0002_standard
,HLV_D_HOLO_certNo_variableCode_dpd30_6m_BD0001_standard
,HLV_D_HOLO_certNo_variableCode_standard_BD003
,HLV_D_HOLO_jk_certNo_fpd1_score
,HLV_D_HOLO_jk_certNo_score_fpd30_v1
,HLV_D_HOLO_jk_certNo_score_fpd7_v1
,HLV_D_HOLO_jk_certNo_varCode_standard_BD0004
,ypy_bhxz_a_fpd30_v1_prob_good
,score_fpd0_v1	
,score_fpd6_v1	
,score_fpd10_v1	
,score_fpd10_v2	
,score_fpd30_v1
,duxiaoman_6
,hengpu_4
,aliyun_5
,baihang_28
,pudao_34
,feicuifen
,wanxiangfen
,pudao_20
,pudao_68
,ruizhi_6
,hengpu_5
,pudao_21
,bh_alic002_1
,bh_alic002_2
,bh_alic002_3
,bh_alic002_4
,t_br_fpd
,t_br_mob4
,t_br2_fpd
,t_br2_mob4
,t_beha3_fpd
,t_beha3_mob4
,dz_fpd
,xz_fpd
from 
    (
    select * 
    from znzz_fintech_ads.dm_f_lxl_test_order_Y_target_2502 as t 
    where dt=date_sub(current_date(), 1) 
      and apply_date='{run_day}'
    ) as t 
------------------离线模型子分-----------------
--贷中离线子分融合模型fpd30标签_分数
left join 
    (
    select order_no, variable_value as HLV_D_HOLO_jk_certNo_varCode_standard_BD0004
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'HLV_D_HOLO_jk_certNo_varCode_standard_BD0004'
      and variable_value is not null 
    ) as t2 on t.order_no=t2.order_no
--fpd30离线子分
left join 
    (
    select order_no, variable_value as HLV_D_HOLO_jk_certNo_score_fpd30_v1
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'HLV_D_HOLO_jk_certNo_score_fpd30_v1'
      and variable_value is not null 
    ) as t3 on t.order_no=t3.order_no
--fpd7离线子分
left join 
    (
    select order_no, variable_value as HLV_D_HOLO_jk_certNo_score_fpd7_v1
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'HLV_D_HOLO_jk_certNo_score_fpd7_v1'
      and variable_value is not null 
    ) as t4 on t.order_no=t4.order_no
--授信全渠道行为特征模型fpd1标签_标准分
left join 
    (
    select order_no, variable_value as HLV_D_HOLO_jk_certNo_fpd1_score
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'HLV_D_HOLO_jk_certNo_fpd1_score'
      and variable_value is not null 
    ) as t6 on t.order_no=t6.order_no
--贷中截面风险dpd30_6m模型
left join 
    (
    select order_no, variable_value as HLV_D_HOLO_certNo_variableCode_dpd30_6m_BD0001_standard
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'HLV_D_HOLO_certNo_variableCode_dpd30_6m_BD0001_standard'
      and variable_value is not null 
    ) as t7 on t.order_no=t7.order_no
--贷中提现风险dpd30_4m模型
left join 
    (
    select order_no, variable_value as HLV_D_HOLO_certNo_variableCode_dpd30_4m_BD0002_standard
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'HLV_D_HOLO_certNo_variableCode_dpd30_4m_BD0002_standard'
      and variable_value is not null 
    ) as t8 on t.order_no=t8.order_no
--贷中行为模型fpd30标签_分数
left join 
    (
    select order_no, variable_value as HLV_D_HOLO_certNo_variableCode_standard_BD003
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'HLV_D_HOLO_certNo_variableCode_standard_BD003'
      and variable_value is not null 
    ) as t9 on t.order_no=t9.order_no 

-- 人行离线子分
left join 
    (
    select 
     id_no_des
    ,score_fpd0_v1	
    ,score_fpd6_v1	
    ,score_fpd10_v1	
    ,score_fpd10_v2	
    ,score_fpd30_v1
    from znzz_fintech_ads.llji_yhx_ascore_model_all_score_flow_fd
    where dt = date_sub('{run_day}', 1)
    ) as t13 on t.id_no_des=t13.id_no_des

------------------三方缓存数据-----------------    
--近100天缓存三方评分数据
left join 
    (
    select 
     id_no_des
    ,duxiaoman_6
    ,hengpu_4
    ,aliyun_5
    ,baihang_28
    ,pudao_34
    ,feicuifen
    ,wanxiangfen
    ,pudao_20
    ,pudao_68
    ,ruizhi_6
    ,hengpu_5
    ,pudao_21
    ,bh_alic002_1
    ,bh_alic002_2
    ,bh_alic002_3
    ,bh_alic002_4
    from znzz_fintech_ads.lxl_r100_three_score_data as t 
    where dt=date_sub('{run_day}', 1)
    ) as t11 on t.id_no_des=t11.id_no_des

------------------无成本或者低成本的实时数据-----------------   
--北京团队子分
left join 
    (
    select
    order_no,
    t_br_fpd,
    t_br_mob4,
    t_br2_fpd,
    t_br2_mob4,
    t_beha3_fpd,
    t_beha3_mob4,
    dz_fpd,
    xz_fpd
    from znzz_fintech_ads.dm_f_cnn_test_tx_allchan_mxf_model as t 
    where apply_date='{run_day}'
    ) as t12 on t.order_no=t12.order_no
  
--百融子分
left join 
    (
    select order_no, variable_value as all_a_br_derived_fpd30_202408_g_p
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'all_a_br_derived_fpd30_202408_g_p'
      and variable_value is not null 
    ) as t14 on t.order_no=t14.order_no 
--百融子分v1
left join 
    (
    select order_no, variable_value as all_a_br_derived_v1_mob4dpd30_202502_st_p
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'all_a_br_derived_v1_mob4dpd30_202502_st_p'
      and variable_value is not null 
    ) as t15 on t.order_no=t15.order_no     
--百融子分v2
left join 
    (
    select order_no, variable_value as all_a_br_derived_v2_fpd30_202411_g_p
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'all_a_br_derived_v2_fpd30_202411_g_p'
      and variable_value is not null 
    ) as t16 on t.order_no=t16.order_no     
--百融子分v3
left join 
    (
    select order_no, variable_value as all_a_br_derived_v3_fpd30_202412_g_p
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'all_a_br_derived_v3_fpd30_202412_g_p'
      and variable_value is not null 
    ) as t17 on t.order_no=t17.order_no  
--洞侦子分
left join 
    (
    select order_no, variable_value as all_a_dz_derived_v1_fpd30_202502_g_p
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'all_a_dz_derived_v1_fpd30_202502_g_p'
      and variable_value is not null 
    ) as t18 on t.order_no=t18.order_no  
--洞侦子分
left join 
    (
    select order_no, variable_value as all_a_dz_derived_v2_fpd30_202502_g_p
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'all_a_dz_derived_v2_fpd30_202502_g_p'
      and variable_value is not null 
    ) as t19 on t.order_no=t19.order_no  
--授信百行洞见fpd30标签202502_好概率
left join 
    (
    select order_no, variable_value as all_a_bhdj_fpd10_v1_p
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'all_a_bhdj_fpd10_v1_p'
      and variable_value is not null 
    ) as t5 on t.order_no=t5.order_no    
--续侦子分
left join 
    (
    select order_no, variable_value as ypy_bhxz_a_fpd30_v1_prob_good
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'ypy_bhxz_a_fpd30_v1_prob_good'
      and variable_value is not null 
    ) as t20 on t.order_no=t20.order_no  
--授信全渠道无成本数据融合模型fp30标签_分数
left join 
    (
    select order_no, variable_value as all_a_app_free_fpd30_202502_s
    from znzz_fintech_ads.lending_model01_scores_vars as t 
    where lending_time='{run_day}'
      and variable_code = 'all_a_app_free_fpd30_202502_s'
      and variable_value is not null 
    ) as t1 on t.order_no=t1.order_no    
;
'''
    print(f'=========================={run_day}=============================')
    df_sample_dict[run_day] = get_data(sql)
    this_day = this_day - timedelta(days=1)


# In[7]:


data_time = pd.DataFrame({'run_day':list(df_sample_dict.keys())})
data_time['run_day'].value_counts().max()


# In[8]:


df_sample_ = pd.concat(df_sample_dict.values(), ignore_index=True)
df_sample_.info(show_counts=True)
df_sample_.head()


# In[9]:


print(df_sample_.shape, df_sample_['order_no'].nunique(), df_sample_['id_no_des'].nunique())


# In[10]:


print(df_sample_['apply_date'].min(), df_sample_['apply_date'].max())


# In[12]:


varsname = df_sample_.columns.to_list()[21:]

print(varsname[:10], varsname[-10:])
print("初始特征变量个数：",len(varsname))


# In[13]:


print(result_path)


# In[16]:


for i, col in enumerate(varsname):
    if df_sample_[col].dtype=='object':
        print(f"======第{i}个变量：{col}========")
        df_sample_[col] = pd.to_numeric(df_sample_[col], errors='coerce')


# In[18]:


pd.set_option('display.max_row',None)
df_sample_.groupby(['apply_date','fpd30'])['order_no'].count().unstack()


# In[19]:


df_sample_.to_csv(result_path + '提现全渠道无成本子分融合模型fpd30标签2410_2411.csv',index=False)
print(result_path + '提现全渠道无成本子分融合模型fpd30标签2410_2411.csv')


# In[ ]:





# In[20]:


# 设置数据存储
task_name = '06_提现全渠道无成本子分融合模型fpd30标签_2410_2411'
timestamp = datetime.now().strftime('%Y%m%d')
directory = f'./result'
if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
result_path = f'{directory}/'


# In[318]:


df_sample = df_sample_.query("fpd30>=0 & diff_days>30").reset_index(drop=True)


# In[319]:


df_sample.loc[df_sample.query("apply_date>='2024-08-01' & apply_date<='2024-08-31'").index, 'data_set']='3_oot1'
df_sample.loc[df_sample.query("apply_date>='2024-09-01' & apply_date<='2024-11-30'").index, 'data_set']='1_train'
df_sample.loc[df_sample.query("apply_date>='2024-12-01' & apply_date<='2024-12-27'").index, 'data_set']='3_oot2'
df_sample['apply_month'] = df_sample['apply_date'].str[0:7]


# In[320]:


target = 'fpd30'


# In[321]:


df_sample.to_csv(result_path + 'model_提现全渠道无成本子分融合模型fpd30标签2410_2411.csv',index=False)
print(result_path + 'model_提现全渠道无成本子分融合模型fpd30标签2410_2411.csv')


# In[6]:


target = 'fpd30'
df_sample = pd.read_csv(result_path + 'model_提现全渠道无成本子分融合模型fpd30标签2410_2411.csv')
print(result_path + 'model_提现全渠道无成本子分融合模型fpd30标签2410_2411.csv')


# In[7]:


df_sample.info(show_counts=True)


# In[9]:


varsname = df_sample.columns[21:66].to_list()
print(len(varsname))


# # 1. 样本概况

# In[33]:


def get_target_summary(df, target, groupby_col):
    """
    对 DataFrame 进行分组聚合，并添加一个汇总行。
    
    参数:
    - df: 待处理的 DataFrame
    - groupby_col: 用于分组的列名
    - agg_cols: 字典，键是列名，值是聚合函数名称（如 'count', 'sum', 'mean'）
    - new_col_name: 字典,键是旧列的名称，值是新列的名称
    
    返回:
    - 包含分组聚合结果和汇总行的新 DataFrame
    """
    # 使用 groupby 和 agg 进行分组和聚合
    grouped = df.groupby(groupby_col)[target].agg(total=lambda x: len(x), 
            bad=lambda x: x.sum(), 
            good=lambda x: (x== 0).sum(), 
            bad_rate=lambda x: x.mean()).reset_index()
    
    # 计算整个 DataFrame 的聚合统计量
    total_summary = df[target].agg(total=lambda x: len(x), 
            bad=lambda x: x.sum(), 
            good=lambda x: (x== 0).sum(), 
            bad_rate=lambda x: x.mean()).to_frame().T
    total_summary[groupby_col] = 'Total'
    
    # 将汇总行添加到分组结果中
    result = pd.concat([grouped, total_summary], ignore_index=True)
    result.rename(columns={groupby_col: 'bins'}, inplace=True)
    
    # 返回结果
    return result


# In[323]:


print(df_sample[target].value_counts())


# In[324]:


df_target_summary_month = get_target_summary(df_sample, target, 'apply_month')
print(df_target_summary_month)


# In[325]:


df_target_summary_set = get_target_summary(df_sample, target, 'data_set')
print(df_target_summary_set)


# In[326]:


task_name


# In[327]:



timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

with pd.ExcelWriter(result_path + f"1_样本概况_{task_name}_{timestamp}.xlsx") as writer:
    df_target_summary_month.to_excel(writer, sheet_name='df_target_summary_month')
    df_target_summary_set.to_excel(writer, sheet_name='df_target_summary_set')
#     df_target_summary.to_excel(writer, sheet_name='df_target_summary')
    
print(f"数据存储完成: {timestamp}")
print(result_path + f"1_样本概况_{task_name}_{timestamp}.xlsx")


# # 2.数据探索性分析

# In[328]:


# 2.1 变量分布
df_explor = toad.detect(df_sample[varsname])
df_explor


# ## 2.1缺失值处理

# In[329]:


for col in varsname:
    if df_sample[col].min()<0:
        print(f"--{col}--")
        df_sample.loc[df_sample[col]<0, col] = np.nan
gc.collect()


# In[10]:


for col in varsname:
    if df_sample[col].min()<0:
        print(f"--{col}--")
        df_sample.loc[df_sample[col]<0, col] = np.nan


# In[331]:


# 2.1 变量分布
df_explor_v1 = toad.detect(df_sample[varsname])
df_explor_v1


# In[333]:


# 2.2 添加最高占比
for i, col in enumerate(varsname):
    if i>=100 and i%500==0:
        print(i)
    df_explor_v1.loc[col, 'mod_null'] = df_sample[col].value_counts(normalize=True, ascending=False, dropna=False).max()
    df_explor_v1.loc[col, 'mod_notna'] = df_sample[col].value_counts(normalize=True, ascending=False).max()


# In[ ]:


timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
df_explor.to_excel(result_path + f"2_数据探索性分析_{task_name}_{timestamp}.xlsx")

print(f"数据存储完成: {timestamp}")
print(result_path + f"2_数据探索性分析_{task_name}_{timestamp}.xlsx")


# ## 2.2 数据探索

# In[334]:



def calculate_missing_rate_by_month(df, columns, groupby_col):
    """
    计算每个月每列的缺失率。
    
    参数:
    - df: 待处理的 DataFrame
    - groupby_col: 分组的列名
    - columns: 需要计算缺失率的列名列表
    
    返回:
    - 包含每个月每列缺失率的新 DataFrame
    """
    # 分组并计算缺失率
    missing_rates = df.groupby(groupby_col)[columns].apply(lambda x: x.isnull().sum()/len(x)).T

    return missing_rates


# In[335]:


# 2.2 缺失率按月分布
columns = varsname
groupby_col = 'apply_month'
df_miss_month = calculate_missing_rate_by_month(df_sample, columns, groupby_col)
df_miss_month.index.name = 'variable'
print(df_miss_month.head())


# In[336]:


# 2.2 缺失率按数据集分布
columns = varsname
groupby_col = 'data_set'
df_miss_set = calculate_missing_rate_by_month(df_sample, columns, groupby_col)
df_miss_set.index.name = 'variable'
print(df_miss_set.head())


# In[337]:


# 2.3 快速查看特征重要性
df_iv = toad.quality(df_sample[varsname+[target]], target, iv_only=True, 
                     method='dt', min_samples=0.05, n_bins=6)
df_iv.index.name = 'variable'
print(df_iv.head())


# In[338]:


timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

with pd.ExcelWriter(result_path + f'2_数据探索统计分析_{task_name}_{timestamp}.xlsx') as writer:
        df_explor.to_excel(writer, sheet_name='df_explor')
        df_explor_v1.to_excel(writer, sheet_name='df_explor_v1')
        df_miss_month.to_excel(writer, sheet_name='df_miss_month')
        df_miss_set.to_excel(writer, sheet_name='df_miss_set')
        df_iv.to_excel(writer, sheet_name='iv') 
print(f"数据存储完成时间：{timestamp}")
print(result_path + f'2_数据探索统计分析_{task_name}_{timestamp}.xlsx')


# # 3.特征粗筛选

# ## 3.1 基于自身属性删除变量

# In[341]:


# 删除近期不可使用的特征(最近月份的缺失率大于等于0.95)
# to_drop_recent = list(df_miss_month[(df_miss_month>=0.90).any(axis=1)].index)
to_drop_recent = []
print("to_drop_recent:", len(to_drop_recent))

# 删除缺失率大于0.95/删除枚举值只有一个/删除方差等于0/删除集中度大于0.95
to_drop_missing = list(df_explor_v1[df_explor_v1.missing.str[:-1].astype(float)/100>=0.90].index)
print("to_drop_missing:", len(to_drop_missing))

to_drop_unique = list(df_explor_v1[df_explor_v1.unique==1].index)
print("to_drop_unique:", len(to_drop_unique))

to_drop_std = list(df_explor_v1[df_explor_v1.std_or_top2==0].index)
print("to_drop_std:", len(to_drop_std))

to_drop_mode = list(df_explor_v1[df_explor_v1.mod_notna>=0.90].index)
print("to_drop_mode:", len(to_drop_mode))

to_drop_iv = list(df_iv[df_iv.iv<=0.01].index)
print("to_drop_iv:", len(to_drop_iv))

to_drop1 = list(set(to_drop_recent + to_drop_missing +  to_drop_unique +  to_drop_std + to_drop_mode + to_drop_iv))
print(f"删除的变量有{len(to_drop1)}个")


# In[342]:


df_iv.loc[to_drop_iv,:]


# In[343]:


varsname_v1 = [col for col in varsname if col not in to_drop1]

print(f"保留的变量有{len(varsname_v1)}个")
print(varsname_v1[:10])


# ## 3.2 基于相关性删除变量
# 

# In[344]:


train_selected, dropped = toad.selection.select(df_sample[varsname_v1+[target]],
                                                target=target, 
                                                empty=0.90, iv=0.01, corr=0.80, 
                                                return_drop=True, exclude=None)
train_selected.shape


# In[345]:


to_drop2 = []
for k, v in dropped.items():
    print(k, ":", len(v))
    to_drop2.extend(list(v))
print(len(set(to_drop2)))


# In[346]:


df_iv.loc[to_drop2,:]


# In[347]:


to_drop2 = []
varsname_v2 = [col for col in varsname_v1 if col not in to_drop2]

print(f"保留的变量有{len(varsname_v2)}个")


# # 4.特征细筛选

# ## 4.1 基于变量稳定性筛选

# In[348]:



def cal_psi_by_month(df_actual, df_expect, cols, month_col, combiner, return_frame = True):
    """
    计算每个月每的psi。
    
    参数:
    - df_actual: 测试集
    - df_expect: 训练集
    - cols: 需要计算稳定性的列名列表
    - month_col: 分组的列名

    返回:
    - 包含每个月的新 DataFrame
    """
    bins_df_list = []
    psi_list = []
    for month_, df_actual_group in df_actual.groupby(month_col):
        if return_frame:
            psi_, bins_df = toad.metrics.PSI(df_actual_group[cols], df_expect[cols], 
                                            combiner = combiner, return_frame = return_frame)
            psi_ = pd.DataFrame({month_: psi_}, index=cols)
            psi_list.append(psi_)
            bins_df['month'] = month_
            bins_df_list.append(bins_df)
        else:
            psi_ = toad.metrics.PSI(df_actual_group[cols], df_expect[cols], 
                                            combiner = combiner, return_frame = return_frame)
            psi_ = pd.DataFrame({month_: psi_}, index=cols)
            psi_list.append(psi_)
        
    # 合并所有结果 DataFrame
    if return_frame:
        psi_df = pd.concat(psi_list, axis=1)
        bins_df = pd.concat(bins_df_list, axis=0)
        
        return (psi_df, bins_df)
    else:
        psi_df = pd.concat(psi_list, axis=1)
        
        return psi_df


def cal_iv_by_month(df, cols, target, month_col, combiner):
    """
    计算每个变量每个月的iv。
    
    参数:
    - df: 待处理的 DataFrame
    - target: Y标签
    - cols: 需要计算iv的列名列表
    - month_col：月份列名
    
    返回:
    - 包含每个月每列iv的新 DataFrame
    """
    df_ = combiner.transform(df[cols+[target, month_col]], labels=True)
    result = pd.DataFrame(columns=sorted(list(df_[month_col].unique())), index=cols)
    for col in cols:
        for month in sorted(list(df_[month_col].unique())):
            data = df_[df_[month_col] == month]
            regroup = data.groupby(col)[target].agg(total=lambda x: x.count(), bad=lambda x: x.sum())
            regroup['good'] = regroup['total'] - regroup['bad']
            regroup['bad_pct'] = regroup['bad']/regroup['bad'].sum() + 1e-10
            regroup['good_pct'] = regroup['good']/regroup['good'].sum() + 1e-10
            regroup['woe'] = np.log(regroup['bad_pct']/regroup['good_pct'])
            regroup['iv'] = (regroup['bad_pct']-regroup['good_pct'])*regroup['woe']
            result.loc[col, month] = regroup['iv'].sum()    
      
    return result


def calculate_vars_distribute(df, cols, target, group_col):    
    """
    参数:
    - df: 待处理的 DataFrame
    - target: Y标签
    - cols: 需要分箱的列名列表
    - group_col：分组列名，如月份、渠道、数据类型
    
    返回:
    - 包含每个月每列iv的新 DataFrame
    """
    result = pd.DataFrame()
    vars = sorted(list(df[group_col].unique()))
    for col in cols:
        for var in vars:
            data = df[df[group_col] == var]
            regroup = data.groupby(col)[target].agg(total=lambda x: x.count(), bad=lambda x: x.sum())
            regroup['good'] = regroup['total'] - regroup['bad']
            regroup['bad_rate'] = regroup['bad']/regroup['total']
            regroup['bad_rate_cum'] = regroup['bad'].cumsum()/regroup['total'].cumsum()
            regroup['total_pct'] = regroup['total']/regroup['total'].sum()
            regroup['bad_pct'] = regroup['bad']/regroup['bad'].sum() + 1e-10
            regroup['good_pct'] = regroup['good']/regroup['good'].sum() + 1e-10
            regroup['bad_pct_cum'] = regroup['bad_pct'].cumsum()
            regroup['good_pct_cum'] = regroup['good_pct'].cumsum()
            regroup['total_pct_cum'] = regroup['total_pct'].cumsum()
            regroup['ks_bin'] = regroup['bad_pct_cum'] - regroup['good_pct_cum']
            regroup['ks'] = regroup['ks_bin'].max()
            regroup['lift_cum'] = regroup['bad_rate_cum']/data[target].mean()
            regroup['lift'] = regroup['bad_rate']/data[target].mean()
            regroup['woe'] = np.log(regroup['bad_pct']/regroup['good_pct'])
            regroup['iv_bins'] = (regroup['bad_pct']-regroup['good_pct'])*regroup['woe']
            regroup['iv'] = regroup['iv_bins'].sum()
            regroup['bins'] = regroup.index
                
            total_summary = data[target].agg(total=lambda x: x.count(), bad=lambda x: x.sum()).to_frame().T
            total_summary['good'] = regroup['total'] - regroup['bad']
            total_summary['bad_rate'] = total_summary['bad']/total_summary['total']
            total_summary['iv'] = regroup['iv_bins'].sum()
            total_summary['ks'] = regroup['ks_bin'].max()
            total_summary['bins'] = 'Total'
            
            regroup = pd.concat([regroup, total_summary], axis=0, ignore_index=True)
            regroup['varsname'] = col
            regroup['groupvars'] = var
            
            usecols = ['groupvars', 'varsname', 'bins', 'total', 'bad', 'good', 'bad_rate', 'bad_rate_cum', 'woe', 'iv', 'iv_bins', 
                       'ks', 'ks_bin', 'lift', 'lift_cum', 'total_pct', 'total_pct_cum', 'bad_pct', 'bad_pct_cum', 'good_pct','good_pct_cum']
            regroup = regroup[usecols]
            result = pd.concat([result, regroup], axis=0, ignore_index=True)

    return result


# In[349]:



# 删除当前索引值所在行的后一行(按从小到大排序,合并都是保留较小值)，配合左闭右开
def DelIndexPlus1(np_regroup, index_value):
np_regroup[index_value,1] = np_regroup[index_value,1] + np_regroup[index_value+1,1]#坏客户
np_regroup[index_value,2] = np_regroup[index_value,2] + np_regroup[index_value+1,2]#好客户
np_regroup = np.delete(np_regroup, index_value+1, axis=0)

return np_regroup
 
# 删除当前索引值所在行(按从小到大排序,合并都是保留较小值)，配合左闭右开
def DelIndex(np_regroup, index_value):
np_regroup[index_value-1,1] = np_regroup[index_value,1] + np_regroup[index_value-1,1]#坏客户
np_regroup[index_value-1,2] = np_regroup[index_value,2] + np_regroup[index_value-1,2]#好客户
np_regroup = np.delete(np_regroup, index_value, axis=0)

return np_regroup 

# 删除/合并客户数为0的箱子
def MergeZero(np_regroup):
#合并好坏客户数连续都为0的箱子
i = 0
while i<=np_regroup.shape[0]-2:
    if (np_regroup[i,1]==0 and np_regroup[i+1,1]==0) or (np_regroup[i,2]==0 and np_regroup[i+1,2]==0):
        np_regroup = DelIndexPlus1(np_regroup,i)
    i = i+1
    
#合并坏客户数为0的箱子
while True:
    if all(np_regroup[:,1]>0) or np_regroup.shape[0]==2:
        break
    bad_zero_index = np.argwhere(np_regroup[:,1]==0)[0][0]
    if bad_zero_index==0:
        np_regroup = DelIndexPlus1(np_regroup, bad_zero_index)
    elif bad_zero_index==np_regroup.shape[0]-1:
        np_regroup = DelIndex(np_regroup, bad_zero_index)
    else:
        bad_1 = np_regroup[bad_zero_index-1,1]
        good_1 = np_regroup[bad_zero_index-1,2]
        badplus1 = np_regroup[bad_zero_index+1,1]
        goodplus1 = np_regroup[bad_zero_index+1,2]           
        if (bad_1/(bad_1+good_1)) <= (badplus1/(badplus1 + goodplus1)):
            np_regroup = DelIndex(np_regroup, bad_zero_index)
        else:
            np_regroup = DelIndexPlus1(np_regroup, bad_zero_index)
#合并好客户数为0的箱子
while True:
    if all(np_regroup[:,2]>0) or np_regroup.shape[0]==2:
        break
    good_zero_index = np.argwhere(np_regroup[:,2]==0)[0][0]
    if good_zero_index==0:
        np_regroup = DelIndexPlus1(np_regroup, good_zero_index)
    elif good_zero_index==np_regroup.shape[0]-1:
        np_regroup = DelIndex(np_regroup, good_zero_index)
    else:
        bad_1 = np_regroup[good_zero_index-1,1]
        good_1 = np_regroup[good_zero_index-1,2]
        badplus1 = np_regroup[good_zero_index+1,1]
        goodplus1 = np_regroup[good_zero_index+1,2]                
        if (bad_1/(bad_1 + good_1)) <= (badplus1/(badplus1 + goodplus1)):
            np_regroup = DelIndexPlus1(np_regroup, good_zero_index)
        else:
            np_regroup = DelIndex(np_regroup, good_zero_index)
            
return np_regroup

#箱子最小占比
def MinPct(np_regroup, threshold=0.05):
while True:
    bins_pct = [(np_regroup[i,1]+np_regroup[i,2])/np_regroup.sum() for i in range(np_regroup.shape[0])]
    min_pct = min(bins_pct)
    if np_regroup.shape[0]==2:
        print(f"箱子最小占比：箱子数达到最小值2个,最小箱子占比{min_pct}")
        break
    if min_pct>=threshold:
        print(f"箱子最小占比：各箱子的样本占比最小值: {threshold}，已满足要求")
        break
    else:
        min_pct_index = bins_pct.index(min(bins_pct))
        if min_pct_index==0:
            np_regroup = DelIndexPlus1(np_regroup, min_pct_index)
        elif min_pct_index == np_regroup.shape[0]-1:
            np_regroup = DelIndex(np_regroup, min_pct_index)
        else:
            BadRate = [np_regroup[i,1]/(np_regroup[i,1]+np_regroup[i,2]) for i in range(np_regroup.shape[0])]
            BadRateDiffMin = [abs(BadRate[i]-BadRate[i+1]) for i in range(np_regroup.shape[0]-1)]
            if BadRateDiffMin[min_pct_index-1]>=BadRateDiffMin[min_pct_index]:
                np_regroup = DelIndexPlus1(np_regroup, min_pct_index)
            else:
                np_regroup = DelIndex(np_regroup, min_pct_index)
return np_regroup


# 箱子的单调性
def MonTone(np_regroup):
while True:
    if np_regroup.shape[0]==2:
        print("箱子单调性：箱子数达到最小值2个")
        break
    BadRate = [np_regroup[i,1]/(np_regroup[i,1]+np_regroup[i,2]) for i in range(np_regroup.shape[0])]
    BadRateMonetone = [BadRate[i]<BadRate[i+1] for i in range(np_regroup.shape[0]-1)]
    #确定是否单调
    if_Montone = len(set(BadRateMonetone))
    #判断跳出循环
    if if_Montone==1:
        print("箱子单调性：各箱子的坏样本率单调")
        break
    else:
        BadRateDiffMin = [abs(BadRate[i]-BadRate[i+1]) for i in range(np_regroup.shape[0]-1)]
        Montone_index = BadRateDiffMin.index(min(BadRateDiffMin))
        np_regroup = DelIndexPlus1(np_regroup, Montone_index)
        
return np_regroup


# 变量分箱，返回分割点，特殊值不参与分箱
def Vars_Bins(data, target, col, cutbins=[]):
df = data[data[target]>=0][[target, col]]
df = df[df[col].notnull()].reset_index(drop=True)
#区间左闭右开
df['bins'] = pd.cut(df[col], cutbins, duplicates='drop', right=False, precision=4, labels=False)
regroup = pd.DataFrame()
regroup['bins'] = df.groupby(['bins'])[col].min()
regroup['total'] = df.groupby(['bins'])[target].count()
regroup['bad'] = df.groupby(['bins'])[target].sum()
regroup['good'] = regroup['total'] - regroup['bad']
regroup.drop(['total'], axis=1, inplace=True)
np_regroup = np.array(regroup)
np_regroup = MergeZero(np_regroup)
np_regroup = MinPct(np_regroup)
np_regroup = MonTone(np_regroup)
cutoffpoints = list(np_regroup[:,0])
# 判断重新分箱后最高集中度占比
mode = [(np_regroup[i,1] + np_regroup[i,2])/np_regroup.sum()>0.95 for i in range(np_regroup.shape[0])]
is_drop_mode = any(mode)
# 判断第一个分割点是否最小值
if df[col].min()==cutoffpoints[0]:
    print(f"变量{col}：最小值所在箱子没有被合并过")
    cutoffpoints=cutoffpoints[1:]

return (cutoffpoints, is_drop_mode)


# In[350]:


# 计算分布前先变量分箱
combiner = toad.transform.Combiner()
combiner.fit(df_sample[varsname_v2+[target]], y=target, 
             method='dt', n_bins=10, min_samples = 0.05, empty_separate=True) 


# In[351]:


existing_bins_dict = combiner.export()
existing_bins_dict


# In[352]:


new_bins_dict = {}
to_drop_mode = []
for i, col in enumerate(varsname_v2):
    print(f"======第{i+1}个变量：{col}=========")
    empty = [x for x in existing_bins_dict[col] if pd.isnull(x)]
    not_empty = [x for x in existing_bins_dict[col] if pd.notnull(x)]
    if len(not_empty)==0:
        continue
    
    cutbins = [float('-inf')] + not_empty + [float('inf')]
    # 确保分箱无0值，单调，最小占比符合要求
    cutbins,  is_drop_mode = Vars_Bins(df_sample, target, col, cutbins=cutbins)
    # 新的分箱分割点，符合toad包要求
    new_bins_dict[col] = cutbins + empty
    # 删除重新分箱后，高度集中的变量
    if is_drop_mode:
        print(f"{col}重新分箱后，集中度占比超95%")
        to_drop_mode.append(col)


# In[353]:


new_bins_dict


# In[354]:


combiner.load(new_bins_dict)


# In[355]:


timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
with open(result_path + f'变量分箱字典_{timestamp}.pkl', 'wb') as f:
    pickle.dump(new_bins_dict, f)
print(result_path + f'变量分箱字典_{timestamp}.pkl')


# In[11]:


combiner = toad.transform.Combiner()
with open(result_path + '变量分箱字典_20250304124846.pkl', 'rb') as f:
    new_bins_dict = pickle.load(f)
    
combiner.load(new_bins_dict)


# In[356]:


to_drop_mode


# In[357]:


# 计算psi
df_psi_by_month = cal_psi_by_month(df_sample, df_sample.query("apply_month=='2024-08'"), varsname_v2,                                    'apply_month', combiner, return_frame = False)
print(df_psi_by_month.head(10))

df_psi_by_set = cal_psi_by_month(df_sample, df_sample.query("data_set=='1_train'"), varsname_v2,                                  'data_set', combiner, return_frame = False)
print(df_psi_by_set.head(10))


# In[358]:


# 计算iv
df_iv_by_month = cal_iv_by_month(df_sample, varsname_v2, target, 'apply_month', combiner)
print(df_iv_by_month.head(10))

df_iv_by_set = cal_iv_by_month(df_sample, varsname_v2, target, 'data_set', combiner)
print(df_iv_by_set.head(10)) 


# In[359]:


df_bins = combiner.transform(df_sample, labels=True)
selected_cols = ['groupvars', 'varsname', 'bins', 'total', 
                 'bad', 'good', 'total_pct', 'bad_pct', 
                 'good_pct', 'bad_rate', 'iv']


# In[12]:


df_bins = combiner.transform(df_sample, labels=True)


# In[360]:


df_group_month = calculate_vars_distribute(df_bins, varsname_v2, target, 'apply_month')[selected_cols] 
print(df_group_month.head())

df_group_set = calculate_vars_distribute(df_bins, varsname_v2, target, 'data_set')[selected_cols]   
print(df_group_set.head())


# In[361]:


# 计算total_pct 和 # bad_rate 以及iv的时间分布
df_total_bad = pd.DataFrame()
pivot_df_iv = pd.DataFrame()
    
for i in varsname_v2:      
    df_tmp = df_group_month.query("varsname==@i & bins!='Total'")
    pivot_df_totalpct = df_tmp.pivot_table(index=['varsname','bins'], columns='groupvars', 
                                           values='total_pct', fill_value=0).reset_index()
    pivot_df_badrate = df_tmp.pivot_table(index=['varsname','bins'], columns='groupvars', 
                                          values='bad_rate', fill_value=0).reset_index()
    pivot_df = pd.merge(pivot_df_totalpct, pivot_df_badrate, how='inner', 
                        on=['varsname','bins'], suffixes=('_total', '_bad'))
    df_total_bad = pd.concat([df_total_bad, pivot_df], axis=0) 

    df_tmp = df_group_month.query("varsname==@i & bins=='Total'")
    df_tmp_iv = df_tmp.pivot_table(index='varsname', columns='groupvars', values='iv').reset_index()
    pivot_df_iv = pd.concat([pivot_df_iv, df_tmp_iv], axis=0)
    
print(df_total_bad.head())
print(pivot_df_iv.head())


# In[362]:


# 计算total_pct 和 # bad_rate 以及iv的时间分布
df_total_bad_set = pd.DataFrame()
pivot_df_iv_set = pd.DataFrame()
    
for i in varsname_v2:      
    df_tmp = df_group_set.query("varsname==@i & bins!='Total'")
    pivot_df_totalpct = df_tmp.pivot_table(index=['varsname','bins'], columns='groupvars', 
                                           values='total_pct', fill_value=0).reset_index()
    pivot_df_badrate = df_tmp.pivot_table(index=['varsname','bins'], columns='groupvars', 
                                          values='bad_rate', fill_value=0).reset_index()
    pivot_df = pd.merge(pivot_df_totalpct, pivot_df_badrate, how='inner', 
                        on=['varsname','bins'], suffixes=('_total', '_bad'))
    df_total_bad_set = pd.concat([df_total_bad_set, pivot_df], axis=0)

    df_tmp = df_group_set.query("varsname==@i & bins=='Total'")
    df_tmp_iv = df_tmp.pivot_table(index='varsname', columns='groupvars', values='iv').reset_index()
    pivot_df_iv_set = pd.concat([pivot_df_iv_set, df_tmp_iv], axis=0)
    
    pivot_df_badrate_iv = pd.merge(pivot_df_badrate, pivot_df_iv_set, how='inner', 
                        on=['varsname'], suffixes=('_bad', '_iv'))
        
print(df_total_bad_set.head() )
print(pivot_df_iv_set.head() )


# ### 删除不稳定特征

# In[363]:


# drop_by_psi_month = list(df_psi_by_month[df_psi_by_month>=0.10].dropna(how='all').index) 
drop_by_psi_set = list(df_psi_by_set[df_psi_by_set>=0.10].dropna(how='all').index)
# drop_by_psi = drop_by_psi_month + drop_by_psi_set
drop_by_psi = drop_by_psi_set
print("drop_by_psi: ", len(drop_by_psi))

# df_iv_by_set.drop(columns=['mean','std','cv'], inplace=True)
# drop_by_iv_month = list(df_iv_by_month[df_iv_by_month<=0.01].dropna(how='all').index)
drop_by_iv_set = list(df_iv_by_set[df_iv_by_set<=0.01].dropna(how='all').index)
# drop_by_iv = drop_by_iv_month + drop_by_iv_set
drop_by_iv = drop_by_iv_set
print("drop_by_iv: ", len(drop_by_iv))

to_drop3 = list(set(drop_by_psi + drop_by_iv))
print("剔除的变量有: ", len(to_drop3))


# In[364]:


df_iv_by_set.loc[drop_by_iv_set,:]


# In[365]:


df_psi_by_set.loc[drop_by_psi_set,:]


# In[366]:


to_drop3 = ['bh_alic002_1','bh_alic002_2','bh_alic002_3']
# len([col for col in to_drop3 if df_miss_set.loc[col, '1_train']<=0.1])
print("剔除的变量有: ", len(to_drop3))


# In[367]:


varsname_v3 = [ col for col in varsname_v2 if col not in to_drop3]
print(f"保留的变量有{len(varsname_v3)}个: ")


# In[13]:


varsname_v3 = ['all_a_app_free_fpd30_202502_s', 'all_a_bhdj_fpd10_v1_p', 'all_a_br_derived_fpd30_202408_g_p', 'all_a_br_derived_v1_mob4dpd30_202502_st_p', 'all_a_br_derived_v2_fpd30_202411_g_p', 'all_a_dz_derived_v2_fpd30_202502_g_p', 'hlv_d_holo_certno_variablecode_dpd30_4m_bd0002_standard', 'hlv_d_holo_certno_variablecode_dpd30_6m_bd0001_standard', 'hlv_d_holo_certno_variablecode_standard_bd003', 'hlv_d_holo_jk_certno_fpd1_score', 'hlv_d_holo_jk_certno_score_fpd30_v1', 'hlv_d_holo_jk_certno_score_fpd7_v1', 'hlv_d_holo_jk_certno_varcode_standard_bd0004', 'ypy_bhxz_a_fpd30_v1_prob_good', 'score_fpd0_v1', 'score_fpd6_v1', 'score_fpd10_v1', 'score_fpd10_v2', 'score_fpd30_v1', 'duxiaoman_6', 'hengpu_4', 'aliyun_5', 'baihang_28', 'pudao_34', 'feicuifen', 'pudao_20', 'pudao_68', 'ruizhi_6', 'hengpu_5', 'pudao_21', 'bh_alic002_4', 't_br2_mob4', 't_beha3_mob4', 'dz_fpd'] + ['t_br_fpd', 't_beha3_fpd', 't_br_mob4', 't_br2_fpd', 'all_a_dz_derived_v1_fpd30_202502_g_p', 'xz_fpd']


# ## 4.2 Y标签相关性删除

# In[368]:


# 计算相关性
exclude = [col for col in df_bins.columns if col not in varsname_v3]

transer = toad.transform.WOETransformer()
df_sample_woe = transer.fit_transform(df_bins, df_bins[target], exclude=exclude)
print(df_sample_woe.shape) 


# In[14]:


# 计算相关性
exclude = [col for col in df_bins.columns if col not in varsname_v3]

transer = toad.transform.WOETransformer()
df_sample_woe = transer.fit_transform(df_bins, df_bins[target], exclude=exclude)
print(df_sample_woe.shape) 


# In[369]:


df_sample_woe.head()


# In[370]:


def find_high_correlation_pairs(df, iv_series, threshold=0.80):
    """
    找出相关系数大于指定阈值的变量对，并排除对角线。保留IV值较大的变量。

    :param df: 输入的DataFrame
    :param iv_series: 包含每个变量的IV值的Series，变量名为行索引
    :param threshold: 相关系数的阈值，默认为0.80
    :return: 包含高相关性变量对及其相关系数的DataFrame，以及保留的变量
    """
    # 计算相关系数矩阵
    corr_matrix = df.copy()
    # 初始化一个空列表来存储高相关性变量对
    high_corr_pairs = []
    # 遍历相关系数矩阵
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                high_corr_pairs.append((var1, var2, corr_value))
    
    # 将结果转换为DataFrame
    high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Var1', 'Var2', 'Correlation'])
    # 初始化一个空列表来存储需要删除的变量
    to_remove = set()
    # 初始化一个空列表，用于记录被剔除的变量（对应每一行）
    removed_vars = []
    # 遍历高相关性变量对，保留IV值较大的变量，删除IV值较小的变量
    for _, row in high_corr_df.iterrows():
        var1 = row['Var1']
        var2 = row['Var2']
        iv1 = iv_series[var1]
        iv2 = iv_series[var2]
        
        if iv1 >= iv2:
            to_remove.add(var2)
            removed_vars.append(var2)
        else:
            to_remove.add(var1)
            removed_vars.append(var1)
    high_corr_df['Removed_Variable'] = removed_vars
    # 返回高相关性变量对及其相关系数，以及保留的变量
    return high_corr_df, list(to_remove)


# In[15]:


# param method: 计算相关系数的方法，可以是'pearson', 'kendall', 'spearman'
df_corr_matrix = df_sample_woe[varsname_v3].corr(method='kendall')
df_corr_matrix.info()


# In[16]:


df_corr_matrix.head()


# In[371]:


# param method: 计算相关系数的方法，可以是'pearson', 'kendall', 'spearman'
df_corr_matrix = df_sample_woe[varsname_v3].corr(method='kendall')
df_corr_matrix


# In[372]:


# 调用函数

df_high_corr, to_drop4 = find_high_correlation_pairs(df_corr_matrix,
                                                     df_iv_by_set['3_oot2'],
                                                     threshold=0.80)

# 查看结果
print("删除的变量有：", len(to_drop4))


# In[373]:


df_high_corr.info()
df_high_corr.head()


# In[374]:


print(to_drop4)


# In[375]:


varsname_v4 = [col for col in varsname_v3 if col not in to_drop4]
print(f"保留变量{len(varsname_v4)}个")
print(varsname_v4)


# In[376]:


def calculate_correlations(df, group_col, varsname, target, method='pointbiserialr'):
    from scipy.stats import pointbiserialr, pearsonr, spearmanr, kendalltau
    
    # 按指定的分组列分组
    grouped = df.groupby(group_col)
    # 初始化一个空的DataFrame来存储所有分组的相关系数
    all_corrs = pd.DataFrame()
    all_pvalue = pd.DataFrame()
    # 遍历每个分组
    for name, group in grouped:      
        # 计算每个变量与目标变量的相关系数
        # 初始化一个空的字典来存储结果
        corr_series = pd.Series(index=varsname)
        corr_series.name = name
        result_pvalue = pd.Series(index=varsname)
        result_pvalue.name = name
        
        binary_var = group[target]
        # 计算每个连续变量与二分类变量之间的点二列相关系数
        for column in varsname:
            if method=='pointbiserialr':
                corr_series[column] =  pointbiserialr(binary_var, group[column])[0]
                result_pvalue[column] =  pointbiserialr(binary_var, group[column])[1]
            elif method=='pearsonr':
                corr_series[column] =  pearsonr(binary_var, group[column])[0]
                result_pvalue[column] =  pearsonr(binary_var, group[column])[1]
            elif method=='spearmanr':
                corr_series[column] =  spearmanr(binary_var, group[column])[0]
                result_pvalue[column] =  spearmanr(binary_var, group[column])[1]
            elif method=='kendalltau':
                corr_series[column] =  kendalltau(binary_var, group[column])[0]
                result_pvalue[column] =  kendalltau(binary_var, group[column])[1]
            else:
                raise ValueError("Invalid method. Choose from 'pointbiserialr','pearson', 'spearman', or 'kendall'.")

        # corr_series = group[varsname].corrwith(group[target], method=method)
        # 将结果添加到总的DataFrame中，并添加分组标识
        all_corrs = pd.concat([all_corrs, corr_series], axis=1)
        all_pvalue = pd.concat([all_pvalue, result_pvalue], axis=1)
    
    # 返回包含所有分组相关系数的DataFrame
    return (all_corrs, all_pvalue)


# In[377]:


# 调用函数
df_corr_vars_target, df_pvalue_vars_target = calculate_correlations(df_sample_woe,
                                                                    'apply_month',
                                                                    varsname_v4,
                                                                    target,
                                                                    method='pointbiserialr'
                                                                   )

# 查看前几行
# df_corr_vars_target


# In[378]:


df_corr_vars_target.info()
df_corr_vars_target.head()


# In[379]:


to_drop5 = list(df_corr_vars_target[df_corr_vars_target.apply(lambda row: (row > 0).any() and (row < 0).any(), axis=1)].index)
print("删除的变量有：", len(to_drop5))


# In[380]:


varsname_v5 = [ col for col in varsname_v4 if col not in to_drop5]
print(f"保留的变量{len(varsname_v5)}个")


# In[17]:


varsname_v5 = ['all_a_app_free_fpd30_202502_s', 'all_a_bhdj_fpd10_v1_p', 'all_a_br_derived_fpd30_202408_g_p', 'all_a_br_derived_v1_mob4dpd30_202502_st_p', 'all_a_br_derived_v2_fpd30_202411_g_p', 'all_a_dz_derived_v2_fpd30_202502_g_p', 'hlv_d_holo_certno_variablecode_dpd30_4m_bd0002_standard', 'hlv_d_holo_certno_variablecode_dpd30_6m_bd0001_standard', 'hlv_d_holo_certno_variablecode_standard_bd003', 'hlv_d_holo_jk_certno_fpd1_score', 'hlv_d_holo_jk_certno_score_fpd30_v1', 'hlv_d_holo_jk_certno_score_fpd7_v1', 'hlv_d_holo_jk_certno_varcode_standard_bd0004', 'ypy_bhxz_a_fpd30_v1_prob_good', 'score_fpd0_v1', 'score_fpd6_v1', 'score_fpd10_v1', 'score_fpd10_v2', 'score_fpd30_v1', 'duxiaoman_6', 'hengpu_4', 'aliyun_5', 'baihang_28', 'pudao_34', 'feicuifen', 'pudao_20', 'pudao_68', 'ruizhi_6', 'hengpu_5', 'pudao_21', 'bh_alic002_4', 't_br2_mob4', 't_beha3_mob4', 'dz_fpd']
varsname_v5.remove('baihang_28')


# In[381]:


to_drop5


# In[382]:


timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

with pd.ExcelWriter(result_path + f'3_变量分析_dis_iv_psi_{task_name}_{timestamp}.xlsx') as writer:
        df_group_month.to_excel(writer, sheet_name='df_group_month') 
        df_group_set.to_excel(writer, sheet_name='df_group_set') 
        df_iv_by_month.to_excel(writer, sheet_name='df_iv_by_month')
        df_iv_by_set.to_excel(writer, sheet_name='df_iv_by_set')
        df_psi_by_month.to_excel(writer, sheet_name='df_psi_by_month')
        df_psi_by_set.to_excel(writer, sheet_name='df_psi_by_set')
        df_pvalue_vars_target.to_excel(writer, sheet_name='df_pvalue_vars_target')
        df_corr_vars_target.to_excel(writer, sheet_name='df_corr_vars_target')
        df_total_bad.to_excel(writer, sheet_name='df_total_bad')
        df_total_bad_set.to_excel(writer, sheet_name='df_total_bad_set')
        pivot_df_iv.to_excel(writer, sheet_name='pivot_df_iv')
        pivot_df_iv_set.to_excel(writer, sheet_name='pivot_df_iv_set') 
        pivot_df_badrate_iv.to_excel(writer, sheet_name='pivot_df_badrate_iv')
print(f"数据存储完成时间：{timestamp}！")        
print(result_path + f'3_变量分析_dis_iv_psi_{task_name}_{timestamp}.xlsx')


# In[383]:


gc.collect()


# In[18]:


df_iv_by_month = pd.read_excel('./result/3_变量分析_dis_iv_psi_06_提现全渠道无成本子分融合模型fpd30标签_2410_2411_20250304133751.xlsx',sheet_name='df_iv_by_month')


# In[19]:


df_iv_by_month.info()
df_iv_by_month.head()


# In[20]:


df_iv_by_month.set_index('Unnamed: 0',inplace=True)
df_iv_by_month.head()


# In[44]:


df_iv_by_set = pd.read_excel('./result/3_变量分析_dis_iv_psi_06_提现全渠道无成本子分融合模型fpd30标签_2410_2411_20250304133751.xlsx',sheet_name='df_iv_by_set')
df_iv_by_set.head()


# In[45]:


df_iv_by_set.set_index('Unnamed: 0',inplace=True)
df_iv_by_set.head()


# # 5.模型训练

# ## 5.0 函数定义

# In[36]:



def model_ks_auc(df, target, y_pred, group_col):
    """
    Args:
        df (dataframe): 含有Y标签和预测分数的数据集
        target (string): Y标签列名
        y_pred (string): 坏概率分数列名
        group_col (string): 分组列名如月份，数据集

    Returns:
        dataframe: AUC和KS值的数据框
    """
    df_ks_auc = pd.DataFrame(index=['KS', 'AUC'])
    for col, group_df in df.groupby(group_col):  
        # 计算 AUC
        group_df = group_df[(group_df[target].notna())&(group_df[y_pred].notna())]
        auc_ = roc_auc_score(group_df[target], group_df[y_pred])      
        fpr, tpr, _ = roc_curve(group_df[target], group_df[y_pred], pos_label=1)
        ks_ = max(abs(tpr-fpr))
        df_ks_auc.loc['KS', col] = ks_
        df_ks_auc.loc['AUC', col] = auc_
#         print(f"{col}：KS值{ks_}，AUC值{auc_}")
    df_ks_auc = df_ks_auc.T
    
    return df_ks_auc



def feature_importance(model):
    if isinstance(model, lgb.Booster):
        print("这是原生接口的模型 (Booster)")
        # 获取特征重要性
        feature_importance_gain = model.feature_importance(importance_type='gain')
        feature_importance_split = model.feature_importance(importance_type='split')
        # 获取特征名称
        feature_names = model.feature_name()
        # 将特征重要性转换为数据框
        df_importance = pd.DataFrame({'gain': feature_importance_gain,
                                      'split': feature_importance_split}, 
                                     index=feature_names)
        df_importance = df_importance.sort_values('gain', ascending=False)
        df_importance.index.name = 'feature'
    elif isinstance(model, (LGBMClassifier, LGBMRegressor)):
        print("这是 sklearn 接口的模型")
        df1_dict = model.get_booster().get_score(importance_type='weight')
        importance_type_split = pd.DataFrame.from_dict(df1_dict, orient='index')
        importance_type_split.columns = ['split']
        importance_type_split = importance_type_split.sort_values('split', ascending=False)
        importance_type_split['split_pct'] = importance_type_split['split'] / importance_type_split['split'].sum()

        df2_dict = model.get_booster().get_score(importance_type='gain')
        importance_type_gain = pd.DataFrame.from_dict(df2_dict, orient='index')
        importance_type_gain.columns = ['gain']
        importance_type_gain = importance_type_gain.sort_values('gain', ascending=False)
        importance_type_gain['gain_pct'] = importance_type_gain['gain'] / importance_type_gain['gain'].sum()

        df_importance = pd.concat([importance_type_gain, importance_type_split], axis=1)
        df_importance = df_importance.sort_values('gain', ascending=False)
        df_importance.index.name = 'feature'
    else:
        print("未知模型类型")
        df_importance = None
    
    return df_importance

# Pickle方式保存和读取模型
def save_model_as_pkl(model, path):
    """
    保存模型到路径path
    :param model: 训练完成的模型
    :param path: 保存的目标路径
    """
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(model, f, protocol=2)
        
# xgb模型保存.bin 格式
def save_model_as_bin(model, save_file_path):
    #保存lgb模型为bin格式
    model.save_model(save_file_path)
    

def load_model_from_pkl(path):
    """
    从路径path加载模型
    :param path: 保存的目标路径
    """
    
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def channel_type(x):
    if x in (209, 213, 229, 233, 235, 236, 240, 241, 244, 226, 227, 231, 234, 245, 246, 247, 248, 249, 251):
        channel='金科渠道'
    elif x==1:
        channel='桔子商城'
    else:
        channel='api渠道'
    return channel

def channel_rate(x):
    if x in (209, 213, 229, 233, 235, 236, 240, 241, 244, 226, 227, 231, 234, 245, 246, 247, 248, 249, 251):
        if x == 227:
            channel='227渠道'
        elif x in (209, 213, 229, 233, 235, 236, 240, 241, 244):
            channel='24利率'
        elif x in (226, 227, 231, 234, 245, 246, 247):
            channel='36利率'
        else:
            channel=None
    else:
        channel=None

    return channel


# ## 5.1 数据预处理

# In[28]:


df_sample['fpd30_1'] = 1 - df_sample['fpd30']
modeltrian_target = 'fpd30_1'


# In[385]:


df_sample['fpd30_1'] = 1 - df_sample['fpd30']


# In[386]:


modeltrian_target = 'fpd30_1'


# In[387]:


df_sample[modeltrian_target].value_counts()


# In[388]:


# 查看训练数据集
df_sample.loc[df_sample.query("data_set not in ('3_oot1','3_oot2')").index, 'data_set']='1_train'
df_sample['data_set'].value_counts()


# In[39]:


df_sample['channel_types'] = df_sample['channel_id'].apply(channel_type)
df_sample['channel_rates'] = df_sample['channel_id'].apply(channel_rate)


# In[390]:


result_path


# ## 5.2 模型训练

# ### 5.2.1 base模型

# In[391]:


### 模型参数
opt_params = {}
opt_params['boosting'] = 'gbdt'
opt_params['objective'] = 'binary'
opt_params['metric'] = 'auc'
opt_params['bagging_freq'] = 1
opt_params['scale_pos_weight'] = 1 
opt_params['seed'] = 1 
opt_params['num_threads'] = -1 
# 调参时设置成不用调参的参数
opt_params['learning_rate'] = 0.1
## 正则参数，防止过拟合
opt_params['bagging_fraction'] = 0.8628008772208227     
opt_params['feature_fraction'] = 0.6177619614753441
opt_params['lambda_l1'] = 0
opt_params['lambda_l2'] = 300
opt_params['early_stopping_rounds'] = 50

# 调参后的参数需要变成整数型
opt_params['num_leaves'] = 21
opt_params['min_data_in_leaf'] = 103
opt_params['max_depth'] = 2
# 调参后的其他参
opt_params['min_gain_to_split'] = 10


# In[392]:


print("最优参数opt_params: ", opt_params)


# In[393]:


print(len(varsname_v5))
print(varsname_v5)


# In[394]:


to_drop4


# In[397]:


# 模型变量
varsname_base = ['all_a_bhdj_fpd10_v1_p','all_a_br_derived_fpd30_202408_g_p',
                 'all_a_br_derived_v1_mob4dpd30_202502_st_p','all_a_br_derived_v2_fpd30_202411_g_p',
                'all_a_dz_derived_v2_fpd30_202502_g_p','ypy_bhxz_a_fpd30_v1_prob_good',
                 't_br2_mob4','t_beha3_mob4','dz_fpd']


# In[398]:


df_sample[varsname_base + to_drop4].info(show_counts=True)


# In[402]:


df_corr_base_drop = df_corr_matrix.loc[varsname_base+to_drop4, varsname_base+to_drop4]
df_corr_base = df_corr_matrix.loc[varsname_base, varsname_base]


# In[403]:


df_iv_base_drop = df_iv_by_set.loc[varsname_base+to_drop4,:]
df_iv_base = df_iv_by_set.loc[varsname_base,:]


# In[456]:


# 相关性保存
with pd.ExcelWriter(result_path + f'4_变量相关性_{task_name}_{timestamp}.xlsx') as writer:
    df_corr_base_drop.to_excel(writer, sheet_name='df_corr_base_drop')
    df_corr_base.to_excel(writer, sheet_name='df_corr_base')
    df_iv_base_drop.to_excel(writer, sheet_name='df_iv_base_drop') 
    df_iv_base.to_excel(writer, sheet_name='df_iv_base')   
print(result_path + f'4_变量相关性_{task_name}_{timestamp}.xlsx')


# In[445]:


# 查看训练数据集
df_sample['data_set'].value_counts()


# In[446]:


# 确定数据集参数后，训练模型
X_train_ = df_sample.query("data_set not in ('3_oot1','3_oot2')")[varsname_base]
y_train_ = df_sample.query("data_set not in ('3_oot1','3_oot2')")[modeltrian_target]
X_train, X_test, y_train, y_test = train_test_split(X_train_,
                                                    y_train_,
                                                    test_size=0.2, 
                                                    random_state=22, 
                                                    stratify=y_train_
                                                   )
print(X_train.shape, X_test.shape)

df_sample.loc[X_train.index, 'data_set']='1_train'
df_sample.loc[X_test.index, 'data_set']='2_test'


# In[447]:


df_sample['data_set'].value_counts()


# In[448]:


# 6，训练/保存/评估模型
# 优化训练模型
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
lgb_model = lgb.train(opt_params, train_set, valid_sets=valid_set, num_boost_round=10000)


# In[449]:


# 优化后评估模型效果
df_sample['y_prob_base'] = lgb_model.predict(df_sample[X_train.columns], num_iteration=lgb_model.best_iteration)
df_sample['y_prob_base'].head()


# In[450]:


# 最初评估模型效果 
df_ks_auc_month_base = model_ks_auc(df_sample, modeltrian_target, 'y_prob_base', 'apply_month')
df_ks_auc_month_base['渠道'] = '全渠道'
tmp = get_target_summary(df_sample, target, 'apply_month').set_index('bins')
df_ks_auc_month_base = pd.concat([tmp, df_ks_auc_month_base], axis=1)
print(df_ks_auc_month_base)


df_ks_auc_set_base = model_ks_auc(df_sample, modeltrian_target, 'y_prob_base', 'data_set')
df_ks_auc_set_base['渠道'] = '全渠道'
tmp = get_target_summary(df_sample, target, 'data_set').set_index('bins')
df_ks_auc_set_base = pd.concat([tmp, df_ks_auc_set_base], axis=1)
print(df_ks_auc_set_base)


# In[451]:


# 最初评估模型效果 
df_ks_auc_month_base_type = pd.DataFrame()
for type_, tmp_df in df_sample.groupby('channel_types'):
    print(f'--------{type_}----------')
    tmp1 = model_ks_auc(tmp_df, modeltrian_target, 'y_prob_base', 'apply_month')
    tmp1['渠道'] = type_
    tmp2 = get_target_summary(tmp_df, target, 'apply_month').set_index('bins')
    df_ks_auc_month_base_type = pd.concat([df_ks_auc_month_base_type,pd.concat([tmp2, tmp1], axis=1)],axis=0)
    
print(df_ks_auc_month_base_type)


# 最初评估模型效果 
df_ks_auc_month_base_rate = pd.DataFrame()
for rate_, tmp_df in df_sample.groupby('channel_rates'):
    print(f'--------{rate_}----------')
    tmp1 = model_ks_auc(tmp_df, modeltrian_target, 'y_prob_base', 'apply_month')
    tmp1['渠道'] = rate_
    tmp2 = get_target_summary(tmp_df, target, 'apply_month').set_index('bins')
    df_ks_auc_month_base_rate = pd.concat([df_ks_auc_month_base_rate,pd.concat([tmp2, tmp1], axis=1)],axis=0)

print(df_ks_auc_month_base_rate)


# In[452]:


# 合并
df_ks_auc_month_base1 = pd.concat([df_ks_auc_month_base, df_ks_auc_month_base_type, df_ks_auc_month_base_rate])
df_ks_auc_month_base1.head()


# In[453]:


# 模型变量重要性
df_importance_base = feature_importance(lgb_model) 
df_importance_base = pd.merge(df_importance_base, df_iv_by_month, how='left', left_index=True,right_index=True)
df_importance_base = df_importance_base.reset_index()
df_importance_base = df_importance_base.rename(columns={'index':'varsname'})
df_importance_base


# In[454]:


# 效果评估后保存模型
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
save_model_as_pkl(lgb_model, result_path + f'{task_name}_base1_{timestamp}.pkl')
save_model_as_bin(lgb_model, result_path + f'{task_name}_base1_{timestamp}.bin')
print(f"模型保存完成！：{timestamp}")
print(result_path + f'{task_name}_base1_{timestamp}.pkl')
print(result_path + f'{task_name}_base1_{timestamp}.bin')

with pd.ExcelWriter(result_path + f'4_模型训练_{task_name}_base1_{timestamp}.xlsx') as writer:
    df_importance_base.to_excel(writer, sheet_name='df_importance_base1')
    df_ks_auc_set_base.to_excel(writer, sheet_name='df_ks_auc_set_base1')
    df_ks_auc_month_base1.to_excel(writer, sheet_name='df_ks_auc_month_base1')      
print(result_path + f'4_模型训练_{task_name}_base1_{timestamp}.xlsx')


# #### 剔除桔子商城样本建模

# In[457]:


df_sample_30 = df_sample.query("channel_id!=1").reset_index(drop=True)


# In[458]:


df_sample_30['data_set'].value_counts()


# In[459]:


# 确定数据集参数后，训练模型
X_train_ = df_sample_30.query("data_set not in ('3_oot1','3_oot2')")[varsname_base]
y_train_ = df_sample_30.query("data_set not in ('3_oot1','3_oot2')")[modeltrian_target]
X_train, X_test, y_train, y_test = train_test_split(X_train_,
                                                    y_train_,
                                                    test_size=0.2, 
                                                    random_state=22, 
                                                    stratify=y_train_
                                                   )
print(X_train.shape, X_test.shape)

df_sample_30.loc[X_train.index, 'data_set']='1_train'
df_sample_30.loc[X_test.index, 'data_set']='2_test'


# In[460]:


df_sample_30['data_set'].value_counts()


# In[461]:


# 6，训练/保存/评估模型
# 优化训练模型
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
lgb_model = lgb.train(opt_params, train_set, valid_sets=valid_set, num_boost_round=10000)


# In[462]:


# 优化后评估模型效果
df_sample_30['y_prob_base'] = lgb_model.predict(df_sample_30[X_train.columns], num_iteration=lgb_model.best_iteration)
df_sample_30['y_prob_base'].head()


# In[463]:


# 优化后评估模型效果-30+客群
df_ks_auc_month_30_base = model_ks_auc(df_sample_30, modeltrian_target, 'y_prob_base', 'apply_month')
df_ks_auc_month_30_base['渠道'] = '全渠道'
tmp = get_target_summary(df_sample_30, target, 'apply_month').set_index('bins')
df_ks_auc_month_30_base = pd.concat([tmp, df_ks_auc_month_30_base], axis=1)
print(df_ks_auc_month_30_base)


df_ks_auc_set_30_base = model_ks_auc(df_sample_30, modeltrian_target, 'y_prob_base', 'data_set')
df_ks_auc_set_30_base['渠道'] = '全渠道'
tmp = get_target_summary(df_sample_30, target, 'data_set').set_index('bins')
df_ks_auc_set_30_base = pd.concat([tmp, df_ks_auc_set_30_base], axis=1)
print(df_ks_auc_set_30_base)


# In[467]:


# 最初评估模型效果 
df_ks_auc_month_30_base_type = pd.DataFrame()
for type_, tmp_df in df_sample_30.groupby('channel_types'):
    print(f'--------{type_}----------')
    tmp1 = model_ks_auc(tmp_df, modeltrian_target, 'y_prob_base', 'apply_month')
    tmp1['渠道'] = type_
    tmp2 = get_target_summary(tmp_df, target, 'apply_month').set_index('bins')
    df_ks_auc_month_30_base_type = pd.concat([df_ks_auc_month_30_base_type,pd.concat([tmp2, tmp1], axis=1)],axis=0)
    
print(df_ks_auc_month_30_base_type)


# 最初评估模型效果 
df_ks_auc_month_30_base_rate = pd.DataFrame()
for rate_, tmp_df in df_sample_30.groupby('channel_rates'):
    print(f'--------{rate_}----------')
    tmp1 = model_ks_auc(tmp_df, modeltrian_target, 'y_prob_base', 'apply_month')
    tmp1['渠道'] = rate_
    tmp2 = get_target_summary(tmp_df, target, 'apply_month').set_index('bins')
    df_ks_auc_month_30_base_rate = pd.concat([df_ks_auc_month_30_base_rate,pd.concat([tmp2, tmp1], axis=1)],axis=0)

print(df_ks_auc_month_30_base_rate)


# In[468]:


# month 
df_ks_auc_month_30_base2 = pd.concat([df_ks_auc_month_30_base, df_ks_auc_month_30_base_type, df_ks_auc_month_30_base_rate])
df_ks_auc_month_30_base2.head()


# In[469]:


# 模型变量重要性
df_importance_base2 = feature_importance(lgb_model) 
df_importance_base2 = pd.merge(df_importance_base2, df_iv_by_month, how='left', left_index=True,right_index=True)
df_importance_base2 = df_importance_base2.reset_index()
df_importance_base2 = df_importance_base2.rename(columns={'index':'varsname'})
df_importance_base2


# In[470]:


# 效果评估后保存模型
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
save_model_as_pkl(lgb_model, result_path + f'{task_name}_base2_{timestamp}.pkl')
save_model_as_bin(lgb_model, result_path + f'{task_name}_base2_{timestamp}.bin')
print(f"模型保存完成！：{timestamp}")
print(result_path + f'{task_name}_base2_{timestamp}.pkl')
print(result_path + f'{task_name}_base2_{timestamp}.bin')

with pd.ExcelWriter(result_path + f'4_模型训练_{task_name}_base2_{timestamp}.xlsx') as writer:
    df_importance_base2.to_excel(writer, sheet_name='df_importance_base2')
    df_ks_auc_set_30_base.to_excel(writer, sheet_name='df_ks_auc_set_30_base2')
    df_ks_auc_month_30_base2.to_excel(writer, sheet_name='df_ks_auc_month_30_base2')      
print(result_path + f'4_模型训练_{task_name}_base2_{timestamp}.xlsx')


# ### 5.2.2 base模型加入行为模型子分

# In[547]:


print(len(varsname_v5))
print(varsname_v5)


# In[548]:


varsname_behave = ['hlv_d_holo_certno_variablecode_dpd30_4m_bd0002_standard',
                   'hlv_d_holo_certno_variablecode_dpd30_6m_bd0001_standard',
                   'hlv_d_holo_certno_variablecode_standard_bd003']


# In[549]:


varsname_base_v1 = varsname_base + varsname_behave
print(len(varsname_base_v1), varsname_base_v1)


# In[550]:


# 查看训练数据集
df_sample['data_set'].value_counts()


# In[551]:


# 查看训练数据集
df_sample.loc[df_sample.query("data_set not in ('3_oot1', '3_oot2')").index, 'data_set']='1_train'
df_sample['data_set'].value_counts()


# In[552]:


# 训练数据集
X_train_ = df_sample.query("data_set not in ('3_oot1','3_oot2')")[varsname_base_v1]
y_train_ = df_sample.query("data_set not in ('3_oot1','3_oot2')")[modeltrian_target]
print(X_train_.shape)

# 确定参数后，确定训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_train_,
                                                    y_train_,
                                                    test_size=0.2, 
                                                    random_state=22, 
                                                    stratify=y_train_
                                                   )
print(X_train.shape, X_test.shape)

df_sample.loc[X_train.index, 'data_set']='1_train'
df_sample.loc[X_test.index, 'data_set']='2_test'
print(df_sample['data_set'].value_counts())


# In[553]:


### 模型参数
opt_params = {}
opt_params['boosting'] = 'gbdt'
opt_params['objective'] = 'binary'
opt_params['metric'] = 'auc'
opt_params['bagging_freq'] = 1
opt_params['scale_pos_weight'] = 1 
opt_params['seed'] = 1 
opt_params['num_threads'] = -1 
# 调参时设置成不用调参的参数
opt_params['learning_rate'] = 0.1
## 正则参数，防止过拟合
opt_params['bagging_fraction'] = 0.8628008772208227     
opt_params['feature_fraction'] = 0.6177619614753441
opt_params['lambda_l1'] = 0
opt_params['lambda_l2'] = 300
opt_params['early_stopping_rounds'] = 50

# 调参后的参数需要变成整数型
opt_params['num_leaves'] = 21
opt_params['min_data_in_leaf'] = 103
opt_params['max_depth'] = 2
# 调参后的其他参
opt_params['min_gain_to_split'] = 10


# In[554]:


# 6，训练/保存/评估模型
# 最初训练模型
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
lgb_model = lgb.train(opt_params, train_set, valid_sets=valid_set, num_boost_round=10000, init_model=None)


# In[555]:


# 最初评估模型效果
df_sample['y_prob_base_v1'] = lgb_model.predict(df_sample[X_train.columns], num_iteration=lgb_model.best_iteration)
df_sample['y_prob_base_v1'].head()


# In[556]:


# 最初评估模型效果 
df_ks_auc_month_base_v1 = model_ks_auc(df_sample, modeltrian_target, 'y_prob_base_v1', 'apply_month')
df_ks_auc_month_base_v1['渠道'] = '全渠道'
tmp = get_target_summary(df_sample, target, 'apply_month').set_index('bins')
df_ks_auc_month_base_v1 = pd.concat([tmp, df_ks_auc_month_base_v1], axis=1)
print(df_ks_auc_month_base_v1)


df_ks_auc_set_base_v1 = model_ks_auc(df_sample, modeltrian_target, 'y_prob_base_v1', 'data_set')
df_ks_auc_set_base_v1['渠道'] = '全渠道'
tmp = get_target_summary(df_sample, target, 'data_set').set_index('bins')
df_ks_auc_set_base_v1 = pd.concat([tmp, df_ks_auc_set_base_v1], axis=1)
print(df_ks_auc_set_base_v1)


# In[557]:



# 最初评估模型效果 
df_ks_auc_month_base_v1_type = pd.DataFrame()
for type_, tmp_df in df_sample.groupby('channel_types'):
    print(f'--------{type_}----------')
    tmp1 = model_ks_auc(tmp_df, modeltrian_target, 'y_prob_base_v1', 'apply_month')
    tmp1['渠道'] = type_
    tmp2 = get_target_summary(tmp_df, target, 'apply_month').set_index('bins')
    df_ks_auc_month_base_v1_type = pd.concat([df_ks_auc_month_base_v1_type,pd.concat([tmp2, tmp1], axis=1)],axis=0)
    
print(df_ks_auc_month_base_v1_type)


# 最初评估模型效果 
df_ks_auc_month_base_v1_rate = pd.DataFrame()
for rate_, tmp_df in df_sample.groupby('channel_rates'):
    print(f'--------{rate_}----------')
    tmp1 = model_ks_auc(tmp_df, modeltrian_target, 'y_prob_base_v1', 'apply_month')
    tmp1['渠道'] = rate_
    tmp2 = get_target_summary(tmp_df, target, 'apply_month').set_index('bins')
    df_ks_auc_month_base_v1_rate = pd.concat([df_ks_auc_month_base_v1_rate,pd.concat([tmp2, tmp1], axis=1)],axis=0)

print(df_ks_auc_month_base_v1_rate)


# In[558]:


# 合并
df_ks_auc_month_base_v1 = pd.concat([df_ks_auc_month_base_v1, df_ks_auc_month_base_v1_type,
                                     df_ks_auc_month_base_v1_rate])


# In[559]:


# 模型变量重要性
df_importance_base_v1 = feature_importance(lgb_model) 
df_importance_base_v1 = pd.merge(df_importance_base_v1, df_iv_by_month, how='left', left_index=True,right_index=True)
df_importance_base_v1 = df_importance_base_v1.reset_index()
df_importance_base_v1 = df_importance_base_v1.rename(columns={'index':'varsname'})
df_importance_base_v1


# In[562]:


# 模型变量相关性
df_corr_base_v1 = df_corr_matrix.loc[varsname_base_v1, varsname_base_v1]
df_iv_base_v1 = df_iv_by_set.loc[varsname_base_v1,:]
# print(df_corr_base_v1.max())


# In[563]:


df_corr_base_v1


# In[564]:



# 效果评估后保存模型
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
save_model_as_pkl(lgb_model, result_path + f'{task_name}_base_v1_{timestamp}.pkl')
save_model_as_bin(lgb_model, result_path + f'{task_name}_base_v1_{timestamp}.bin')
print(f"模型保存完成！：{timestamp}")
print(result_path + f'{task_name}_base_v1_{timestamp}.pkl')
print(result_path + f'{task_name}_base_v1_{timestamp}.bin')

with pd.ExcelWriter(result_path + f'5_模型优化_{task_name}_base_v1_{timestamp}.xlsx') as writer:
    df_importance_base_v1.to_excel(writer, sheet_name='df_importance_base_v1')
    df_ks_auc_month_base_v1.to_excel(writer, sheet_name='df_ks_auc_month_base_v1')
    df_ks_auc_set_base_v1.to_excel(writer, sheet_name='df_ks_auc_set_base_v1')
    df_corr_base_v1.to_excel(writer, sheet_name='df_corr_base_v1')
    df_iv_base_v1.to_excel(writer, sheet_name='df_iv_base_v1')  

print(result_path + f'5_模型优化_{task_name}_base_v1_{timestamp}.xlsx')


# ### 5.2.3 base模型加入征信模型子分

# In[21]:


print(len(varsname_v5))
print(varsname_v5)


# In[22]:


varsname_pboc = ['hlv_d_holo_jk_certno_score_fpd7_v1',
                 'score_fpd0_v1', 'score_fpd6_v1', 'score_fpd10_v1', 'score_fpd10_v2', 'score_fpd30_v1']


# In[23]:


varsname_base_v1 = ['all_a_bhdj_fpd10_v1_p', 'all_a_br_derived_fpd30_202408_g_p', 'all_a_br_derived_v1_mob4dpd30_202502_st_p', 'all_a_br_derived_v2_fpd30_202411_g_p', 'all_a_dz_derived_v2_fpd30_202502_g_p', 'ypy_bhxz_a_fpd30_v1_prob_good', 't_br2_mob4', 't_beha3_mob4', 'dz_fpd', 'hlv_d_holo_certno_variablecode_dpd30_4m_bd0002_standard', 'hlv_d_holo_certno_variablecode_dpd30_6m_bd0001_standard', 'hlv_d_holo_certno_variablecode_standard_bd003']
print(len(varsname_base_v1),varsname_base_v1)


# In[24]:


varsname_base_v2 = varsname_base_v1 + varsname_pboc
print(len(varsname_base_v2),varsname_base_v2)


# In[25]:


# 查看训练数据集
df_sample['data_set'].value_counts()


# In[26]:


# 查看训练数据集
df_sample.loc[df_sample.query("data_set not in ('3_oot1', '3_oot2')").index, 'data_set']='1_train'
df_sample['data_set'].value_counts()


# In[29]:


# 训练数据集
X_train_ = df_sample.query("data_set not in ('3_oot1','3_oot2')")[varsname_base_v2]
y_train_ = df_sample.query("data_set not in ('3_oot1','3_oot2')")[modeltrian_target]
print(X_train_.shape)


# In[30]:


# 确定参数后，确定训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_train_,
                                                    y_train_,
                                                    test_size=0.2, 
                                                    random_state=22, 
                                                    stratify=y_train_
                                                   )
print(X_train.shape, X_test.shape)

df_sample.loc[X_train.index, 'data_set']='1_train'
df_sample.loc[X_test.index, 'data_set']='2_test'
print(df_sample['data_set'].value_counts())


# In[31]:



### 模型参数
opt_params = {}
opt_params['boosting'] = 'gbdt'
opt_params['objective'] = 'binary'
opt_params['metric'] = 'auc'
opt_params['bagging_freq'] = 1
opt_params['scale_pos_weight'] = 1 
opt_params['seed'] = 1 
opt_params['num_threads'] = -1 
# 调参时设置成不用调参的参数
opt_params['learning_rate'] = 0.1
## 正则参数，防止过拟合
opt_params['bagging_fraction'] = 0.8628008772208227     
opt_params['feature_fraction'] = 0.6177619614753441
opt_params['lambda_l1'] = 0
opt_params['lambda_l2'] = 300
opt_params['early_stopping_rounds'] = 50

# 调参后的参数需要变成整数型
opt_params['num_leaves'] = 21
opt_params['min_data_in_leaf'] = 103
opt_params['max_depth'] = 2
# 调参后的其他参
opt_params['min_gain_to_split'] = 10


# In[32]:



# 6，训练/保存/评估模型
# 最初训练模型
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
lgb_model = lgb.train(opt_params, train_set, valid_sets=valid_set, num_boost_round=10000, init_model=None)


# In[34]:


# 最初评估模型效果
df_sample['y_prob_base_v2'] = lgb_model.predict(df_sample[X_train.columns], num_iteration=lgb_model.best_iteration)
df_sample['y_prob_base_v2'].head()  


# In[37]:



# 最初评估模型效果 
df_ks_auc_month_base_v2 = model_ks_auc(df_sample, modeltrian_target, 'y_prob_base_v2', 'apply_month')
df_ks_auc_month_base_v2['渠道'] = '全渠道'
tmp = get_target_summary(df_sample, target, 'apply_month').set_index('bins')
df_ks_auc_month_base_v2 = pd.concat([tmp, df_ks_auc_month_base_v2], axis=1)
print(df_ks_auc_month_base_v2)


df_ks_auc_set_base_v2 = model_ks_auc(df_sample, modeltrian_target, 'y_prob_base_v2', 'data_set')
df_ks_auc_set_base_v2['渠道'] = '全渠道'
tmp = get_target_summary(df_sample, target, 'data_set').set_index('bins')
df_ks_auc_set_base_v2 = pd.concat([tmp, df_ks_auc_set_base_v2], axis=1)
print(df_ks_auc_set_base_v2)


# In[40]:



# 最初评估模型效果 
df_ks_auc_month_base_v2_type = pd.DataFrame()
for type_, tmp_df in df_sample.groupby('channel_types'):
    print(f'--------{type_}----------')
    tmp1 = model_ks_auc(tmp_df, modeltrian_target, 'y_prob_base_v2', 'apply_month')
    tmp1['渠道'] = type_
    tmp2 = get_target_summary(tmp_df, target, 'apply_month').set_index('bins')
    df_ks_auc_month_base_v2_type = pd.concat([df_ks_auc_month_base_v2_type,pd.concat([tmp2, tmp1], axis=1)],axis=0)
    
print(df_ks_auc_month_base_v2_type)


# 最初评估模型效果 
df_ks_auc_month_base_v2_rate = pd.DataFrame()
for rate_, tmp_df in df_sample.groupby('channel_rates'):
    print(f'--------{rate_}----------')
    tmp1 = model_ks_auc(tmp_df, modeltrian_target, 'y_prob_base_v2', 'apply_month')
    tmp1['渠道'] = rate_
    tmp2 = get_target_summary(tmp_df, target, 'apply_month').set_index('bins')
    df_ks_auc_month_base_v2_rate = pd.concat([df_ks_auc_month_base_v2_rate,pd.concat([tmp2, tmp1], axis=1)],axis=0)

print(df_ks_auc_month_base_v2_rate)


# In[41]:


# 合并
df_ks_auc_month_base_v2 = pd.concat([df_ks_auc_month_base_v2, df_ks_auc_month_base_v2_type, df_ks_auc_month_base_v2_rate])
df_ks_auc_month_base_v2.head()


# In[42]:



# 模型变量重要性
df_importance_base_v2 = feature_importance(lgb_model) 
df_importance_base_v2 = pd.merge(df_importance_base_v2, df_iv_by_month, how='left', left_index=True,right_index=True)
df_importance_base_v2 = df_importance_base_v2.reset_index()
df_importance_base_v2 = df_importance_base_v2.rename(columns={'index':'varsname'})
df_importance_base_v2


# In[46]:


# 模型变量相关性

df_corr_base_v2 = df_corr_matrix.loc[varsname_base_v2, varsname_base_v2]
df_iv_base_v2 = df_iv_by_set.loc[varsname_base_v2,:]


# In[47]:



# 效果评估后保存模型
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
save_model_as_pkl(lgb_model, result_path + f'{task_name}_base_v2_{timestamp}.pkl')
save_model_as_bin(lgb_model, result_path + f'{task_name}_base_v2_{timestamp}.bin')
print(f"模型保存完成！：{timestamp}")
print(result_path + f'{task_name}_base_v2_{timestamp}.pkl')
print(result_path + f'{task_name}_base_v2_{timestamp}.bin')

with pd.ExcelWriter(result_path + f'5_模型优化_{task_name}_base_v2_{timestamp}.xlsx') as writer:
    df_importance_base_v2.to_excel(writer, sheet_name='df_importance_base_v2')
    df_ks_auc_month_base_v2.to_excel(writer, sheet_name='df_ks_auc_month_base_v2')
    df_ks_auc_set_base_v2.to_excel(writer, sheet_name='df_ks_auc_set_base_v2')
    df_corr_base_v2.to_excel(writer, sheet_name='df_corr_base_v2')
    df_iv_base_v2.to_excel(writer, sheet_name='df_iv_base_v2')  

print(result_path + f'5_模型优化_{task_name}_base_v2_{timestamp}.xlsx')


# ### 5.2.3 加入三方缓存数据

# In[48]:


print(len(varsname_v5))
print(varsname_v5)


# In[49]:


varsname_three = ['duxiaoman_6', 'hengpu_4', 'aliyun_5', 'pudao_34', 'feicuifen', 'pudao_20',
                  'pudao_68', 'ruizhi_6', 'pudao_21']
                   
varsname_base_v3 = varsname_base_v2 + varsname_three
print(len(varsname_base_v3), varsname_base_v3)


# In[50]:


# 查看训练数据集
df_sample['data_set'].value_counts()


# In[52]:


# 查看训练数据集
df_sample.loc[df_sample.query("data_set not in ('3_oot1', '3_oot2')").index, 'data_set']='1_train'
df_sample['data_set'].value_counts()


# In[53]:


# 训练数据集
X_train_ = df_sample.query("data_set not in ('3_oot1','3_oot2')")[varsname_base_v3]
y_train_ = df_sample.query("data_set not in ('3_oot1','3_oot2')")[modeltrian_target]

# 确定参数后，确定训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_train_,
                                                    y_train_,
                                                    test_size=0.2, 
                                                    random_state=22, 
                                                    stratify=y_train_
                                                   )
print(X_train.shape, X_test.shape)

df_sample.loc[X_train.index, 'data_set']='1_train'
df_sample.loc[X_test.index, 'data_set']='2_test'
print(df_sample['data_set'].value_counts())


# In[54]:



### 模型参数
opt_params = {}
opt_params['boosting'] = 'gbdt'
opt_params['objective'] = 'binary'
opt_params['metric'] = 'auc'
opt_params['bagging_freq'] = 1
opt_params['scale_pos_weight'] = 1 
opt_params['seed'] = 1 
opt_params['num_threads'] = -1 
# 调参时设置成不用调参的参数
opt_params['learning_rate'] = 0.1
## 正则参数，防止过拟合
opt_params['bagging_fraction'] = 0.8628008772208227     
opt_params['feature_fraction'] = 0.6177619614753441
opt_params['lambda_l1'] = 0
opt_params['lambda_l2'] = 300
opt_params['early_stopping_rounds'] = 50

# 调参后的参数需要变成整数型
opt_params['num_leaves'] = 21
opt_params['min_data_in_leaf'] = 103
opt_params['max_depth'] = 2
# 调参后的其他参
opt_params['min_gain_to_split'] = 10


# In[55]:



# 6，训练/保存/评估模型
# 最初训练模型
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
lgb_model = lgb.train(opt_params, train_set, valid_sets=valid_set, num_boost_round=10000, init_model=None)


# In[56]:



# 最初评估模型效果
df_sample['y_prob_base_v3'] = lgb_model.predict(df_sample[X_train.columns], num_iteration=lgb_model.best_iteration)
df_sample['y_prob_base_v3'].head() 


# In[57]:



# 最初评估模型效果 
df_ks_auc_month_base_v3 = model_ks_auc(df_sample, modeltrian_target, 'y_prob_base_v3', 'apply_month')
df_ks_auc_month_base_v3['渠道'] = '全渠道'
tmp = get_target_summary(df_sample, target, 'apply_month').set_index('bins')
df_ks_auc_month_base_v3 = pd.concat([tmp, df_ks_auc_month_base_v3], axis=1)
print(df_ks_auc_month_base_v3)


df_ks_auc_set_base_v3 = model_ks_auc(df_sample, modeltrian_target, 'y_prob_base_v3', 'data_set')
df_ks_auc_set_base_v3['渠道'] = '全渠道'
tmp = get_target_summary(df_sample, target, 'data_set').set_index('bins')
df_ks_auc_set_base_v3 = pd.concat([tmp, df_ks_auc_set_base_v3], axis=1)
print(df_ks_auc_set_base_v3)


# In[58]:



# 最初评估模型效果 
df_ks_auc_month_base_v3_type = pd.DataFrame()
for type_, tmp_df in df_sample.groupby('channel_types'):
    print(f'--------{type_}----------')
    tmp1 = model_ks_auc(tmp_df, modeltrian_target, 'y_prob_base_v3', 'apply_month')
    tmp1['渠道'] = type_
    tmp2 = get_target_summary(tmp_df, target, 'apply_month').set_index('bins')
    df_ks_auc_month_base_v3_type = pd.concat([df_ks_auc_month_base_v3_type,pd.concat([tmp2, tmp1], axis=1)],axis=0)
    
print(df_ks_auc_month_base_v3_type)


# 最初评估模型效果 
df_ks_auc_month_base_v3_rate = pd.DataFrame()
for rate_, tmp_df in df_sample.groupby('channel_rates'):
    print(f'--------{rate_}----------')
    tmp1 = model_ks_auc(tmp_df, modeltrian_target, 'y_prob_base_v3', 'apply_month')
    tmp1['渠道'] = rate_
    tmp2 = get_target_summary(tmp_df, target, 'apply_month').set_index('bins')
    df_ks_auc_month_base_v3_rate = pd.concat([df_ks_auc_month_base_v3_rate,pd.concat([tmp2, tmp1], axis=1)],axis=0)

print(df_ks_auc_month_base_v3_rate)


# In[59]:



# 合并
df_ks_auc_month_base_v3 = pd.concat([df_ks_auc_month_base_v3, df_ks_auc_month_base_v3_type, df_ks_auc_month_base_v3_rate])
df_ks_auc_month_base_v3.head()


# In[61]:



# 模型变量重要性
df_importance_base_v3 = feature_importance(lgb_model) 
df_importance_base_v3 = pd.merge(df_importance_base_v3, df_iv_by_month, how='left', left_index=True,right_index=True)
df_importance_base_v3 = df_importance_base_v3.reset_index()
df_importance_base_v3 = df_importance_base_v3.rename(columns={'index':'varsname'})
df_importance_base_v3


# In[62]:


# 模型相关性
df_corr_base_v3 = df_corr_matrix.loc[varsname_base_v3, varsname_base_v3]
df_iv_base_v3 = df_iv_by_set.loc[varsname_base_v3,:]

df_corr_base_v3


# In[63]:



# 效果评估后保存模型
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
save_model_as_pkl(lgb_model, result_path + f'{task_name}_base_v3_{timestamp}.pkl')
save_model_as_bin(lgb_model, result_path + f'{task_name}_base_v3_{timestamp}.bin')
print(f"模型保存完成！：{timestamp}")
print(result_path + f'{task_name}_base_v3_{timestamp}.pkl')
print(result_path + f'{task_name}_base_v3_{timestamp}.bin')

with pd.ExcelWriter(result_path + f'5_模型优化_{task_name}_base_v3_{timestamp}.xlsx') as writer:
    df_importance_base_v3.to_excel(writer, sheet_name='df_importance_base_v3')
    df_ks_auc_month_base_v3.to_excel(writer, sheet_name='df_ks_auc_month_base_v3')
    df_ks_auc_set_base_v3.to_excel(writer, sheet_name='df_ks_auc_set_base_v3')
    df_corr_base_v3.to_excel(writer, sheet_name='df_corr_base_v3')
    df_iv_base_v3.to_excel(writer, sheet_name='df_iv_base_v3')  

print(result_path + f'5_模型优化_{task_name}_base_v3_{timestamp}.xlsx')


# ### 5.2.4 加入融合模型子分

# In[64]:


print(len(varsname_v5))
print(varsname_v5)


# In[66]:


varsname_merge_score = ['all_a_app_free_fpd30_202502_s',
                   'hlv_d_holo_jk_certno_fpd1_score',
                   'hlv_d_holo_jk_certno_varcode_standard_bd0004',
                       'hlv_d_holo_jk_certno_score_fpd30_v1']
                   
varsname_base_v4 = varsname_base_v3 + varsname_merge_score
print(len(varsname_base_v4),varsname_base_v4)


# In[67]:


# 查看训练数据集
df_sample['data_set'].value_counts()


# In[68]:




# 查看训练数据集
df_sample.loc[df_sample.query("data_set not in ('3_oot1', '3_oot2')").index, 'data_set']='1_train'
df_sample['data_set'].value_counts()


# In[69]:



# 训练数据集
X_train_ = df_sample.query("data_set not in ('3_oot1','3_oot2')")[varsname_base_v4]
y_train_ = df_sample.query("data_set not in ('3_oot1','3_oot2')")[modeltrian_target]
print(X_train_.shape)

# 确定参数后，确定训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_train_,
                                                    y_train_,
                                                    test_size=0.2, 
                                                    random_state=22, 
                                                    stratify=y_train_
                                                   )
print(X_train.shape, X_test.shape)

df_sample.loc[X_train.index, 'data_set']='1_train'
df_sample.loc[X_test.index, 'data_set']='2_test'
print(df_sample['data_set'].value_counts())


# In[70]:



### 模型参数
opt_params = {}
opt_params['boosting'] = 'gbdt'
opt_params['objective'] = 'binary'
opt_params['metric'] = 'auc'
opt_params['bagging_freq'] = 1
opt_params['scale_pos_weight'] = 1 
opt_params['seed'] = 1 
opt_params['num_threads'] = -1 
# 调参时设置成不用调参的参数
opt_params['learning_rate'] = 0.1
## 正则参数，防止过拟合
opt_params['bagging_fraction'] = 0.8628008772208227     
opt_params['feature_fraction'] = 0.6177619614753441
opt_params['lambda_l1'] = 0
opt_params['lambda_l2'] = 300
opt_params['early_stopping_rounds'] = 50

# 调参后的参数需要变成整数型
opt_params['num_leaves'] = 21
opt_params['min_data_in_leaf'] = 103
opt_params['max_depth'] = 2
# 调参后的其他参
opt_params['min_gain_to_split'] = 10


# In[71]:



# 6，训练/保存/评估模型
# 最初训练模型
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
lgb_model = lgb.train(opt_params, train_set, valid_sets=valid_set, num_boost_round=10000, init_model=None)


# In[72]:


# 最初评估模型效果
df_sample['y_prob_base_v4'] = lgb_model.predict(df_sample[X_train.columns], num_iteration=lgb_model.best_iteration)
df_sample['y_prob_base_v4'].head()   


# In[73]:



# 最初评估模型效果 
df_ks_auc_month_base_v4 = model_ks_auc(df_sample, modeltrian_target, 'y_prob_base_v4', 'apply_month')
df_ks_auc_month_base_v4['渠道'] = '全渠道'
tmp = get_target_summary(df_sample, target, 'apply_month').set_index('bins')
df_ks_auc_month_base_v4 = pd.concat([tmp, df_ks_auc_month_base_v4], axis=1)
print(df_ks_auc_month_base_v4)


df_ks_auc_set_base_v4 = model_ks_auc(df_sample, modeltrian_target, 'y_prob_base_v4', 'data_set')
df_ks_auc_set_base_v4['渠道'] = '全渠道'
tmp = get_target_summary(df_sample, target, 'data_set').set_index('bins')
df_ks_auc_set_base_v4 = pd.concat([tmp, df_ks_auc_set_base_v4], axis=1)
print(df_ks_auc_set_base_v4)


# In[74]:



# 最初评估模型效果 
df_ks_auc_month_base_v4_type = pd.DataFrame()
for type_, tmp_df in df_sample.groupby('channel_types'):
    print(f'--------{type_}----------')
    tmp1 = model_ks_auc(tmp_df, modeltrian_target, 'y_prob_base_v4', 'apply_month')
    tmp1['渠道'] = type_
    tmp2 = get_target_summary(tmp_df, target, 'apply_month').set_index('bins')
    df_ks_auc_month_base_v4_type = pd.concat([df_ks_auc_month_base_v4_type,pd.concat([tmp2, tmp1], axis=1)],axis=0)
    
print(df_ks_auc_month_base_v4_type)


# 最初评估模型效果 
df_ks_auc_month_base_v4_rate = pd.DataFrame()
for rate_, tmp_df in df_sample.groupby('channel_rates'):
    print(f'--------{rate_}----------')
    tmp1 = model_ks_auc(tmp_df, modeltrian_target, 'y_prob_base_v4', 'apply_month')
    tmp1['渠道'] = rate_
    tmp2 = get_target_summary(tmp_df, target, 'apply_month').set_index('bins')
    df_ks_auc_month_base_v4_rate = pd.concat([df_ks_auc_month_base_v4_rate,pd.concat([tmp2, tmp1], axis=1)],axis=0)

print(df_ks_auc_month_base_v4_rate)


# In[75]:


# 合并
df_ks_auc_month_base_v4 = pd.concat([df_ks_auc_month_base_v4, df_ks_auc_month_base_v4_type, df_ks_auc_month_base_v4_rate])
df_ks_auc_month_base_v4.head()


# In[76]:


# 模型变量重要性
df_importance_base_v4 = feature_importance(lgb_model) 
df_importance_base_v4 = pd.merge(df_importance_base_v4, df_iv_by_month, how='left', left_index=True,right_index=True)
df_importance_base_v4 = df_importance_base_v4.reset_index()
df_importance_base_v4 = df_importance_base_v4.rename(columns={'index':'varsname'})
df_importance_base_v4


# In[77]:


# 模型相关性
df_corr_base_v4 = df_corr_matrix.loc[varsname_base_v4, varsname_base_v4]
df_iv_base_v4 = df_iv_by_set.loc[varsname_base_v4,:]


# In[78]:



# 效果评估后保存模型
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
save_model_as_pkl(lgb_model, result_path + f'{task_name}_base_v4_{timestamp}.pkl')
save_model_as_bin(lgb_model, result_path + f'{task_name}_base_v4_{timestamp}.bin')
print(f"模型保存完成！：{timestamp}")
print(result_path + f'{task_name}_base_v4_{timestamp}.pkl')
print(result_path + f'{task_name}_base_v4_{timestamp}.bin')

with pd.ExcelWriter(result_path + f'5_模型优化_{task_name}_base_v4_{timestamp}.xlsx') as writer:
    df_importance_base_v4.to_excel(writer, sheet_name='df_importance_base_v4')
    df_ks_auc_month_base_v4.to_excel(writer, sheet_name='df_ks_auc_month_base_v4')
    df_ks_auc_set_base_v4.to_excel(writer, sheet_name='df_ks_auc_set_base_v4')
    df_corr_base_v4.to_excel(writer, sheet_name='df_corr_base_v4')
    df_iv_base_v4.to_excel(writer, sheet_name='df_iv_base_v4')  

print(result_path + f'5_模型优化_{task_name}_base_v4_{timestamp}.xlsx')


# ## 5.3 模型效果对比

# ### 5.3.1数据处理

# In[518]:


# usecols= ['order_no','channel_id', 'lending_time','apply_month', 'mob',\
#           'maxdpd', 'fpd', 'fpd10', 'fpd30', 'mob4dpd30', 'diff_days'] + varsname_v5
df_model_vars = pd.read_csv('./result/洞侦续侦模型fpd30/桔子商城fpd30授信实时模型_建模数据集_250110.csv')
df_model_vars.info(show_counts=True)
df_model_vars.head()


# In[520]:


df_model_vars[varsname_v6] = df_model_vars[varsname_v6].replace(-1, np.nan)
gc.collect()


# In[521]:


# df_model_vars[varsname_v5].describe().T


# In[708]:


# 衍生Y标签
print(df_model_vars['fpd'].min(), df_model_vars['fpd'].max())
print(df_model_vars['fpd30'].min(), df_model_vars['fpd30'].max())


# In[522]:


# 衍生Y标签
# df_model_vars['fpd10'] = df_model_vars['fpd'].apply(lambda x: 1 if x>10 else 0)
# df_model_vars['fpd20'] = df_model_vars['fpd'].apply(lambda x: 1 if x>20 else 0)


# In[525]:


df_model_vars.groupby(["lending_time",'fpd10'])['order_no'].count().unstack()


# In[526]:


# df_model_vars.loc[df_model_vars.query("lending_time>='2024-10-16'").index, 'fpd30'] = -1
# df_model_vars.loc[df_model_vars.query("lending_time>='2024-10-26'").index, 'fpd20'] = -1


# In[527]:


# df_model_vars.drop(index=df_model_vars.query("lending_time>='2024-11-06'").index, inplace=True)


# In[528]:


# df_model_vars = df_model_vars.reset_index(drop=True)


# In[530]:


# # 添加客群标签
# def diff_days_(x):
#     if x<=30:
#         days = 'T30-'
#     elif x>30:
#         days = 'T30+'
#     else:
#         days = np.nan
#     return days

# df_model_vars['客群'] = df_model_vars['diff_days'].apply(diff_days_)


# In[531]:


df_model_vars['diff_days'].max()


# In[532]:


# df_model_vars.info(show_counts=True)


# In[534]:


# 第一次模型打分 
lgb_model= load_model_from_pkl('./result/桔子商城授信模型fpd30/桔子商城授信模型fpd30_v3_20250110145432.pkl')
df_model_vars['y_prob_v3'] = lgb_model.predict(df_model_vars[varsname_v6],
                                               num_iteration=lgb_model.best_iteration)


# In[720]:


result_path


# In[537]:


# 最终模型打分
lgb_model= load_model_from_pkl('./result/桔子商城授信模型fpd30/桔子商城授信模型fpd30_20250110152634.pkl')
df_model_vars['y_prob_v1'] = lgb_model.predict(df_model_vars[['standard_score','y_prob_v3']],
                                               num_iteration=lgb_model.best_iteration)


# In[1]:


lgb_model.feathure_name()


# In[571]:


# 其他提现模型数据 
sql="""
select t1.order_no,channel_id, fpd10,fpd30,prob
from
(
    select order_no,channel_id, fpd10,fpd30
    from znzz_fintech_ads.dm_f_lxl_test_order_Y_target as t 
    where dt=date_sub(current_date(), 1) 
      and lending_time>='2024-11-01'
      and lending_time<='2024-11-30'
) as t1 
inner join 
(select order_no,prob
from znzz_fintech_ads.fkmodel_ascore_fico_fpd10_v1_score as t 
where dt>='2024-11-01'
)as t2 on t1.order_no=t2.order_no
;

"""
df_tx_bj = get_data(sql)
df_tx_bj.info(show_counts=True)
df_tx_bj.head() 


# In[576]:


df_tx_bj['channel_type'] = df_tx_bj['channel_id'].astype(int).apply(channel_type)
df_tx_bj['channel_type'].value_counts()


# In[577]:


df_tx_bj['channel_rate'] = df_tx_bj['channel_id'].astype(int).apply(channel_rate)
df_tx_bj['channel_rate'].value_counts()


# In[587]:


fpr, tpr, _ = roc_curve(df_tx_bj.query("channel_type=='桔子商城'")['fpd30'], df_tx_bj.query("channel_type=='桔子商城'")['prob'], pos_label=1)
from sklearn.metrics import auc
auc_value = auc(fpr, tpr)
ks_value = max(abs(tpr - fpr))
print(auc_value, ks_value)


# In[540]:


# selected_cols = df_tx_bj.columns.to_list()[3:]
# df_tx_bj[selected_cols].describe().T


# In[541]:


# df_tx_bj_copy = df_tx_bj.copy()


# In[542]:


# result_path


# In[543]:


# df_tx_bj.to_csv(result_path + '全渠道其他提现模型分数_241218.csv')
# print(result_path + '全渠道其他提现模型分数_241218.csv')


# In[544]:


# # 好分数转为坏分数
# for i, col in enumerate(selected_cols):
#     print(f'第{i}个变量：{col}')
#     df_tx_bj[col] = 1 - df_tx_bj[col]


# In[545]:


# print(df_model_vars.shape, df_tx_bj.shape)


# In[546]:


# df_evalue = pd.merge(df_model_vars, df_tx_bj, how='left',on=['order_no'])
# print(df_evalue.shape, df_evalue['order_no'].nunique())


# In[547]:


# df_evalue.info(show_counts=True)


# In[548]:


# df_evalue.drop(columns=['apply_date','channel_id_y'], inplace=True)
# df_evalue.rename(columns={'channel_id_x':'channel_id'},inplace=True)


# ### 5.3.2 效果对比

# In[549]:


# 小数转换百分数
def to_percentage(x):
    if isinstance(x, (float)) and pd.notnull(x):
        return f"{x * 100:.2f}%"
    return x

def float_format(x):
    if isinstance(x, (float)) and pd.notnull(x):
        return '%.3f' %x
    return x

def cal_data_item(df, label_col, score_col, percentile=0.95):
    from sklearn.metrics import auc
    fpr, tpr, _ = roc_curve(df[label_col], df[score_col], pos_label=1)
    auc_value = auc(fpr, tpr)
    ks_value = max(abs(tpr - fpr))
    badrate = df[label_col].mean()
    
    if percentile>=0.90:#概率分数是坏分数，计算最坏5%客群的lift
        pct_n = df[score_col].quantile(percentile)
        pct_n_badrate = df[df[score_col]>pct_n][label_col].mean()
    elif percentile<=0.10:#概率分数是好分数，计算最坏5%客群的lift
        pct_n = df[score_col].quantile(percentile)
        pct_n_badrate = df[df[score_col]<pct_n][label_col].mean()
    else:
        print("请根据概率分数是好分数还是坏分数，决定分位数的位置")
    
    if badrate>0 and pct_n_badrate>0:
        lift_n = pct_n_badrate/badrate
    else:
        lift_n = np.nan
    return pd.Series({'KS': ks_value, 'AUC': auc_value, 'top5lift':lift_n})


# 计算KS
def cal_ks_auc(df, groupkeys, model_score_label_dict):
    # groupkeys: 分组字段
    # model_score_label_dict: value: score_list: 得分字段列表, key: label_list: 标签字段列表
    # df: 有标签和得分的数据框
    # 输出KS、AUC
    
    ks_auc_result = pd.DataFrame()
    if not isinstance(groupkeys, list):
        groupkeys = [groupkeys]
    for label_, score_list in model_score_label_dict.items():
        data1 = df[df[label_]>=0]
        total_bad = data1.groupby(groupkeys)[label_].agg(total=lambda x: len(x), 
                                                        bad=lambda x: x.sum(), 
                                                        badrate=lambda x: x.mean())
        total_bad['badrate'] = total_bad['badrate'].apply(to_percentage)
        total_bad.insert(loc=0, column='target_type', value=label_,                          allow_duplicates=False)
        
        ks_auc_list = []
        for score_ in score_list:
            data = df[(df[label_]>=0) & (df[score_].notnull())]
            tmp_ks_auc = data.groupby(groupkeys).apply(cal_data_item,                                                        label_col=label_,                                                        score_col=score_)
            tmp_ks_auc = tmp_ks_auc.rename(columns={'KS':f'KS_{score_}',
                                                    'AUC':f'AUC_{score_}',
                                                    'top5lift':f'top5lift_{score_}'})
            ks_auc_list.append(tmp_ks_auc)
        df_ks_auc = pd.concat(ks_auc_list, axis=1)
        ks_columns = [col for col in df_ks_auc.columns if 'KS' in col]
        AUC_columns = [col for col in df_ks_auc.columns if 'AUC' in col]
        lift_columns = [col for col in df_ks_auc.columns if 'top5lift' in col]
        df_ks_auc[ks_columns] = df_ks_auc[ks_columns].applymap(float_format)
        df_ks_auc[AUC_columns] = df_ks_auc[AUC_columns].applymap(float_format)
        df_ks_auc[lift_columns] = df_ks_auc[lift_columns].applymap(float_format)        
        df_ks_auc = df_ks_auc[ks_columns + AUC_columns + lift_columns]
        
        df_ks_auc = pd.concat([total_bad, df_ks_auc], axis=1)
        df_ks_auc = df_ks_auc.reset_index()
        
        ks_auc_result = pd.concat([ks_auc_result, df_ks_auc], axis=0, ignore_index=True)
        print(f'==============完成标签：{label_}===============')
    
    return ks_auc_result


# In[559]:


colsname = ['bad_score','y_prob_v3','y_prob_v1']

print(colsname)
target_list = ['fpd10', 'fpd20', 'fpd30']
labels_models_dict = {target: colsname for target in target_list}
print(labels_models_dict)


# In[551]:


df_model_vars['channel_id'].head()


# In[732]:


# groupkeys1 = ['apply_month']
# df_ksauc_all_v1 = cal_ks_auc(df_evalue, groupkeys1, labels_models_dict)
# df_ksauc_all_v1.insert(loc=(len(groupkeys1)), column='渠道', value='全渠道', allow_duplicates=False)
# df_ksauc_all_v1.insert(loc=(len(groupkeys1)+1), column='客群', value='全体', allow_duplicates=False)
# df_ksauc_all_v1.head()


# In[553]:


def channel_type(x):
    if x in (209, 213, 229, 233, 235, 236, 240, 241, 244, 226, 227, 231, 234, 245, 246, 247):
        channel='金科渠道'
    elif x==1:
        channel='桔子商城'
    else:
        channel='api渠道'
    return channel

def channel_rate(x):
    if x in (209, 213, 229, 233, 235, 236, 240, 241, 244, 226, 227, 231, 234, 245, 246, 247):
        if x == 227:
            channel='227'
        elif x in (209, 213, 229, 233, 235, 236, 240, 241, 244):
            channel='24利率'
        elif x in (226, 227, 231, 234, 245, 246, 247):
            channel='36利率'
        else:
            channel=None
    else:
        channel=None

    return channel


# In[554]:


df_evalue = df_model_vars.copy()


# In[555]:


df_evalue['channel_type'] = df_evalue['channel_id'].apply(channel_type)
df_evalue['channel_type'].value_counts()


# In[556]:


df_evalue['channel_rate'] = df_evalue['channel_id'].apply(channel_rate)
df_evalue['channel_rate'].value_counts()


# In[735]:


# df_evalue['apply_month_new'] = df_evalue['apply_month']
# df_evalue.loc[df_evalue.query("lending_time>='2024-09-01' & lending_time<='2024-09-20'").index, 'apply_month_new']='2024-09_1train'
# df_evalue.loc[df_evalue.query("lending_time>='2024-09-21' & lending_time<='2024-09-30'").index, 'apply_month_new']='2024-09_3oot'


# In[560]:



groupkeys2 = ['apply_month', 'channel_type']
df_ksauc_all_v2 = cal_ks_auc(df_evalue, groupkeys2, labels_models_dict)
# df_ksauc_all_v2.insert(loc=(len(groupkeys2)), column='客群', value='全体', allow_duplicates=False)
df_ksauc_all_v2.head()


# In[749]:



# groupkeys3 = ['apply_month_new', '客群']
# df_ksauc_all_v3 = cal_ks_auc(df_evalue, groupkeys3, labels_models_dict)
# df_ksauc_all_v3.insert(loc=(len(groupkeys3)-1), column='渠道', value='全渠道', allow_duplicates=False)
# df_ksauc_all_v3.head()


# In[558]:



groupkeys4 = ['apply_month', 'channel_rate']
df_ksauc_all_v4 = cal_ks_auc(df_evalue, groupkeys4, labels_models_dict)
df_ksauc_all_v4.head()


# In[588]:


df_ksauc_all_v2.query("channel_type=='桔子商城'")


# In[564]:


df_ksauc_all_v4.query("channel_rate=='227'")


# In[589]:


# target_list = ['fpd10', 'fpd20', 'fpd30']
# labels_models_dict_2 = {target: ['y_prob'] for target in target_list}
# print(labels_models_dict_2)


# In[590]:


# groupkeys5 = ['apply_month_new', '渠道', '客群', 'category']
# df_ksauc_all_v5 = cal_ks_auc(df_evalue, groupkeys5, labels_models_dict_2)
# df_ksauc_all_v5.head()


# In[591]:



# groupkeys6 = ['apply_month_new', '渠道', 'category']
# df_ksauc_all_v6 = cal_ks_auc(df_evalue, groupkeys6, labels_models_dict_2)
# df_ksauc_all_v6.insert(loc=(len(groupkeys6)), column='客群', value='全体', allow_duplicates=False)
# df_ksauc_all_v6.head()


# In[592]:


df_ksauc_all_1 = pd.concat([df_ksauc_all_v2,df_ksauc_all_v4], axis=0)
# df_ksauc_all_2 = pd.concat([df_ksauc_all_v5,df_ksauc_all_v6], axis=0)


# In[593]:


# result_path


# In[594]:



timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

with pd.ExcelWriter(result_path + f'6_效果对比分析_{task_name}_{timestamp}.xlsx') as writer:
    df_ksauc_all_1.to_excel(writer, sheet_name='df_ksauc_all_1')
#     df_ksauc_all_2.to_excel(writer, sheet_name='df_ksauc_all_2')
print(f"数据存储完成！{timestamp}")
print(result_path + f'6_模型对比分析_{task_name}_{timestamp}.xlsx')


# # 6. 评分分布

# In[374]:


result_path


# In[375]:


df_sample.to_csv(result_path + r'全渠道实时提现行为模型fpd30样本.csv',index=False)


# In[376]:


score = 'y_prob_v3'


# In[377]:


df_sample['apply_month'].value_counts()


# In[378]:


c = toad.transform.Combiner()
c.fit(df_sample.query("apply_month=='2024-07'")[[score, target]], y=target, method='quantile', n_bins=20) 
df_sample['score_bins'] = c.transform(df_sample[score], labels=True)


# In[379]:


df_sample['score_bins'].head()


# In[380]:


score_psi_by_month = cal_psi_by_month(df_sample, df_sample.query("apply_month=='2024-08'"), 
                                                [score], 'apply_month_new', c, return_frame = False)
print(score_psi_by_month)

# score_psi_by_dataset = cal_psi_by_month(df_sample, df_sample.query("apply_month=='2024-07'"), 
#                                                 [score], 'data_set', c, return_frame = False)
# print(score_psi_by_dataset)


# In[381]:


def get_model_psi(df, cols, month_col, combiner):
    # 获取所有唯一的月份
    months = sorted(list(set(df[month_col])))
    # 初始化一个空的 DataFrame 来存储 PSI 值
    psi_matrix = pd.DataFrame(index=months, columns=months, dtype=float)
    # 循环计算每个月份与其他月份之间的PSI
    for i, month_i in enumerate(months):
        for j, month_j in enumerate(months):
            if i != j:
                # 从原始数据集中提取特定月份的数据
                df_actual_i = df[df[month_col] == month_i]
                df_expect_j = df[df[month_col] == month_j]
                # 调用函数计算PSI
                psi_ = toad.metrics.PSI(df_actual_i[cols], df_expect_j[cols], 
                                        combiner = combiner, return_frame = False)
                # 将结果存入矩阵
                psi_matrix.loc[month_i, month_j] = psi_
            else:
                # 对角线上的值设为 NaN 或 0，表示同一月份的 PSI
                psi_matrix.loc[month_i, month_j] = 0.0
    
    return psi_matrix


# In[382]:


df_psi_matrix = get_model_psi(df_sample, score, 'apply_month', c)

# 打印最终的 PSI 矩阵
print(df_psi_matrix)


# In[383]:


score_group_by_dataset = calculate_vars_distribute(df_sample, ['score_bins'], target, 'data_set') 
score_group_by_dataset = score_group_by_dataset[['groupvars', 'bins', 'total', 'bad',
                                                 'good', 'bad_rate', 'bad_rate_cum',
                                                 'total_pct_cum', 'ks_bin', 'lift', 'lift_cum']]


# In[384]:


timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

with pd.ExcelWriter(result_path + f'6_评分分布_{task_name}_{timestamp}.xlsx') as writer:
    score_psi_by_month.to_excel(writer, sheet_name='score_psi_by_month')
#     score_psi_by_dataset.to_excel(writer, sheet_name='score_psi_by_dataset')
#     df_score_group_by_month.to_excel(writer, sheet_name='df_score_group_by_month')
#     score_group_by_month.to_excel(writer, sheet_name='score_group_by_month')
#     df_score_group_by_dataset.to_excel(writer, sheet_name='df_score_group_by_dataset')
    score_group_by_dataset.to_excel(writer, sheet_name='score_group_by_dataset')
#     score_group_by_dataset_1.to_excel(writer, sheet_name='score_group_by_dataset_1')
print(f"数据存储完成！:{timestamp}")
print(result_path + f'6_评分分布_{task_name}_{timestamp}.xlsx')




#==============================================================================
# File: 1.3.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:31:54 2019

@author: zixing.mei
"""

import pandas as pd  
import numpy as np  
import os  
#为画图指定路径  
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'  
#读取数据  
data = pd.read_excel( './data/ data_for_tree.xlsx')  
data.head()  

org_lst = ['uid','create_dt','oil_actv_dt','class_new','bad_ind']
agg_lst = ['oil_amount','discount_amount','sale_amount','amount','pay_amount','coupon_amount','payment_coupon_amount']
dstc_lst = ['channel_code','oil_code','scene','source_app','call_source']

df = data[org_lst].copy()
df[agg_lst] = data[agg_lst].copy()
df[dstc_lst] = data[dstc_lst].copy()

base = df[org_lst].copy()
base = base.drop_duplicates(['uid'],keep = 'first')

gn = pd.DataFrame()  
for i in agg_lst:  
    #计算个数  
    tp = pd.DataFrame(df.groupby('uid').apply(
                                       lambda df:len(df[i])).reset_index())  
    tp.columns = ['uid',i + '_cnt']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #求历史特征值大于零的个数  
    tp = pd.DataFrame(df.groupby('uid').apply(
                          lambda df:np.where(df[i]>0,1,0).sum()).reset_index())  
    tp.columns = ['uid',i + '_num']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #对历史数据求和  
    tp = pd.DataFrame(df.groupby('uid').apply(
                                  lambda df:np.nansum(df[i])).reset_index())  
    tp.columns = ['uid',i + '_tot']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #对历史数据求均值  
    tp = pd.DataFrame(df.groupby('uid').apply(
                                    lambda df:np.nanmean(df[i])).reset_index())  
    tp.columns = ['uid',i + '_avg']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #对历史数据求最大值  
    tp = pd.DataFrame(df.groupby('uid').apply(
                                     lambda df:np.nanmax(df[i])).reset_index())  
    tp.columns = ['uid',i + '_max']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #对历史数据求最小值  
    tp = pd.DataFrame(df.groupby('uid').apply(
                                    lambda df:np.nanmin(df[i])).reset_index())  
    tp.columns = ['uid',i + '_min']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #对历史数据求方差  
    tp = pd.DataFrame(df.groupby('uid').apply(
                                     lambda df:np.nanvar(df[i])).reset_index())  
    tp.columns = ['uid',i + '_var']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #对历史数据求极差  
    tp = pd.DataFrame(df.groupby('uid').apply(
                lambda df:np.nanmax(df[i])-np.nanmin(df[i]) ).reset_index())  
    tp.columns = ['uid',i + '_ran']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #对历史数据求变异系数,为防止除数为0，利用0.01进行平滑  
    tp = pd.DataFrame(df.groupby('uid').apply(lambda df:np.nanmean(df[i])/(np.nanvar(df[i])+0.01))).reset_index()  
    tp.columns = ['uid',i + '_cva']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left') 

gc = pd.DataFrame()  
for i in dstc_lst:  
    tp = pd.DataFrame(df.groupby('uid').apply(
                                   lambda df: len(set(df[i]))).reset_index())  
    tp.columns = ['uid',i + '_dstc']  
    if gc.empty == True:  
        gc = tp  
    else:  
        gc = pd.merge(gc,tp,on = 'uid',how = 'left')

fn =  base.merge(gn,on='uid').merge(gc,on='uid')  
fn = pd.merge(fn,gc,on= 'uid')   
fn.shape 

x = fn.drop(['uid','oil_actv_dt','create_dt','bad_ind','class_new'],axis = 1)
y = fn.bad_ind.copy()

from sklearn import tree  
dtree = tree.DecisionTreeRegressor(max_depth = 2,min_samples_leaf = 500,min_samples_split = 5000)  
dtree = dtree.fit(x,y) 

import pydotplus   
from IPython.display import Image  
from sklearn.externals.six import StringIO  
import os  
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

dot_data = StringIO()  
tree.export_graphviz(dtree, out_file=dot_data,  
                         feature_names=x.columns,  
                         class_names=['bad_ind'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())   
Image(graph.create_png())  

dff1 = fn.loc[(fn.amount_tot>9614.5)&(fn.coupon_amount_cnt>6)].copy()  
dff1['level'] = 'past_A'  
dff2 = fn.loc[(fn.amount_tot>9614.5)&(fn.coupon_amount_cnt<=6)].copy()  
dff2['level'] = 'past_B'  
dff3 = fn.loc[fn.amount_tot<=9614.5].copy()  
dff3['level'] = 'past_C'



#==============================================================================
# File: 2 (1).py
#==============================================================================

#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: utils.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-04-20
'''

import json
import math
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, r2_score, recall_score, precision_score, f1_score

FILLNA = -999


# auc
def get_auc(target, y_pred):
    if target.nunique() != 2:
        raise ValueError('the target is not 2 classier target')
    else:
        return roc_auc_score(target, y_pred)


# ks
def get_ks(target, y_pred):
    df = pd.DataFrame({
        'y_pred': y_pred,
        'target': target,
    })
    crossfreq = pd.crosstab(df['y_pred'], df['target'])
    crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
    crossdens['ks'] = abs(crossdens[0] - crossdens[1])
    ks = max(crossdens['ks'])
    return ks


def to_score(x, A=404.65547022, B=72.1347520444):
    if x <= 0.001:
        x = 0.001
    elif x >= 0.999:
        x = 0.999

    result = round(A - B * math.log(x / (1 - x)))

    if result < 0:
        result = 0
    if result > 1200:
        result = 1200
    result = 1200 - result
    return result


def filter_miss(df, miss_threshold=0.9):
    '''

    :param df: 数据集
    :param miss_threshold: 缺失率大于等于该阈值的变量剔除
    :return:
    '''
    names_list = []
    for name, series in df.items():
        n = series.isnull().sum()
        miss_q = n / series.size
        if miss_q < miss_threshold:
            names_list.append(name)
    return names_list


def dump_model_to_file(model, path):
    pickle.dump(model, open(path, "wb"))


def load_model_from_file(path):
    return pickle.load(open(path, 'rb'))


def read_sql_string_from_file(path):
    with open(path, 'r', encoding='utf-8') as fb:
        sql = fb.read()
        return sql


def save_json(res_dict, file, indent=4):
    if isinstance(file, str):
        of = open(file, 'w')

    with of as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=indent)


def load_json(file):
    if isinstance(file, str):
        of = open(file, 'r')

    with of as f:
        res_dict = json.load(f)

    return res_dict


def unpack_tuple(x):
    if len(x) == 1:
        return x[0]
    else:
        return x


def psi(no_base, base, return_frame=False):
    '''
    psi计算
    :param no_base: 非基准数据集
    :param base: 基准数据集
    :param return_frame: 是否返回详细的psi数据集
    :return:
    '''
    psi = list()
    frame = list()

    if isinstance(no_base, pd.DataFrame):
        for col in no_base:
            p, f = calc_psi(no_base[col], base[col])
            psi.append(p)
            frame.append(f)

        psi = pd.Series(psi, index=no_base.columns)

        frame = pd.concat(
            frame,
            keys=no_base.columns,
            names=['columns', 'id'],
        ).reset_index()
        frame = frame.drop(columns='id')
    else:
        psi, frame = calc_psi(no_base, base)

    res = (psi,)

    if return_frame:
        res += (frame,)

    return unpack_tuple(res)


def calc_psi(no_base, base):
    '''
    psi计算的具体逻辑
    :param no_base: 非基准数据集
    :param base: 基准数据集
    :return:
    '''
    no_base_prop = pd.Series(no_base).value_counts(normalize=True, dropna=False)
    base_prop = pd.Series(base).value_counts(normalize=True, dropna=False)

    psi = np.sum((no_base_prop - base_prop) * np.log(no_base_prop / base_prop))

    frame = pd.DataFrame({
        'no_base': no_base_prop,
        'base': base_prop,
    })
    frame.index.name = 'value'

    return psi, frame.reset_index()


# corr
def get_corr(df):
    return df.corr()


# accuracy
def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


# precision
def get_precision(y_true, y_pred):
    if len(np.unique(y_true)) == 2:
        return precision_score(y_true, y_pred)
    else:
        return precision_score(y_true, y_pred, average='macro')


# recall
def get_recall(y_true, y_pred):
    if len(np.unique(y_true)) == 2:
        return recall_score(y_true, y_pred)
    else:
        return precision_score(y_true, y_pred, average='macro')


# f1
def get_f1(y_true, y_pred):
    if len(np.unique(y_true)) == 2:
        return f1_score(y_true, y_pred)
    else:
        return f1_score(y_true, y_pred, average='macro')


# r2
def r2(preds, target):
    return r2_score(target, preds)


# vif
def get_vif(X: pd.DataFrame, y: pd.Series):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    vif = 1 / (1 - r2)
    return vif


def get_best_threshold(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    ks = list(tpr - fpr)
    thresh = threshold[ks.index(max(ks))]
    return thresh


def get_bad_rate(df):
    return df.sum() / df.count()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))



#==============================================================================
# File: 2 (2).py
#==============================================================================

"""Top-level package for autoTreeModel."""

__author__ = """RyanZheng"""
__email__ = 'zhengruiping000@163.com'

from .auto_build_tree_model import AutoBuildTreeModel
from .feature_selection_2_treemodel import ShapSelectFeature, corr_select_feature, psi
from .plot_metrics import get_optimal_cutoff, plot_ks, plot_roc, plot_pr, plot_pr_f1, calc_celue_cm, calc_plot_metrics


__version__ = "0.1.6"
VERSION = __version__



#==============================================================================
# File: 2 (3).py
#==============================================================================

#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: auto_build_tree_model.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-04-27
'''

import gc
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .bayes_opt_tuner import classifiers_model_auto_tune_params
from .feature_selection_2_treemodel import feature_select, stepwise_del_feature, psi
from .utils import get_ks, get_auc, to_score

warnings.filterwarnings('ignore')

from .logger_utils import Logger

log = Logger(level='info', name=__name__).logger


class AutoBuildTreeModel():
    def __init__(self, datasets, fea_names, target, key='key', data_type='type',
                 no_feature_names=['key', 'target', 'apply_time', 'type'], ml_res_save_path='model_result',
                 to_score_a_b={'A': 404.65547022, 'B': 72.1347520444}):

        if data_type not in datasets:
            raise KeyError('train、test数据集标识的字段名不存在！或未进行数据集的划分，请将数据集划分为train、test！！！')

        data_type_ar = np.unique(datasets[data_type])
        if 'train' not in data_type_ar:
            raise KeyError("""没有开发样本，数据集标识字段{}没有`train`该取值！！！""".format(data_type))

        if 'test' not in data_type_ar:
            raise KeyError("""没有验证样本，数据集标识字段{}没有`test`该取值！！！""".format(data_type))

        if target not in datasets:
            raise KeyError('样本中没有目标变量y值！！！')

        # fea_names = [i for i in fea_names if i != key and i != target]
        fea_names = [i for i in fea_names if i not in no_feature_names]
        log.info('数据集变量个数 : {}'.format(len(fea_names)))
        log.info('fea_names is : {}'.format(fea_names))

        self.datasets = datasets
        self.fea_names = fea_names
        self.target = target
        self.key = key
        self.no_feature_names = no_feature_names
        self.ml_res_save_path = os.path.join(ml_res_save_path, time.strftime('%Y%m%d%H%M%S_%S', time.localtime()))
        self.to_score_a_b = to_score_a_b
        self.min_child_samples = max(round(len(datasets[datasets['type'] == 'train']) * 0.02),
                                     50)  # 一个叶子上数据的最小数量. 可以用来处理过拟合
        # self.min_child_samples = max(round(len(datasets['dev']) * 0.02), 50)  # 一个叶子上数据的最小数量. 可以用来处理过拟合

        os.makedirs(self.ml_res_save_path, exist_ok=True)

    def fit(self, is_feature_select=True, is_auto_tune_params=True, is_stepwise_del_feature=True,
            feature_select_method='shap', method_threhold=0.001,
            corr_threhold=0.8, psi_threhold=0.2):
        '''

        Args:
            is_feature_select:
            is_auto_tune_params:
            feature_select_method:
            method_threhold:
            corr_threhold:
            psi_threhold:

        Returns: xgboost.sklearn.XGBClassifier或lightgbm.sklearn.LGBClassifier；list
            返回最优模型，入模变量list

        '''
        log.info('*' * 30 + '开始自动建模' + '*' * 30)

        log.info('*' * 30 + '获取变量名和数据集' + '*' * 30)
        fea_names = self.fea_names.copy()
        dev_data = self.datasets[self.datasets['type'] == 'train']
        nodev_data = self.datasets[self.datasets['type'] == 'test']

        del self.datasets;
        gc.collect()

        # dev_data = self.datasets['dev']
        # nodev_data = self.datasets['nodev']

        # params = {
        #     'learning_rate': 0.05,
        #     'n_estimators': 200,
        #     'max_depth': 3,
        #     'min_child_weight': 5,
        #     'gamma': 7,
        #     'subsample': 0.7,
        #     'colsample_bytree': 0.9,
        #     'colsample_bylevel': 0.7,
        #     'reg_alpha': 10,
        #     'reg_lambda': 10,
        #     'scale_pos_weight': 1
        # }
        # log.info('默认参数 {}'.format(params))
        #
        # log.info('构建基础模型')

        if is_feature_select:
            log.info('需要进行变量筛选')
            fea_names = feature_select({'dev': dev_data, 'nodev': nodev_data}, fea_names, self.target,
                                       feature_select_method, method_threhold,
                                       corr_threhold,
                                       psi_threhold)

        if is_auto_tune_params:
            log.info('需要进行自动调参')
            best_model = classifiers_model_auto_tune_params(train_data=(dev_data[fea_names], dev_data[self.target]),
                                                            test_data=(nodev_data[fea_names], nodev_data[self.target]))
            params = best_model.get_params()

        if is_stepwise_del_feature:
            log.info('需要逐步的删除变量')
            _, fea_names = stepwise_del_feature({'dev': dev_data, 'nodev': nodev_data}, fea_names, self.target, params)

        # 最终模型
        log.info('使用自动调参选出来的最优参数+筛选出来的变量，构建最终模型')
        log.info('最终变量的个数{}, 最终变量{}'.format(len(fea_names), fea_names))
        log.info('自动调参选出来的最优参数{}'.format(params))
        xgb_clf = XGBClassifier(**params)
        xgb_clf.fit(dev_data[fea_names], dev_data[self.target])

        # ###
        # pred_nodev = xgb_clf.predict_proba(nodev_data[fea_names])[:, 1]
        # pred_dev = xgb_clf.predict_proba(dev_data[fea_names])[:, 1]
        # df_pred_nodev = pd.DataFrame({'target': nodev_data[self.target], 'p': pred_nodev}, index=nodev_data.index)
        # df_pred_dev = pd.DataFrame({'target': dev_data[self.target], 'p': pred_dev}, index=dev_data.index)
        # ###

        # ###
        # df_pred_nodev = nodev_data[self.no_feature_names + fea_names]
        # df_pred_dev = dev_data[self.no_feature_names + fea_names]
        # df_pred_nodev['p'] = xgb_clf.predict_proba(df_pred_nodev[fea_names])[:, 1]
        # df_pred_dev['p'] = xgb_clf.predict_proba(df_pred_dev[fea_names])[:, 1]
        # ###

        ###
        df_pred_nodev = nodev_data[self.no_feature_names]
        df_pred_dev = dev_data[self.no_feature_names]
        df_pred_nodev['p'] = xgb_clf.predict_proba(nodev_data[fea_names])[:, 1]
        df_pred_dev['p'] = xgb_clf.predict_proba(dev_data[fea_names])[:, 1]
        ###

        # 计算auc、ks、psi
        test_ks = get_ks(df_pred_nodev[self.target], df_pred_nodev['p'])
        train_ks = get_ks(df_pred_dev[self.target], df_pred_dev['p'])
        test_auc = get_auc(df_pred_nodev[self.target], df_pred_nodev['p'])
        train_auc = get_auc(df_pred_dev[self.target], df_pred_dev['p'])

        q_cut_list = np.arange(0, 1, 1 / 20)
        bins = np.append(np.unique(np.quantile(df_pred_nodev['p'], q_cut_list)), df_pred_nodev['p'].max() + 0.1)
        df_pred_nodev['range'] = pd.cut(df_pred_nodev['p'], bins=bins, precision=0, right=False).astype(str)
        df_pred_dev['range'] = pd.cut(df_pred_dev['p'], bins=bins, precision=0, right=False).astype(str)
        nodev_psi = psi(df_pred_nodev['range'], df_pred_dev['range'])
        res_dict = {'dev_auc': train_auc, 'nodev_auc': test_auc, 'dev_ks': train_ks, 'nodev_ks': test_ks,
                    'nodev_dev_psi': nodev_psi}
        log.info('auc & ks & psi: {}'.format(res_dict))
        log.info('*' * 30 + '自动构建模型完成！！！' + '*' * 30)

        ##############
        log.info('*' * 30 + '建模相关结果开始保存！！！' + '*' * 30)
        joblib.dump(xgb_clf.get_booster(), os.path.join(self.ml_res_save_path, 'xgb.ml'))
        joblib.dump(xgb_clf, os.path.join(self.ml_res_save_path, 'xgb_sk.ml'))
        json.dump(xgb_clf.get_params(), open(os.path.join(self.ml_res_save_path, 'xgb.params'), 'w'))
        xgb_clf.get_booster().dump_model(os.path.join(self.ml_res_save_path, 'xgb.txt'))
        pd.DataFrame([res_dict]).to_csv(os.path.join(self.ml_res_save_path, 'xgb_auc_ks_psi.csv'), index=False)


        pd.DataFrame(list(xgb_clf.get_booster().get_fscore().items()),
                     columns=['fea_names', 'weight']
                     ).sort_values('weight', ascending=False).set_index('fea_names').to_csv(
            os.path.join(self.ml_res_save_path, 'xgb_featureimportance.csv'))

        nodev_data[self.no_feature_names + fea_names].head(500).to_csv(os.path.join(self.ml_res_save_path, 'xgb_input.csv'),
                                                                       index=False)

        ##############pred to score
        df_pred_nodev['score'] = df_pred_nodev['p'].map(
            lambda x: to_score(x, self.to_score_a_b['A'], self.to_score_a_b['B']))
        df_pred_dev['score'] = df_pred_dev['p'].map(
            lambda x: to_score(x, self.to_score_a_b['A'], self.to_score_a_b['B']))
        ##############pred to score


        df_pred_nodev.append(df_pred_dev).to_csv(os.path.join(self.ml_res_save_path, 'xgb_pred_to_report_data.csv'),
                                                 index=False)

        log.info('*' * 30 + '建模相关结果保存完成！！！保存路径为：{}'.format(self.ml_res_save_path) + '*' * 30)

        return xgb_clf, fea_names


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    ##**************************************************随机生成的数据例子**************************************************
    ##**************************************************随机生成的数据例子**************************************************
    ##**************************************************随机生成的数据例子**************************************************
    X, y = make_classification(n_samples=1000, n_features=30, n_classes=2, random_state=328)
    data = pd.DataFrame(X)
    data['target'] = y
    data['key'] = [i for i in range(len(data))]
    data.columns = ['f0_radius', 'f0_texture', 'f0_perimeter', 'f0_area', 'f0_smoothness',
                    'f0_compactness', 'f0_concavity', 'f0_concave_points', 'f0_symmetry',
                    'f0_fractal_dimension', 'f1_radius_error', 'f1_texture_error', 'f1_perimeter_error',
                    'f2_area_error', 'f2_smoothness_error', 'f2_compactness_error', 'f2_concavity_error',
                    'f2_concave_points_error', 'f2_symmetry_error', 'f2_fractal_dimension_error',
                    'f3_radius', 'f3_texture', 'f3_perimeter', 'f3_area', 'f3_smoothness',
                    'f3_compactness', 'f3_concavity', 'f3_concave_points', 'f3_symmetry',
                    'f3_fractal_dimension', 'target', 'key']

    dev, nodev = train_test_split(data, test_size=0.3, random_state=328)
    dev['type'] = 'train'
    nodev['type'] = 'test'
    data = dev.append(nodev)

    ###TODO 注意修改
    client_batch = 'TT00p1'
    key, target, data_type = 'key', 'target', 'type'  # key是主键字段名，target是目标变量y的字段名，data_type是train、test数据集标识的字段名
    ml_res_save_path = '../examples/example_model_result/{}'.format(
        client_batch)
    ###TODO 注意修改

    ###TODO 下面代码基本可以不用动
    # 初始化
    autobtmodel = AutoBuildTreeModel(datasets=data,  # 训练模型的数据集
                                     fea_names=list(data.columns),  # 数据集的字段名
                                     target=target,  # 目标变量y字段名
                                     key=key,  # 主键字段名
                                     data_type=data_type,  # train、test数据集标识的字段名
                                     no_feature_names=[key, target, data_type],  # 数据集中不用于开发模型的特征字段名，即除了x特征的其它字段名
                                     ml_res_save_path=ml_res_save_path,  # 建模相关结果保存路径
                                     )

    # 训练模型
    model, in_model_fea = autobtmodel.fit(is_feature_select=True,  # 特征筛选
                                          is_auto_tune_params=True,  # 是否自动调参
                                          is_stepwise_del_feature=True,  # 是进行逐步的变量删除
                                          feature_select_method='shap',  # 特征筛选指标
                                          method_threhold=0.001,  # 特征筛选指标阈值
                                          corr_threhold=0.8,  # 相关系数阈值
                                          psi_threhold=0.1,  # PSI阈值
                                          )
    ###TODO 上面代码基本可以不用动
    ##**************************************************随机生成的数据例子**************************************************
    ##**************************************************随机生成的数据例子**************************************************
    ##**************************************************随机生成的数据例子**************************************************

    # ##**************************************************虚构现实数据例子**************************************************
    # ##**************************************************虚构现实数据例子**************************************************
    # ##**************************************************虚构现实数据例子**************************************************
    #
    # ###TODO 注意修改，读取建模数据
    # data = pd.read_csv(
    #     '../examples/example_data/TT01p1_id_y_fea_to_model.csv')
    # ###TODO 注意修改，读取建模数据
    #
    # ###TODO 注意修改
    # client_batch = 'TT01p1'
    # key, target, data_type = 'id', 'target', 'type'  # key是主键字段名，target是目标变量y的字段名，data_type是train、test数据集标识的字段名
    # ml_res_save_path = '../examples/example_model_result/{}'.format(
    #     client_batch)
    # ###TODO 注意修改
    #
    # ###TODO 下面代码基本可以不用动
    # # 初始化
    # autobtmodel = AutoBuildTreeModel(datasets=data,  # 训练模型的数据集
    #                                  fea_names=list(data.columns),  # 数据集的字段名
    #                                  target=target,  # 目标变量y字段名
    #                                  key=key,  # 主键字段名
    #                                  data_type=data_type,  # train、test数据集标识的字段名
    #                                  no_feature_names=[key, target, data_type] + ['apply_time'],
    #                                  # 数据集中不用于开发模型的特征字段名，即除了x特征的其它字段名
    #                                  ml_res_save_path=ml_res_save_path,  # 建模相关结果保存路径
    #                                  )
    #
    # # 训练模型
    # model, in_model_fea = autobtmodel.fit(is_feature_select=True,  # 特征筛选
    #                                       is_auto_tune_params=True,  # 是否自动调参
    #                                       is_stepwise_del_feature=True,  # 是进行逐步的变量删除
    #                                       feature_select_method='shap',  # 特征筛选指标
    #                                       method_threhold=0.001,  # 特征筛选指标阈值
    #                                       corr_threhold=0.8,  # 相关系数阈值
    #                                       psi_threhold=0.1,  # PSI阈值
    #                                       )
    # ###TODO 上面代码基本可以不用动
    #
    # ##**************************************************虚构现实数据例子**************************************************
    # ##**************************************************虚构现实数据例子**************************************************
    # ##**************************************************虚构现实数据例子**************************************************



#==============================================================================
# File: 2 (4).py
#==============================================================================

"""Main module."""



#==============================================================================
# File: 2 (5).py
#==============================================================================

#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: bayes_opt_tuner.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-04-25
'''

import time
import warnings

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .utils import get_ks, get_auc, get_accuracy, get_recall, get_precision, get_f1, r2

warnings.filterwarnings('ignore')

from .logger_utils import Logger

log = Logger(level='info', name=__name__).logger


class ModelTune():
    def __init__(self):
        self.base_model = None
        self.best_model = None
        self.tune_params = None
        self.loss = np.inf
        self.default_params = None
        self.int_params = None
        self.init_params = None
        self.metrics = None
        self._metrics_score = []
        self.scores = []

    def get_model(self):
        return self.best_model

    def _map_metrics(self):
        mapper = {
            'accuracy': get_accuracy,
            'f1': get_f1,
            'precision': get_precision,
            'recall': get_recall,
            'r2': r2,
            'auc': get_auc,
            'ks': get_ks
        }

        for metric in self.metrics:
            if metric not in mapper:
                raise ValueError('指定的指标 ''`{}` 不支持'.format(metric))
            self._metrics_score.append(mapper[metric])
            self.scores.append(0.)

    def fit(self, train_data=(), test_data=()
            , init_points=30, iterations=120, metrics=[]):
        '''

        Args:
            train_data:
            test_data:
            init_points:
            iterations:
            metrics:

        Returns:

        '''

        if len(metrics) > 0:
            self.metrics = metrics
        self._map_metrics()

        X_train, y_train = train_data
        X_test, y_test = test_data

        def loss_fun(train_result, test_result, weight=0.3):

            # return test_result - 2 ** abs(test_result - train_result)
            return test_result - 2 ** abs(test_result - train_result) * weight

        # def loss_fun(train_result, test_result):
        #     train_result = train_result * 100
        #     test_result = test_result * 100
        #
        #     return train_result - 2 ** abs(train_result - test_result)

        def obj_fun(**params):
            for param in self.int_params:
                params[param] = int(round(params[param]))

            model = self.base_model(**params, **self.default_params)
            model.fit(X_train, y_train)

            pred_test = model.predict_proba(X_test)[:, 1]
            pred_train = model.predict_proba(X_train)[:, 1]

            # test_auc = get_auc(y_test, pred_test)
            # train_auc = get_auc(y_train, pred_train)
            # print('test_auc is : ', test_auc)
            # print('train_auc is : ', train_auc)

            test_ks = get_ks(y_test, pred_test)
            train_ks = get_ks(y_train, pred_train)
            # print('test_ks is : ', test_ks)
            # print('train_ks is : ', train_ks)

            # maximize = loss_fun(train_auc, test_auc)
            maximize = loss_fun(train_ks, test_ks)
            # print('max_result is : ', maximize)
            # max_result = loss_fun(train_ks, test_ks) * 2 + loss_fun(train_auc, test_auc)

            loss = -maximize
            if loss < self.loss:
                self.loss = loss
                self.best_model = model
                # print('best model result is {}'.format(loss))
                # print('best model params is : ')
                # print(self.best_model.get_params())
                for i, _metric in enumerate(self._metrics_score):
                    self.scores[i] = _metric(y_test, pred_test)
            # print('current obj_fun result is : ', maximize)

            return maximize

        params_optimizer = BayesianOptimization(obj_fun, self.tune_params, random_state=1)
        log.info('需要优化的超参数是 : {}'.format(params_optimizer.space.keys))

        log.info('开始优化超参数!!!')
        start = time.time()

        params_optimizer.maximize(init_points=1, n_iter=0, acq='ei',
                                  xi=0.0)
        params_optimizer.probe(self.init_params, lazy=True)

        # params_optimizer.probe(self.init_params, lazy=True)

        params_optimizer.maximize(init_points=0, n_iter=0)

        params_optimizer.maximize(init_points=init_points, n_iter=iterations, acq='ei',
                                  xi=0.0)  # init_points：探索开始探索之前的迭代次数；iterations：方法试图找到最大值的迭代次数
        # params_optimizer.maximize(init_points=init_points, n_iter=iterations, acq='ucb', xi=0.0, alpha=1e-6)
        end = time.time()
        log.info('优化参数结束!!! 共耗时{} 分钟'.format((end - start) / 60))
        log.info('最优参数是 : {}'.format(params_optimizer.max['params']))
        log.info('{} model 最大化的结果 : {}'.format(type(self.best_model), params_optimizer.max['target']))


class ClassifierModel(ModelTune):
    def __init__(self):
        super().__init__()
        self.metrics = ['auc', 'ks']


class RegressorModel(ModelTune):
    def __init__(self):
        super().__init__()
        self.metrics = ['r2', 'rmse']


class XGBClassifierTuner(ClassifierModel):
    def __init__(self):
        super().__init__()

        self.base_model = XGBClassifier
        self.tune_params = {
            'learning_rate': (0.01, 0.15),
            'n_estimators': (90, 300),
            'max_depth': (2, 7),
            'min_child_weight': (1, 300),
            'subsample': (0.4, 1.0),
            'colsample_bytree': (0.3, 1.0),
            'colsample_bylevel': (0.5, 1.0),
            'gamma': (0, 20.0),
            'reg_alpha': (0, 20.0),
            'reg_lambda': (0, 20.0),
            # 'scale_pos_weight': (1, 5),
            # 'max_delta_step': (0, 10)
        }

        self.default_params = {
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'nthread': -1
        }

        self.init_params = {
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': 3,
            'min_child_weight': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.9,
            'colsample_bylevel': 0.7,
            'gamma': 7,
            'reg_alpha': 10,
            'reg_lambda': 10,
            # 'scale_pos_weight': 1
        }

        self.int_params = ['max_depth', 'n_estimators']


class LGBClassifierTuner(ClassifierModel):
    '''
    英文版：https://lightgbm.readthedocs.io/en/latest/Parameters.html

    中文版：https://lightgbm.apachecn.org/#/docs/6

    其他注解：https://medium.com/@gabrieltsen
    '''

    def __init__(self):
        super().__init__()

        self.base_model = LGBMClassifier
        self.tune_params = {
            'max_depth': (3, 15),
            'num_leaves': (16, 128),
            'learning_rate': (0.01, 0.2),
            'reg_alpha': (0, 100),
            'reg_lambda': (0, 100),
            'min_child_samples': (1, 100),
            'min_child_weight': (0.01, 100),
            'colsample_bytree': (0.5, 1),
            'subsample': (0.5, 1),
            'subsample_freq': (2, 10),
            'n_estimators': (90, 500),

        }

        self.default_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'random_state': 1024,
            'n_jobs': -1,
            'num_threads': -1,
            'verbose': -1,

        }

        self.init_params = {
            'max_depth': -1,
            "num_leaves": 31,
            "learning_rate": 0.02,
            "reg_alpha": 0.85,
            "reg_lambda": 3,
            "min_child_samples": 20,  # TODO注意修改 .......sklearn:min_child_samples    原生:min_data、min_data_in_leaf
            "min_child_weight": 0.05,  # sklearn:min_child_weight    原生:min_hessian、min_sum_hessian_in_leaf
            "colsample_bytree": 0.9,  # sklearn:colsample_bytree   原生:feature_fraction
            "subsample": 0.8,  # sklearn:subsample  原生:bagging_fraction
            "subsample_freq": 2,  # sklearn:subsample_freq  原生:bagging_freq
            "n_estimators": 100  # sklearn:n_estimators  原生:num_boost_round、num_iterations
        }

        self.int_params = ['max_depth', 'num_leaves', 'min_child_samples', 'n_estimators', 'subsample_freq']


classifiers_dic = {
    # 'lr': LogisticRegressionTuner,
    # 'rf': RandomForestClassifierTuner,
    'xgb': XGBClassifierTuner,
    'lgb': LGBClassifierTuner
}


def classifiers_model_auto_tune_params(models=['xgb'], metrics=[], train_data=(), test_data=()
                                       , init_points=30, iterations=120, verbose=1):
    '''

    Args:
        models:
        metrics:
        train_data:
        test_data:
        init_points:
        iterations:
        verbose:

    Returns:

    '''
    best_model = None
    if not isinstance(models, list):
        raise AttributeError('models参数必须是一个列表, ', '但实际是 {}'.format(type(models)))
    if len(models) == 0:
        models = list(classifiers_dic.keys())
    classifiers = []
    for model in models:
        if model in classifiers_dic:
            classifiers.append(classifiers_dic[model])
    loss = np.inf
    _model = None
    for classifier in classifiers:
        if verbose:
            log.info("优化 {}...".format(classifier()))
        _model = classifier()
        _model.fit(train_data=train_data,
                   test_data=test_data
                   , init_points=init_points, iterations=iterations, metrics=metrics)

        _loss = _model.loss
        if verbose:
            _show_fit_log(_model)
        if _loss < loss:
            loss = _loss
            best_model = _model

    return best_model.get_model()


def _show_fit_log(model):
    _out = '  最优结果: '
    _out += ' loss: {:.3}'.format(model.loss)
    _out += ' 测试集 '
    for i, _metric in enumerate(model.metrics):
        _out += ' {}: {:.3}'.format(_metric[:3],
                                    model.scores[i])
    log.info(_out)


if __name__ == '__main__':
    X = pd.read_pickle('X_train.pkl')
    X = pd.DataFrame(X)
    y = pd.read_pickle('y_train.pkl')
    y = pd.Series(y)
    X_test = pd.read_pickle('X_test.pkl')
    X_test = pd.DataFrame(X_test)
    y_test = pd.read_pickle('y_test.pkl')
    y_test = pd.Series(y_test)

    ####build model
    best_model = classifiers_model_auto_tune_params(train_data=(X, y), test_data=(X_test, y_test), verbose=1,
                                                    init_points=1,
                                                    iterations=2)
    # best_model = classifiers_model_auto_tune_params(train_data=(X, y), test_data=(X_test, y_test), verbose=1)
    print('classifiers_model run over!!!')
    print(type(best_model))
    print(best_model.get_params())
    train_pred_y = best_model.predict_proba(X)[:, 1]
    test_pred_y = best_model.predict_proba(X_test)[:, 1]
    ####build model

    #####build model
    # best_model = LGBMClassifier()
    # best_model.fit(X,y)
    # print(best_model.get_params())
    # train_pred_y = best_model.predict_proba(X)[:, 1]
    # test_pred_y = best_model.predict_proba(X_test)[:, 1]
    #####build model

    # #####build model
    # import lightgbm as lgb
    #
    # init_params = {
    #     "boosting_type": "gbdt",
    #     "objective": "binary",
    #     "metric": "auc",
    # }
    # best_model = lgb.train(params=init_params, train_set=lgb.Dataset(X, y), valid_sets=lgb.Dataset(X_test, y_test))
    # best_model.save_model('lgb.txt')
    # json_model = best_model.dump_model()
    # import json
    #
    # with open('lgb.json', 'w') as f:
    #     json.dump(json_model, f)
    # train_pred_y = best_model.predict(X)
    # test_pred_y = best_model.predict(X_test)
    # #####build model

    train_auc = get_auc(y, train_pred_y)
    test_auc = get_auc(y_test, test_pred_y)
    train_ks = get_ks(y, train_pred_y)
    test_ks = get_ks(y_test, test_pred_y)
    print('train_auc is : ', train_auc, 'test_auc is : ', test_auc)
    print('train_ks is : ', train_ks, 'test_ks is : ', test_ks)

    # # #####build model
    # params = {
    #     'learning_rate': 0.05,
    #     'n_estimators': 200,
    #     'max_depth': 3,
    #     'min_child_weight': 5,
    #     'gamma': 7,
    #     'subsample': 0.7,
    #     'colsample_bytree': 0.9,
    #     'colsample_bylevel': 0.7,
    #     'reg_alpha': 10,
    #     'reg_lambda': 10,
    #     'scale_pos_weight': 1
    # }
    #
    # clf = XGBClassifier(**params)
    # clf.fit(X, y)
    # estimator = clf.get_booster()
    # temp = estimator.save_raw()[4:]
    # # #####build model

    ####构建数据
    # from sklearn.datasets import make_classification
    # from sklearn.model_selection import train_test_split
    # import pickle
    # X, y = make_classification(n_samples=10000, random_state=1024)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #
    # with open('X_train.pkl', 'wb') as f:
    #     f.write(pickle.dumps(X_train))
    # with open('y_train.pkl', 'wb') as f:
    #     f.write(pickle.dumps(y_train))
    # with open('X_test.pkl', 'wb') as f:
    #     f.write(pickle.dumps(X_test))
    # with open('y_test.pkl', 'wb') as f:
    #     f.write(pickle.dumps(y_test))



#==============================================================================
# File: 2 (6).py
#==============================================================================

"""Console script for autotreemodel."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for autotreemodel."""
    click.echo("Replace this message by putting your code into "
               "autotreemodel.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover



#==============================================================================
# File: 2 (7).py
#==============================================================================

#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: feature_selection_2_treemodel.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-04-26
'''

import warnings

import numpy as np
import pandas as pd
import shap
from tqdm import tqdm
from xgboost import XGBClassifier

from .utils import get_ks

warnings.filterwarnings('ignore')

from .logger_utils import Logger

log = Logger(level='info', name=__name__).logger


class ShapSelectFeature:
    def __init__(self, estimator, linear=False, estimator_is_fit_final=False):
        self.estimator = estimator
        self.linear = linear
        self.weight = None
        self.estimator_is_fit_final = estimator_is_fit_final

    def fit(self, X, y, exclude=None):
        '''

        Args:
            X:
            y:
            exclude:

        Returns:

        '''
        if exclude is not None:
            X = X.drop(columns=exclude)
        if not self.estimator_is_fit_final:
            self.estimator.fit(X, y)
        if self.linear:
            explainer = shap.LinearExplainer(self.estimator, X)
        else:
            estimator = self.estimator.get_booster()
            temp = estimator.save_raw()[4:]
            estimator.save_raw = lambda: temp
            explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)
        shap_abs = np.abs(shap_values)
        shap_importance_list = shap_abs.mean(0)
        self.weight = pd.DataFrame(shap_importance_list, index=X.columns, columns=['weight'])
        return self.weight


def corr_select_feature(frame, by='auc', threshold=0.95, return_frame=False):
    '''

    Args:
        frame:
        by:
        threshold:
        return_frame:

    Returns:

    '''
    if not isinstance(by, (str, pd.Series)):

        if isinstance(by, pd.DataFrame):
            if by.shape[1] == 1:
                by = pd.Series(by.iloc[:, 0].values, index=by.index)
            else:
                by = pd.Series(by.iloc[:, 1].values, index=by.iloc[:, 0].values)
            # by = pd.Series(by.iloc[:, 1].values, index=frame.columns)
        else:
            by = pd.Series(by, index=frame.columns)

    # 给重要性排下序
    by.sort_values(ascending=False, inplace=True)
    # print('给重要性排下序：', by)

    # df = frame.copy()

    by.index = by.index.astype(type(list(frame.columns)[0]))
    df_corr = frame[list(by.index)].fillna(-999).corr().abs()  # 填充
    # df_corr = frame[list(by.index)].corr().abs()

    ix, cn = np.where(np.triu(df_corr.values, 1) > threshold)

    del_all = []

    if len(ix):

        for i in df_corr:

            if i not in del_all:
                # 找出与当前特征的相关性大于域值的特征
                del_tmp = list(df_corr[i][(df_corr[i] > threshold) & (df_corr[i] != 1)].index)

                # 比较当前特征与需要删除的特征的特征重要性
                if del_tmp:
                    by_tmp = by.loc[del_tmp]
                    del_l = list(by_tmp[by_tmp <= by.loc[i]].index)
                    del_all.extend(del_l)

    del_f = list(set(del_all))

    if return_frame:
        r = frame.drop(columns=del_f)
        return (del_f, r)

    return del_f


def psi(no_base, base, return_frame=False):
    '''
    psi计算
    Args:
        no_base:非基准数据集
        base:基准数据集
        return_frame:是否返回详细的psi数据集

    Returns:
        float或Series
    '''
    psi = list()
    frame = list()

    if isinstance(no_base, pd.DataFrame):
        for col in no_base:
            p, f = calc_psi(no_base[col], base[col])
            psi.append(p)
            frame.append(f)

        psi = pd.Series(psi, index=no_base.columns)

        frame = pd.concat(
            frame,
            keys=no_base.columns,
            names=['columns', 'id'],
        ).reset_index()
        frame = frame.drop(columns='id')
    else:
        psi, frame = calc_psi(no_base, base)

    res = (psi,)

    if return_frame:
        res += (frame,)

    return unpack_tuple(res)


def unpack_tuple(x):
    if len(x) == 1:
        return x[0]
    else:
        return x


def calc_psi(no_base, base):
    '''
    psi计算的具体逻辑
    Args:
        no_base: 非基准数据集
        base: 基准数据集

    Returns:
        float或DataFrame
    '''
    no_base_prop = pd.Series(no_base).value_counts(normalize=True, dropna=False)
    base_prop = pd.Series(base).value_counts(normalize=True, dropna=False)

    psi = np.sum((no_base_prop - base_prop) * np.log(no_base_prop / base_prop))

    frame = pd.DataFrame({
        'no_base': no_base_prop,
        'base': base_prop,
    })
    frame.index.name = 'value'

    return psi, frame.reset_index()


def feature_select(datasets, fea_names, target, feature_select_method='shap', method_threhold=0.001,
                   corr_threhold=0.8, psi_threhold=0.1, params={}):
    '''

    Args:
        datasets:
        fea_names:
        target:
        feature_select_method:
        method_threhold:
        corr_threhold:
        psi_threhold:
        params:

    Returns:

    '''
    dev_data = datasets['dev']
    nodev_data = datasets['nodev']

    params = {
        'learning_rate': params.get('learning_rate', 0.05),
        'n_estimators': params.get('n_estimators', 200),
        'max_depth': params.get('max_depth', 3),
        'min_child_weight': params.get('min_child_weight', 5),
        'subsample': params.get('subsample', 0.7),
        'colsample_bytree': params.get('colsample_bytree', 0.9),
        'colsample_bylevel': params.get('colsample_bylevel', 0.7),
        'gamma': params.get('gamma', 7),
        'reg_alpha': params.get('reg_alpha', 10),
        'reg_lambda': params.get('reg_lambda', 10)
    }

    xgb_clf = XGBClassifier(**params)
    xgb_clf.fit(dev_data[fea_names], dev_data[target])

    if feature_select_method == 'shap':
        shap_model = ShapSelectFeature(estimator=xgb_clf, estimator_is_fit_final=True)
        fea_weight = shap_model.fit(dev_data[fea_names], dev_data[target])
        fea_weight.sort_values(by='weight', inplace=True)
        fea_weight = fea_weight[fea_weight['weight'] >= method_threhold]
        log.info('Shap阈值: {}'.format(method_threhold))
        log.info('Shap剔除的变量个数: {}'.format(len(fea_names) - fea_weight.shape[0]))
        log.info('Shap保留的变量个数: {}'.format(fea_weight.shape[0]))
        fea_names = list(fea_weight.index)
        log.info('*' * 50 + 'Shap筛选变量' + '*' * 50)


    elif feature_select_method == 'feature_importance':
        fea_weight = pd.DataFrame(list(xgb_clf.get_booster().get_score(importance_type='gain').items()),
                                  columns=['fea_names', 'weight']
                                  ).sort_values('weight').set_index('fea_names')
        fea_weight = fea_weight[fea_weight['weight'] >= method_threhold]
        log.info('feature_importance阈值: {}'.format(method_threhold))
        log.info('feature_importance剔除的变量个数: {}'.format(len(fea_names) - fea_weight.shape[0]))
        fea_names = list(fea_weight.index)
        log.info('feature_importance保留的变量个数: {}'.format(fea_names))
        log.info('*' * 50 + 'feature_importance筛选变量' + '*' * 50)

    if corr_threhold:
        del_fea_list = corr_select_feature(dev_data[fea_names], by=fea_weight, threshold=0.8)
        log.info('相关性阈值: {}'.format(corr_threhold))
        log.info('相关性剔除的变量个数: {}'.format(len(del_fea_list)))
        fea_names = [i for i in fea_names if i not in del_fea_list]
        # fea_names = list(set(fea_names) - set(del_fea_list))
        log.info('相关性保留的变量个数: {}'.format(len(fea_names)))
        log.info('*' * 50 + '相关性筛选变量' + '*' * 50)

    if psi_threhold:
        psi_df = psi(dev_data[fea_names], nodev_data[fea_names]).sort_values(0)
        psi_df = psi_df.reset_index()
        psi_df = psi_df.rename(columns={'index': 'fea_names', 0: 'psi'})
        psi_list = psi_df[psi_df.psi < psi_threhold].fea_names.tolist()
        log.info('PSI阈值: {}'.format(psi_threhold))
        log.info('PSI剔除的变量个数: {}'.format(len(fea_names) - len(psi_list)))
        fea_names = [i for i in fea_names if i in psi_list]
        # fea_names = list(set(fea_names) and set(psi_list))
        log.info('PSI保留的变量个数: {}'.format(len(fea_names)))
        log.info('*' * 50 + 'PSI筛选变量' + '*' * 50)

    return fea_names


def stepwise_del_feature(datasets, fea_names, target, params={}):
    '''

    Args:
        datasets:
        fea_names:
        target:
        params:

    Returns:

    '''
    log.info("开始逐步删除变量")
    dev_data = datasets['dev']
    nodev_data = datasets['nodev']
    stepwise_del_params = {
        'learning_rate': params.get('learning_rate', 0.05),
        'n_estimators': params.get('n_estimators', 200),
        'max_depth': params.get('max_depth', 3),
        'min_child_weight': params.get('min_child_weight', 5),
        'subsample': params.get('subsample', 0.7),
        'colsample_bytree': params.get('colsample_bytree', 0.9),
        'colsample_bylevel': params.get('colsample_bylevel', 0.7),
        'gamma': params.get('gamma', 7),
        'reg_alpha': params.get('reg_alpha', 10),
        'reg_lambda': params.get('reg_lambda', 10)
    }

    xgb_clf = XGBClassifier(**stepwise_del_params)
    xgb_clf.fit(dev_data[fea_names], dev_data[target])

    pred_test = xgb_clf.predict_proba(nodev_data[fea_names])[:, 1]
    pred_train = xgb_clf.predict_proba(dev_data[fea_names])[:, 1]

    test_ks = get_ks(nodev_data[target], pred_test)
    train_ks = get_ks(dev_data[target], pred_train)
    log.info('test_ks is : {}'.format(test_ks))
    log.info('train_ks is : {}'.format(train_ks))

    train_number, oldks, del_list = 0, test_ks, list()
    log.info('train_number: {}, test_ks: {}'.format(train_number, test_ks))

    # while True:
    #     flag = True
    #     for fea_name in tqdm(fea_names):
    #         print('变量{}进行逐步：'.format(fea_name))
    #         names = [fea for fea in fea_names if fea_name != fea]
    #         print('变量names is：', names)
    #         xgb_clf.fit(dev_data[names], dev_data[target])
    #         train_number += 1
    #         pred_test = xgb_clf.predict_proba(nodev_data[names])[:, 1]
    #         test_ks = get_ks(nodev_data[target], pred_test)
    #         if test_ks >= oldks:
    #             oldks = test_ks
    #             flag = False
    #             del_list.append(fea_name)
    #             log.info(
    #                 '等于或优于之前结果 train_number: {}, test_ks: {} by feature: {}'.format(train_number, test_ks, fea_name))
    #             fea_names = names
    #     if flag:
    #         print('=====================又重新逐步==========')
    #         break
    #     log.info("结束逐步删除变量 train_number: %s, test_ks: %s del_list: %s" % (train_number, oldks, del_list))
    #     print('oldks is ：',oldks)
    #     print('fea_names is : ',fea_names)

    for fea_name in tqdm(fea_names):
        names = [fea for fea in fea_names if fea_name != fea]
        xgb_clf.fit(dev_data[names], dev_data[target])
        train_number += 1
        pred_test = xgb_clf.predict_proba(nodev_data[names])[:, 1]
        test_ks = get_ks(nodev_data[target], pred_test)
        if test_ks >= oldks:
            oldks = test_ks
            del_list.append(fea_name)
            log.info(
                '等于或优于之前结果 train_number: {}, test_ks: {} by feature: {}'.format(train_number, test_ks, fea_name))
            fea_names = names
    log.info("结束逐步删除变量 train_number: %s, test_ks: %s del_list: %s" % (train_number, oldks, del_list))

    ########################
    log.info('逐步剔除的变量个数: {}'.format(del_list))
    fea_names = [i for i in fea_names if i not in del_list]
    # fea_names = list(set(fea_names) - set(del_list))
    log.info('逐步保留的变量个数: {}'.format(len(fea_names)))
    log.info('*' * 50 + '逐步筛选变量' + '*' * 50)

    return del_list, fea_names



#==============================================================================
# File: 2 (8).py
#==============================================================================

#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: logger_utils.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-04-20
'''

import logging


# 日志输出
class Logger():
    # 日志级别关系映射
    level_relations = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    def __init__(self, level="info", name=None,
                 fmt="%(asctime)s - %(name)s[line:%(lineno)d] - %"
                     "(levelname)s: %(message)s"):
        logging.basicConfig(level=self.level_relations.get(level), format=fmt)
        self.logger = logging.getLogger(name)



#==============================================================================
# File: 2 (9).py
#==============================================================================

#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: plot_metrics.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2022-08-26
'''

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve


def get_optimal_cutoff(fpr_recall, tpr_precision, threshold, is_f1=False):
    if is_f1:
        youdenJ_f1score = (2 * tpr_precision * fpr_recall) / (tpr_precision + fpr_recall)
    else:
        youdenJ_f1score = tpr_precision - fpr_recall
    point_index = np.argmax(youdenJ_f1score)
    optimal_threshold = threshold[point_index]
    point = [fpr_recall[point_index], tpr_precision[point_index]]
    return optimal_threshold, point


def plot_ks(y_true: pd.Series, y_pred: pd.Series, output_path='', title=''):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks = max(tpr - fpr)  ###计算ks的值

    plt.figure(figsize=(6, 6))
    x = np.arange(len(thresholds)) / len(thresholds)
    plt.plot(x, tpr, lw=1)
    plt.plot(x, fpr, lw=1)
    plt.plot(x, tpr - fpr, lw=1, linestyle='--', label='KS curve (KS = %0.4f)' % ks)

    optimal_th, optimal_point = get_optimal_cutoff(fpr_recall=fpr, tpr_precision=tpr, threshold=thresholds)
    optimal_th_index = np.where(thresholds == optimal_th)
    plt.plot(optimal_th_index[0][0] / len(thresholds), ks, marker='o', color='r')
    plt.text(optimal_th_index[0][0] / len(thresholds), ks, (float('%.4f' % optimal_point[0]),
                                                            float('%.4f' % optimal_point[1])),
             ha='right', va='top', fontsize=12)
    plt.text(optimal_th_index[0][0] / len(thresholds), ks, f'threshold:{optimal_th:.4f}', fontsize=12)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Thresholds Index')
    plt.ylabel('TPR FPR KS')
    name = '{} KS Curve'.format(title)
    plt.title(name)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(output_path + name, bbox_inches='tight')
    plt.show()
    return {'KS': ks, 'KS最大值-threshold': optimal_th}


def plot_roc(y_true: pd.Series, y_pred: pd.Series, output_path='', title=''):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_value = roc_auc_score(y_true, y_pred)  ###计算auc的值

    lw = 2
    plt.figure(figsize=(6, 6))

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.4f)' % auc_value)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    optimal_th, optimal_point = get_optimal_cutoff(fpr_recall=fpr, tpr_precision=tpr, threshold=thresholds)
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], (float('%.4f' % optimal_point[0]),
                                                  float('%.4f' % optimal_point[1])),
             ha='right', va='top', fontsize=12)
    plt.text(optimal_point[0], optimal_point[1], f'threshold:{optimal_th:.4f}', fontsize=12)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    name = '{} ROC Curve'.format(title)
    plt.title(name)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(output_path + name, bbox_inches='tight')
    plt.show()
    return {'AUC': auc_value}


def plot_pr(y_true: pd.Series, y_pred: pd.Series, output_path='', title=''):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1score = (2 * precision * recall) / (precision + recall)  ###计算F1score
    max_f1score = max(f1score)

    lw = 2
    plt.figure(figsize=(6, 6))

    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PR curve (F1score = %0.4f)' % max_f1score)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    optimal_th, optimal_point = get_optimal_cutoff(fpr_recall=recall, tpr_precision=precision, threshold=thresholds,
                                                   is_f1=True)
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], (float('%.4f' % optimal_point[0]),
                                                  float('%.4f' % optimal_point[1])),
             ha='right', va='top', fontsize=12)
    plt.text(optimal_point[0], optimal_point[1], f'threshold:{optimal_th:.4f}', fontsize=12)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    name = '{} PR Curve'.format(title)
    plt.title(name)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(output_path + name, bbox_inches='tight')
    plt.show()
    return {'F1_Score最大值': max_f1score, 'F1_Score最大值-threshold': optimal_th, '模型拐点': optimal_th, '阀值': optimal_th,
            'Precision': optimal_point[1], 'Recall': optimal_point[0], 'F1_Score': max_f1score}


def plot_pr_f1(y_true: pd.Series, y_pred: pd.Series, output_path='', title=''):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    thresholds = np.insert(thresholds, 0, 0, axis=None)
    f1score = (2 * precision * recall) / (precision + recall)  ###计算F1score

    x = np.arange(len(thresholds)) / len(thresholds)

    pr_f1_dict = {'Precision': precision, 'Recall': recall, 'F1_score': f1score}

    for i in pr_f1_dict:
        plt.figure(figsize=(6, 6))

        plt.plot(x, pr_f1_dict[i], lw=1)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Thresholds Index')
        plt.ylabel('{}'.format(i))
        name = '{} {} Curve'.format(title, i)
        plt.title(name)
        plt.savefig(output_path + name, bbox_inches='tight')
        plt.show()

    return {'Thresholds': list(thresholds), '模型召回率': list(recall), '模型精准率': list(precision), '模型F1-score': list(f1score)}


def calc_celue_cm(df: pd.DataFrame, target='target', to_bin_col='p'):
    q_cut_list = np.arange(0, 1, 1 / 10) + 0.1
    confusion_matrix_df = pd.DataFrame()
    for i in q_cut_list:
        df['pred_label'] = np.where(df[to_bin_col] >= i, 1, 0)
        tmp_list = []
        tmp_list.append(i)

        tn, fp, fn, tp = confusion_matrix(np.array(df[target]), np.array(df['pred_label'])).ravel()

        tmp_list.extend([tp, fp, tn, fn])

        confusion_matrix_df = confusion_matrix_df.append(pd.DataFrame(tmp_list).T)

    # confusion_matrix_df.columns = ['阈值', 'TP', 'FP', 'TN', 'FN']
    confusion_matrix_df.columns = ['阈值', '实际正样本-预测为正样本', '实际负样本-预测为正样本', '实际负样本-预测为负样本', '实际正样本-预测为负样本']
    confusion_matrix_df.set_index('阈值', inplace=True)
    confusion_matrix_df['sum'] = confusion_matrix_df.apply(lambda x: x.sum(), axis=1)

    # return confusion_matrix_df
    return confusion_matrix_df.to_dict()


def save_json(res_dict, file, indent=4):
    if isinstance(file, str):
        of = open(file, 'w')

    with of as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=indent)


def load_json(file):
    if isinstance(file, str):
        of = open(file, 'r')

    with of as f:
        res_dict = json.load(f)

    return res_dict


def calc_plot_metrics(df: pd.DataFrame, to_bin_col='p', target='target', curve_save_path=''):
    data = {k: v for k, v in df.groupby('type')}
    data.update({'all': df})

    for data_type, type_df in data.items():
        res_save_path = os.path.join(curve_save_path, data_type)
        os.makedirs(res_save_path, exist_ok=True)

        res_dict = {}
        res_dict.update(plot_roc(type_df[target], type_df[to_bin_col], res_save_path, data_type))
        res_dict.update(plot_ks(type_df[target], type_df[to_bin_col], res_save_path, data_type))
        res_dict.update(plot_pr(type_df[target], type_df[to_bin_col], res_save_path, data_type))
        res_dict.update(calc_celue_cm(type_df, target, to_bin_col))
        res_dict.update(plot_pr_f1(type_df[target], type_df[to_bin_col], res_save_path, data_type))

        ###相关指标保存json格式
        save_json(res_dict, os.path.join(res_save_path, '{}_res_json.json'.format(data_type)))


if __name__ == "__main__":
    ######读取数据
    data_path = 'TD47p25combine_td_to_report_data.csv'
    df = pd.read_csv(data_path)
    df = df[df['label'].notnull()]

    print(len(df))

    ######结果保存路径
    curve_save_path = 'curve_result'
    calc_plot_metrics(df=df, to_bin_col='td', target='label', curve_save_path=curve_save_path)



#==============================================================================
# File: 2.2.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:50:38 2019

@author: zixing.mei
"""

def Num(feature,mth):  
    df=data.loc[:,feature +'1': feature +str(mth)]  
    auto_value=np.where(df>0,1,0).sum(axis=1)  
    return feature +'_num'+str(mth),auto_value 
 
def Avg(feature, mth):  
    df=data.loc[:,feature +'1': feature +str(mth)]  
    auto_value=np.nanmean(df,axis = 1 )  
    return feature +'_avg'+str(mth),auto_value  
   
def Msg(feature, mth):  
    df=data.loc[:,feature +'1': feature +str(mth)]  
    df_value=np.where(df>0,1,0)  
    auto_value=[]  
    for i in range(len(df_value)):  
        row_value=df_value[i,:]  
        if row_value.max()<=0:  
            indexs='0'  
            auto_value.append(indexs)  
        else:  
            indexs=1  
            for j in row_value:  
                if j>0:  
                    break  
                indexs+=1  
            auto_value.append(indexs)  
    return feature +'_msg'+str(mth),auto_value 

def Cav(feature, mth):  
    df=data.loc[:,feature +'1':inv+str(mth)]  
    auto_value = df[feature +'1']/np.nanmean(df,axis = 1 )   
    return feature +'_cav'+str(mth),auto_value 

def Mai(feature, mth):  
    arr=np.array(data.loc[:,feature +'1': feature +str(mth)])       
    auto_value = []  
    for i in range(len(arr)):  
        df_value = arr[i,:]  
        value_lst = []  
        for k in range(len(df_value)-1):  
            minus = df_value[k] - df_value[k+1]  
            value_lst.append(minus)  
        auto_value.append(np.nanmax(value_lst))       
    return feature +'_mai'+str(mth),auto_value 

def Ran(feature, mth):  
    df=data.loc[:,feature +'1': feature +str(mth)]  
    auto_value = np.nanmax(df,axis = 1 )  -  np.nanmin(df,axis = 1 )
    return feature +'_ran'+str(mth),auto_value   




#==============================================================================
# File: 2.3.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:52:55 2019

@author: zixing.mei
"""

from sklearn.preprocessing import OneHotEncoder   
enc = OneHotEncoder()  
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])   
enc.transform([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]).toarray()  

import math  
#离散型变量 WOE编码  
class charWoe(object):  
    def __init__(self, datasets, dep, weight, vars):  
                #数据集字典，{'dev':训练集,'val':测试集,'off':跨时间验证集}  
        self.datasets = datasets 
        self.devf = datasets.get("dev", "") #训练集  
        self.valf = datasets.get("val", "") #测试集  
        self.offf = datasets.get("off", "") #跨时间验证集  
        self.dep = dep #标签  
        self.weight = weight #样本权重  
        self.vars = vars #参与建模的特征名  
        self.nrows, self.ncols = self.devf.shape #样本数，特征数  
  
    def char_woe(self):  
        #得到每一类样本的个数，且加入平滑项使得bad和good都不为0  
        dic = dict(self.devf.groupby([self.dep]).size())  
        good  = dic.get(0, 0) + 1e-10
        bad =  dic.get(1, 0) + 1e-10  
        #对每一个特征进行遍历。  
        for col in self.vars:  
            #得到每一个特征值对应的样本数。  
            data = dict(self.devf[[col, self.dep]].groupby(
                                                  [col, self.dep]).size())  
            ''' 
            当前特征取值超过100个的时候，跳过当前取值。 
            因为取值过多时，WOE分箱的效率较低，建议对特征进行截断。 
            出现频率过低的特征值统一赋值，放入同一箱内。 
            '''  
            if len(data) > 100:  
                print(col, "contains too many different values...")
                continue  
            #打印取值个数  
            print(col, len(data))  
            dic = dict()  
            #k是特征名和特征取值的组合，v是样本数  
            for (k, v) in data.items():  
                #value为特征名，dp为特征取值  
                value, dp = k  
                #如果找不到key设置为一个空字典  
                dic.setdefault(value, {})   
                #字典中嵌套字典  
                dic[value][int(dp)] = v  
            for (k, v) in dic.items():  
                dic[k] = {str(int(k1)): v1 for (k1, v1) in v.items()}  
                dic[k]["cnt"] = sum(v.values())  
                bad_rate = round(dic[k].get("1", 0)/ dic[k]["cnt"], 5)
                dic[k]["bad_rate"] = bad_rate  
            #利用定义的函数进行合并。  
            dic = self.combine_box_char(dic)  
            #对每个特征计算WOE值和IV值  
            for (k, v) in dic.items():  
                a = v.get("0", 1) / good + 1e-10  
                b = v.get("1", 1) / bad + 1e-10  
                dic[k]["Good"] = v.get("0", 0)  
                dic[k]["Bad"] = v.get("1", 0)  
                dic[k]["woe"] = round(math.log(a / b), 5)  
            ''' 
            按照分箱后的点进行分割， 
            计算得到每一个特征值的WOE值， 
            将原始特征名加上'_woe'后缀，并赋予WOE值。 
            '''  
            for (klis, v) in dic.items():  
                for k in klis.split(","):  
                    #训练集进行替换  
                    self.devf.loc[self.devf[col]==k,
                                                    "%s_woe" % col] = v["woe"]
                    #测试集进行替换  
                    if not isinstance(self.valf, str):  
                        self.valf.loc[self.valf[col]==k,
                                                     "%s_woe" % col] = v["woe"]
                    #跨时间验证集进行替换  
                    if not isinstance(self.offf, str):  
                        self.offf.loc[self.offf[col]==k,                     
                                                     "%s_woe" % col] = v["woe"]
        #返回新的字典，其中包含三个数据集。  
        return {"dev": self.devf, "val": self.valf, "off": self.offf}
  
    def combine_box_char(self, dic):  
        ''' 
        实施两种分箱策略。 
        1.不同箱之间负样本占比差异最大化。 
        2.每一箱的样本量不能过少。 
        '''  
        #首先合并至10箱以内。按照每一箱负样本占比差异最大化原则进行分箱。  
        while len(dic) >= 10:  
            #k是特征值，v["bad_rate"]是特征值对应的负样本占比
            bad_rate_dic = {k: v["bad_rate"] 
                                             for (k, v) in dic.items()}  
            #按照负样本占比排序。因为离散型变量 是无序的，
                        #可以直接写成负样本占比递增的形式。  
            bad_rate_sorted = sorted(bad_rate_dic.items(),
                                                         key=lambda x: x[1])
            #计算每两箱之间的负样本占比差值。
                        #准备将差值最小的两箱进行合并。  
            bad_rate = [bad_rate_sorted[i+1][1]-
                                      bad_rate_sorted[i][1] for i in 
                                      range(len(bad_rate_sorted)-1)]
            min_rate_index = bad_rate.index(min(bad_rate))  
            #k1和k2是差值最小的两箱的key.  
            k1, k2 = bad_rate_sorted[min_rate_index][0],\
                                     bad_rate_sorted[min_rate_index+1][0]  
            #得到重新划分后的字典，箱的个数比之前少一。  
            dic["%s,%s" % (k1, k2)] = dict()  
            dic["%s,%s" % (k1, k2)]["0"] = dic[k1].get("0", 0)\
                                                            + dic[k2].get("0", 0)
            dic["%s,%s" % (k1, k2)]["1"] = dic[k1].get("1", 0) \
                                                            + dic[k2].get("1", 0)
            dic["%s,%s" % (k1, k2)]["cnt"] = dic[k1]["cnt"]\
                                                              + dic[k2]["cnt"]
            dic["%s,%s" % (k1, k2)]["bad_rate"] = round(
                                    dic["%s,%s" % (k1, k2)]["1"] / 
                                    dic["%s,%s" % (k1, k2)]["cnt"],5)  
            #删除旧的key。  
            del dic[k1], dic[k2]  
        ''' 
        结束循环后，箱的个数应该少于10。 
        下面实施第二种分箱策略。 
        将样本数量少的箱合并至其他箱中，以保证每一箱的样本数量不要太少。 
        '''  
        #记录当前样本最少的箱的个数。      
        min_cnt = min([v["cnt"] for v in dic.values()])  
        #当样本数量小于总样本的5%或者总箱的个数大于5的时候，对箱进行合并  
        while min_cnt < self.nrows * 0.05 and len(dic) > 5:  
            min_key = [k for (k, v) in dic.items() 
                                     if v["cnt"] == min_cnt][0]  
            bad_rate_dic = {k: v["bad_rate"] 
                                          for (k, v) in dic.items()}  
            bad_rate_sorted = sorted(bad_rate_dic.items(),
                                              key=lambda x: x[1])  
            keys = [k[0] for k in bad_rate_sorted]  
            min_index = keys.index(min_key)  
            ''''' 
            同样想保持合并后箱之间的负样本占比差异最大化。 
            由于箱的位置不同，按照三种不同情况进行分类讨论。 
            '''  
            #如果是第一箱，和第二项合并  
            if min_index == 0:  
                k1, k2 = keys[:2]  
            #如果是最后一箱，和倒数第二箱合并  
            elif min_index == len(dic) - 1:  
                k1, k2 = keys[-2:]  
            #如果是中间箱，和bad_rate值相差最小的箱合并  
            else:  
                bef_bad_rate = dic[min_key]["bad_rate"]\
                                             -dic[keys[min_index - 1]]["bad_rate"]
                aft_bad_rate = dic[keys[min_index+1]]["bad_rate"] - dic[min_key]["bad_rate"]
                if bef_bad_rate < aft_bad_rate:  
                    k1, k2 = keys[min_index - 1], min_key
                else:  
                    k1, k2 = min_key, keys[min_index + 1]
            #得到重新划分后的字典，箱的个数比之前少一。  
            dic["%s,%s" % (k1, k2)] = dict()  
            dic["%s,%s" % (k1, k2)]["0"] = dic[k1].get("0", 0) \
                                                             + dic[k2].get("0", 0)
            dic["%s,%s" % (k1, k2)]["1"] = dic[k1].get("1", 0)\
                                                             + dic[k2].get("1", 0)
            dic["%s,%s" % (k1, k2)]["cnt"] = dic[k1]["cnt"]\
                                                                  +dic[k2]["cnt"]
            dic["%s,%s" % (k1, k2)]["bad_rate"] = round(
                                                dic["%s,%s" % (k1, k2)]["1"] / 
                                                dic["%s,%s" % (k1, k2)]["cnt"],5)
            #删除旧的key。  
            del dic[k1], dic[k2]  
            #当前最小的箱的样本个数  
            min_cnt = min([v["cnt"] for v in dic.values()])  
        return dic  




#==============================================================================
# File: 2.4.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:08:42 2019

@author: zixing.mei
"""
import math

def sloveKS(self, model, X, Y, Weight):  
    Y_predict = [s[1] for s in model.predict_proba(X)]  
    nrows = X.shape[0]  
    #还原权重  
    lis = [(Y_predict[i], Y.values[i], Weight[i]) for i in range(nrows)]
    #按照预测概率倒序排列  
    ks_lis = sorted(lis, key=lambda x: x[0], reverse=True)        
    KS = list()  
    bad = sum([w for (p, y, w) in ks_lis if y > 0.5])  
    good = sum([w for (p, y, w) in ks_lis if y <= 0.5])  
    bad_cnt, good_cnt = 0, 0  
    for (p, y, w) in ks_lis:  
        if y > 0.5:  
            #1*w 即加权样本个数  
            bad_cnt += w                
        else:  
            #1*w 即加权样本个数  
            good_cnt += w               
        ks = math.fabs((bad_cnt/bad)-(good_cnt/good))  
        KS.append(ks)  
    return max(KS) 

def slovePSI(self, model, dev_x, val_x):  
    dev_predict_y = [s[1] for s in model.predict_proba(dev_x)]  
    dev_nrows = dev_x.shape[0]  
    dev_predict_y.sort()  
    #等频分箱成10份  
    cutpoint = [-100] + [dev_predict_y[int(dev_nrows/10*i)] 
                         for i in range(1, 10)] + [100]  
    cutpoint = list(set(cutpoint))  
    cutpoint.sort()
    val_predict_y = [s[1] for s in list(model.predict_proba(val_x))]  
    val_nrows = val_x.shape[0]  
    PSI = 0  
    #每一箱之间分别计算PSI  
    for i in range(len(cutpoint)-1):  
        start_point, end_point = cutpoint[i], cutpoint[i+1]  
        dev_cnt = [p for p in dev_predict_y 
                                 if start_point <= p < end_point]  
        dev_ratio = len(dev_cnt) / dev_nrows + 1e-10  
        val_cnt = [p for p in val_predict_y 
                                 if start_point <= p < end_point]  
        val_ratio = len(val_cnt) / val_nrows + 1e-10  
        psi = (dev_ratio - val_ratio) * math.log(dev_ratio/val_ratio)
        PSI += psi  
    return PSI  

import xgboost as xgb  
from xgboost import plot_importance  
  
class xgBoost(object):  
    def __init__(self, datasets, uid, dep, weight, 
                                  var_names, params, max_del_var_nums=0):
        self.datasets = datasets  
        #样本唯一标识，不参与建模  
        self.uid = uid       
        #二分类标签  
        self.dep = dep     
        #样本权重  
        self.weight = weight      
        #特征列表  
        self.var_names = var_names    
        #参数字典，未指定字段使用默认值  
        self.params = params     
        #单次迭代最多删除特征的个数  
        self.max_del_var_nums = max_del_var_nums    
        self.row_num = 0  
        self.col_num = 0  
  
    def training(self, min_score=0.0001, modelfile="", output_scores=list()):  
        lis = self.var_names[:]  
        dev_data = self.datasets.get("dev", "")  #训练集  
        val_data = self.datasets.get("val", "")  #测试集  
        off_data = self.datasets.get("off", "")  #跨时间验证集
                #从字典中查找参数值，没有则使用第二项作为默认值  
        model = xgb.XGBClassifier(
                           learning_rate=self.params.get("learning_rate", 0.1),
              n_estimators=self.params.get("n_estimators", 100),  
              max_depth=self.params.get("max_depth", 3),  
              min_child_weight=self.params.get("min_child_weight", 1),subsample=self.params.get("subsample", 1),  
              objective=self.params.get("objective", 
                                                             "binary:logistic"),
              nthread=self.params.get("nthread", 10),  
              scale_pos_weight=self.params.get("scale_pos_weight", 1),
              random_state=0,  
              n_jobs=self.params.get("n_jobs", 10),  
              reg_lambda=self.params.get("reg_lambda", 1),  
              missing=self.params.get("missing", None) )  
        while len(lis) > 0:   
            #模型训练  
            model.fit(X=dev_data[self.var_names], y=dev_data[self.dep])  
            #得到特征重要性  
            scores = model.feature_importances_     
            #清空字典  
            lis.clear()      
            ''' 
            当特征重要性小于预设值时， 
            将特征放入待删除列表。 
            当列表长度超过预设最大值时，跳出循环。 
            即一次只删除限定个数的特征。 
            '''  
            for (idx, var_name) in enumerate(self.var_names):  
                #小于特征重要性预设值则放入列表  
                if scores[idx] < min_score:    
                    lis.append(var_name)  
                #达到预设单次最大特征删除个数则停止本次循环  
                if len(lis) >= self.max_del_var_nums:     
                    break  
            #训练集KS  
            devks = self.sloveKS(model, dev_data[self.var_names],
                                       dev_data[self.dep], dev_data[self.weight])
            #初始化ks值和PSI  
            valks, offks, valpsi, offpsi = 0.0, 0.0, 0.0, 0.0 
            #测试集KS和PSI  
            if not isinstance(val_data, str):  
                valks = self.sloveKS(model,
                                                      val_data[self.var_names], 
                                                      val_data[self.dep], 
                                                      val_data[self.weight])  
                valpsi = self.slovePSI(model,
                                                        dev_data[self.var_names],
                                                        val_data[self.var_names])
            #跨时间验证集KS和PSI  
            if not isinstance(off_data, str):  
                offks = self.sloveKS(model,
                                                  off_data[self.var_names],
                                                  off_data[self.dep],
                                                  off_data[self.weight])  
                offpsi = self.slovePSI(model,
                                                     dev_data[self.var_names],
                                                     off_data[self.var_names])  
            #将三个数据集的KS和PSI放入字典  
            dic = {"devks": float(devks), 
                                 "valks": float(valks),
                                  "offks": offks,  
                 "valpsi": float(valpsi),
                                  "offpsi": offpsi}  
            print("del var: ", len(self.var_names), 
                                       "-->", len(self.var_names) - len(lis),
                                       "ks: ", dic, ",".join(lis))
        self.var_names = [var_name for var_name in self.var_names if var_name not in lis]
        plot_importance(model)  
        #重新训练，准备进入下一循环  
        model = xgb.XGBClassifier(
                             learning_rate=self.params.get("learning_rate", 0.1),
               n_estimators=self.params.get("n_estimators", 100),
                 max_depth=self.params.get("max_depth", 3),  
                 min_child_weight=self.params.get("min_child_weight",1),
               subsample=self.params.get("subsample", 1),  
               objective=self.params.get("objective", 
                                                        "binary:logistic"),  
               nthread=self.params.get("nthread", 10),  
               scale_pos_weight=self.params.get("scale_pos_weight",1),
               random_state=0,  
               n_jobs=self.params.get("n_jobs", 10),  
               reg_lambda=self.params.get("reg_lambda", 1),  
               missing=self.params.get("missing", None))  











#==============================================================================
# File: 2.5.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:11:59 2019

@author: zixing.mei
"""

def target_value(self,old_devks,old_offks,target,devks,offks,w=0.2):  
    '''  
    如果参数设置为"best"，使用最优调参策略， 
    否则使用跨时间测试集KS最大策略。 
    '''  
    if target == "best":  
        return offks-abs(devks-offks)*w
    else:  
        return offks  

def check_params(self, dev_data, off_data, params, param, train_number, step, target, 
                                                            targetks, old_devks, old_offks):  
    ''' 
    当前向搜索对调参策略有提升时， 
    继续前向搜索。 
    否则进行后向搜索 
    '''  
    while True:  
        try:  
            if params[param] + step > 0:  
                params[param] += step  
                model = xgb.XGBClassifier(
                                   max_depth=params.get("max_depth", 3),
                                   learning_rate=params.get("learning_rate", 0.05),
                                   n_estimators=params.get("n_estimators", 100),
                                   min_child_weight=params.get(
                                                       "min_child_weight", 1),
                                   subsample=params.get("subsample", 1),  
                                   scale_pos_weight=params.get(
                                   "scale_pos_weight", 1),
                                   nthread=10,n_jobs=10, random_state=0)  
                model.fit(dev_data[self.var_names],
                                              dev_data[self.dep],
                                              dev_data[self.weight])  
                devks = self.sloveKS(model, 
                                                       dev_data[self.var_names], 
                                                       dev_data[self.dep], 
                                                       dev_data[self.weight])  
                offks = self.sloveKS(model, 
                                                       off_data[self.var_names], 
                                                       off_data[self.dep], 
                                                       off_data[self.weight])  
                train_number += 1  
                targetks_n = self.target_value(
                                                      old_devks=old_devks, 
                                                      old_offks=old_offks, 
                                                      target=target,  
                                                      devks=devks, 
                                                      offks=offks)  
                if targetks < targetks_n:  
                    targetks = targetks_n  
                    old_devks = devks  
                    old_offks = offks  
                else:  
                    break  
            else:  
                break  
        except:  
            break  
    params[param] -= step  
    return params, targetks, train_number  

def auto_choose_params(self, target="offks"):  
    """ 
    "mzh1": offks + (offks - devks) * 0.2 最大化   
        "mzh2": (offks + (offks - devks) * 0.2)**2 最大化 
        其余取值均使用跨时间测试集offks  最大化
    当业务稳定性较差时，应将0.2改为更大的值 
    """  
    dev_data = self.datasets.get("dev", "")  
    off_data = self.datasets.get("off", "")  
    #设置参数初始位置  
    params = {  
        "max_depth": 5,  
        "learning_rate": 0.09,  
        "n_estimators": 120,  
        "min_child_weight": 50,  
        "subsample": 1,  
        "scale_pos_weight": 1,  
        "reg_lambda": 21  
    }  
    model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),  
                                  learning_rate=params.get("learning_rate", 0.05),
                 n_estimators=params.get("n_estimators", 100),
                 min_child_weight=params.get("min_child_weight",1),
                 subsample=params.get("subsample", 1),
                 scale_pos_weight=params.get("scale_pos_weight",1),
                 reg_lambda=params.get("reg_lambda", 1),
                 nthread=8, n_jobs=8, random_state=7)  
    model.fit(dev_data[self.var_names], 
                      dev_data[self.dep],
                      dev_data[self.weight])  
    devks = self.sloveKS(model, 
                               dev_data[self.var_names], 
                               dev_data[self.dep], 
                               dev_data[self.weight])  
    offks = self.sloveKS(model,
                                    off_data[self.var_names], 
                                    off_data[self.dep], 
                                    off_data[self.weight])  
    train_number = 0  
    #设置调参步长  
    dic = {  
        "learning_rate": [0.05, -0.05],  
        "max_depth": [1, -1],  
        "n_estimators": [20, 5, -5, -20],  
        "min_child_weight": [20, 5, -5, -20],  
        "subsample": [0.05, -0.05],  
        "scale_pos_weight": [20, 5, -5, -20],  
        "reg_lambda": [10, -10]  
    }  
    #启用调参策略  
    targetks = self.target_value(old_devks=devks, 
                                       old_offks=offks, target=target, 
                                       devks=devks, offks=offks)  
    old_devks = devks  
    old_offks = offks  
    #按照参数字典，双向搜索最优参数  
    while True:  
        targetks_lis = []  
        for (key, values) in dic.items():  
            for v in values:  
                if v + params[key] > 0:  
                    params, targetks, train_number = \
                                                       self.check_params(dev_data, 
                                                       off_data, params, 
                                                       key, train_number,  
                            v, target, targetks, 
                                                       old_devks, old_offks)  
                    targetks_n = self.target_value(
                                                         old_devks=old_devks, 
                                                         old_offks=old_offks, 
                                                         target=target,  
                             devks=devks, offks=offks)
                    if targetks < targetks_n:  
                        old_devks = devks  
                        old_offks = offks  
                        targetks_lis.append(targetks)  
        if not targetks_lis:  
            break  
    print("Best params: ", params)  
    model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),  
                   learning_rate=params.get("learning_rate", 0.05),
                  n_estimators=params.get("n_estimators", 100),
                 min_child_weight=params.get("min_child_weight",1),
                 subsample=params.get("subsample", 1),  
                 scale_pos_weight=params.get("scale_pos_weight",1),
                 reg_lambda=params.get("reg_lambda", 1),  
                 nthread=10, n_jobs=10, random_state=0)  
    model.fit(dev_data[self.var_names], 
                  dev_data[self.dep], dev_data[self.weight])  




#==============================================================================
# File: 2.6.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:13:32 2019

@author: zixing.mei
"""

def auto_delete_vars(self):  
    dev_data = self.datasets.get("dev", "")  
    off_data = self.datasets.get("off", "")  
    params = self.params  
    model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),  
                 learning_rate=params.get("learning_rate", 0.05),
                 n_estimators=params.get("n_estimators", 100),
                 min_child_weight=params.get("min_child_weight",1),
                  subsample=params.get("subsample", 1),  
                  scale_pos_weight=params.get("scale_pos_weight",1),
                 reg_lambda=params.get("reg_lambda", 1),  
                 nthread=8, n_jobs=8, random_state=7)  
    model.fit(dev_data[self.var_names], 
                  dev_data[self.dep], dev_data[self.weight])  
    offks = self.sloveKS(model, off_data[self.var_names], 
                               off_data[self.dep], off_data[self.weight])  
    train_number = 0  
    print("train_number: %s, offks: %s" % (train_number, offks))  
    del_list = list()  
    oldks = offks  
    while True:  
        bad_ind = True  
        for var_name in self.var_names:  
            #遍历每一个特征  
            model=xgb.XGBClassifier(
                                  max_depth=params.get("max_depth", 3),  
                 learning_rate=params.get("learning_rate",0.05),
                 n_estimators=params.get("n_estimators", 100), 
                 min_child_weight=params.get("min_child_weight",1),
                 subsample=params.get("subsample", 1),  
                 scale_pos_weight=params.get("scale_pos_weight",1),
                 reg_lambda=params.get("reg_lambda", 1),  
                 nthread=10,n_jobs=10,random_state=7)  
            #将当前特征从模型中去掉  
            names = [var for var in self.var_names 
                                    if var_name != var]  
            model.fit(dev_data[names], dev_data[self.dep], 
                                  dev_data[self.weight])  
            train_number += 1  
            offks = self.sloveKS(model, off_data[names], 
                                     off_data[self.dep], off_data[self.weight])
            ''' 
            比较KS是否有提升， 
            如果有提升或者武明显变化， 
            则可以将特征去掉 
            '''  
            if offks >= oldks:  
                oldks = offks  
                bad_ind = False  
                del_list.append(var_name)  
                self.var_names = names  
            else:  
                continue
        if bad_ind:  
            break  
    print("(End) train_n: %s, offks: %s del_list_vars: %s" 
                  % (train_number, offks, del_list))  




#==============================================================================
# File: 2.7.py
#==============================================================================

import pandas as pd  
from sklearn.metrics import roc_auc_score,roc_curve,auc  
from sklearn import metrics  
from sklearn.linear_model import LogisticRegression  
import numpy as np  
data = pd.read_csv('Acard.txt')  
data.head()  
data.obs_mth.unique()
train = data[data.obs_mth != '2018-11-30'].reset_index().copy()  
val = data[data.obs_mth == '2018-11-30'].reset_index().copy()  
  
feature_lst = ['person_info','finance_info','credit_info','act_info']  
x = train[feature_lst]  
y = train['bad_ind']  
  
val_x =  val[feature_lst]  
val_y = val['bad_ind']  
  
lr_model = LogisticRegression(C=0.1,class_weight='balanced')  
lr_model.fit(x,y) 
 
y_pred = lr_model.predict_proba(x)[:,1]  
fpr_lr_train,tpr_lr_train,_ = roc_curve(y,y_pred)  
train_ks = abs(fpr_lr_train - tpr_lr_train).max()  
print('train_ks : ',train_ks)  
  
y_pred = lr_model.predict_proba(val_x)[:,1]  
fpr_lr,tpr_lr,_ = roc_curve(val_y,y_pred)  
val_ks = abs(fpr_lr - tpr_lr).max()  
print('val_ks : ',val_ks)  

from matplotlib import pyplot as plt  
plt.plot(fpr_lr_train,tpr_lr_train,label = 'train LR')  
plt.plot(fpr_lr,tpr_lr,label = 'evl LR')  
plt.plot([0,1],[0,1],'k--')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC Curve')  
plt.legend(loc = 'best')  
plt.show()  
model = lr_model  
row_num, col_num = 0, 0  
bins = 20  
Y_predict = [s[1] for s in model.predict_proba(val_x)]  
Y = val_y  
nrows = Y.shape[0]  
lis = [(Y_predict[i], Y[i]) for i in range(nrows)]  
ks_lis = sorted(lis, key=lambda x: x[0], reverse=True)  
bin_num = int(nrows/bins+1)  
bad = sum([1 for (p, y) in ks_lis if y > 0.5])  
good = sum([1 for (p, y) in ks_lis if y <= 0.5])  
bad_cnt, good_cnt = 0, 0  
KS = []  
BAD = []  
GOOD = []  
BAD_CNT = []  
GOOD_CNT = []  
BAD_PCTG = []  
BADRATE = []  
dct_report = {}  
for j in range(bins):  
    ds = ks_lis[j*bin_num: min((j+1)*bin_num, nrows)]  
    bad1 = sum([1 for (p, y) in ds if y > 0.5])  
    good1 = sum([1 for (p, y) in ds if y <= 0.5])  
    bad_cnt += bad1  
    good_cnt += good1  
    bad_pctg = round(bad_cnt/sum(val_y),3)  
    badrate = round(bad1/(bad1+good1),3)  
    ks = round(math.fabs((bad_cnt / bad) - (good_cnt / good)),3)  
    KS.append(ks)  
    BAD.append(bad1)  
    GOOD.append(good1)  
    BAD_CNT.append(bad_cnt)  
    GOOD_CNT.append(good_cnt)  
    BAD_PCTG.append(bad_pctg)  
    BADRATE.append(badrate)  
    dct_report['KS'] = KS  
    dct_report['负样本个数'] = BAD  
    dct_report['正样本个数'] = GOOD  
    dct_report['负样本累计个数'] = BAD_CNT  
    dct_report['正样本累计个数'] = GOOD_CNT  
    dct_report['捕获率'] = BAD_PCTG  
    dct_report['负样本占比'] = BADRATE  
val_repot = pd.DataFrame(dct_report)  
print(val_repot)  

from pyecharts.charts import *  
from pyecharts import options as opts  
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei']  
np.set_printoptions(suppress=True)  
pd.set_option('display.unicode.ambiguous_as_wide', True)  
pd.set_option('display.unicode.east_asian_width', True)  
line = (  
  
    Line()  
    .add_xaxis(list(val_repot.index))  
    .add_yaxis(  
        "分组坏人占比",  
        list(val_repot.BADRATE),  
        yaxis_index=0,  
        color="red",  
    )  
    .set_global_opts(  
        title_opts=opts.TitleOpts(title="行为评分卡模型表现"),  
    )  
    .extend_axis(  
        yaxis=opts.AxisOpts(  
            name="累计坏人占比",  
            type_="value",  
            min_=0,  
            max_=0.5,  
            position="right",  
            axisline_opts=opts.AxisLineOpts(  
                linestyle_opts=opts.LineStyleOpts(color="red")  
            ),  
            axislabel_opts=opts.LabelOpts(formatter="{value}"),  
        )  
  
    )  
    .add_xaxis(list(val_repot.index))  
    .add_yaxis(  
        "KS",  
        list(val_repot['KS']),  
        yaxis_index=1,  
        color="blue",  
        label_opts=opts.LabelOpts(is_show=False),  
    )  
)  
line.render_notebook()  

print('变量名单：',feature_lst)  
print('系数：',lr_model.coef_)  
print('截距：',lr_model.intercept_)  

import math
#算分数onekey   
def score(person_info,finance_info,credit_info,act_info):  
    xbeta = person_info * ( 3.49460978) \
                  + finance_info * ( 11.40051582 ) \
                  + credit_info * (2.45541981) \
                  + act_info * ( -1.68676079) \
                  -0.34484897   
    score = 650-34* (xbeta)/math.log(2)  
    return score  
val['score'] = val.apply(lambda x : 
                            score(x.person_info,x.finance_info,x.
                            credit_info,x.act_info) ,axis=1)  
fpr_lr,tpr_lr,_ = roc_curve(val_y,val['score'])  
val_ks = abs(fpr_lr - tpr_lr).max()  
print('val_ks : ',val_ks)  

#对应评级区间  
def level(score):  
    level = 0  
    if score <= 600:  
        level = "D"  
    elif score <= 640 and score > 600 :   
        level = "C"  
    elif score <= 680 and score > 640:  
        level = "B"  
    elif  score > 680 :  
        level = "A"  
    return level  
val['level'] = val.score.map(lambda x : level(x) )  
print(val.level.groupby(val.level).count()/len(val))  

import XGBoost as xgb  
data = pd.read_csv('Acard.txt')  
df_train = data[data.obs_mth != '2018-11-30'].reset_index().copy()  
val = data[data.obs_mth == '2018-11-30'].reset_index().copy()  
lst = ['person_info','finance_info','credit_info','act_info']  
  
train = data[data.obs_mth != '2018-11-30'].reset_index().copy()  
evl = data[data.obs_mth == '2018-11-30'].reset_index().copy()  
  
x = train[lst]  
y = train['bad_ind']  
  
evl_x =  evl[lst]  
evl_y = evl['bad_ind']  
  
#定义XGB函数  
def XGB_test(train_x,train_y,test_x,test_y):  
    from multiprocessing import cpu_count  
    clf = xgb.XGBClassifier(
        boosting_type='gbdt', num_leaves=31, 
                reg_Ap=0.0, reg_lambda=1,  
        max_depth=2, n_estimators=800,
                max_features = 140, objective='binary',  
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,  
        learning_rate=0.05, min_child_weight=50,
                random_state=None,n_jobs=cpu_count()-1,  
        num_iterations = 800 #迭代次数  
    )  
    clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],
                eval_metric='auc',early_stopping_rounds=100)  
    print(clf.n_features_)  
    return clf,clf.best_score_[ 'valid_1']['auc']  

#模型训练
model,auc = XGB_test(x,y,evl_x,evl_y) 
#训练集预测
y_pred = model.predict_proba(x)[:,1]  
fpr_xgb_train,tpr_xgb_train,_ = roc_curve(y,y_pred)  
train_ks = abs(fpr_xgb_train - tpr_xgb_train).max()  
print('train_ks : ',train_ks)  
#跨时间验证集预测
y_pred = model.predict_proba(evl_x)[:,1]  
fpr_xgb,tpr_xgb,_ = roc_curve(evl_y,y_pred)  
evl_ks = abs(fpr_xgb - tpr_xgb).max()  
print('evl_ks : ',evl_ks)  
#画出ROC曲线并计算KS值
from matplotlib import pyplot as plt  
plt.plot(fpr_xgb_train,tpr_xgb_train,label = 'train LR')  
plt.plot(fpr_xgb,tpr_xgb,label = 'evl LR')  
plt.plot([0,1],[0,1],'k--')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC Curve')  
plt.legend(loc = 'best')  
plt.show()  

row_num, col_num = 0, 0  
bins = 20  
Y_predict = evl['score']  
Y = evl_y  
nrows = Y.shape[0]  
lis = [(Y_predict[i], Y[i]) for i in range(nrows)]  
ks_lis = sorted(lis, key=lambda x: x[0], reverse=True)  
bin_num = int(nrows/bins+1)  
bad = sum([1 for (p, y) in ks_lis if y > 0.5])  
good = sum([1 for (p, y) in ks_lis if y <= 0.5])  
bad_cnt, good_cnt = 0, 0  
KS = []  
BAD = []  
GOOD = []  
BAD_CNT = []  
GOOD_CNT = []  
BAD_PCTG = []  
BADRATE = []  
dct_report = {}  
for j in range(bins):  
    ds = ks_lis[j*bin_num: min((j+1)*bin_num, nrows)]  
    bad1 = sum([1 for (p, y) in ds if y > 0.5])  
    good1 = sum([1 for (p, y) in ds if y <= 0.5])  
    bad_cnt += bad1  
    good_cnt += good1  
    bad_pctg = round(bad_cnt/sum(evl_y),3)  
    badrate = round(bad1/(bad1+good1),3)  
    ks = round(math.fabs((bad_cnt / bad) - (good_cnt / good)),3)  
    KS.append(ks)  
    BAD.append(bad1)  
    GOOD.append(good1)  
    BAD_CNT.append(bad_cnt)  
    GOOD_CNT.append(good_cnt)  
    BAD_PCTG.append(bad_pctg)  
    BADRATE.append(badrate)  
    dct_report['KS'] = KS  
    dct_report['BAD'] = BAD  
    dct_report['GOOD'] = GOOD  
    dct_report['BAD_CNT'] = BAD_CNT  
    dct_report['GOOD_CNT'] = GOOD_CNT  
    dct_report['BAD_PCTG'] = BAD_PCTG  
    dct_report['BADRATE'] = BADRATE  
val_repot = pd.DataFrame(dct_report)  
print(val_repot)

def score(pred):   
    score = 600+50*(math.log2((1- pred)/ pred))  
    return score  
evl['xbeta'] = model.predict_proba(evl_x)[:,1]     
evl['score'] = evl.apply(lambda x : score(x.xbeta) ,axis=1)  
fpr_lr,tpr_lr,_ = roc_curve(evl_y,evl['score'])  
evl_ks = abs(fpr_lr - tpr_lr).max()  
print('val_ks : ',evl_ks) 

# 自定义损失函数，需要提供损失函数的一阶导和二阶导  
def loglikelood(preds, dtrain):  
    labels = dtrain.get_label()  
    preds = 1.0 / (1.0 + np.exp(-preds))  
    grad = preds - labels  
    hess = preds * (1.0-preds)  
    return grad, hess  
  
# 自定义前20%正样本占比最大化评价函数  
def binary_error(preds, train_data):  
    labels = train_data.get_label()  
    dct = pd.DataFrame({'pred':preds,'percent':preds,'labels':labels})  
    #取百分位点对应的阈值  
    key = dct['percent'].quantile(0.2)  
    #按照阈值处理成二分类任务  
    dct['percent']= dct['percent'].map(lambda x :1 if x <= key else 0)    
    #计算评价函数，权重默认0.5，可以根据情况调整  
    result = np.mean(dct[dct.percent== 1]['labels'] == 1)*0.5 \
               + np.mean((dct.labels - dct.pred)**2)*0.5  
    return 'error',result  
  
watchlist  = [(dtest,'eval'), (dtrain,'train')]  
param = {'max_depth':3, 'eta':0.1, 'silent':1}  
num_round = 100  
# 自定义损失函数训练  
bst = xgb.train(param, dtrain, num_round, watchlist, loglikelood, binary_error) 



#==============================================================================
# File: 3.3.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:25:58 2019

@author: zixing.mei
"""

import pandas as pd  
from sklearn.metrics import roc_auc_score,roc_curve,auc  
from sklearn.model_selection import train_test_split  
from sklearn import metrics  
from sklearn.linear_model import LogisticRegression  
from sklearn.svm import LinearSVC  
import numpy as np  
import random  
import math  
from sklearn.calibration import CalibratedClassifierCV  
data = pd.read_excel('./data/tra_sample.xlsx')  
data.head()  
feature_lst = ['zx_score','msg_cnt','phone_num_cnt','register_days']    
train = data[data.type == 'target'].reset_index().copy()    
diff = data[data.type == 'origin'].reset_index().copy()    
val = data[data.type == 'offtime'].reset_index().copy()    
  
''' 
TrainS 目标域样本   
TrainA 源域样本   
LabelS 目标域标签   
LabelA 源域标签   
'''  
  
train = train.loc[:1200]    
    
trans_S = train[feature_lst].copy()    
label_S = train['bad_ind'].copy()    
    
trans_A = diff[feature_lst].copy()    
label_A = diff['bad_ind'].copy()    
    
val_x =  val[feature_lst].copy()    
val_y = val['bad_ind'].copy()    
    
test = val_x.copy()   
lr_model = LogisticRegression(C=0.1,class_weight = 'balanced',solver = 'liblinear')  
lr_model.fit(trans_S,label_S)  
  
y_pred = lr_model.predict_proba(trans_S)[:,1]  
fpr_lr_train,tpr_lr_train,_ = roc_curve(label_S,y_pred)  
train_ks = abs(fpr_lr_train - tpr_lr_train).max()  
print('train_ks : ',train_ks)  
  
y_pred = lr_model.predict_proba(test)[:,1]  
fpr_lr,tpr_lr,_ = roc_curve(val_y,y_pred)  
val_ks = abs(fpr_lr - tpr_lr).max()  
print('val_ks : ',val_ks)  
  
from matplotlib import pyplot as plt  
plt.plot(fpr_lr_train,tpr_lr_train,label = 'train LR')  
plt.plot(fpr_lr,tpr_lr,label = 'evl LR')  
plt.plot([0,1],[0,1],'k--')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC Curve')  
plt.legend(loc = 'best')  
plt.show()  
trans_data = np.concatenate((trans_A, trans_S), axis=0)  
trans_label = np.concatenate((label_A, label_S), axis=0)  
  
lr_model = LogisticRegression(C=0.3,class_weight = 'balanced',solver = 'liblinear')  
lr_model.fit(trans_A,label_A)  
  
y_pred = lr_model.predict_proba(trans_data)[:,1]  
fpr_lr_train,tpr_lr_train,_ = roc_curve(trans_label,y_pred)  
train_ks = abs(fpr_lr_train - tpr_lr_train).max()  
print('train_ks : ',train_ks)  
  
y_pred = lr_model.predict_proba(test)[:,1]  
fpr_lr,tpr_lr,_ = roc_curve(val_y,y_pred)  
val_ks = abs(fpr_lr - tpr_lr).max()  
print('val_ks : ',val_ks)  
  
from matplotlib import pyplot as plt  
plt.plot(fpr_lr_train,tpr_lr_train,label = 'train LR')  
plt.plot(fpr_lr,tpr_lr,label = 'evl LR')  
plt.plot([0,1],[0,1],'k--')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC Curve')  
plt.legend(loc = 'best')  
plt.show()  
import numpy as np      
import pandas as pd    
from sklearn.linear_model import LogisticRegression     
from sklearn.metrics import roc_curve     
    
def Tr_lr_boost(trans_A,trans_S,label_A,label_S,test,label_test,
                  N=500,early_stopping_rounds =100):    
    """   
        逻辑回归的学习率、权重的大小，影响整体收敛的快慢    
        H 测试样本分类结果    
        TrainS 目标域样本    
        TrainA 源域样本    
        LabelS 目标域标签    
        LabelA 源域标签    
        Test  测试样本    
        N 迭代次数   
        early_stopping_rounds 提前停止轮次 
    """   
    #计算weight      
    def calculate_P(weights, label):      
        total = np.sum(weights)      
        return np.asarray(weights / total, order='C')      
          
    #用逻辑回归作为基分类器，输出概率      
    def train_classify(trans_data, trans_label, test_data, P):      
        clf = LogisticRegression(C=0.3,class_weight = 'balanced',solver='liblinear')      
        clf.fit(trans_data, trans_label, sample_weight=P[:, 0])      
        return clf.predict_proba(test_data)[:,1],clf      
          
    #计算在目标域上面的错误率      
    def calculate_error_rate(label_R, label_H, weight):      
        total = np.sum(weight)      
        return np.sum(weight[:, 0] / total * np.abs(label_R - label_H))      
          
    #根据逻辑回归输出的score的得到标签，注意这里不能用predict直接输出标签      
    def put_label(score_H,thred):      
        new_label_H = []      
        for i in score_H:      
            if i <= thred:      
                new_label_H.append(0)      
            else:      
                new_label_H.append(1)      
        return new_label_H      
          
    #指定迭代次数，相当于集成模型中基模型的数量      
         
          
    #拼接数据集    
    trans_data = np.concatenate((trans_A, trans_S), axis=0)      
    trans_label = np.concatenate((label_A, label_S), axis=0)      
        
    #三个数据集样本数    
    row_A = trans_A.shape[0]      
    row_S = trans_S.shape[0]      
    row_T = test.shape[0]      
        
    #三个数据集合并为打分数据集    
    test_data = np.concatenate((trans_data, test), axis=0)      
          
    # 初始化权重      
    weights_A = np.ones([row_A, 1])/row_A      
    weights_S = np.ones([row_S, 1])/row_S*2      
    weights = np.concatenate((weights_A, weights_S), axis=0)      
        
    #按照公式初始化beta值    
    bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))      
          
        
    # 存每一次迭代的bata值=error_rate / (1 - error_rate)      
    bata_T = np.zeros([1, N])      
    # 存储每次迭代的标签    
    result_label = np.ones([row_A + row_S + row_T, N])       
          
    trans_data = np.asarray(trans_data, order='C')      
    trans_label = np.asarray(trans_label, order='C')      
    test_data = np.asarray(test_data, order='C')      
        
    #最优KS      
    best_ks = -1      
    #最优基模型数量          
    best_round = -1    
    #最优模型      
    best_model = -1     
         
    """ 
    初始化结束    
    正式开始训练  
    """     
        
    for i in range(N):      
        P = calculate_P(weights, trans_label)      
          
        result_label[:, i],model = train_classify(trans_data, trans_label, test_data, P)  
        score_H = result_label[row_A:row_A + row_S, i]      
        pctg = np.sum(trans_label)/len(trans_label)      
        thred = pd.DataFrame(score_H).quantile(1-pctg)[0]      
        
        label_H = put_label(score_H,thred)      
        
        #计算在目标域上的错误率    
        error_rate = calculate_error_rate(label_S, label_H,   
                                                    weights[row_A:row_A + row_S, :])  
        # 防止过拟合     
        if error_rate > 0.5:      
            error_rate = 0.5      
        if error_rate == 0:      
            N = i      
            break       
                
        bata_T[0, i] = error_rate / (1 - error_rate)      
          
        # 调整目标域样本权重      
        for j in range(row_S):      
            weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i],  \
                                      (-np.abs(result_label[row_A + j, i] - label_S[j])))
          
        # 调整源域样本权重      
        for j in range(row_A):      
            weights[j] = weights[j] * np.power(bata,   
                                               np.abs(result_label[j, i] - label_A[j]))  
        y_pred = result_label[(row_A + row_S):,i]      
        fpr_lr_train,tpr_lr_train,_ = roc_curve(label_test,y_pred)      
        train_ks = abs(fpr_lr_train - tpr_lr_train).max()      
        print('test_ks : ',train_ks,'当前第',i+1,'轮')      
              
        # 不再使用后一半学习器投票，而是只保留效果最好的逻辑回归模型      
        if train_ks > best_ks :      
            best_ks = train_ks      
            best_round = i      
            best_model = model    
        # 当超过eadrly_stopping_rounds轮KS不再提升后，停止训练  
        if best_round < i - early_stopping_rounds:  
            break  
    return best_ks,best_round,best_model   
    
# 训练并得到最优模型best_model    
best_ks,best_round,best_model = Tr_lr_boost(trans_A,trans_S,label_A,label_S,
                                            test,label_test=val_y,N=300,
                                            early_stopping_rounds=20) 

y_pred = best_model.predict_proba(trans_S)[:,1]  
fpr_lr_train,tpr_lr_train,_ = roc_curve(label_S,y_pred)  
train_ks = abs(fpr_lr_train - tpr_lr_train).max()  
print('train_ks : ',train_ks)  
  
y_pred = best_model.predict_proba(test)[:,1]  
fpr_lr,tpr_lr,_ = roc_curve(val_y,y_pred)  
val_ks = abs(fpr_lr - tpr_lr).max()  
print('val_ks : ',val_ks)  
  
from matplotlib import pyplot as plt  
plt.plot(fpr_lr_train,tpr_lr_train,label = 'train LR')  
plt.plot(fpr_lr,tpr_lr,label = 'evl LR')  
plt.plot([0,1],[0,1],'k--')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC Curve')  
plt.legend(loc = 'best')  
plt.show()  




#==============================================================================
# File: 3.4.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:30:41 2019

@author: zixing.mei
"""

import numpy as np  
from scipy.linalg.misc import norm  
from scipy.sparse.linalg import eigs  
  
def JDA(Xs,Xt,Ys,Yt,k=100,lamda=0.1,ker='primal',gamma=1.0,data='default'):  
    X = np.hstack((Xs , Xt))  
    X = np.diag(1/np.sqrt(np.sum(X**2)))  
    (m,n) = X.shape  
    #源域样本量  
    ns = Xs.shape[1]  
    #目标域样本量  
    nt = Xt.shape[1]  
    #分类个数  
    C = len(np.unique(Ys))  
    # 生成MMD矩阵  
    e1 = 1/ns*np.ones((ns,1))  
    e2 = 1/nt*np.ones((nt,1))  
    e = np.vstack((e1,e2))  
    M = np.dot(e,e.T)*C  
      
    #除了0，空，False以外都可以运行  
    if any(Yt) and len(Yt)==nt:  
        for c in np.reshape(np.unique(Ys) ,-1 ,1):  
            e1 = np.zeros((ns,1))  
            e1[Ys == c] = 1/len(Ys[Ys == c])  
            e2 = np.zeros((nt,1))  
            e2[Yt ==c] = -1/len(Yt[Yt ==c])  
            e = np.hstack((e1 ,e2))  
            e = e[np.isinf(e) == 0]  
            M = M+np.dot(e,e.T)  
      
    #矩阵迹求平方根          
    M = M/norm(M ,ord = 'fro' )  
      
    # 计算中心矩阵  
    H = np.eye(n) - 1/(n)*np.ones((n,n))  
      
    # Joint Distribution Adaptation: JDA  
    if ker == 'primal':  
        #特征值特征向量  
        A = eigs(np.dot(np.dot(X,M),X.T)+lamda*np.eye(m),
                           k=k, M=np.dot(np.dot(X,H),X.T),  which='SM')  
        Z = np.dot(A.T,X)  
    else:  
        pass  
    return A,Z  




#==============================================================================
# File: 3.5.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:31:33 2019

@author: zixing.mei
"""

import numpy as np  
from scipy import sparse as sp  
def DAELM(Train_s,Train_t,Test_t,NL,Type="CLASSIFIER" , Num_hid=100 ,Active_Function="sig"):  
    ''' 
    Train_s：源域训练集
    Train_t：目标域训练集
    Test_t：目标域测试集
    Type：模型类型（分类："CLASSIFIER"，回归："REGRESSION"）  
    Num_hid：隐层神经元个数，默认100个 
    Active_Function：映射函数（" sigmoid ":sigmoid函数, "sin":正弦函数）
    NL：模型选择  
    '''  
      
    Cs = 0.01  
    Ct = 0.01  
      
    #回归或分类  
    REGRESSION=0  
    CLASSIFIER=1  
      
    #训练数据  
    train_data = Train_s  
    T = train_data[:,0].T  
    P = train_data[:,1:train_data.shape[1]].T  
    del train_data  
      
    #目标域数据  
    train_target_dt = Train_t  
    Tt = train_target_dt[:,0].T  
    Pt = train_target_dt[:,1:train_target_dt.shape[1]].T  
      
    #测试集数据  
    test_data = Test_t  
    TVT = test_data[:,0].T  
    TE0 = test_data[:,0].T  
    TVP = test_data[:,2:test_data.shape[1]].T  
    del test_data  
      
    Num_train = P.shape[1]  
    Num_train_Target = Pt.shape[1]  
    Num_test = TVP.shape[1]  
    Num_input= P.shape[0]  
      
    if Type is not "REGRESSION":  
        sorted_target = np.sort(np.hstack((T ,  TVT)))  
        label = np.zeros((1,1))  
        label[0,0] = sorted_target[0,0]  
        j = 0  
        for i in range(2,(Num_train+Num_test+1)):  
            if sorted_target[0,i-1] != label[0,j-1]:  
                j=j+1  
                label[0,j-1] = sorted_target[0,i-1]  
                  
        number_class = j+1  
        Num_output = number_class  
          
  
        temp_T = np.zeros(Num_output , Num_train)  
        for i in range(1,Num_train+1):  
            for j in range(1,number_class+1):  
                if label(0,j-1) == T(0,i-1):  
                    break  
            temp_T[j-1 , i-1] = 1  
        T = temp_T*2-1  
  
        Tt_m = np.zeros(Num_output , Num_train_Target)  
        for i in range(1,Num_train_Target+1):  
            for j in range(1 , number_class+1):  
                if label[0,j-1] == Tt[0,i-1]:  
                    break  
            Tt_m[j-1 , i-1] = 1  
        Tt = Tt_m*2-1  
          
  
        temp_TV_T = np.zeros(Num_output,Num_test)  
        for i in range(1,Num_test):  
            for j in range(1,number_class+1):  
                if label(0,j-1) == TVT(0,i-1):  
                    break  
            temp_TV_T[j-1 , i-1] = 1  
        TVT = temp_TV_T*2-1  
          
    InputWeight = np.random.rand(Num_hid,Num_input)*2-1  
    Bis_hid = np.random.rand(Num_hid ,1)  
    H_m = InputWeight*P  
    Ht_m = InputWeight*Pt  
    del P  
    del Pt  
      
    ind = np.ones(1,Num_train)  
    indt = np.ones(1,Num_train_Target)  
    BiasMatrix = Bis_hid[:,ind-1]  
    BiasMatrixT = Bis_hid[:,indt-1]  
    H_m = H_m + BiasMatrix  
    Ht_m=Ht_m+BiasMatrixT  
      
    if Active_Function == "sigmoid":  
        H = 1/(1+np.exp(-H_m))  
        Ht = 1/(1+np.exp(-Ht_m))  
    if Active_Function == "sin":  
        H = np.sin(H_m)  
        Ht = np.sin(Ht_m)  
    if Active_Function != " sigmoid " and Active_Function!="sin":  
        pass  
      
    del H_m  
    del Ht_m  
      
    n = Num_hid  
      
    #DAELM模型  
    H=H.T  
    Ht=Ht.T  
    T=T.T  
    Tt=Tt.T  
      
    if NL == 0:  
        A = Ht*H.T  
        B = Ht*Ht.T+np.eye(Num_train_Target)/Ct  
        C=H*Ht.T  
        D=H*H.T+np.eye(Num_train)/Cs  
        ApT=np.linalg.inv(B)*Tt-np.linalg.inv(B)*A* \
                       np.linalg.inv(C*np.linalg.inv(B)*A-D)*(C*np.linalg.inv(B)*Tt-T)
        ApS=inv(C*np.linalg.inv(B)*A-D)*(C*np.linalg.inv(B)*Tt-T)  
        OutputWeight=H.T*ApS+Ht.T*ApT  
    else:  
        OutputWeight=np.linalg.inv(np.eye(n)+Cs*H.t*H+Ct*Ht.T*Ht)*(Cs*H.T*T+Ct*Ht.T*Tt)  
      
    #计算准确率  
      
    Y=(H * OutputWeight).T  
      
    H_m_test=InputWeight*TVP  
    ind = np.ones(1,Num_hid)  
    BiasMatrix=Bis_hid[:,ind-1]  
    H_m_test = H_m_test+BiasMatrix  
    if Active_Function == "sig":  
        H_test = 1/(1+np.exp(-H_m_test))  
    if Active_Function == "sin":  
        H_test = np.sin(H_m_test)  
          
    TY = (H_test.T*OutputWeight).T  
      
    #返回测试集结果  
    if Type =="CLASSIFIER":  
        return TY  
    else:  
        pass  




#==============================================================================
# File: 3.6.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:32:43 2019

@author: zixing.mei
"""

from sklearn.metrics import roc_auc_score as AUC  
import pandas as pd  
import numpy as np  
   
class Tra_learn3ft (object):  
    """ 
        一种多模型融合的Tradaboost变体 
        使用三个模型同时进行样本筛选，目的是减小variance 
        clfA 模型A 
        clfB 模型B 
        clfC 模型C 
        step 预计去掉的样本比例 
        max_turns最大迭代次数 
    """  
    def __init__(self,clfA,clfB,clfC,step,max_turns=5):  
        self.clfA = clfA  
        self.clfB = clfB  
        self.clfC = clfC  
        self.step = step  
        self.max_turns = max_turns  
        self.scoreA = 0  
        self.scoreB = 0  
        self.scoreC = 0  
  
    def tlearn(self,dev,test,val,bad_ind,featureA,featureB,featureC,drop_rate):  
        """ 
            dev 训练集 源域 
            test 测试集 辅助域 
            val 验证集 
            bad_ind 标签 
            featureA 特征组A 
            featureB 特征组B 
            featureC 特征组C 
        """  
        print(len(featureA),len(featureB),len(featureC))  
        result = pd.DataFrame()  
        temp_test = test  
        features = list(set(featureA+featureB+featureC))  
        turn = 1  
        while( turn <= self.max_turns):  
            new = pd.DataFrame()  
              
            """ 
                模型A对特征组featureA训练， 
                并预测得到dev和test和val的概率 
                以及test上的分类结果（分数分布在0.8*(min+max)两侧）
            """  
            self.clfA.fit(dev[featureA],dev[bad_ind])  
            predA= self.clfA.predict_proba(dev[featureA])[:,1]   
            probA = self.clfA.predict_proba(test[featureA])[:,1]  
            preA = (probA > (np.max(probA)+np.min(probA))*0.8)  
            valid_a = self.clfA.predict_proba(val[featureA])[:,1]   
            """ 
                模型B对特征组featureB训练， 
                并预测得到dev和test和val的概率 
                以及test上的分类结果（分数分布在0.8*(min+max)两侧）
            """  
            self.clfB.fit(dev[featureB],dev[bad_ind])  
            predB = self.clfB.predict_proba(dev[featureB])[:,1]  
            probB = self.clfB.predict_proba(test[featureB])[:,1]  
            preB = (probA > (np.max(probB)+np.min(probB))*0.8)  
            valid_b = self.clfB.predict_proba(val[featureB])[:,1]  
            """ 
                模型C对特征组featureC训练， 
                并预测得到dev和test和val的概率 
                以及test上的分类结果（分数分布在0.8*(min+max)两侧） 
            """              
            self.clfC.fit(dev[featureC],dev[bad_ind])  
            predC= self.clfC.predict_proba(dev[featureC])[:,1]  
            probC = self.clfC.predict_proba(test[featureC])[:,1]  
            preC = (probC > (np.max(probC)+np.min(probC))*0.8)  
            valid_c = self.clfC.predict_proba(val[featureC])[:,1]  
            """ 
                分别计算三个模型在val上的AUC 
                模型加权融合的策略：以单模型的AUC作为权重
            """  
            valid_scoreA = AUC(val[bad_ind],valid_a)  
            valid_scoreB = AUC(val[bad_ind],valid_b)  
            valid_scoreC = AUC(val[bad_ind],valid_c)  
            valid_score = AUC(val[bad_ind], valid_a*valid_scoreA
                                             +valid_b*valid_scoreB + valid_c*valid_scoreC)
              
            """ 
                index1 三个模型在test上的预测概率相同的样本 
                sum_va 三个模型AUC之和为分母做归一化 
                prob 测试集分类结果融合， 
                index1（分类结果）*AUC（权重）/sum_va（归一化分母） 
                index2 分类结果升序排列，取出两端的test样本 
                new 筛选后样本集 
            """  
            index1 = (preA==preB) & (preA==preC)  
            sum_va = valid_scoreA+valid_scoreB+valid_scoreC  
            prob = (probC[index1]*valid_scoreC+probA[index1]*valid_scoreA  
                    +probB[index1]*valid_scoreB)/sum_va  
            Ap_low = np.sort(prob)[int(len(prob)*turn/2.0/self.max_turns)]-0.01  
            Ap_high= np.sort(prob)[int(len(prob)*
                                                          (1-turn/2.0/self.max_turns))]+0.01
            index2 = ((prob>Ap_high) | (prob<Ap_low))    
            new['no'] = test['no'][index1][index2]      
            new['pred'] = prob[index2]  
            result = result.append(new)  
            """ 
                rightSamples 同时满足index1和index2条件的预测概率 
                score_sim 三个模型在test上的预测结果差异和 
            """  
            rightSamples = test[index1][index2]  
            rightSamples[bad_ind] = preA[index1][index2]  
  
            score_sim = np.sum(abs(probA-probB)+
                                             abs(probA-probC) +abs(probB-probC)+0.1)/len(probA)
            """ 
                从数据集dev中取出step之后的部分样本并计算AUC 
                valid_score 前文三模型加权融合的AUC 
                得到drop 
            """  
            true_y = dev.iloc[self.step:][bad_ind]  
            dev_prob = predA[self.step:]*valid_scoreA+ predB[self.step:]*valid_scoreB + predC[self.step:]*valid_scoreC  
                              
            dev_score = AUC(true_y,dev_prob)  
              
            drop = self.max_turns/(1+ drop_rate*
                                                      np.exp(-self.max_turns)*valid_score)
            """ 
                使用Traddaboost相同的权重调整方法， 
                挑选权重大于阈值的样本。 
            """  
            loss_bias = 0  
            if(self.step>0):  
                true_y = dev.iloc[0:self.step][bad_ind]  
                temp = predA[0:self.step]*valid_scoreA  \  
                        + predB[0:self.step]*valid_scoreB  \  
                        + predC[0:self.step]*valid_scoreC  
                temp = (temp+0.1)/(max(temp)+0.2)#归一化  
                temp = (true_y-1)*np.log(1-temp)-true_y*np.log(temp)#样本权重  
                loc = int(min(self.step,len(rightSamples)*drop+2)
                                                             *np.random.rand())#去除样本的比例  
                loss_bias =  np.sort(temp)[-loc]  
                temp = np.append(temp,np.zeros(len(dev)-self.step)-99)  
                remain_index = (temp <= loss_bias)  
                self.step = self.step-sum(1-remain_index)  
            else:  
                remain_index = []  
                  
            """ 
                得到新的test 
            """  
            dev = dev[remain_index].append(rightSamples[features+[bad_ind,'no']])  
            test = test[~test.index.isin(rightSamples.index)]  
            turn += 1  
        """ 
            计算原始test上的AUC 
        """  
        probA = self.clfA.predict_proba(test[featureA])[:,1]  
        pA = self.clfA.predict_proba(temp_test[featureA])[:,1]  
        valid_a = self.clfA.predict_proba(val[featureA])[:,1]  
  
        probB = self.clfB.predict_proba(test[featureB])[:,1]  
        valid_b = self.clfB.predict_proba(val[featureB])[:,1]  
        pB = self.clfB.predict_proba(temp_test[featureB])[:,1]  
  
        probC = self.clfC.predict_proba(test[features])[:,1]  
        valid_c = self.clfC.predict_proba(val[features])[:,1]  
        pC = self.clfC.predict_proba(temp_test[features])[:,1]  
  
        self.scoreA = AUC(val[bad_ind],valid_a)  
        self.scoreB = AUC(val[bad_ind],valid_b)  
        self.scoreC = AUC(val[bad_ind],valid_c)  

        return pA,pB,pC  




#==============================================================================
# File: 4.2.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:50:39 2019

@author: zixing.mei
"""

import xgboost as xgb  
from sklearn.datasets import load_digits # 训练数据  
xgb_params_01 = {}  
digits_2class = load_digits(2)  
X_2class = digits_2class['data']  
y_2class = digits_2class['target']  
dtrain_2class = xgb.DMatrix(X_2class, label=y_2class)
# 训练三棵树的模型  
gbdt_03 = xgb.train(xgb_params_01, dtrain_2class, num_boost_round=3) 
# 以前面三棵树的模型为基础，从第四棵树开始训练  
gbdt_03a = xgb.train(xgb_params_01, dtrain_2class, num_boost_round=7, xgb_model=gbdt_03)  





#==============================================================================
# File: 4.4.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:35:39 2019

@author: zixing.mei
"""

import matplotlib.pyplot as plt  
import seaborn as sns; sns.set()  
import numpy as np  
#产生实验数据  
from sklearn.datasets.samples_generator import make_blobs  
X, y_true = make_blobs(n_samples=700, centers=4,  
             cluster_std=0.5, random_state=2019) 
X = X[:, ::-1] #方便画图  
  
from sklearn.mixture import GaussianMixture as GMM  
gmm = GMM(n_components=4).fit(X) #指定聚类中心个数为4  
labels = gmm.predict(X)  
plt.scatter(X[:, 0], X[:, 1], c=labels, s=5, cmap='viridis')  
probs = gmm.predict_proba(X)  
print(probs[:10].round(2))  
size = probs.max(1)  
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size)  

from matplotlib.patches import Ellipse  
#给定的位置和协方差画一个椭圆
def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    #将协方差转换为主轴
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    #画出椭圆
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
#画图
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=4, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=4, zorder=2)
    ax.axis('equal')
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_  , gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
 
from sklearn.datasets import make_moons  
Xmoon, ymoon = make_moons(100, noise=.04, random_state=0)  
plt.scatter(Xmoon[:, 0], Xmoon[:, 1]) 
gmm2 = GMM(n_components=2, covariance_type='full', random_state=0)  
plot_gmm(gmm2, Xmoon) 
gmm10 = GMM(n_components=10, covariance_type='full', random_state=0)  
plot_gmm(gmm10, Xmoon, label=False)  
Xnew = gmm10.sample(200)[0]
plt.scatter(Xnew[:, 0], Xnew[:, 1])  




#==============================================================================
# File: 4.5.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:41:08 2019

@author: zixing.mei
"""

n_components = np.arange(1, 21)  
models = [GMM(n, covariance_type='full', 
                random_state=0).fit(Xmoon) for n in n_components]  
plt.plot(n_components, [m.bic(Xmoon) for m in models], label='BIC')  
plt.plot(n_components, [m.aic(Xmoon) for m in models], label='AIC')  
plt.legend(loc='best')  
plt.xlabel('n_components')  



#==============================================================================
# File: 5.3.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:42:49 2019

@author: zixing.mei
"""

class imbalanceData():    
      
    """  
      处理不均衡数据 
        train训练集 
        test测试集 
        mmin低分段错分比例 
        mmax高分段错分比例 
        bad_ind样本标签 
        lis不参与建模变量列表 
    """  
    def __init__(self, train,test,mmin,mmax, bad_ind,lis=[]):  
        self.bad_ind = bad_ind  
        self.train_x = train.drop([bad_ind]+lis,axis=1)  
        self.train_y = train[bad_ind]  
        self.test_x = test.drop([bad_ind]+lis,axis=1)  
        self.test_y = test[bad_ind]  
        self.columns = list(self.train_x.columns)  
        self.keep = self.columns + [self.bad_ind]  
        self.mmin = 0.1  
        self.mmax = 0.7  
      
    ''''' 
        设置不同比例， 
        针对头部和尾部预测不准的样本，进行加权处理。 
        0.1为噪声的权重，不参与过采样。 
        1为正常样本权重，参与过采样。 
    '''  
    def weight(self,x,y):  
        if x == 0 and y < self.mmin:  
            return 0.1  
        elif x == 1 and y > self.mmax:  
            return 0.1  
        else:  
            return 1  
    ''''' 
        用一个LightGBM算法和weight函数进行样本选择 
        只取预测准确的部分进行后续的smote过采样 
    '''  
    def data_cleaning(self):  
        lgb_model,lgb_auc  = self.lgb_test()  
        sample = self.train_x.copy()  
        sample[self.bad_ind] = self.train_y  
        sample['pred'] = lgb_model.predict_proba(self.train_x)[:,1]  
        sample = sample.sort_values(by=['pred'],ascending=False).reset_index()  
        sample['rank'] = np.array(sample.index)/len(sample)  
        sample['weight'] = sample.apply(lambda x:self.weight(x.bad_ind,x['rank']),
                                                                        axis = 1)
        osvp_sample = sample[sample.weight == 1][self.keep]  
        osnu_sample = sample[sample.weight < 1][self.keep]     
        train_x_osvp = osvp_sample[self.columns]  
        train_y_osvp = osvp_sample[self.bad_ind]  
        return train_x_osvp,train_y_osvp,osnu_sample  
  
    ''''' 
        实施smote过采样 
    '''  
    def apply_smote(self):  
        ''''' 
            选择样本，只对部分样本做过采样 
            train_x_osvp,train_y_osvp 为参与过采样的样本 
            osnu_sample为不参加过采样的部分样本 
        '''  
        train_x_osvp,train_y_osvp,osnu_sample = self.data_cleaning()  
        rex,rey = self.smote(train_x_osvp,train_y_osvp)  
        print('badpctn:',rey.sum()/len(rey))  
        df_rex = pd.DataFrame(rex)  
        df_rex.columns =self.columns  
        df_rex['weight'] = 1  
        df_rex[self.bad_ind] = rey  
        df_aff_ovsp = df_rex.append(osnu_sample)  
        return df_aff_ovsp  
  
    ''''' 
        定义LightGBM函数 
    '''  
    def lgb_test(self):  
        import lightgbm as lgb  
        clf =lgb.LGBMClassifier(boosting_type = 'gbdt',  
                               objective = 'binary',  
                               metric = 'auc',  
                               learning_rate = 0.1,  
                               n_estimators = 24,  
                               max_depth = 4,  
                               num_leaves = 25,  
                               max_bin = 40,  
                               min_data_in_leaf = 5,  
                               bagging_fraction = 0.6,  
                               bagging_freq = 0,  
                               feature_fraction = 0.8,  
                               )  
        clf.fit(self.train_x,self.train_y,eval_set=[(self.train_x,self.train_y),
                                                                  (self.test_x,self.test_y)],
                                                                    eval_metric = 'auc')
        return clf,clf.best_score_['valid_1']['auc']  
  
    ''''' 
        调用imblearn中的smote函数 
    '''  
    def smote(self,train_x_osvp,train_y_osvp,m=4,K=15,random_state=0):  
        from imblearn.over_sampling import SMOTE  
        smote = SMOTE(k_neighbors=K, kind='borderline1', m_neighbors=m, n_jobs=1,  
                out_step='deprecated', random_state=random_state, ratio=None,  
                      svm_estimator='deprecated')  
        rex,rey = smote.fit_resample(train_x_osvp,train_y_osvp)  
        return rex,rey  
df_aff_ovsp = imbalanceData(train=train,test=evl,mmin=0.3,mmax=0.7, bad_ind='bad_ind',
                            lis=['index', 'uid', 'td_score', 'jxl_score', 'mj_score',
                                 'rh_score', 'zzc_score', 'zcx_score','obs_mth']).apply_smote()
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import roc_curve  
  
lr_model = LogisticRegression(C=0.05,class_weight='balanced')  
lr_model.fit(x,y)  
  
y_pred = lr_model.predict_proba(x)[:,1]  
fpr_lr_train,tpr_lr_train,_ = roc_curve(y,y_pred)  
train_ks = abs(fpr_lr_train - tpr_lr_train).max()  
print('train_ks : ',train_ks)  
  
y_pred = lr_model.predict_proba(evl_x)[:,1]  
fpr_lr,tpr_lr,_ = roc_curve(evl_y,y_pred)  
evl_ks = abs(fpr_lr - tpr_lr).max()  
print('evl_ks : ',evl_ks)  
  
from matplotlib import pyplot as plt  
plt.plot(fpr_lr_train,tpr_lr_train,label = 'train LR')  
plt.plot(fpr_lr,tpr_lr,label = 'evl LR')  
plt.plot([0,1],[0,1],'k--')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC Curve')  
plt.legend(loc = 'best')  
plt.show() 




#==============================================================================
# File: 5.4.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:46:56 2019

@author: zixing.mei
"""

import numpy as np  
from utils import *  
import pandas as pd  
import sklearn.svm as svm 
from collections import Counter  
class TSVM(object):  
    def __init__(self):  
         # 分别对应有label的样本权重和无label的样本权重
        self.Cu = 0.001   
        self.Cl = 1  
    def fit(self,train_data):  
        # 将数据集中的第一个正例，和第一个负例作为真实标记样本，其余视为无标记。  
        pos_one = train_data[train_data[:,0] == 1][0]  
        pos_other = train_data[train_data[:,0] == 1][1:]  
        neg_one = train_data[train_data[:,0] == -1][0]  
        neg_other = train_data[train_data[:,0] == -1][1:]  
        train = np.vstack((pos_one,neg_one))   
         #S用于对数据进行测试
        self.other = np.vstack((pos_other,neg_other))   
        # 训练一个初始的分类器，设置不均衡参数  
        self.clf =  svm.SVC(C=1.5, kernel=self.kernel)
        self.clf.fit(train[:,1:],train[:,0])  
        pred_y = self.clf.predict(self.other[:,1:])  
          
        X = np.vstack((train,self.other))  
         # 将预测结果放到SVM模型中进行训练  
        y = np.vstack((train[:,0].reshape(-1,1), pred_y.reshape(-1,1)))[:,0]
        self.w = np.ones(train_data.shape[0])  
        self.w[len(train):] = self.Cu  
        while self.Cu < self.Cl:  
            print(X.shape,y.shape,self.w.shape)  
            self.clf.fit(X[:,1:],y,sample_weight = self.w)  
            while True:  
                   #返回的是带符号的距离
                dist = self.clf.decision_function(X[:,1:])   
                xi = 1 - y * dist  
                #取出预判为正例和负例的id  
                xi_posi, xi_negi = np.where(y[2:]>0),np.where(y[2:]<0)
                xi_pos , xi_neg = xi[xi_posi],xi[xi_negi]
                xi_pos_maxi = np.argmax(xi_pos)  
                xi_neg_maxi = np.argmax(xi_neg)  
                xi_pos_max = xi_pos[xi_pos_maxi]  
                xi_neg_max = xi_neg[xi_neg_maxi]  
                #不断地拿两个距离最大的点进行交换。
                   #交换策略：两个点中至少有一个误分类。 
                if xi_pos_max >0 and xi_neg_max > 0 \
                     and (xi_pos_max + xi_neg_max) > 2:
                    # 交换类别  
                    y[xi_pos_maxi],y[xi_neg_maxi] = \
                      y[xi_neg_maxi],y[xi_pos_maxi]
                    self.clf.fit(X[:,1:],y, sample_weight = self.w)  
                else:  
                    break  
            self.Cu = min(2 * self.Cu ,self.Cl)  
            # 交换权重  
            self.w[len(train):] = self.Cu  
    def predict(self):
        pred_y = self.clf.predict(self.other[:,1:])
        return 1 - np.mean(pred_y == self.other[:,0])

import numpy as np    
import matplotlib.pyplot as plt    
from sklearn.semi_supervised import label_propagation    
from sklearn.datasets import make_moons  
  
# 生成弧形数据    
n_samples = 200     
X, y  = make_moons(n_samples, noise=0.04, random_state=0)    
outer, inner = 0, 1    
labels = np.full(n_samples, -1.)    
labels[0] = outer    
labels[-1] = inner    
# 使用LP算法实现标签传递   
label_spread = label_propagation.LabelSpreading(kernel='rbf')    
label_spread.fit(X, labels)    
    
# 输出标签    
output_labels = label_spread.transduction_    
plt.figure(figsize=(8.5, 4))    
plt.subplot(1, 2, 1)    
plt.scatter(X[labels == outer, 0],   
            X[labels == outer, 1], color='navy',    
      marker='s', lw=0, label="outer labeled", s=10)    
plt.scatter(X[labels == inner, 0], X[labels == inner, 1],   
            color='c', marker='s', lw=0, label='inner labeled', s=10)    
plt.scatter(X[labels == -1, 0], X[labels == -1, 1],   
            color='darkorange', marker='.', label='unlabeled')    
plt.legend(scatterpoints=1, shadow=False, loc='upper right')    
plt.title("Raw data (2 classes=outer and inner)")    
    
plt.subplot(1, 2, 2)    
output_label_array = np.asarray(output_labels)    
outer_numbers = np.where(output_label_array == outer)[0]    
inner_numbers = np.where(output_label_array == inner)[0]    
plt.scatter(X[outer_numbers, 0], X[outer_numbers, 1], color='navy',    
      marker='s', lw=0, s=10, label="outer learned")    
plt.scatter(X[inner_numbers, 0], X[inner_numbers, 1], color='c',    
      marker='s', lw=0, s=10, label="inner learned")    
plt.legend(scatterpoints=1, shadow=False, loc='upper right')    
plt.title("Labels learned with Label Spreading (KNN)")    
    
plt.subplots_adjust(left=0.07, bottom=0.07, right=0.9, top=0.92)    
plt.show() 



#==============================================================================
# File: 6.3.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:51:40 2019

@author: zixing.mei
"""

from pyod.models.lof import LOF    
  
 #训练异常检测模型，然后输出训练集样本的异常分  
clf = LOF(n_neighbors=20, algorithm='auto', leaf_size=30, 
            metric='minkowski', p=2,metric_params=None, 
            contamination=0.1, n_jobs=1)  
clf.fit(x)   
  
#异常分  
out_pred = clf.predict_proba(x,method ='linear')[:,1]    
train['out_pred'] = out_pred    
  
#异常分在0.9百分位以下的样本删掉   
key = train['out_pred'].quantile(0.9)

x = train[train.out_pred< key][feature_lst]
y = train[train.out_pred < key]['bad_ind']   
   
val_x = val[feature_lst]    
val_y = val['bad_ind']    
  
#重新训练模型   
lr_model = LogisticRegression(C=0.1,class_weight='balanced')    
lr_model.fit(x,y)    
y_pred = lr_model.predict_proba(x)[:,1]    
fpr_lr_train,tpr_lr_train,_ = roc_curve(y,y_pred)    
train_ks = abs(fpr_lr_train - tpr_lr_train).max()    
print('train_ks : ',train_ks)    
    
y_pred = lr_model.predict_proba(val_x)[:,1]    
fpr_lr,tpr_lr,_ = roc_curve(val_y,y_pred)    
val_ks = abs(fpr_lr - tpr_lr).max()    
print('val_ks : ',val_ks)    
  
from matplotlib import pyplot as plt    
plt.plot(fpr_lr_train,tpr_lr_train,label = 'train LR')    
plt.plot(fpr_lr,tpr_lr,label = 'evl LR')    
plt.plot([0,1],[0,1],'k--')    
plt.xlabel('False positive rate')    
plt.ylabel('True positive rate')    
plt.title('ROC Curve')    
plt.legend(loc = 'best')    
plt.show()  




#==============================================================================
# File: 6.4.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:54:36 2019

@author: zixing.mei
"""

from pyod.models.iforest import IForest
clf = IForest(behaviour='new', bootstrap=False, contamination=0.1, max_features=1.0,
                max_samples='auto', n_estimators=500, n_jobs=-1, random_state=None,verbose=0)
clf.fit(x)
out_pred = clf.predict_proba(x,method ='linear')[:,1]
train['out_pred'] = out_pred
train['for_pred'] = np.where(train.out_pred>0.7,'负样本占比','正样本占比')
dic = dict(train.groupby(train.for_pred).bad_ind.agg(np.sum)/ \
           train.bad_ind.groupby(train.for_pred).count())
pd.DataFrame(dic,index=[0])

clf = IForest(behaviour='new', bootstrap=False, contamination=0.1, max_features=1.0,
                max_samples='auto', n_estimators=500, n_jobs=-1, random_state=None,verbose=0)
clf.fit(x)
y_pred = clf.predict_proba(x,method ='linear')[:,1]    
fpr_lr_train,tpr_lr_train,_ = roc_curve(y,y_pred)    
train_ks = abs(fpr_lr_train - tpr_lr_train).max()    
print('train_ks : ',train_ks)    
y_pred = clf.predict_proba(val_x,method ='linear')[:,1]    
fpr_lr,tpr_lr,_ = roc_curve(val_y,y_pred)    
val_ks = abs(fpr_lr - tpr_lr).max()    
print('val_ks : ',val_ks)   
from matplotlib import pyplot as plt    
plt.plot(fpr_lr_train,tpr_lr_train,label = 'train LR')    
plt.plot(fpr_lr,tpr_lr,label = 'evl LR')    
plt.plot([0,1],[0,1],'k--')    
plt.xlabel('False positive rate')    
plt.ylabel('True positive rate')    
plt.title('ROC Curve')    
plt.legend(loc = 'best')    
plt.show()  




#==============================================================================
# File: 7.1.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:55:52 2019

@author: zixing.mei
"""

from sklearn.metrics import accuracy_score  
import lightgbm as lgb 

#'regression_l1'等价于MAE损失函数
lgb_param_l1 = {  
    'learning_rate': 0.01,  
    'boosting_type': 'gbdt',  
    'objective': 'regression_l1',   
    'min_child_samples': 46,  
    'min_child_weight': 0.02,  
    'feature_fraction': 0.6,  
    'bagging_fraction': 0.8,  
    'bagging_freq': 2,  
    'num_leaves': 31,  
    'max_depth': 5,  
    'lambda_l2': 1,  
    'lambda_l1': 0,  
    'n_jobs': -1,  
}  
  
#'regression_l2'等价于MSE损失函数  
lgb_param_l2 = { 
    'learning_rate': 0.01,  
    'boosting_type': 'gbdt',  
    'objective': 'regression_l2',  
    'feature_fraction': 0.7,  
    'bagging_fraction': 0.7,  
    'bagging_freq': 2,  
    'num_leaves': 52,  
    'max_depth': 5,  
    'lambda_l2': 1,  
    'lambda_l1': 0,  
    'n_jobs': -1,  
}  
# 第一种参数预测  
clf1=lgb.LGBMRegressor(**lgb_params1)  
clf.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_val,y_val)],
                                  eval_metric='mae',early_stopping_rounds=200)
#预测的划分出来的测试集的标签  
pred_val1=clf1.predict(X_val,num_iteration=clf.best_iteration_)   
vali_mae1=accuracy_score(y_val,np.round(pred_val1))  
#预测的未带标签的测试集的标签  
pred_test1=clf.predcit(test[feature_name],num_iteration=clf.best_iteration_)   
# 第二种参数预测 
clf2=lgb.LGBMRegressor(**lgb_params2)
clf2.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_val,y_val)],
                                   eval_metric='rmse',early_stopping_rounds=200)  
#预测的划分出来的测试集的标签  
pred_val2=clf2.predict(X_val,num_iteration=clf2.best_iteration_)  
vali_mae2=accuracy_score(y_val,np.round(pred_val2))  
#预测的未带标签的测试集的标签  
pred_test2=clf.predcit(test_featur,num_iteration=clf2.best_iteration_)   
# 模型参数进行融合之后的结果  
pred_test=pd.DataFrame()  
pred_test['ranks']=list(range(50000))  
pred_test['result']=1  
pred_test.loc[pred_test.ranks<400,'result'] = 
           pred_test1.loc[pred_test1.ranks< 400,'pred_mae'].values *0.4 
           + pred_test2.loc[pred_test2.ranks< 400,'pred_mse'].values * 0.6  
pred_test.loc[pred_test.ranks>46000,'result'] = 
              pred_test1.loc[pred_test1.ranks> 46000,'pred_mae'].values *0.4
              + pred_test2.loc[pred_test2.ranks> 46000,'pred_mse'].values * 0.6




#==============================================================================
# File: 7.2.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:57:09 2019

@author: zixing.mei
"""

import lightgbm as lgb  
import random  
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error  
from sklearn.linear_model import LogisticRegression  
from sklearn import metrics  
from sklearn.metrics import roc_curve  
from matplotlib import pyplot as plt  
import math  
  
df_train = data[data.obs_mth != '2018-11-30'].reset_index().copy()    
df_test = data[data.obs_mth == '2018-11-30'].reset_index().copy()    
NUMERIC_COLS = ['person_info','finance_info','credit_info','act_info']
from sklearn.preprocessing import OneHotEncoder,LabelEncoder  
  
lgb_train = lgb.Dataset(df_train[NUMERIC_COLS], 
                          df_train['bad_ind'], free_raw_data=False)  
params = {  
    'num_boost_round': 50,  
    'boosting_type': 'gbdt',  
    'objective': 'binary',  
    'num_leaves': 2,  
    'metric': 'auc',  
    'max_depth':1,  
    'feature_fraction':1,  
    'bagging_fraction':1, } 
model = lgb.train(params,lgb_train)  
leaf = model.predict(df_train[NUMERIC_COLS],pred_leaf=True)  
lgb_enc = OneHotEncoder()  
#生成交叉特征
lgb_enc.fit(leaf)
#和原始特征进行合并
data_leaf = np.hstack((lgb_enc.transform(leaf).toarray(),df_train[NUMERIC_COLS]))  
leaf_test = model.predict(df_test[NUMERIC_COLS],pred_leaf=True)  
lgb_enc = OneHotEncoder()  
lgb_enc.fit(leaf_test)  
data_leaf_test = np.hstack((lgb_enc.transform(leaf_test).toarray(),
                              df_test[NUMERIC_COLS]))  
train = data_leaf.copy()  
train_y = df_train['bad_ind'].copy()  
val = data_leaf_test.copy()  
val_y = df_test['bad_ind'].copy()  
lgb_lm = LogisticRegression(penalty='l2',C=0.2, class_weight='balanced',solver='liblinear')
lgb_lm.fit(train, train_y)  
y_pred_lgb_lm_train = lgb_lm.predict_proba(train)[:, 1]  
fpr_lgb_lm_train, tpr_lgb_lm_train, _ = roc_curve(train_y,y_pred_lgb_lm_train)
y_pred_lgb_lm = lgb_lm.predict_proba(val)[:,1]  
fpr_lgb_lm,tpr_lgb_lm,_ = roc_curve(val_y,y_pred_lgb_lm)  
plt.figure(1)  
plt.plot([0, 1], [0, 1], 'k--')  
plt.plot(fpr_lgb_lm_train,tpr_lgb_lm_train,label='LGB + LR train')  
plt.plot(fpr_lgb_lm, tpr_lgb_lm, label='LGB + LR test')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC curve')  
plt.legend(loc='best')  
plt.show()  
print('LGB+LR train ks:',abs(fpr_lgb_lm_train - tpr_lgb_lm_train).max(),
                               'LGB+LR AUC:', metrics.auc(fpr_lgb_lm_train, tpr_lgb_lm_train))
print('LGB+LR test ks:',abs(fpr_lgb_lm - tpr_lgb_lm).max(),
                              'LGB+LR AUC:', metrics.auc(fpr_lgb_lm, tpr_lgb_lm))
dff_train = pd.DataFrame(train)  
dff_train.columns = [ 'ft' + str(x) for x in range(train.shape[1])]  
  
dff_val = pd.DataFrame(val)  
dff_val.columns = [ 'ft' + str(x) for x in range(val.shape[1])]  
#生成可以传入PSI的数据集  
def make_psi_data(dff_train):  
    dftot = pd.DataFrame()  
    for col in dff_train.columns:  
        zero= sum(dff_train[col] == 0)  
        one= sum(dff_train[col] == 1)  
        ftdf = pd.DataFrame(np.array([zero,one]))  
        ftdf.columns = [col]  
        if len(dftot) == 0:  
            dftot = ftdf.copy()  
        else:  
            dftot[col] = ftdf[col].copy()  
    return dftot  
psi_data_train = make_psi_data(dff_train)  
psi_data_val = make_psi_data(dff_val) 
def var_PSI(dev_data, val_data):  
    dev_cnt, val_cnt = sum(dev_data), sum(val_data)  
    if dev_cnt * val_cnt == 0:  
        return 0  
    PSI = 0  
    for i in range(len(dev_data)):  
        dev_ratio = dev_data[i] / dev_cnt  
        val_ratio = val_data[i] / val_cnt + 1e-10  
        psi = (dev_ratio - val_ratio) * math.log(dev_ratio/val_ratio)
        PSI += psi  
    return PSI  
psi_dct = {}  
for col in dff_train.columns:  
    psi_dct[col] = var_PSI(psi_data_train[col],psi_data_val[col]) 
f = zip(psi_dct.keys(),psi_dct.values())  
f = sorted(f,key = lambda x:x[1],reverse = False)  
psi_df = pd.DataFrame(f)  
psi_df.columns = pd.Series(['变量名','PSI'])  
feature_lst = list(psi_df[psi_df['PSI']<psi_df.quantile(0.6)[0]]['变量名'])  
train = dff_train[feature_lst].copy()  
train_y = df_train['bad_ind'].copy()  
val = dff_val[feature_lst].copy()  
val_y = df_test['bad_ind'].copy()  
lgb_lm = LogisticRegression(C = 0.3,class_weight='balanced',solver='liblinear')
lgb_lm.fit(train, train_y)  
y_pred_lgb_lm_train = lgb_lm.predict_proba(train)[:, 1]  
fpr_lgb_lm_train, tpr_lgb_lm_train, _ = roc_curve(train_y, y_pred_lgb_lm_train)
y_pred_lgb_lm = lgb_lm.predict_proba(val)[:, 1]  
fpr_lgb_lm, tpr_lgb_lm, _ = roc_curve(val_y, y_pred_lgb_lm)  
plt.figure(1)  
plt.plot([0, 1], [0, 1], 'k--')  
plt.plot(fpr_lgb_lm_train, tpr_lgb_lm_train, label='LGB + LR train')  
plt.plot(fpr_lgb_lm, tpr_lgb_lm, label='LGB + LR test')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC curve')  
plt.legend(loc='best')  
plt.show()  
print('LGB+LR train ks:',abs(fpr_lgb_lm_train - tpr_lgb_lm_train).max(),
                               'LGB+LR AUC:', metrics.auc(fpr_lgb_lm_train, tpr_lgb_lm_train))
print('LGB+LR test ks:',abs(fpr_lgb_lm - tpr_lgb_lm).max(),'LGB+LR AUC:',
                              metrics.auc(fpr_lgb_lm, tpr_lgb_lm))
x = train  
y = train_y  
  
val_x =  val  
val_y = val_y  
  
#定义lgb函数  
def LGB_test(train_x,train_y,test_x,test_y):  
    from multiprocessing import cpu_count  
    clf = lgb.LGBMClassifier(  
        boosting_type='gbdt', num_leaves=31, reg_Ap=0.0, reg_lambda=1,
        max_depth=2, n_estimators=800,max_features=140,objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,  
        learning_rate=0.05, min_child_weight=50,
              random_state=None,n_jobs=cpu_count()-1,)  
    clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],
                eval_metric='auc',early_stopping_rounds=100)  
    return clf,clf.best_score_[ 'valid_1']['auc']  
#训练模型
model,auc = LGB_test(x,y,val_x,val_y)                      
  
#模型贡献度放在feture中  
feature = pd.DataFrame(  
            {'name' : model.booster_.feature_name(),  
            'importance' : model.feature_importances_  
          }).sort_values(by = ['importance'],ascending = False) 
feature_lst2 = list(feature[feature.importance>5].name)
train = dff_train[feature_lst2].copy()  
train_y = df_train['bad_ind'].copy()  
val = dff_val[feature_lst2].copy()  
val_y = df_test['bad_ind'].copy()  
lgb_lm = LogisticRegression(C = 0.3,class_weight='balanced',solver='liblinear')
lgb_lm.fit(train, train_y)  
  
y_pred_lgb_lm_train = lgb_lm.predict_proba(train)[:, 1]  
fpr_lgb_lm_train, tpr_lgb_lm_train, _ = roc_curve(train_y, y_pred_lgb_lm_train)
  
y_pred_lgb_lm = lgb_lm.predict_proba(val)[:, 1]  
fpr_lgb_lm, tpr_lgb_lm, _ = roc_curve(val_y, y_pred_lgb_lm)  
  
plt.figure(1)  
plt.plot([0, 1], [0, 1], 'k--')  
plt.plot(fpr_lgb_lm_train, tpr_lgb_lm_train, label='LGB + LR train')  
plt.plot(fpr_lgb_lm, tpr_lgb_lm, label='LGB + LR test')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC curve')  
plt.legend(loc='best')  
plt.show()  
print('LGB+LR train ks:',abs(fpr_lgb_lm_train - tpr_lgb_lm_train).max(),
      'LGB+LR AUC:', metrics.auc(fpr_lgb_lm_train, tpr_lgb_lm_train))  
print('LGB+LR test ks:',abs(fpr_lgb_lm - tpr_lgb_lm).max(),'LGB+LR AUC:', 
      metrics.auc(fpr_lgb_lm, tpr_lgb_lm))  




#==============================================================================
# File: 7.3.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:00:19 2019

@author: zixing.mei
"""

import torch      
import torch.nn as nn      
import random    
from sklearn.model_selection import train_test_split    
import torchvision.transforms as transforms      
import torchvision.datasets as dsets      
from torch.autograd import Variable      
      
random_st = random.choice(range(10000))      
train_images, test_images = train_test_split(train_images,test_size=0.15,   
                        random_state=random_st)      
    
train_data = MyDataset(train_images)        
test_data = MyDataset(test_images)      
    
train_loader = torch.utils.data.DataLoader(train_data, batch_size=50,   
                       shuffle=True, num_workers=0)  
test_loader = torch.utils.data.DataLoader(test_data, batch_size=25,   
                      huffle=False, num_workers=0)  
#搭建LSTM网络    
class Rnn(nn.Module):      
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):      
        super(Rnn, self).__init__()      
        self.n_layer = n_layer      
        self.hidden_dim = hidden_dim      
        self.LSTM = nn.LSTM(in_dim, hidden_dim,   
                   n_layer,batch_first=True)      
        self.linear = nn.Linear(hidden_dim,n_class)      
        self.sigmoid = nn.Sigmoid()       
       
    def forward(self, x):      
        x = x.sum(dim = 1)      
        out, _ = self.LSTM(x)      
        out = out[:, -1, :]      
        out = self.linear(out)      
        out = self.sigmoid(out)      
        return out  
#28个特征，42个月切片，2个隐层，2分类        
model = Rnn(28,42,2,2)       
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
model = model.to(device)       
#使用二分类对数损失函数      
criterion = nn.SoftMarginLoss(reduction='mean')         
opt = torch.optim.Adam(model.parameters())        
total_step = len(train_loader)        
total_step_test = len(test_loader)    
num_epochs = 50    

for epoch in range(num_epochs):      
    train_label = []      
    train_pred = []      
    model.train()      
    for i, (images, labels) in enumerate(train_loader):      
        images = images.to(device)      
        labels = labels.to(device)      
        #网络训练    
        out = model(images)      
        loss = criterion(out, labels)      
        opt.zero_grad()      
        loss.backward()      
        opt.step()      
        #每一百轮打印一次    
        if i%100 == 0:      
            print('train epoch: {}/{}, round: {}/{},loss: {}'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss))  
        #真实标记和预测值    
        train_label.extend(labels.cpu().numpy().flatten().tolist())      
        train_pred.extend(out.detach().cpu().numpy().flatten().tolist())    
    #计算真正率和假正率    
    fpr_lm_train, tpr_lm_train, _ = roc_curve(np.array(train_label),   
                                                      np.array(train_pred))      
    #计算KS和AUC     
    print('train epoch: {}/{}, KS: {}, ROC: {}'.format(      
        epoch + 1, num_epochs,abs(fpr_lm_train - tpr_lm_train).max(),  
               metrics.auc(fpr_lm_train, tpr_lm_train)))      
        
    test_label = []      
    test_pred = []      
        
    model.eval()      
    #计算测试集上的KS值和AUC值    
    for i, (images, labels) in enumerate(test_loader):      
            
        images = images.to(device)      
        labels = labels.to(device)      
        out = model(images)      
        loss = criterion(out, labels)      
            
        #计算KS和AUC      
        if i%100 == 0:      
            print('test epoch: {}/{}, round: {}/{},loss: {}'.format(
                    epoch + 1, num_epochs,i + 1, total_step_test, loss))      
        test_label.extend(labels.cpu().numpy().flatten().tolist())      
        test_pred.extend(out.detach().cpu().numpy().flatten().tolist())    
        
    fpr_lm_test, tpr_lm_test, _ = roc_curve(np.array(test_label),   
                                                    np.array(test_pred))      
        
    print('test epoch: {}/{}, KS: {}, ROC: {}'.format( epoch + 1,
                                                        num_epochs,
                                                   abs(fpr_lm_test - tpr_lm_test).max(),
                                                      metrics.auc(fpr_lm_test - tpr_lm_test)))




#==============================================================================
# File: 7.4.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:01:35 2019

@author: zixing.mei
"""

#加载xlearn包    
import xlearn as xl    
#调用FM模型    
fm_model = xl.create_fm()    
# 训练集    
fm_model.setTrain("train.txt")    
# 设置验证集    
fm_model.setValidate("test.txt")   
# 分类问题：acc(Accuracy);prec(precision);f1(f1 score);auc(AUC score)    
param = {'task':'binary','lr':0.2,'lambda':0.002,'metric':'auc'}    
fm_model.fit(param, "model.out")   
fm_model.setSigmoid()    
fm_model.predict("model.out","output.txt")    
fm_model.setTXTModel("model.txt")  




#==============================================================================
# File: 7.5.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:02:19 2019

@author: zixing.mei
"""

from heamy.dataset import Dataset  
from heamy.estimator import Regressor  
from heamy.pipeline import ModelsPipeline  
import pandas as pd  
import xgboost as xgb  
from sklearn.metrics import roc_auc_score  
import lightgbm as lgb  
from sklearn.linear_model import LinearRegression  
from sklearn.ensemble import ExtraTreesClassifier  
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn import svm  
import numpy as np  
  
def xgb_model1(X_train, y_train, X_test, y_test=None):  
    # xgboost1  
    params = {'booster': 'gbtree',  
              'objective':'rank:pairwise',  
              'eval_metric' : 'auc',  
              'eta': 0.02,  
              'max_depth': 5,  # 4 3  
              'colsample_bytree': 0.7,#0.8  
              'subsample': 0.7,  
              'min_child_weight': 1,  # 2 3  
              'seed': 1111,  
              'silent':1  
              }  
    dtrain = xgb.DMatrix(X_train, label=y_train)  
    dvali = xgb.DMatrix(X_test)  
    model = xgb.train(params, dtrain, num_boost_round=800)  
    predict = model.predict_proba(dvali)  
    minmin = min(predict)  
    maxmax = max(predict)  
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
    return vfunc(predict)  
  
def xgb_model2(X_train, y_train, X_test, y_test=None):  
    # xgboost2  
    params = {'booster': 'gbtree',  
              'objective':'rank:pairwise',  
              'eval_metric' : 'auc',  
              'eta': 0.015,  
              'max_depth': 5,  # 4 3  
              'colsample_bytree': 0.7,#0.8  
              'subsample': 0.7,  
              'min_child_weight': 1,  # 2 3  
              'seed': 11,  
              'silent':1  
              }  
    dtrain = xgb.DMatrix(X_train, label=y_train)  
    dvali = xgb.DMatrix(X_test)  
    model = xgb.train(params, dtrain, num_boost_round=1200)  
    predict = model.predict_proba (dvali)  
    minmin = min(predict)  
    maxmax = max(predict)  
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
    return vfunc(predict)  
  
def xgb_model3(X_train, y_train, X_test, y_test=None):  
    # xgboost3  
    params = {'booster': 'gbtree',  
              'objective':'rank:pairwise',  
              'eval_metric' : 'auc',  
              'eta': 0.01,  
              'max_depth': 5,  # 4 3  
              'colsample_bytree': 0.7,#0.8  
              'subsample': 0.7,  
              'min_child_weight': 1,  # 2 3  
              'seed': 1,  
              'silent':1  
              }  
    dtrain = xgb.DMatrix(X_train, label=y_train)  
    dvali = xgb.DMatrix(X_test)  
    model = xgb.train(params, dtrain, num_boost_round=2000)  
    predict = model.predict_proba (dvali)  
    minmin = min(predict)  
    maxmax = max(predict)  
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
    return vfunc(predict)  
  
def et_model(X_train, y_train, X_test, y_test=None):  
    #ExtraTree  
    model = ExtraTreesClassifier(max_features='log2',n_estimators=1000,n_jobs=1).fit(X_train,y_train)
    predict = model.predict_proba(X_test)[:,1] 
    minmin = min(predict)  
    maxmax = max(predict)  
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
    return vfunc(predict)     
  
def gbdt_model(X_train, y_train, X_test, y_test=None):  
    #GBDT  
    model = GradientBoostingClassifier(learning_rate=0.02,max_features=0.7,
                                             n_estimators=700,max_depth=5).fit(X_train,y_train)
    predict = model.predict_proba(X_test)[:,1]  
    minmin = min(predict)  
    maxmax = max(predict)  
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
    return vfunc(predict)  
  
def logistic_model(X_train, y_train, X_test, y_test=None):  
    #逻辑回归  
    model = LogisticRegression(penalty = 'l2').fit(X_train,y_train)  
    predict = model.predict_proba(X_test)[:,1] 
    minmin = min(predict)  
    maxmax = max(predict)  
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
    return vfunc(predict)  
 
  
def lgb_model(X_train, y_train, X_test, y_test=None):  
    #LightGBM  
    lgb_train=lgb.Dataset(X_train,y_train,categorical_feature={'sex','merriage','income',
                                                                   'qq_bound','degree',
                                                                   'wechat_bound',
                                                                   'account_grade','industry'})
    lgb_test = lgb.Dataset(X_test,categorical_feature={'sex','merriage','income','qq_bound',
                                                             'degree','wechat_bound',
                                                             'account_grade','industry'})  
    params = {  
        'task': 'train',  
        'boosting_type': 'gbdt',  
        'objective': 'binary',  
        'metric':'auc',  
        'num_leaves': 25,  
        'learning_rate': 0.01,  
        'feature_fraction': 0.7,  
        'bagging_fraction': 0.7,  
        'bagging_freq': 5,  
        'min_data_in_leaf':5,  
        'max_bin':200,  
        'verbose': 0,  
    }  
    gbm = lgb.train(params,  
    lgb_train,  
    num_boost_round=2000)  
    predict = gbm.predict_proba(X_test)  
    minmin = min(predict)  
    maxmax = max(predict)  
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
    return vfunc(predict)  
  
def svm_model(X_train, y_train, X_test, y_test=None):  
    #支持向量机  
    model = svm.SVC(C=0.8,kernel='rbf',gamma=20,
                          decision_function_shape='ovr').fit(X_train,y_train)
    predict = model.predict_proba(X_test)[:,1]  
   	minmin = min(predict)  
   	maxmax = max(predict)  
  	vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
  	return vfunc(predict)  
  
import pandas as pd  
import numpy as np  
from minepy import MINE  
  
""" 
  从csv文件中，加载8个模型的预测分数 
"""  
xgb1_result = pd.read_csv('xgb1.csv')  
xgb2_result = pd.read_csv('xgb2.csv')  
xgb3_result = pd.read_csv('xgb3.csv')  
et_result = pd.read_csv('et_model.csv')  
svm_result = pd.read_csv('svm.csv')  
lr_result = pd.read_csv('lr.csv')  
lgb_result = pd.read_csv('lgb.csv')  
gbdt_result = pd.read_csv('gbdt.csv')  
  
res = []  
res.append(xgb1_result.score.values)  
res.append(xgb2_result.score.values)  
res.append(xgb3_result.score.values)  
res.append(et_result.score.values)  
res.append(svm_result.score.values)  
res.append(lr_result.score.values)  
res.append(lgb_result.score.values)  
res.append(gbdt_result.score.values)  
  
""" 
  计算向量两两之间的MIC值 
"""  
cm = []  
for i in range(7):  
    tmp = []  
    for j in range(7):  
        m = MINE()  
        m.compute_score(res[i], res[j])  
        tmp.append(m.mic())  
    cm.append(tmp)  
  
""" 
    绘制MIC图像 
"""  
  
fs = ['xgb1','xgb2','xgb3','et','svm','lr','lgb','gbdt']  
  
import matplotlib.pyplot as plt  
  
def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  
    plt.title(title)  
    plt.colorbar()  
    tick_marks = np.arange(8)  
    plt.xticks(tick_marks, fs, rotation=45)  
    plt.yticks(tick_marks, fs)  
    plt.tight_layout()  
  
plot_confusion_matrix(cm, title='mic')  
plt.show() 
model_xgb2 = Regressor(dataset= dataset, estimator=xgb_feature2,name='xgb2',use_cache=False) 
model_lr = Regressor(dataset= dataset, estimator=logistic_model,name='lr',use_cache=False) 
model_lgb = Regressor(dataset= dataset, estimator=lgb_model,name='lgb',use_cache=False)  
model_ gbdt = Regressor(dataset= dataset, estimator=gbdt_model,name='gbdt',use_cache=False)
pipeline = ModelsPipeline(model_xgb2, model_lr, model_lgb, model_svm)  
stack_data = pipeline.stack(k=5, seed=0, add_diff=False, full_test=True)  
stacker = Regressor(dataset=stack_data,estimator=LinearRegression,
                      parameters={'fit_intercept': False})  
predict_result = stacker.predict()
val = pd.read_csv('val_list.csv')
val['PROB'] = predict_result
minmin, maxmax = min(val ['PROB']),max(val ['PROB'])
val['PROB'] = val['PROB'].map(lambda x:(x-minmin)/(maxmax-minmin))
val['PROB'] = val['PROB'].map(lambda x:'%.4f' % x)




#==============================================================================
# File: 8.2.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:05:34 2019

@author: zixing.mei
"""

import networkx as nx  
import pandas as pd  
import matplotlib.pyplot as plt  
  
edge_list=pd.read_csv('./data/stack_network_links.csv')  
G=nx.from_pandas_edgelist(edge_list,edge_attr='value' )  
plt.figure(figsize=(15,10))  
nx.draw(G,with_labels=True,  
        edge_color='grey',  
        node_color='pink',  
        node_size = 500,  
        font_size = 40,  
        pos=nx.spring_layout(G,k=0.2))  
#度  
nx.degree(G) 
 
import networkx as nx  
nx.eigenvector_centrality(G)  

import networkx as nx  
nx.pagerank(G,Ap=0.9)  

import networkx as nx  
nx.betweenness_centrality(G)  

import networkx as nx  
nx.closeness_centrality(G)  

preds = nx.jaccard_coefficient(G, [('azure','.net')])  
for u, v, p in preds:  
    print('(%s, %s) -> %.8f' % (u, v, p)) 




#==============================================================================
# File: 8.3.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:07:41 2019

@author: zixing.mei
"""

import networkx as nx    
import numpy as np    
from sklearn.model_selection import train_test_split    
from sklearn.neighbors import KNeighborsClassifier    
from sklearn.svm import SVC    
#给定真实标签    
G = nx.karate_club_graph()    
groundTruth = [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,0,0,0,0,1,1]
#定义邻接矩阵，将网络节点转换成n*n的方阵   
def graphmatrix(G):    
    n = G.number_of_nodes()    
    temp = np.zeros([n,n])    
    for edge in G.edges():    
        temp[int(edge[0])][int(edge[1])] = 1     
        temp[int(edge[1])][int(edge[0])] = 1    
    return temp    
    
edgeMat = graphmatrix(G)    
    
x_train, x_test, y_train, y_test = train_test_split(edgeMat, 
                                     groundTruth, test_size=0.6, random_state=0)
#使用线性核svm分类器进行训练    
clf = SVC(kernel="linear")    
    
clf.fit(x_train, y_train)    
predicted= clf.predict(x_test)     
print(predicted)    
    
score = clf.score(x_test, y_test)    
print(score)   

import networkx as nx  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
#二值化，默认用0.5作为阈值，可以根据业务标签分布调整   
def binary(nodelist, threshold=0.5):  
    for i in range(len(nodelist)):  
        if( nodelist[i] > threshold ): nodelist[i] = 1.0  
        else: nodelist[i] = 0  
    return nodelist  
  
G = nx.karate_club_graph()  
groundTruth = [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,0,0,0,0,1,1]  
max_iter = 2 #迭代次数  
nodes = list(G.nodes())   
nodes_list = {nodes[i]: i for i in range(0, len(nodes))}  
  
vote = np.zeros(len(nodes))  
x_train, x_test, y_train, y_test = train_test_split(nodes, 
                                       groundTruth, test_size=0.7, random_state=1)
  
vote[x_train] = y_train  
vote[x_test] = 0.5 #初始化概率为0.5  
  
for i in range(max_iter):  
    #只用前一次迭代的值  
    last = np.copy(vote)  
    for u in G.nodes():
        if( u in x_train ):
            continue  
        temp = 0.0  
        for item in G.neighbors(u):  
            #对所有邻居求和  
            temp = temp + last[nodes_list[item]]  
        vote[nodes_list[u]] = temp/len(list(G.neighbors(u)))  
 
#二值化得到分类标签   
temp = binary(vote)  
  
pred = temp[x_test]  
#计算准确率   
print(accuracy_score(y_test, pred))  
import networkx as nx  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
from sklearn import preprocessing  
from scipy import sparse  
  
G = nx.karate_club_graph()  
groundTruth = [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,0,0,0,0,1,1]  
  
def graphmatrix(G):  
    #节点抽象成边  
    n = G.number_of_nodes()  
    temp = np.zeros([n,n])  
    for edge in G.edges():  
        temp[int(edge[0])][int(edge[1])] = 1   
        temp[int(edge[1])][int(edge[0])] = 1  
    return temp  
  
def propagation_matrix(G):  
    #矩阵标准化  
    degrees = G.sum(axis=0)  
    degrees[degrees==0] += 1  # 避免除以0  
      
    D2 = np.identity(G.shape[0])  
    for i in range(G.shape[0]):  
        D2[i,i] = np.sqrt(1.0/degrees[i])  
      
    S = D2.dot(G).dot(D2)  
    return S  
#定义取最大值的函数   
def vec2label(Y):  
    return np.argmax(Y,axis=1)  
  
edgematrix = graphmatrix(G)  
S = propagation_matrix(edgematrix)  
  
Ap = 0.8  
cn = 2  
max_iter = 10  
  
#定义迭代函数  
F = np.zeros([G.number_of_nodes(),2])  
X_train, X_test, y_train, y_test = train_test_split(list(G.nodes()), 
                                       groundTruth, test_size=0.7, random_state=1)
for (node, label) in zip(X_train, y_train):  
    F[node][label] = 1  
  
Y = F.copy()  
  
for i in range(max_iter):  
    F_old = np.copy(F)  
    F = Ap*np.dot(S, F_old) + (1-Ap)*Y  
  
temp = vec2label(F)  
pred = temp[X_test]  
print(accuracy_score(y_test, pred))  




#==============================================================================
# File: 8.4.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:11:33 2019

@author: zixing.mei
"""

import networkx as nx  
from networkx.algorithms import community  
import itertools  
  
G = nx.karate_club_graph()  
comp = community.girvan_newman(G)     
# 令社区个数为4，这样会依次得到K=2，K=3，K=4时候的划分结果
k = 4  
limited = itertools.takewhile(lambda c: len(c) <= k, comp)  
for communities in limited:  
    print(tuple(sorted(c) for c in communities)) 
    
    
import networkx as nx  
import community   
G = nx.karate_club_graph()  
part = community.best_partition(G)  
print(len(part)) 


import math  
import numpy as np  
from sklearn import metrics  
def NMI(A,B):  
    total = len(A)  
    X = set(A)  
    Y = set(B)  
    #计算互信息MI  
    MI = 0  
    eps = 1.4e-45  
    for x in X:  
        for y in Y:  
            AOC = np.where(A==x)  
            BOC = np.where(B==y)  
            ABOC = np.intersect1d(AOC,BOC)  
            px = 1.0*len(AOC[0])/total  
            py = 1.0*len(BOC[0])/total  
            pxy = 1.0*len(ABOC)/total  
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)  
    # 标准化互信息NMI  
    Hx = 0  
    for x in X:  
        AOC = 1.0*len(np.where(A==x)[0])  
        Hx = Hx - (AOC/total)*math.log(AOC/total+eps,2)  
    Hy = 0  
    for y in Y:  
        BOC = 1.0*len(np.where(B==y)[0])  
        Hy = Hy - (BOC/total)*math.log(BOC/total+eps,2)  
    NMI = 2.0*MI/(Hx+Hy)  
    return NMI  
#测试   
if __name__ == '__main__':  
    A = np.array([1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3])  
    B = np.array([1,2,1,1,1,1,1,2,2,2,2,3,1,1,3,3,3]) 
       #调用自定义的NMI函数 
    print(NMI(A,B))  
        #调用sklearn封装好的NMI函数
    print(metrics.normalized_mutual_info_score(A,B))




#==============================================================================
# File: 8.5.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:23:41 2019

@author: zixing.mei
"""

import numpy as np  
  
def get_cost(X, U, V, lamb=0):  
    '''''  
        计算损失函数     
        J = |X-UV|+ lamb*(|U|+|V|) 
        输入: X [n, d], U [n, m], V [m, d]  
    '''  
    UV = np.dot(U, V)   
    cost1 = np.sum((X - UV)**2)  
    cost2 = np.sum(U**2) + np.sum(V**2)  
    res = cost1 + lamb*cost2  
    return res  
  
def Matrix_Factor(X, m, lamb=0.1, learnRate=0.01):  
    '''''  
        损失函数定义 
        J = |X-UV| + lamb*(|U|+|V|) 
        输入: X [n, d]  
        输出: U [n, m], V [m, n] 
    '''  
    maxIter = 100  
    n, d = X.shape  
    #随机初始化  
    U = np.random.random([n, m])/n  
    V = np.random.random([m, d])/m  
    # 迭代  
    iter_num = 1   
    while iter_num < maxIter:  
        #计算U的偏导  
        dU = 2*( -np.dot(X, V.T) + np.linalg.multi_dot([U, V, V.T]) + lamb*U )
        U = U - learnRate * dU  
        #计算V的偏导  
        dV = 2*( -np.dot(U.T, X) + np.linalg.multi_dot([U.T, U, V]) + lamb*V )
        V = V - learnRate * dV  
        iter_num += 1  
    return U, V  


import numpy as np  
import networkx as nx  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import roc_curve,auc  
from matplotlib import pyplot as plt   
import random
#加载数据 
G = nx.karate_club_graph()  
groundTruth = [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]  
  
#构造邻接矩阵  
def graph2matrix(G):  
    n = G.number_of_nodes()  
    res = np.zeros([n,n])  
    for edge in G.edges():  
        res[int(edge[0])][int(edge[1])] = 1   
        res[int(edge[1])][int(edge[0])] = 1  
    return res  
  
#生成网络  
G = nx.karate_club_graph()  
G = graph2matrix(G)  
  
#迭代20次  
[U, V] = Matrix_Factor(G, 20)  
#划分训练集测试集  
X_train, X_test, y_train, y_test = train_test_split(U,groundTruth,test_size=0.7,random_state=1)
#逻辑回归训练  
lgb_lm = LogisticRegression(penalty='l2',C=0.2,class_weight='balanced',solver='liblinear')  
lgb_lm.fit(X_train, y_train)   
   
y_pred_lgb_lm_train = lgb_lm.predict_proba(X_train)[:, 1]    
fpr_lgb_lm_train, tpr_lgb_lm_train, _ = roc_curve(y_train,y_pred_lgb_lm_train)  
  
y_pred_lgb_lm = lgb_lm.predict_proba(X_test)[:,1]    
fpr_lgb_lm,tpr_lgb_lm,_ = roc_curve(y_test,y_pred_lgb_lm)    
  
#计算KS值并绘制ROC曲线  
plt.figure(1)    
plt.plot([0, 1], [0, 1], 'k--')    
plt.plot(fpr_lgb_lm_train,tpr_lgb_lm_train,label='LGB + LR train')    
plt.plot(fpr_lgb_lm, tpr_lgb_lm, label='LGB + LR test')    
plt.xlabel('False positive rate')    
plt.ylabel('True positive rate')    
plt.title('ROC curve')    
plt.legend(loc='best')    
plt.show()    
print('train ks:',abs(fpr_lgb_lm_train - tpr_lgb_lm_train).max(),  
                'test AUC:',auc(fpr_lgb_lm_train, tpr_lgb_lm_train))  
print('test ks:',abs(fpr_lgb_lm - tpr_lgb_lm).max(),  
               ' test AUC:', auc(fpr_lgb_lm, tpr_lgb_lm)) 

def rondom_walk (self,length, start_node):  
    walk = [start_node]  
    while len(walk) < length:  
        temp = walk[-1]  
        temp_nbrs = list(self.G.neighbors(temp))  
        if len(temp_nbrs) > 0:  
            walk.append(random.choice(temp_nbrs))  
        else:  
            break  
    return walk  

#Node2Vec
import networkx as nx
from node2vec import Node2Vec
 
# 自定义图
graph = nx.fast_gnp_random_graph(n=100, p=0.5)
 
# 预计算概率并生成行走
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)  
 
# 嵌入节点
model = node2vec.fit(window=10, min_count=1, batch_words=4)  
 
# 寻找最相似节点
model.wv.most_similar('2')
 
# 保存节点嵌入结果
model.wv.save_word2vec_format('EMBEDDING_FILENAME')
 
# 保存模型
model.save('EMBEDDING_MODEL_FILENAME')
 
# 用Hadamard方法嵌入边
from node2vec.edges import HadamardEmbedder
 
edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
 
# 快速查找嵌入
edges_embs[('1', '2')]

# 在单独的实例中获取所有边
edges_kv = edges_embs.as_keyed_vectors()
 
# 寻找最相似边
edges_kv.most_similar(str(('1', '2')))
 
# 保存边嵌入结果
edges_kv.save_word2vec_format('EDGES_EMBEDDING_FILENAME')






#==============================================================================
# File: 8.6.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:51:29 2019

@author: zixing.mei
"""

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import networkx as nx  
def normalize(A , symmetric=True):  
    # A = A+I  
    A = A + torch.eye(A.size(0))  
    # 所有节点的度  
    d = A.sum(1)  
    if symmetric:  
        #D = D^-1/2  
        D = torch.diag(torch.pow(d , -0.5))  
        return D.mm(A).mm(D)  
    else :  
        # D=D^-1  
        D =torch.diag(torch.pow(d,-1))  
        return D.mm(A)  
class GCN(nn.Module):  
    ''' 
    Z = AXW 
    '''  
    def __init__(self , A, dim_in , dim_out):  
        super(GCN,self).__init__()  
        self.A = A  
        self.fc1 = nn.Linear(dim_in ,dim_in,bias=False)  
        self.fc2 = nn.Linear(dim_in,dim_in//2,bias=False)  
        self.fc3 = nn.Linear(dim_in//2,dim_out,bias=False)  
  
    def forward(self,X):  
        ''' 
        计算三层GCN 
        '''  
        X = F.relu(self.fc1(self.A.mm(X)))  
        X = F.relu(self.fc2(self.A.mm(X)))  
        return self.fc3(self.A.mm(X))  
#获得数据    
G = nx.karate_club_graph()    
A = nx.adjacency_matrix(G).todense()    
#矩阵A需要标准化    
A_normed = normalize(torch.FloatTensor(A/1.0),True)    
    
N = len(A)    
X_dim = N    
    
# 没有节点的特征，简单用一个单位矩阵表示所有节点    
X = torch.eye(N,X_dim)    
# 正确结果    
Y = torch.zeros(N,1).long()    
# 计算loss的时候要去掉没有标记的样本    
Y_mask = torch.zeros(N,1,dtype=torch.uint8)    
# 一个分类给一个样本    
Y[0][0]=0    
Y[N-1][0]=1    
#有样本的地方设置为1    
Y_mask[0][0]=1    
Y_mask[N-1][0]=1    
    
#真实的空手道俱乐部的分类数据    
Real = torch.zeros(34 , dtype=torch.long)    
for i in [1,2,3,4,5,6,7,8,11,12,13,14,17,18,20,22] :    
    Real[i-1] = 0    
for i in [9,10,15,16,19,21,23,24,25,26,27,28,29,30,31,32,33,34] :    
    Real[i-1] = 1    
    
#  GCN模型    
gcn = GCN(A_normed ,X_dim,2)    
#选择adam优化器    
gd = torch.optim.Adam(gcn.parameters())    
    
for i in range(300):    
    #转换到概率空间    
    y_pred =F.softmax(gcn(X),dim=1)    
    #下面两行计算cross entropy    
    loss = (-y_pred.log().gather(1,Y.view(-1,1)))    
    #仅保留有标记的样本    
    loss = loss.masked_select(Y_mask).mean()    
    
    #梯度下降    
    #清空前面的导数缓存    
    gd.zero_grad()    
    #求导    
    loss.backward()    
    #一步更新    
    gd.step()    
    
    if i%100==0 :    
        _,mi = y_pred.max(1)    
        print(mi)    
        #计算精确度    
print((mi == Real).float().mean())




#==============================================================================
# File: autotreemodel.py
#==============================================================================

"""Main module."""



#==============================================================================
# File: auto_build_scorecard.py
#==============================================================================

#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: auto_build_scorecard.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-05-20
'''

import os
import time
import warnings

import numpy as np
import pandas as pd
import xlsxwriter

import autobmt

warnings.filterwarnings('ignore')

log = autobmt.Logger(level='info', name=__name__).logger


class AutoBuildScoreCard:

    def __init__(self, datasets, fea_names, target, key='key', data_type='type',
                 no_feature_names=['key', 'target', 'apply_time', 'type'], ml_res_save_path='model_result',
                 AB={}, data_dict=None, positive_corr=False):

        if data_type not in datasets:
            raise KeyError('train、test数据集标识的字段名不存在！或未进行数据集的划分，请将数据集划分为train、test！！！')

        datasets[data_type] = datasets[data_type].map(str.lower)

        self.data_type_ar = np.unique(datasets[data_type])
        if 'train' not in self.data_type_ar:
            raise KeyError("""没有开发样本，数据集标识字段{}没有`train`该取值！！！""".format(data_type))

        if 'test' not in self.data_type_ar:
            raise KeyError("""没有验证样本，数据集标识字段{}没有`test`该取值！！！""".format(data_type))

        if target not in datasets:
            raise KeyError('样本中没有目标变量y值！！！')

        if data_dict is not None and isinstance(data_dict, pd.DataFrame):
            if not data_dict.columns.isin(['feature', 'cn']).all():
                raise KeyError("原始数据字典中没有feature或cn字段，请保证同时有feature字段和cn字段")

        # fea_names = [i for i in fea_names if i != key and i != target]
        fea_names = [i for i in fea_names if i not in no_feature_names]
        log.info('数据集变量个数 : {}'.format(len(fea_names)))
        log.info('fea_names is : {}'.format(fea_names))

        self.datasets = datasets
        self.fea_names = fea_names
        self.target = target
        self.key = key
        self.no_feature_names = no_feature_names
        self.data_dict = data_dict
        self.ml_res_save_path = os.path.join(ml_res_save_path, time.strftime('%Y%m%d%H%M%S_%S', time.localtime()))
        self.AB = AB
        self.positive_corr = positive_corr  # 分数与模型预测的概率值是否正相关。默认False，负相关，即概率约高，分数越低

        os.makedirs(self.ml_res_save_path, exist_ok=True)

    def fit(self, empty_threhold=0.9, iv_threhold=0.02, corr_threhold=0.7, psi_threhold=0.1,
            dev_nodev_iv_diff_threhold=0.08):
        log.info('样本数据集类型：{}'.format(self.data_type_ar))
        log.info('样本行列情况：{}'.format(self.datasets.shape))
        log.info('样本正负占比情况：')
        log.info(self.datasets[self.target].value_counts() / len(self.datasets))

        excel_file_name_path = "{}/build_model_log.xlsx".format(self.ml_res_save_path)
        workbook = xlsxwriter.Workbook(excel_file_name_path, {'nan_inf_to_errors': True})
        excel_utils = autobmt.Report2Excel(excel_file_name_path, workbook=workbook)
        log.info('读取样本&特征数据集：{} |为样本数据，其他为特征数据'.format(self.no_feature_names))
        # =========================读取数据集结束=========================

        time_start = time.time()

        log.info('Step 1: EDA，整体数据探索性数据分析')
        all_data_eda = autobmt.detect(self.datasets)
        all_data_eda.to_excel('{}/all_data_eda.xlsx'.format(self.ml_res_save_path))
        excel_utils.df_to_excel(all_data_eda, '1.EDA')

        log.info('Step 2: 特征粗筛选开始')
        # 进行特征初步选择
        # 分箱方法支持 dt、chi、equal_freq、kmeans
        fs_dic = {
            "empty": {'threshold': empty_threhold},
            # "const": {'threshold': 0.95},
            "psi": {'threshold': psi_threhold},
            "iv": {'threshold': iv_threhold},
            # "iv_diff": {'threshold': dev_nodev_iv_diff_threhold},
            "corr": {'threshold': corr_threhold},

        }

        # TODO 考虑将self.datasets更换为train_data
        fs = autobmt.FeatureSelection(df=self.datasets, target=self.target, exclude_columns=self.no_feature_names,
                                      match_dict=self.data_dict,
                                      params=fs_dic)
        selected_df, selected_features, select_log_df, fbfb = fs.select()
        summary = autobmt.calc_var_summary(
            fbfb.transform(selected_df[selected_df['type'] == 'train'])[selected_features + [self.target]],
            fbfb.export(), target=self.target, need_bin=False)
        autobmt.del_df(self.datasets)

        log.info('Step 2: 特征粗筛选结束')

        log.info('Step 3: 对剩下的变量调用最优分箱')
        log.info('剩下的变量个数 : {}'.format(len(selected_features)))
        log.info('剩下的变量 : {}'.format(selected_features))

        # train_data = selected_df[selected_df['type'] == 'train']

        # 训练集
        fb, best_binning_result = autobmt.best_binning(selected_df[selected_df['type'] == 'train'],
                                                       x_list=selected_features,
                                                       target=self.target)
        df_bin = fb.transform(selected_df, labels=False)
        train_bin = df_bin[df_bin['type'] == 'train']
        test_bin = df_bin[df_bin['type'] == 'test']
        if 'oot' in self.data_type_ar:
            oot_bin = df_bin[df_bin['type'] == 'oot']

        train_var_summary = autobmt.calc_var_summary(train_bin[selected_features + [self.target]], fb.export(),
                                                     target=self.target, need_bin=False)

        # 测试集
        # test_bin = fb.transform(test_data, labels=False)
        test_var_summary = autobmt.calc_var_summary(test_bin[selected_features + [self.target]], fb.export(),
                                                    target=self.target, need_bin=False)

        if 'oot' in self.data_type_ar:
            # oot_bin = fb.transform(oot_data, labels=False)
            oot_var_summary = autobmt.calc_var_summary(oot_bin[selected_features + [self.target]], fb.export(),
                                                       target=self.target, need_bin=False)

        # 计算拐点
        train_data_inflection_df = autobmt.compare_inflection_point(train_var_summary)
        train_data_inflection_df.columns = train_data_inflection_df.columns.map(lambda x: 'train_' + x)

        test_data_inflection_df = autobmt.compare_inflection_point(test_var_summary)
        test_data_inflection_df.columns = test_data_inflection_df.columns.map(lambda x: 'test_' + x)
        data_inflection_df = pd.concat([train_data_inflection_df, test_data_inflection_df], axis=1)
        if 'oot' in self.data_type_ar:
            oot_data_inflection_df = autobmt.compare_inflection_point(oot_var_summary)
            oot_data_inflection_df.columns = oot_data_inflection_df.columns.map(lambda x: 'oot_' + x)
            data_inflection_df = pd.concat([data_inflection_df, oot_data_inflection_df], axis=1)

        # 存储特征粗筛内容
        # 存储最优分箱内容
        excel_utils.df_to_excel(select_log_df, '2.feature_selection')
        autobmt.var_summary_to_excel(summary, workbook, '3.binning_summary')
        autobmt.var_summary_to_excel(train_var_summary, workbook, '4.best_train_binning_summary')
        autobmt.var_summary_to_excel(test_var_summary, workbook, '5.best_test_binning_summary')
        if 'oot' in self.data_type_ar:
            autobmt.var_summary_to_excel(oot_var_summary, workbook, '6.best_oot_binning_summary')
        excel_utils.df_to_excel(best_binning_result, '7.best_binning_log')
        excel_utils.df_to_excel(data_inflection_df.reset_index(), '8.data_inflection_point')
        # 转woe
        log.info('Step 4: 对剩下的变量进行woe转换')
        woetf = autobmt.WoeTransformer()
        train_woe = woetf.fit_transform(train_bin, train_bin[self.target], exclude=self.no_feature_names)
        test_woe = woetf.transform(test_bin)
        if 'oot' in self.data_type_ar:
            oot_woe = woetf.transform(oot_bin)

        # TODO list：iv_psi_missing_df再加上coef、pvalue、vif
        # TODO list：1、通过稳定性筛选特征 2、由于分箱转woe值后，变量之间的共线性会变强，通过相关性再次筛选特征

        log.info('Step 5: 对woe转换后的变量进行stepwise')
        if 'oot' in self.data_type_ar:
            in_model_data = pd.concat([train_woe, test_woe, oot_woe])
        else:
            in_model_data = pd.concat([train_woe, test_woe])

        var_bin_woe = autobmt.fea_woe_dict_format(woetf.export(), fb.export())

        # 将woe转化后的数据做逐步回归
        final_data = autobmt.stepwise(train_woe, target=self.target, estimator='ols', direction='both',
                                      criterion='aic',
                                      exclude=self.no_feature_names)

        # 确定建模要用的变量
        selected_features = [fea for fea in final_data.columns if fea not in self.no_feature_names]

        log.info('Step 6: 用逻辑回归构建模型')
        # 用逻辑回归建模
        from sklearn.linear_model import LogisticRegression

        while True:  # 循环的目的是保证入模变量的系数都为整
            lr = LogisticRegression()
            lr.fit(final_data[selected_features], final_data[self.target])
            drop_var = np.array(selected_features)[np.where(lr.coef_ < 0)[1]]
            if len(drop_var) == 0:
                break
            selected_features = list(set(selected_features) - set(drop_var))

        in_model_data['p'] = lr.predict_proba(in_model_data[selected_features])[:, 1]

        ###
        psi_v = autobmt.psi(test_woe[selected_features], train_woe[selected_features])
        psi_v.name = 'train_test_psi'
        train_iv = train_var_summary[['var_name', 'IV']].rename(columns={'IV': 'train_iv'}).drop_duplicates().set_index(
            'var_name')
        test_iv = test_var_summary[['var_name', 'IV']].rename(columns={'IV': 'test_iv'}).drop_duplicates().set_index(
            'var_name')
        var_miss = select_log_df[['feature', 'cn', 'miss_rate']].drop_duplicates().set_index('feature')

        if 'oot' in self.data_type_ar:
            psi_o = autobmt.psi(oot_woe[selected_features], train_woe[selected_features])
            psi_o.name = 'train_oot_psi'
            oot_iv = oot_var_summary[['var_name', 'IV']].rename(columns={'IV': 'oot_iv'}).drop_duplicates().set_index(
                'var_name')

        coef_s = {}
        for idx, key in enumerate(selected_features):
            coef_s[key] = lr.coef_[0][idx]
        var_coef = pd.Series(coef_s, name='coef')
        var_vif = autobmt.get_vif(final_data[selected_features])
        statsm = autobmt.StatsModel(estimator='ols', intercept=True)
        t_p_c_value = statsm.stats(final_data[selected_features], final_data[self.target])
        p_value = pd.Series(t_p_c_value['p_value'], name='p_value')
        t_value = pd.Series(t_p_c_value['t_value'], name='t_value')

        if 'oot' in self.data_type_ar:
            var_info = pd.concat(
                [var_coef, p_value, t_value, var_vif, train_iv, test_iv, oot_iv, psi_v, psi_o, var_miss],
                axis=1).dropna(subset=['coef'])
        else:
            var_info = pd.concat([var_coef, p_value, t_value, var_vif, train_iv, test_iv, psi_v, var_miss],
                                 axis=1).dropna(subset=['coef'])
        ###

        log.info('Step 7: 构建评分卡')
        card = autobmt.ScoreCard(
            combiner=fb,
            transer=woetf,
            # class_weight = 'balanced',
            # C=0.1,
            # base_score = 600,
            # odds = 15 ,
            # pdo = 50,
            # rate = 2
            AB=self.AB  # 自定义的大A，大B
        )

        card.fit(final_data[selected_features], final_data[self.target])

        log.info('Step 8: 持久化模型，分箱点，woe值，评分卡结构======>开始')

        autobmt.dump_to_pkl(fb, os.path.join(self.ml_res_save_path, 'fb.pkl'))
        autobmt.dump_to_pkl(woetf, os.path.join(self.ml_res_save_path, 'woetf.pkl'))
        autobmt.dump_to_pkl(selected_features, os.path.join(self.ml_res_save_path, 'in_model_var.pkl'))

        woetf.export(to_json=os.path.join(self.ml_res_save_path, 'var_bin_woe.json'))
        woetf.export(to_json=os.path.join(self.ml_res_save_path, 'var_bin_woe_format.json'), var_bin_woe=var_bin_woe)
        fb.export(to_json=os.path.join(self.ml_res_save_path, 'var_split_point.json'), bin_format=False)
        fb.export(to_json=os.path.join(self.ml_res_save_path, 'var_split_point_format.json'))
        card.export(to_json=os.path.join(self.ml_res_save_path, 'scorecard.json'))

        woetf.export(to_csv=os.path.join(self.ml_res_save_path, 'var_bin_woe.csv'))
        woetf.export(to_csv=os.path.join(self.ml_res_save_path, 'var_bin_woe_format.csv'), var_bin_woe=var_bin_woe)
        fb.export(to_csv=os.path.join(self.ml_res_save_path, 'var_split_point.csv'), bin_format=False)
        fb.export(to_csv=os.path.join(self.ml_res_save_path, 'var_split_point_format.csv'))
        scorecard_structure = card.export(to_dataframe=True)
        scorecard_structure.to_csv(os.path.join(self.ml_res_save_path, 'scorecard.csv'), index=False)

        autobmt.dump_to_pkl(lr, os.path.join(self.ml_res_save_path, 'lrmodel.pkl'))
        autobmt.dump_to_pkl(card, os.path.join(self.ml_res_save_path, 'scorecard.pkl'))
        if self.AB:
            in_model_data['score'] = in_model_data['p'].map(
                lambda x: autobmt.to_score(x, self.AB['A'], self.AB['B'], self.positive_corr))
        else:
            in_model_data['score'] = in_model_data['p'].map(lambda x: autobmt.to_score(x, self.positive_corr))
        output_report_data = in_model_data[self.no_feature_names + ['p', 'score']]

        output_report_data.to_csv(os.path.join(self.ml_res_save_path, 'lr_pred_to_report_data.csv'), index=False)
        selected_df[self.no_feature_names + selected_features].head(500).to_csv(
            os.path.join(self.ml_res_save_path, 'lr_test_input.csv'), index=False)
        lr_auc_ks_psi = autobmt.get_auc_ks_psi(output_report_data)
        lr_auc_ks_psi.to_csv(os.path.join(self.ml_res_save_path, 'lr_auc_ks_psi.csv'),
                             index=False)
        log.info('Step 8: 持久化模型，分箱点，woe值，评分卡结构======>结束')

        log.info('Step 9: 持久化建模中间结果到excel，方便复盘')
        # corr
        train_woe_corr_df = train_woe[selected_features].corr().reset_index()
        test_woe_corr_df = test_woe[selected_features].corr().reset_index()
        if 'oot' in self.data_type_ar:
            oot_woe_corr_df = oot_woe[selected_features].corr().reset_index()

        # 建模中间结果存储
        # 将模型最终评估结果合并到每一步的评估中

        excel_utils.df_to_excel(var_info.reset_index().rename(columns={'index': 'var_name'}), '9.var_info')
        excel_utils.df_to_excel(scorecard_structure, 'scorecard_structure')

        in_model_var_train_summary = train_var_summary[train_var_summary['var_name'].isin(selected_features)]
        in_model_var_test_summary = test_var_summary[test_var_summary['var_name'].isin(selected_features)]
        if 'oot' in self.data_type_ar:
            in_model_var_oot_summary = oot_var_summary[oot_var_summary['var_name'].isin(selected_features)]
        autobmt.var_summary_to_excel(in_model_var_train_summary, workbook, '11.in_model_var_train_summary')
        autobmt.var_summary_to_excel(in_model_var_test_summary, workbook, '12.in_model_var_test_summary')
        if 'oot' in self.data_type_ar:
            autobmt.var_summary_to_excel(in_model_var_oot_summary, workbook, '13.in_model_var_oot_summary')

        excel_utils.df_to_excel(train_woe_corr_df, '14.train_woe_df_corr')
        excel_utils.df_to_excel(test_woe_corr_df, '15.test_woe_df_corr')
        if 'oot' in self.data_type_ar:
            excel_utils.df_to_excel(oot_woe_corr_df, '16.oot_woe_df_corr')
        excel_utils.close_workbook()

        if 'oot' in self.data_type_ar:
            autobmt.plot_var_bin_summary(
                {'train': in_model_var_train_summary, 'test': in_model_var_test_summary,
                 'oot': in_model_var_oot_summary},
                selected_features, target=self.target, by=self.data_type_ar, file_path=excel_file_name_path,
                sheet_name='plot_var_bin_summary')
        else:
            autobmt.plot_var_bin_summary(
                {'train': in_model_var_train_summary, 'test': in_model_var_test_summary},
                selected_features, target=self.target, by=self.data_type_ar, file_path=excel_file_name_path,
                sheet_name='plot_var_bin_summary')

        time_end = time.time()
        time_c = time_end - time_start
        log.info('*' * 30 + '建模相关结果保存完成！！！保存路径为：{}'.format(self.ml_res_save_path) + '*' * 30)
        log.info('模型效果：\n{}'.format(lr_auc_ks_psi))
        log.info('time cost {} s'.format(time_c))

        return lr, selected_features

    @classmethod
    def predict(cls, to_pred_df=None, model_path=None):
        if to_pred_df is None:
            raise ValueError('需要进行预测的数据集不能为None，请指定数据集！！！')
        if model_path is None:
            raise ValueError('模型路径不能为None，请指定模型文件路径！！！')

        fb = autobmt.load_from_pkl(os.path.join(model_path, 'fb.pkl'))
        woetf = autobmt.load_from_pkl(os.path.join(model_path, 'woetf.pkl'))
        lrmodel = autobmt.load_from_pkl(os.path.join(model_path, 'lrmodel.pkl'))
        selected_features = autobmt.load_from_pkl(os.path.join(model_path, 'in_model_var.pkl'))

        bin_data = fb.transform(to_pred_df)
        woe_data = woetf.transform(bin_data)
        to_pred_df['p'] = lrmodel.predict_proba(woe_data[selected_features])[:, 1]

        return to_pred_df



#==============================================================================
# File: auto_build_tree_model.py
#==============================================================================

#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: auto_build_tree_model.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-11-17
'''

import gc
import json
import os
import time
import warnings

import joblib
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import xgboost as xgb

import autobmt

warnings.filterwarnings('ignore')

from .logger_utils import Logger

log = Logger(level='info', name=__name__).logger


class AutoBuildTreeModel():
    def __init__(self, datasets, fea_names, target, key='key', data_type='type',
                 no_feature_names=['key', 'target', 'apply_time', 'type'], ml_res_save_path='model_result',
                 AB={}, positive_corr=False):

        if data_type not in datasets:
            raise KeyError('train、test数据集标识的字段名不存在！或未进行数据集的划分，请将数据集划分为train、test！！！')

        data_type_ar = np.unique(datasets[data_type])
        if 'train' not in data_type_ar:
            raise KeyError("""没有开发样本，数据集标识字段{}没有`train`该取值！！！""".format(data_type))

        if 'test' not in data_type_ar:
            raise KeyError("""没有验证样本，数据集标识字段{}没有`test`该取值！！！""".format(data_type))

        if target not in datasets:
            raise KeyError('样本中没有目标变量y值！！！')

        # fea_names = [i for i in fea_names if i != key and i != target]
        fea_names = [i for i in fea_names if i not in no_feature_names]
        log.info('数据集变量个数 : {}'.format(len(fea_names)))
        log.info('fea_names is : {}'.format(fea_names))

        self.datasets = datasets
        self.fea_names = fea_names
        self.target = target
        self.key = key
        self.no_feature_names = no_feature_names
        self.ml_res_save_path = os.path.join(ml_res_save_path, time.strftime('%Y%m%d%H%M%S_%S', time.localtime()))
        self.AB = AB
        self.positive_corr = positive_corr  # 分数与模型预测的概率值是否正相关。默认False，负相关，即概率约高，分数越低
        self.min_child_samples = max(round(len(datasets[datasets['type'] == 'train']) * 0.02),
                                     50)  # 一个叶子上数据的最小数量. 可以用来处理过拟合

        os.makedirs(self.ml_res_save_path, exist_ok=True)

    def fit(self, is_feature_select=True, is_auto_tune_params=True, is_stepwise_del_feature=True,
            feature_select_method='shap', method_threhold=0.001,
            corr_threhold=0.8, psi_threhold=0.2):
        '''

        Args:
            is_feature_select:
            is_auto_tune_params:
            feature_select_method:
            method_threhold:
            corr_threhold:
            psi_threhold:

        Returns: xgboost.sklearn.XGBClassifier或lightgbm.sklearn.LGBClassifier；list
            返回最优模型，入模变量list

        '''
        log.info('*' * 30 + '开始自动建模' + '*' * 30)

        log.info('*' * 30 + '获取变量名和数据集' + '*' * 30)
        fea_names = self.fea_names.copy()
        dev_data = self.datasets[self.datasets['type'] == 'train']
        nodev_data = self.datasets[self.datasets['type'] == 'test']

        del self.datasets;
        gc.collect()

        # dev_data = self.datasets['dev']
        # nodev_data = self.datasets['nodev']

        params = {
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': 3,
            #'min_child_weight': max(round(len(dev_data) * 0.01), 50),
            'min_child_weight': 5,
            'gamma': 7,
            'subsample': 0.7,
            'colsample_bytree': 0.9,
            'colsample_bylevel': 0.7,
            'reg_alpha': 10,
            'reg_lambda': 10,
            'scale_pos_weight': 1
        }
        log.info('默认参数 {}'.format(params))

        best_model = XGBClassifier(**params)
        # best_model.fit(dev_data[fea_names], dev_data[self.target])
        log.info('构建基础模型')

        if is_feature_select:
            log.info('需要进行变量筛选')
            fea_names = autobmt.feature_select({'dev': dev_data, 'nodev': nodev_data}, fea_names, self.target,
                                               feature_select_method, method_threhold,
                                               corr_threhold,
                                               psi_threhold)

        if is_auto_tune_params:
            log.info('需要进行自动调参')
            best_model = autobmt.classifiers_model_auto_tune_params(
                train_data=(dev_data[fea_names], dev_data[self.target]),
                test_data=(nodev_data[fea_names], nodev_data[self.target]))
            params = best_model.get_params()

        if is_stepwise_del_feature:
            log.info('需要逐步的删除变量')
            _, fea_names = autobmt.stepwise_del_feature(best_model, {'dev': dev_data, 'nodev': nodev_data}, fea_names,
                                                        self.target,
                                                        params)

        # 最终模型
        log.info('使用自动调参选出来的最优参数+筛选出来的变量，构建最终模型')
        log.info('最终变量的个数{}, 最终变量{}'.format(len(fea_names), fea_names))
        log.info('自动调参选出来的最优参数{}'.format(params))
        # xgb_clf = XGBClassifier(**params)
        # xgb_clf.fit(dev_data[fea_names], dev_data[self.target])
        best_model.fit(dev_data[fea_names], dev_data[self.target])

        # ###
        # pred_nodev = xgb_clf.predict_proba(nodev_data[fea_names])[:, 1]
        # pred_dev = xgb_clf.predict_proba(dev_data[fea_names])[:, 1]
        # df_pred_nodev = pd.DataFrame({'target': nodev_data[self.target], 'p': pred_nodev}, index=nodev_data.index)
        # df_pred_dev = pd.DataFrame({'target': dev_data[self.target], 'p': pred_dev}, index=dev_data.index)
        # ###

        # ###
        # df_pred_nodev = nodev_data[self.no_feature_names + fea_names]
        # df_pred_dev = dev_data[self.no_feature_names + fea_names]
        # df_pred_nodev['p'] = xgb_clf.predict_proba(df_pred_nodev[fea_names])[:, 1]
        # df_pred_dev['p'] = xgb_clf.predict_proba(df_pred_dev[fea_names])[:, 1]
        # ###

        ###
        df_pred_nodev = nodev_data[self.no_feature_names]
        df_pred_dev = dev_data[self.no_feature_names]
        df_pred_nodev['p'] = best_model.predict_proba(nodev_data[fea_names])[:, 1]
        df_pred_dev['p'] = best_model.predict_proba(dev_data[fea_names])[:, 1]
        ###

        # 计算auc、ks、psi
        test_ks = autobmt.get_ks(df_pred_nodev[self.target], df_pred_nodev['p'])
        train_ks = autobmt.get_ks(df_pred_dev[self.target], df_pred_dev['p'])
        test_auc = autobmt.get_auc(df_pred_nodev[self.target], df_pred_nodev['p'])
        train_auc = autobmt.get_auc(df_pred_dev[self.target], df_pred_dev['p'])

        q_cut_list = np.arange(0, 1, 1 / 20)
        bins = np.append(np.unique(np.quantile(df_pred_nodev['p'], q_cut_list)), df_pred_nodev['p'].max() + 0.1)
        df_pred_nodev['range'] = pd.cut(df_pred_nodev['p'], bins=bins, precision=0, right=False).astype(str)
        df_pred_dev['range'] = pd.cut(df_pred_dev['p'], bins=bins, precision=0, right=False).astype(str)
        nodev_psi = autobmt.psi(df_pred_nodev['range'], df_pred_dev['range'])
        res_dict = {'dev_auc': train_auc, 'nodev_auc': test_auc, 'dev_ks': train_ks, 'nodev_ks': test_ks,
                    'nodev_dev_psi': nodev_psi}
        log.info('auc & ks & psi: {}'.format(res_dict))
        log.info('*' * 30 + '自动构建模型完成！！！' + '*' * 30)

        ##############
        log.info('*' * 30 + '建模相关结果开始保存！！！' + '*' * 30)
        joblib.dump(best_model._Booster, os.path.join(self.ml_res_save_path, 'xgb.ml'))
        joblib.dump(best_model, os.path.join(self.ml_res_save_path, 'xgb_sk.ml'))
        autobmt.dump_to_pkl(best_model._Booster, os.path.join(self.ml_res_save_path, 'xgb.pkl'))
        autobmt.dump_to_pkl(best_model, os.path.join(self.ml_res_save_path, 'xgb_sk.pkl'))
        json.dump(best_model.get_params(), open(os.path.join(self.ml_res_save_path, 'xgb.params'), 'w'))
        best_model._Booster.dump_model(os.path.join(self.ml_res_save_path, 'xgb.txt'))
        pd.DataFrame([res_dict]).to_csv(os.path.join(self.ml_res_save_path, 'xgb_auc_ks_psi.csv'), index=False)

        pd.DataFrame(list(best_model._Booster.get_fscore().items()),
                     columns=['fea_names', 'weight']
                     ).sort_values('weight', ascending=False).set_index('fea_names').to_csv(
            os.path.join(self.ml_res_save_path, 'xgb_featureimportance.csv'))

        nodev_data[self.no_feature_names + fea_names].head(500).to_csv(
            os.path.join(self.ml_res_save_path, 'xgb_test_input.csv'),
            index=False)

        ##############pred to score
        df_pred_nodev['score'] = df_pred_nodev['p'].map(
            lambda x: autobmt.to_score(x, self.AB['A'], self.AB['B'], self.positive_corr))
        df_pred_dev['score'] = df_pred_dev['p'].map(
            lambda x: autobmt.to_score(x, self.AB['A'], self.AB['B'], self.positive_corr))
        ##############pred to score

        df_pred_nodev.append(df_pred_dev).to_csv(os.path.join(self.ml_res_save_path, 'xgb_pred_to_report_data.csv'),
                                                 index=False)

        log.info('*' * 30 + '建模相关结果保存完成！！！保存路径为：{}'.format(self.ml_res_save_path) + '*' * 30)

        return best_model, fea_names

    @classmethod
    def predict(cls, to_pred_df=None, model_path=None):
        if to_pred_df is None:
            raise ValueError('需要进行预测的数据集不能为None，请指定数据集！！！')
        if model_path is None:
            raise ValueError('模型路径不能为None，请指定模型文件路径！！！')

        try:
            model = joblib.load(os.path.join(model_path, 'xgb.ml'))
        except:
            model = pickle.load(open(os.path.join(model_path, 'xgb.pkl'), 'rb'))

        try:
            model_feature_names = model.feature_names
        except:
            model_feature_names = model.get_booster().feature_names
            model = model.get_booster()

        to_pred_df['p'] = model.predict(xgb.DMatrix(to_pred_df[model_feature_names]))

        return to_pred_df



#==============================================================================
# File: auto_build_tree_model_lgb.py
#==============================================================================

#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: auto_build_tree_model.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-11-17
'''

import gc
import json
import os
import time
import warnings

import joblib
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

import autobmt

warnings.filterwarnings('ignore')

from .logger_utils import Logger

log = Logger(level='info', name=__name__).logger


class AutoBuildTreeModelLGB():
    def __init__(self, datasets, fea_names, target, key='key', data_type='type',
                 no_feature_names=['key', 'target', 'apply_time', 'type'], ml_res_save_path='model_result',
                 AB={}, positive_corr=False):

        if data_type not in datasets:
            raise KeyError('train、test数据集标识的字段名不存在！或未进行数据集的划分，请将数据集划分为train、test！！！')

        data_type_ar = np.unique(datasets[data_type])
        if 'train' not in data_type_ar:
            raise KeyError("""没有开发样本，数据集标识字段{}没有`train`该取值！！！""".format(data_type))

        if 'test' not in data_type_ar:
            raise KeyError("""没有验证样本，数据集标识字段{}没有`test`该取值！！！""".format(data_type))

        if target not in datasets:
            raise KeyError('样本中没有目标变量y值！！！')

        # fea_names = [i for i in fea_names if i != key and i != target]
        fea_names = [i for i in fea_names if i not in no_feature_names]
        log.info('数据集变量个数 : {}'.format(len(fea_names)))
        log.info('fea_names is : {}'.format(fea_names))

        self.datasets = datasets
        self.fea_names = fea_names
        self.target = target
        self.key = key
        self.no_feature_names = no_feature_names
        self.ml_res_save_path = os.path.join(ml_res_save_path, time.strftime('%Y%m%d%H%M%S_%S', time.localtime()))
        self.AB = AB
        self.positive_corr = positive_corr  # 分数与模型预测的概率值是否正相关。默认False，负相关，即概率约高，分数越低
        self.min_child_samples = max(round(len(datasets[datasets['type'] == 'train']) * 0.02),
                                     50)  # 一个叶子上数据的最小数量. 可以用来处理过拟合

        os.makedirs(self.ml_res_save_path, exist_ok=True)

    def fit(self, is_feature_select=True, is_auto_tune_params=True, is_stepwise_del_feature=True,
            feature_select_method='shap', method_threhold=0.001,
            corr_threhold=0.8, psi_threhold=0.2):
        '''

        Args:
            is_feature_select:
            is_auto_tune_params:
            feature_select_method:
            method_threhold:
            corr_threhold:
            psi_threhold:

        Returns: xgboost.sklearn.XGBClassifier或lightgbm.sklearn.LGBClassifier；list
            返回最优模型，入模变量list

        '''
        log.info('*' * 30 + '开始自动建模' + '*' * 30)

        log.info('*' * 30 + '获取变量名和数据集' + '*' * 30)
        fea_names = self.fea_names.copy()
        dev_data = self.datasets[self.datasets['type'] == 'train']
        nodev_data = self.datasets[self.datasets['type'] == 'test']

        del self.datasets;
        gc.collect()

        # dev_data = self.datasets['dev']
        # nodev_data = self.datasets['nodev']

        params = {
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': 3,
            #'min_child_weight': max(round(len(dev_data) * 0.01), 50),
            'min_child_weight': 5,
            'gamma': 7,
            'subsample': 0.7,
            'colsample_bytree': 0.9,
            'colsample_bylevel': 0.7,
            'reg_alpha': 10,
            'reg_lambda': 10,
            'scale_pos_weight': 1
        }
        log.info('默认参数 {}'.format(params))

        best_model = XGBClassifier(**params)
        # best_model.fit(dev_data[fea_names], dev_data[self.target])
        log.info('构建基础模型')

        if is_feature_select:
            log.info('需要进行变量筛选')
            fea_names = autobmt.feature_select({'dev': dev_data, 'nodev': nodev_data}, fea_names, self.target,
                                               feature_select_method, method_threhold,
                                               corr_threhold,
                                               psi_threhold)

        if is_auto_tune_params:
            log.info('需要进行自动调参')
            best_model = autobmt.classifiers_model_auto_tune_params(models=['lgb'],
                                                                    train_data=(
                                                                        dev_data[fea_names], dev_data[self.target]),
                                                                    test_data=(
                                                                        nodev_data[fea_names], nodev_data[self.target]))
            params = best_model.get_params()

        if is_stepwise_del_feature:
            log.info('需要逐步的删除变量')
            _, fea_names = autobmt.stepwise_del_feature(best_model, {'dev': dev_data, 'nodev': nodev_data}, fea_names,
                                                        self.target,
                                                        params)

        # 最终模型
        log.info('使用自动调参选出来的最优参数+筛选出来的变量，构建最终模型')
        log.info('最终变量的个数{}, 最终变量{}'.format(len(fea_names), fea_names))
        log.info('自动调参选出来的最优参数{}'.format(params))
        # xgb_clf = XGBClassifier(**params)
        # xgb_clf.fit(dev_data[fea_names], dev_data[self.target])
        best_model.fit(dev_data[fea_names], dev_data[self.target])

        # ###
        # pred_nodev = xgb_clf.predict_proba(nodev_data[fea_names])[:, 1]
        # pred_dev = xgb_clf.predict_proba(dev_data[fea_names])[:, 1]
        # df_pred_nodev = pd.DataFrame({'target': nodev_data[self.target], 'p': pred_nodev}, index=nodev_data.index)
        # df_pred_dev = pd.DataFrame({'target': dev_data[self.target], 'p': pred_dev}, index=dev_data.index)
        # ###

        # ###
        # df_pred_nodev = nodev_data[self.no_feature_names + fea_names]
        # df_pred_dev = dev_data[self.no_feature_names + fea_names]
        # df_pred_nodev['p'] = xgb_clf.predict_proba(df_pred_nodev[fea_names])[:, 1]
        # df_pred_dev['p'] = xgb_clf.predict_proba(df_pred_dev[fea_names])[:, 1]
        # ###

        ###
        df_pred_nodev = nodev_data[self.no_feature_names]
        df_pred_dev = dev_data[self.no_feature_names]
        df_pred_nodev['p'] = best_model.predict_proba(nodev_data[fea_names])[:, 1]
        df_pred_dev['p'] = best_model.predict_proba(dev_data[fea_names])[:, 1]
        ###

        # 计算auc、ks、psi
        test_ks = autobmt.get_ks(df_pred_nodev[self.target], df_pred_nodev['p'])
        train_ks = autobmt.get_ks(df_pred_dev[self.target], df_pred_dev['p'])
        test_auc = autobmt.get_auc(df_pred_nodev[self.target], df_pred_nodev['p'])
        train_auc = autobmt.get_auc(df_pred_dev[self.target], df_pred_dev['p'])

        q_cut_list = np.arange(0, 1, 1 / 20)
        bins = np.append(np.unique(np.quantile(df_pred_nodev['p'], q_cut_list)), df_pred_nodev['p'].max() + 0.1)
        df_pred_nodev['range'] = pd.cut(df_pred_nodev['p'], bins=bins, precision=0, right=False).astype(str)
        df_pred_dev['range'] = pd.cut(df_pred_dev['p'], bins=bins, precision=0, right=False).astype(str)
        nodev_psi = autobmt.psi(df_pred_nodev['range'], df_pred_dev['range'])
        res_dict = {'dev_auc': train_auc, 'nodev_auc': test_auc, 'dev_ks': train_ks, 'nodev_ks': test_ks,
                    'nodev_dev_psi': nodev_psi}
        log.info('auc & ks & psi: {}'.format(res_dict))
        log.info('*' * 30 + '自动构建模型完成！！！' + '*' * 30)

        ##############
        log.info('*' * 30 + '建模相关结果开始保存！！！' + '*' * 30)
        joblib.dump(best_model._Booster, os.path.join(self.ml_res_save_path, 'lgb.ml'))
        joblib.dump(best_model, os.path.join(self.ml_res_save_path, 'lgb_sk.ml'))
        autobmt.dump_to_pkl(best_model._Booster, os.path.join(self.ml_res_save_path, 'lgb.pkl'))
        autobmt.dump_to_pkl(best_model, os.path.join(self.ml_res_save_path, 'lgb_sk.pkl'))
        json.dump(best_model.get_params(), open(os.path.join(self.ml_res_save_path, 'lgb.params'), 'w'))
        best_model._Booster.save_model(os.path.join(self.ml_res_save_path, 'lgb.txt'))
        json.dump(best_model._Booster.dump_model(), open(os.path.join(self.ml_res_save_path, 'lgb.json'), 'w'))
        pd.DataFrame([res_dict]).to_csv(os.path.join(self.ml_res_save_path, 'lgb_auc_ks_psi.csv'), index=False)

        pd.DataFrame(list(tuple(zip(best_model._Booster.feature_name(), best_model._Booster.feature_importance()))),
                     columns=['fea_names', 'weight']
                     ).sort_values('weight', ascending=False).set_index('fea_names').to_csv(
            os.path.join(self.ml_res_save_path, 'lgb_featureimportance.csv'))

        nodev_data[self.no_feature_names + fea_names].head(500).to_csv(
            os.path.join(self.ml_res_save_path, 'lgb_test_input.csv'),
            index=False)

        ##############pred to score
        df_pred_nodev['score'] = df_pred_nodev['p'].map(
            lambda x: autobmt.to_score(x, self.AB['A'], self.AB['B'], self.positive_corr))
        df_pred_dev['score'] = df_pred_dev['p'].map(
            lambda x: autobmt.to_score(x, self.AB['A'], self.AB['B'], self.positive_corr))
        ##############pred to score

        df_pred_nodev.append(df_pred_dev).to_csv(os.path.join(self.ml_res_save_path, 'lgb_pred_to_report_data.csv'),
                                                 index=False)

        log.info('*' * 30 + '建模相关结果保存完成！！！保存路径为：{}'.format(self.ml_res_save_path) + '*' * 30)

        return best_model, fea_names

    @classmethod
    def predict(cls, to_pred_df=None, model_path=None):
        if to_pred_df is None:
            raise ValueError('需要进行预测的数据集不能为None，请指定数据集！！！')
        if model_path is None:
            raise ValueError('模型路径不能为None，请指定模型文件路径！！！')

        try:
            model = joblib.load(os.path.join(model_path, 'lgb.ml'))
        except:
            model = pickle.load(open(os.path.join(model_path, 'lgb.pkl'), 'rb'))

        try:
            model_feature_names = model.feature_name()
        except:
            model_feature_names = model._Booster.feature_name()
            model = model._Booster

        to_pred_df['p'] = model.predict(to_pred_df[model_feature_names])

        return to_pred_df



#==============================================================================
# File: bayes_opt_tuner.py
#==============================================================================

#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: bayes_opt_tuner.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-04-25
'''

import time
import warnings

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .metrics import get_ks, get_auc
from .utils import get_accuracy, get_recall, get_precision, get_f1, r2

warnings.filterwarnings('ignore')

from .logger_utils import Logger

log = Logger(level='info', name=__name__).logger


class ModelTune():
    def __init__(self):
        self.base_model = None
        self.best_model = None
        self.tune_params = None
        self.loss = np.inf
        self.default_params = None
        self.int_params = None
        self.init_params = None
        self.metrics = None
        self._metrics_score = []
        self.scores = []

    def get_model(self):
        return self.best_model

    def _map_metrics(self):
        mapper = {
            'accuracy': get_accuracy,
            'f1': get_f1,
            'precision': get_precision,
            'recall': get_recall,
            'r2': r2,
            'auc': get_auc,
            'ks': get_ks
        }

        for metric in self.metrics:
            if metric not in mapper:
                raise ValueError('指定的指标 ''`{}` 不支持'.format(metric))
            self._metrics_score.append(mapper[metric])
            self.scores.append(0.)

    def fit(self, train_data=(), test_data=()
            , init_points=30, iterations=120, metrics=[]):
        '''

        Args:
            train_data:
            test_data:
            init_points:
            iterations:
            metrics:

        Returns:

        '''

        if len(metrics) > 0:
            self.metrics = metrics
        self._map_metrics()

        X_train, y_train = train_data
        X_test, y_test = test_data

        def loss_fun(train_result, test_result, weight=0.3):

            # return test_result - 2 ** abs(test_result - train_result)
            return test_result - 2 ** abs(test_result - train_result) * weight

        # def loss_fun(train_result, test_result):
        #     train_result = train_result * 100
        #     test_result = test_result * 100
        #
        #     return train_result - 2 ** abs(train_result - test_result)

        def obj_fun(**params):
            for param in self.int_params:
                params[param] = int(round(params[param]))

            model = self.base_model(**params, **self.default_params)
            model.fit(X_train, y_train)

            pred_test = model.predict_proba(X_test)[:, 1]
            pred_train = model.predict_proba(X_train)[:, 1]

            # test_auc = get_auc(y_test, pred_test)
            # train_auc = get_auc(y_train, pred_train)
            # print('test_auc is : ', test_auc)
            # print('train_auc is : ', train_auc)

            test_ks = get_ks(y_test, pred_test)
            train_ks = get_ks(y_train, pred_train)
            # print('test_ks is : ', test_ks)
            # print('train_ks is : ', train_ks)

            # maximize = loss_fun(train_auc, test_auc)
            maximize = loss_fun(train_ks, test_ks)
            # print('max_result is : ', maximize)
            # max_result = loss_fun(train_ks, test_ks) * 2 + loss_fun(train_auc, test_auc)

            loss = -maximize
            if loss < self.loss:
                self.loss = loss
                self.best_model = model
                # print('best model result is {}'.format(loss))
                # print('best model params is : ')
                # print(self.best_model.get_params())
                for i, _metric in enumerate(self._metrics_score):
                    self.scores[i] = _metric(y_test, pred_test)
            # print('current obj_fun result is : ', maximize)

            return maximize

        params_optimizer = BayesianOptimization(obj_fun, self.tune_params, random_state=1)
        log.info('需要优化的超参数是 : {}'.format(params_optimizer.space.keys))

        log.info('开始优化超参数!!!')
        start = time.time()

        params_optimizer.maximize(init_points=1, n_iter=0, acq='ei',
                                  xi=0.0)
        params_optimizer.probe(self.init_params, lazy=True)

        # params_optimizer.probe(self.init_params, lazy=True)

        params_optimizer.maximize(init_points=0, n_iter=0)

        params_optimizer.maximize(init_points=init_points, n_iter=iterations, acq='ei',
                                  xi=0.0)  # init_points：探索开始探索之前的迭代次数；iterations：方法试图找到最大值的迭代次数
        # params_optimizer.maximize(init_points=init_points, n_iter=iterations, acq='ucb', xi=0.0, alpha=1e-6)
        end = time.time()
        log.info('优化参数结束!!! 共耗时{} 分钟'.format((end - start) / 60))
        log.info('最优参数是 : {}'.format(params_optimizer.max['params']))
        log.info('{} model 最大化的结果 : {}'.format(type(self.best_model), params_optimizer.max['target']))


class ClassifierModel(ModelTune):
    def __init__(self):
        super().__init__()
        self.metrics = ['auc', 'ks']


class RegressorModel(ModelTune):
    def __init__(self):
        super().__init__()
        self.metrics = ['r2', 'rmse']


class XGBClassifierTuner(ClassifierModel):
    def __init__(self):
        super().__init__()

        self.base_model = XGBClassifier
        self.tune_params = {
            'learning_rate': (0.01, 0.15),
            'n_estimators': (90, 300),
            'max_depth': (2, 7),
            'min_child_weight': (1, 300),
            'subsample': (0.4, 1.0),
            'colsample_bytree': (0.3, 1.0),
            'colsample_bylevel': (0.5, 1.0),
            'gamma': (0, 20.0),
            'reg_alpha': (0, 20.0),
            'reg_lambda': (0, 20.0),
            # 'scale_pos_weight': (1, 5),
            # 'max_delta_step': (0, 10)
        }

        self.default_params = {
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'nthread': -1
        }

        self.init_params = {
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': 3,
            'min_child_weight': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.9,
            'colsample_bylevel': 0.7,
            'gamma': 7,
            'reg_alpha': 10,
            'reg_lambda': 10,
            # 'scale_pos_weight': 1
        }

        self.int_params = ['max_depth', 'n_estimators']


class LGBClassifierTuner(ClassifierModel):
    '''
    英文版：https://lightgbm.readthedocs.io/en/latest/Parameters.html

    中文版：https://lightgbm.apachecn.org/#/docs/6

    其他注解：https://medium.com/@gabrieltsen
    '''

    def __init__(self):
        super().__init__()

        self.base_model = LGBMClassifier
        self.tune_params = {
            'max_depth': (3, 15),
            'num_leaves': (16, 128),
            'learning_rate': (0.01, 0.2),
            'reg_alpha': (0, 100),
            'reg_lambda': (0, 100),
            'min_child_samples': (1, 100),
            'min_child_weight': (0.01, 100),
            'colsample_bytree': (0.5, 1),
            'subsample': (0.5, 1),
            'subsample_freq': (2, 10),
            'n_estimators': (90, 500),

        }

        self.default_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'random_state': 1024,
            'n_jobs': -1,
            'num_threads': -1,
            'verbose': -1,

        }

        self.init_params = {
            'max_depth': -1,
            "num_leaves": 31,
            "learning_rate": 0.02,
            "reg_alpha": 0.85,
            "reg_lambda": 3,
            "min_child_samples": 20,  # TODO注意修改 .......sklearn:min_child_samples    原生:min_data、min_data_in_leaf
            "min_child_weight": 0.05,  # sklearn:min_child_weight    原生:min_hessian、min_sum_hessian_in_leaf
            "colsample_bytree": 0.9,  # sklearn:colsample_bytree   原生:feature_fraction
            "subsample": 0.8,  # sklearn:subsample  原生:bagging_fraction
            "subsample_freq": 2,  # sklearn:subsample_freq  原生:bagging_freq
            "n_estimators": 100  # sklearn:n_estimators  原生:num_boost_round、num_iterations
        }

        self.int_params = ['max_depth', 'num_leaves', 'min_child_samples', 'n_estimators', 'subsample_freq']


classifiers_dic = {
    # 'lr': LogisticRegressionTuner,
    # 'rf': RandomForestClassifierTuner,
    'xgb': XGBClassifierTuner,
    'lgb': LGBClassifierTuner
}


def classifiers_model_auto_tune_params(models=['xgb'], metrics=[], train_data=(), test_data=()
                                       , init_points=30, iterations=120, verbose=1):
    '''

    Args:
        models:
        metrics:
        train_data:
        test_data:
        init_points:
        iterations:
        verbose:

    Returns:

    '''
    best_model = None
    if not isinstance(models, list):
        raise AttributeError('models参数必须是一个列表, ', '但实际是 {}'.format(type(models)))
    if len(models) == 0:
        models = list(classifiers_dic.keys())
    classifiers = []
    for model in models:
        if model in classifiers_dic:
            classifiers.append(classifiers_dic[model])
    loss = np.inf
    _model = None
    for classifier in classifiers:
        if verbose:
            log.info("优化 {}...".format(classifier()))
        _model = classifier()
        _model.fit(train_data=train_data,
                   test_data=test_data
                   , init_points=init_points, iterations=iterations, metrics=metrics)

        _loss = _model.loss
        if verbose:
            _show_fit_log(_model)
        if _loss < loss:
            loss = _loss
            best_model = _model

    return best_model.get_model()


def _show_fit_log(model):
    _out = '  最优结果: '
    _out += ' loss: {:.3}'.format(model.loss)
    _out += ' 测试集 '
    for i, _metric in enumerate(model.metrics):
        _out += ' {}: {:.3}'.format(_metric[:3],
                                    model.scores[i])
    log.info(_out)


if __name__ == '__main__':
    X = pd.read_pickle('X_train.pkl')
    X = pd.DataFrame(X)
    y = pd.read_pickle('y_train.pkl')
    y = pd.Series(y)
    X_test = pd.read_pickle('X_test.pkl')
    X_test = pd.DataFrame(X_test)
    y_test = pd.read_pickle('y_test.pkl')
    y_test = pd.Series(y_test)

    ####build model
    best_model = classifiers_model_auto_tune_params(train_data=(X, y), test_data=(X_test, y_test), verbose=1,
                                                    init_points=1,
                                                    iterations=2)
    # best_model = classifiers_model_auto_tune_params(train_data=(X, y), test_data=(X_test, y_test), verbose=1)
    print('classifiers_model run over!!!')
    print(type(best_model))
    print(best_model.get_params())
    train_pred_y = best_model.predict_proba(X)[:, 1]
    test_pred_y = best_model.predict_proba(X_test)[:, 1]
    ####build model

    #####build model
    # best_model = LGBMClassifier()
    # best_model.fit(X,y)
    # print(best_model.get_params())
    # train_pred_y = best_model.predict_proba(X)[:, 1]
    # test_pred_y = best_model.predict_proba(X_test)[:, 1]
    #####build model

    # #####build model
    # import lightgbm as lgb
    #
    # init_params = {
    #     "boosting_type": "gbdt",
    #     "objective": "binary",
    #     "metric": "auc",
    # }
    # best_model = lgb.train(params=init_params, train_set=lgb.Dataset(X, y), valid_sets=lgb.Dataset(X_test, y_test))
    # best_model.save_model('lgb.txt')
    # json_model = best_model.dump_model()
    # import json
    #
    # with open('lgb.json', 'w') as f:
    #     json.dump(json_model, f)
    # train_pred_y = best_model.predict(X)
    # test_pred_y = best_model.predict(X_test)
    # #####build model

    train_auc = get_auc(y, train_pred_y)
    test_auc = get_auc(y_test, test_pred_y)
    train_ks = get_ks(y, train_pred_y)
    test_ks = get_ks(y_test, test_pred_y)
    print('train_auc is : ', train_auc, 'test_auc is : ', test_auc)
    print('train_ks is : ', train_ks, 'test_ks is : ', test_ks)

    # # #####build model
    # params = {
    #     'learning_rate': 0.05,
    #     'n_estimators': 200,
    #     'max_depth': 3,
    #     'min_child_weight': 5,
    #     'gamma': 7,
    #     'subsample': 0.7,
    #     'colsample_bytree': 0.9,
    #     'colsample_bylevel': 0.7,
    #     'reg_alpha': 10,
    #     'reg_lambda': 10,
    #     'scale_pos_weight': 1
    # }
    #
    # clf = XGBClassifier(**params)
    # clf.fit(X, y)
    # estimator = clf.get_booster()
    # temp = estimator.save_raw()[4:]
    # # #####build model

    ####构建数据
    # from sklearn.datasets import make_classification
    # from sklearn.model_selection import train_test_split
    # import pickle
    # X, y = make_classification(n_samples=10000, random_state=1024)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #
    # with open('X_train.pkl', 'wb') as f:
    #     f.write(pickle.dumps(X_train))
    # with open('y_train.pkl', 'wb') as f:
    #     f.write(pickle.dumps(y_train))
    # with open('X_test.pkl', 'wb') as f:
    #     f.write(pickle.dumps(X_test))
    # with open('y_test.pkl', 'wb') as f:
    #     f.write(pickle.dumps(y_test))



#==============================================================================
# File: B卡建模-Lr-174渠道 (1).py
#==============================================================================

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import toad
import os 
from datetime import datetime
from sklearn.model_selection import train_test_split
from IPython.core.interactiveshell import InteractiveShell
import warnings
import gc
from statistics import mode
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, cross_validate, cross_val_score
import time
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from pypmml import Model

warnings.filterwarnings("ignore")
InteractiveShell.ast_node_interactivity = 'all'
pd.set_option('display.max_row',None)
pd.set_option('display.width',1000)


# In[2]:


os.getcwd()


# In[3]:


# 运行函数脚本
get_ipython().run_line_magic('run', 'function.ipynb')


# # 1.读取数据集

# In[4]:


filepath = r'D:\liuyedao\B卡开发\三方数据匹配'
data = pd.read_csv(filepath + r'\order_20230927.csv') 


# In[53]:


df_base = data.query("channel_id==174 & apply_date<'2023-07-01' & apply_date>='2022-10-01'")
# 删除全部是空值的列 
df_base.dropna(how='all', axis=1, inplace=True)
df_base.dropna(how='all', axis=1, inplace=True)
df_base.reset_index(drop=True, inplace=True)
df_base.info()
df_base.head()


# In[54]:


to_drop = list(df_base.select_dtypes(include='object').columns)
print(to_drop)


# In[55]:


df_base.drop(['operationType', 'swift_number', 'name', 'mobileEncrypt', 'orderNo', 'idCardEncrypt'],
             axis=1,inplace=True)


# In[56]:


# 小于0为异常值，转为空值
for col in df_base.select_dtypes(include='float64').columns[1:]:
    df_base[col] = df_base[col].mask(df_base[col]<0)


# In[57]:


df_behavior = pd.read_csv(r'D:\liuyedao\B卡开发\lr_xgb_174\b卡衍生变量_20231013.csv')


# In[58]:


df_base = pd.merge(df_base, df_behavior, how='left', on='user_id')
print(df_base.shape, df_base.order_no.nunique())


# In[59]:


df_pudao_3 = pd.read_csv(r'D:\liuyedao\B卡开发\lr_xgb_174\order_pudao_3_diff_20231013.csv')
df_bairong_1 = pd.read_csv(r'D:\liuyedao\B卡开发\lr_xgb_174\order_bairong_1_diff_20231013.csv')


# In[60]:


df_base = pd.merge(df_base, df_pudao_3, how='left', on='order_no')
print(df_base.shape, df_base.order_no.nunique())


# In[61]:


df_base = pd.merge(df_base, df_bairong_1, how='left', on='order_no')
print(df_base.shape, df_base.order_no.nunique())


# In[62]:


df_model = df_base.query("apply_date>='2022-10-01' & apply_date<'2023-06-01' & target in [0.0, 1.0]")


# In[63]:


path = r'D:\liuyedao\B卡开发\lr_result_174\\'


# In[64]:


tmp_df_model = toad.detect(df_model)
tmp_df_model.to_excel(path + 'df_explore_20231023.xlsx')


# ## 训练数据集

# In[65]:


df_train = df_base.query("apply_date>='2022-10-01' & apply_date<'2023-05-01' & target in [0.0, 1.0]")
# df_train.groupby(['lending_month','target'])['order_no'].count().unstack()


# In[66]:


print(df_train.target.value_counts())
print(df_train.target.value_counts(normalize=True))


# ## oot测试数据集

# In[67]:


df_oot = df_base.query("apply_date>='2023-05-01' & apply_date<'2023-06-01' & target in [0.0, 1.0]")
print(df_oot.target.value_counts())
print(df_oot.target.value_counts(normalize=True))


# In[68]:


print(df_train.shape, df_oot.shape)


# # 2.数据处理

# In[69]:


train = df_train.copy()
oot = df_oot.copy()

# oot = df_train.copy()
# train = df_oot.copy()


# In[70]:


to_drop = train.columns[0:6].to_list()
print(to_drop)


# In[71]:


train = train.drop(to_drop, axis=1)
oot = oot.drop(to_drop, axis=1)


# In[72]:


to_drop = list(train.select_dtypes(include='object').columns)
print(to_drop)


# In[ ]:


# train = train.drop(to_drop, axis=1)
# oot = oot.drop(to_drop, axis=1)


# In[73]:


train.info()


# In[74]:


oot.info()


# In[75]:


print(train.target.value_counts())
print(oot.target.value_counts())


# In[76]:


print(train.target.value_counts(normalize=True))
print(oot.target.value_counts(normalize=True))


# # 3.数据探索分析

# In[ ]:


# # 小于0为异常值，转为空值
# train = train.mask(train<0)
# oot = oot.mask(oot<0)


# In[77]:


# 删除全部是空值的列
train.dropna(how='all', axis=1, inplace=True)
oot.dropna(how='all', axis=1, inplace=True)


# In[78]:


train_explore = toad.detect(train.drop('target', axis=1))
oot_explore = toad.detect(oot.drop('target', axis=1))

modes = train.drop('target', axis=1).apply(mode)
mode_counts = train.drop('target', axis=1).eq(modes).sum()
train_modes = pd.DataFrame({'mode':modes, 'mode_num':mode_counts}, index=list(train.columns[1:]))

modes = oot.drop('target', axis=1).apply(mode)
mode_counts = oot.drop('target', axis=1).eq(modes).sum()
oot_modes = pd.DataFrame({'mode':modes, 'mode_num':mode_counts}, index=list(oot.columns[1:]))

train_isna = pd.DataFrame(train.drop('target', axis=1).isnull().sum(), columns=['missing_num'])
oot_isna = pd.DataFrame(oot.drop('target', axis=1).isnull().sum(), columns=['missing_num'])


train_iv = toad.quality(train,'target',iv_only=False)
oot_iv = toad.quality(oot,'target',iv_only=False)


# In[79]:


train_df_explore = pd.concat([train_explore, train_modes, train_isna, train_iv.drop('unique',axis=1)], axis=1)

train_df_explore['no_null_num'] = train_df_explore['size'] - train_df_explore['missing_num']
train_df_explore['miss_rate'] = train_df_explore['missing_num'] / train_df_explore['size']
train_df_explore['mode_pct_all'] = train_df_explore['mode_num']/train_df_explore['size']
train_df_explore['mode_pct_notna'] = train_df_explore['mode_num']/train_df_explore['no_null_num']


# In[80]:


oot_df_explore = pd.concat([oot_explore, oot_modes, oot_isna, oot_iv.drop('unique',axis=1)], axis=1)

oot_df_explore['no_null_num'] = oot_df_explore['size'] - oot_df_explore['missing_num']
oot_df_explore['miss_rate'] = oot_df_explore['missing_num'] / oot_df_explore['size']
oot_df_explore['mode_pct_all'] = oot_df_explore['mode_num']/oot_df_explore['size']
oot_df_explore['mode_pct_notna'] = oot_df_explore['mode_num']/oot_df_explore['no_null_num']


# In[81]:


df_iv = pd.merge(train_iv, oot_iv, how='inner',left_index=True, right_index=True,suffixes=['_train','_oot'])
df_iv['diff_iv'] = df_iv['iv_oot']-df_iv['iv_train']
df_iv['rate_iv'] = df_iv['iv_oot']/df_iv['iv_train'] - 1


# In[82]:


writer=pd.ExcelWriter(path + 'B卡_探索性分析_58同城_20231023.xlsx')

train_df_explore.to_excel(writer,sheet_name='train_df_explore')
oot_df_explore.to_excel(writer,sheet_name='oot_df_explore')

# train_df_iv.to_excel(writer,sheet_name='train_df_iv')
# oot_df_iv.to_excel(writer,sheet_name='oot_df_iv')

df_iv.to_excel(writer,sheet_name='df_iv')

writer.save()


# # 4.特征粗筛选

# In[83]:


# 删除缺失率大于0.85/删除枚举值只有一个/删除方差等于0/删除集中度大于0.85

to_drop_missing = list(train_df_explore[train_df_explore.miss_rate>=0.85].index)
print(len(to_drop_missing))
to_drop_unique = list(train_df_explore[train_df_explore.unique==1].index)
print(len(to_drop_unique))
to_drop_std = list(train_df_explore[train_df_explore.std_or_top2==0].index)
print(len(to_drop_std))
to_drop_mode = list(train_df_explore[train_df_explore.mode_pct_notna>=0.85].index)
print(len(to_drop_mode))
to_drop_mode2 = list(train_df_explore[train_df_explore.mode_pct_all>=0.85].index)
print(len(to_drop_mode2))

to_drop_train = list(set(to_drop_missing+to_drop_unique+to_drop_std+to_drop_mode+to_drop_mode2))
print(len(to_drop_train))


# In[84]:


# 删除缺失率大于0.85/删除枚举值只有一个/删除方差等于0/删除集中度大于0.85
to_drop_missing = list(oot_df_explore[oot_df_explore.miss_rate>=0.85].index)
print(len(to_drop_missing))
to_drop_unique = list(oot_df_explore[oot_df_explore.unique==1].index)
print(len(to_drop_unique))
to_drop_std = list(oot_df_explore[oot_df_explore.std_or_top2==0].index)
print(len(to_drop_std))
to_drop_mode = list(oot_df_explore[oot_df_explore.mode_pct_notna>=0.85].index)
print(len(to_drop_mode))
to_drop_mode2 = list(oot_df_explore[oot_df_explore.mode_pct_all>=0.85].index)
print(len(to_drop_mode))

to_drop_oot = list(set(to_drop_missing+to_drop_unique+to_drop_std+to_drop_mode+to_drop_mode2))
print(len(to_drop_oot))


# In[85]:


train_1 = train.drop(to_drop_train, axis=1)
print(train_1.shape)

oot_1 = oot.drop(to_drop_oot, axis=1)
print(oot_1.shape)


# In[197]:


# 共同的变量
sim_cols = list(set(train_1.drop('target',axis=1).columns).intersection(set(oot_1.drop('target',axis=1).columns)))
print(len(sim_cols))

train_2 = train_1[['target']+sim_cols]
oot_2 = oot_1[['target']+sim_cols]
print(train_2.shape, oot_2.shape)


# In[198]:


# psi/iv稳定性筛选
to_drop_iv = list(df_iv.query("iv_train<=0.02").index)
print(len(to_drop_iv))

# psi = toad.metrics.PSI(train_2.drop('target', axis=1), oot_2.drop('target',axis=1))
# to_drop_psi = list(psi[psi>=0.25].index)
# print(len(to_drop_psi))

to_drop = []
for col in list(set(to_drop_iv)):
    if col in train_2.columns:
        to_drop.append(col)
print(len(to_drop))

train_2.drop(to_drop, axis=1, inplace=True)
print(train_2.shape)


# In[ ]:





# In[199]:


# iv值/相关性筛选
train_selected, dropped = toad.selection.select(train_2, target='target', empty=0.85, iv=0.02,
                                                corr=1.0,
                                                return_drop=True, exclude=None)
train_selected.shape


# In[200]:


for i in dropped.keys():
    print("变量筛选维度：{}， 共剔除变量{}".format(i, len(dropped[i])))


# In[201]:


# 人工剔除
to_drop_man = ['model_score_01_baihang_1','model_score_01_q_tianchuang_1','model_score_01_r_tianchuang_1',
           'model_score_01_1_bileizhen_1',
           'model_score_01_2_bileizhen_1','model_score_01_3_bileizhen_1','value_011_moxingfen_7',
           'value_012_pudao_5',
           'model_score_01_pudao_11','model_score_01_pudao_12','model_score_01_pudao_16',
           'model_score_01_2_bairong_8','model_score_01_3_bairong_8','model_score_01_7_bairong_8',
           'model_score_01_8_bairong_8']

to_drop = []
for col in to_drop_man:
    if col in train_selected.columns:
        to_drop.append(col)
print(len(to_drop))

train_selected.drop(to_drop, axis=1, inplace=True)
print(train_selected.shape)


# ## 变量分箱

# In[181]:


# cols = [
#  'als_m12_cell_nbank_ca_orgnum_diff'
# ,'model_score_01_rong360_4'
# ,'model_score_01_moxingfen_7'
# ,'value_054_bairong_1'
# ,'value_060_baihang_6'
# ,'als_d15_id_nbank_nsloan_orgnum'
# ,'model_score_01_ruizhi_6'
# ,'als_fst_id_nbank_inteday'
# ,'value_026_bairong_12'
# ]

# train_selected = train[['target']+cols]


# In[202]:


# 第一次分箱
c = toad.transform.Combiner()
c.fit(train_selected, y='target', method='chi', min_samples=0.05, n_bins=10, empty_separate=True) 
bins_result = c.export()


# In[184]:


adj_bins = {
    'als_m12_cell_nbank_ca_orgnum_diff': [-88888, 1.0, 2.0],
    'model_score_01_rong360_4': [-88888, 0.055878434330225, 0.074964128434658, 0.10986643284559],
    'model_score_01_tianchuang_7': [-88888, 555.0, 595.0],
    'als_m3_cell_bank_max_inteday_diff': [-0.5, 46.0],
    'model_score_01_moxingfen_7': [-88888, 530.0],
    'value_054_bairong_1': [-88888, 3.0, 5.0],
    'value_060_baihang_6': [-88888, 2.0, 3.0, 4.0],
    'als_d15_id_nbank_nsloan_orgnum': [-88888, 3.0, 4.0],
    'model_score_01_ruizhi_6': [-88888, 670.0, 774.0, 842.0],
    'als_fst_id_nbank_inteday': [-88888, 343.0, 357.0],
    'value_026_bairong_12': [-88888, 34.1395]
}

# 更新分箱
c.update(adj_bins) 


# In[216]:


train_selected_bins = c.transform(train_selected.fillna(-99999), labels=True)
oot_selected_bins = c.transform(oot[train_selected.columns].fillna(-99999), labels=True)


# In[204]:


train_selected_bins = c.transform(train_selected, labels=True)
oot_selected_bins = c.transform(oot[train_selected.columns], labels=True)


# In[217]:


print(train_selected_bins.shape,oot_selected_bins.shape)


# In[218]:


bins_dict_train = {}
for col in train_selected_bins.columns[1:]:
    bins_dict_train[col] = regroup(train_selected_bins, col, target='target')
    
df_result_train = pd.concat(list(bins_dict_train.values()), axis=0, ignore_index =True)


# In[219]:


bins_dict_oot = {}
for col in oot_selected_bins.columns[1:]:
    bins_dict_oot[col] = regroup(oot_selected_bins, col, target='target')
    
df_result_oot = pd.concat(list(bins_dict_oot.values()), axis=0, ignore_index =True)


# In[220]:


df_result = pd.merge(df_result_train, df_result_oot, how='left', on=['varsname','bins'],
                     suffixes=['_train','_oot'])


# In[221]:


df_result.to_excel(path + 'B卡_chi_58同城_20231023_all_单调.xlsx',index=False)


# In[ ]:


to_drop_bins1 = []
for col in list(bins_result.keys()):
    if len(bins_result[col])==1:
        to_drop_bins1.append(col)
print(len(to_drop_bins1))
to_drop_bins2 = list(set(df_result[(df_result['iv_train']<0.02)]['varsname']))
print(len(to_drop_bins2))
to_drop_bins = list(set(to_drop_bins1 + to_drop_bins2))
print(len(to_drop_bins))


# In[ ]:


train_selected_1 = train_selected.drop(to_drop_bins, axis=1)
train_selected_1.shape


# In[ ]:


train_selected_bins_1 = train_selected_bins[train_selected_1.columns]
oot_selected_bins_1 = oot_selected_bins[train_selected_bins_1.columns]


# In[ ]:


from toad.plot import bin_plot, badrate_plot


# In[ ]:


# col = list(train_selected_bins_1.drop('target',axis=1).columns)[1]

# bin_plot(train_selected_bins_1, x=col, target='target')
# bin_plot(oot_selected_bins_1, x=col, target='target')


# In[209]:


# 连续变量分箱主函数
def ContinueVarBins(data, col, flag='target', cutbins=[]):
    df = data[[flag, col]]
    df = df[~df[col].isnull()].reset_index(drop=True)
    df['bins'] = pd.cut(df[col], cutbins, duplicates='drop', right=False, precision=4)
    regroup = pd.DataFrame()
    regroup['bins'] = df.groupby(['bins'])[col].max()
    regroup['total'] = df.groupby(['bins'])[flag].count()
    regroup['bad'] = df.groupby(['bins'])[flag].sum()
    regroup['good'] = regroup['total'] - regroup['bad']
    regroup.drop(['total'], axis=1, inplace=True)
    np_regroup = np.array(regroup)
    np_regroup = MergeZero(np_regroup)
    np_regroup = MinPct(np_regroup)
    np_regroup = MonTone(np_regroup)
    regroup = pd.DataFrame(index=np.arange(np_regroup.shape[0]))
    regroup['bins'] = np_regroup[:,0]
    regroup['total'] = np_regroup[:,1] + np_regroup[:,2]
    regroup['bad'] = np_regroup[:,1]
    regroup['good'] = np_regroup[:,2]
    cutoffpoints = list(np_regroup[:,0])
    cutoffpoints = [float('-inf')] + cutoffpoints
    # 最大值分割点，转换最小值分割点
    df['bins_new'] = pd.cut(df[col], cutoffpoints, duplicates='drop', right=True, precision=4)
    tmp = pd.DataFrame()
    tmp['bins'] = df.groupby(['bins_new'])[col].min()
    cutoffpoints = list(tmp['bins'])  
    
    return cutoffpoints


# In[213]:


# 自动调整分箱
adj_bins = {}
for col in list(train_selected.columns)[1:]:
    tmp = np.isnan(bins_result[col])
    cutbins = [bins_result[col][x] for x in range(len(tmp)) if not tmp[x]]
    if len(cutbins)>0:
        cutbins = [float('-inf')] + cutbins + [float('inf')]
        cutoffpoints = ContinueVarBins(train_selected, col, flag='target', cutbins=cutbins)
        tmp_value1 = train_selected[col].min()
        tmp_value2 = cutoffpoints[0]
        if tmp_value1==tmp_value2:
            adj_bins[col] = [-88888] + cutoffpoints[1:]
#             print("分割点有最小值", col)
        else:
            adj_bins[col] = [-88888] + cutoffpoints
    else:
        print("无分割点", col)


# In[214]:


# 更新分箱
c.update(adj_bins)


# In[146]:


# 调整分箱:空值单独一箱
train_selected.fillna(-99999, inplace=True)


# In[215]:


adj_bins


# In[137]:


train_selected.head()


# In[ ]:


# c.export()
# c.load(dict)
# c.transform(dataframe, labels=False)


# ## WOE转换

# In[222]:


transer = toad.transform.WOETransformer()
train_woe = transer.fit_transform(c.transform(train_selected.fillna(-99999)), train_selected['target'],
                                  exclude=['target'])
train_woe.shape


# In[223]:


train_selected_woe, dropped = toad.selection.select(train_woe, target='target', empty=0.85,
                                                    iv=0.02, corr=0.7, return_drop=True, exclude=None)
train_selected_woe.shape


# In[224]:


for i in dropped.keys():
    print("变量筛选维度：{}， 共剔除变量{}".format(i, len(dropped[i])))


# In[225]:


df_result_finally = pd.merge(df_result, pd.DataFrame({'varsname':list(train_selected_woe.columns)}),
                             how='inner', on='varsname')
df_result_finally.to_excel(path + 'B卡_chi_58同城_20231023_all_单调_iv_corr_筛选.xlsx',index=False)


# In[160]:


cols = [
 'model_score_01_3_xinyongsuanli_1'
,'als_m1_cell_nbank_cons_allnum'
,'value_002_moxingfen_7'
,'als_m12_cell_nbank_selfnum_diff'
,'als_m6_id_coon_orgnum_diff'
,'model_score_01_tianchuang_7'
,'als_d15_cell_nbank_cons_allnum_diff'
,'model_score_01_tengxun_1'
,'value_087_baihang_6'
,'als_m12_cell_nbank_max_monnum_diff'
,'value_068_baihang_6'
,'ppdi_m3_cell_nbank_loan_orgnum'
,'als_m3_cell_coon_allnum_diff'
,'als_m12_cell_bank_ret_orgnum'
,'model_score_01_moxingfen_7'
,'als_m3_id_nbank_nsloan_orgnum'
,'value_050_bairong_1'
,'als_m12_cell_nbank_week_allnum_diff'
,'als_d15_id_nbank_nsloan_orgnum'
,'value_031_bairong_12'
,'als_m3_cell_nbank_cons_allnum_diff'
,'als_d15_cell_nbank_selfnum'
,'als_m12_id_nbank_ca_allnum_diff'
,'model_score_01_rong360_4'
,'als_m3_id_cooff_allnum_diff'
,'als_m12_cell_nbank_nsloan_allnum_diff'
,'als_m1_cell_nbank_night_allnum_diff'
,'als_m12_id_max_inteday'
,'value_060_baihang_6'
,'als_m3_cell_caon_allnum_diff'
,'model_score_01_ruizhi_6'
,'als_m6_cell_oth_orgnum_diff'
,'value_019_baihang_6'
,'model_score_01_tianchuang_8'
,'model_score_01_ronghuijinke_3'
,'als_m6_cell_nbank_oth_orgnum_diff'
,'als_m3_id_nbank_night_allnum_diff'
,'value_057_baihang_6'
,'als_m12_cell_avg_monnum_diff'
,'model_score_01_v2_duxiaoman_1'
,'als_m12_id_pdl_orgnum_diff'
,'als_m12_id_cooff_allnum_diff'
,'als_m6_id_nbank_ca_orgnum_diff'
]
train_selected_woe = train_selected_woe[['target']+cols]


# In[226]:


oot_woe = transer.transform(c.transform(oot[train_selected_woe.columns].fillna(-99999)))
oot_selected_woe = oot_woe[list(train_selected_woe.columns)]
oot_selected_woe.shape


# In[227]:


psi = toad.metrics.PSI(train_selected_woe.drop('target',axis=1), oot_selected_woe.drop('target',axis=1))
to_drop_psi = list(psi[psi>0.25].index)
print(len(to_drop_psi))


# In[228]:


train_selected_woe_psi = train_selected_woe.drop(to_drop_psi, axis=1)
oot_selected_woe_psi = oot_selected_woe[train_selected_woe_psi.columns]


# In[229]:


df_result_finally = pd.merge(df_result, pd.DataFrame({'varsname':list(train_selected_woe_psi.columns)}),
                             how='inner', on='varsname')
df_result_finally.to_excel(path + 'B卡_chi_58同城_20231023_all_单调_psi_筛选.xlsx',index=False)


# ## 逐步回归

# In[322]:


cols = [
 'model_score_01_3_xinyongsuanli_1'
,'als_m1_cell_nbank_cons_allnum'
,'value_027_bairong_12'
,'als_m12_cell_nbank_selfnum_diff'
,'als_m6_id_coon_orgnum_diff'
,'als_m3_id_coon_allnum_diff'
,'als_m3_id_nbank_else_orgnum_diff'
,'als_m12_id_coon_allnum_diff'
,'als_d15_id_nbank_allnum'
,'als_m1_id_pdl_allnum'
,'als_m12_cell_nbank_else_orgnum_diff'
,'value_058_baihang_6'
,'model_score_01_tianchuang_7'
,'als_m3_id_nbank_week_orgnum_diff'
,'als_m3_cell_nbank_avg_monnum_diff'
,'create_time_diff_x'
,'model_score_01_tengxun_1'
,'als_m6_id_nbank_nsloan_orgnum_diff'
,'als_m3_cell_pdl_allnum'
,'als_m3_id_nbank_max_inteday_diff'
,'value_087_baihang_6'
,'als_m6_cell_nbank_else_orgnum_diff'
,'als_m12_cell_nbank_max_monnum_diff'
,'als_m3_cell_coon_orgnum_diff'
,'als_m1_cell_nbank_oth_allnum_diff'
,'als_m1_id_nbank_oth_allnum'
,'als_m3_cell_nbank_min_monnum_diff'
,'model_score_01_moxingfen_7'
,'ppdi_m3_cell_nbank_fin_orgnum'
,'value_050_bairong_1'
,'als_d15_cell_nbank_cons_orgnum_diff'
,'als_m12_cell_nbank_week_allnum_diff'
,'model_score_01_hangliezhi_1'
,'als_d15_id_nbank_oth_allnum_diff'
,'value_031_bairong_12'
,'als_m6_id_nbank_oth_orgnum_diff'
,'als_d15_cell_nbank_orgnum_diff'
,'als_m6_id_max_inteday'
,'als_m1_cell_nbank_orgnum_diff'
,'als_m3_cell_nbank_cons_allnum_diff'
,'als_m12_id_caon_orgnum_diff'
,'value_025_baihang_6'
,'als_m12_id_avg_monnum_diff'
,'als_m6_cell_avg_monnum_diff'
,'value_021_baihang_6'
,'als_d15_cell_nbank_selfnum'
,'als_m3_id_nbank_cons_orgnum_diff'
,'als_m12_cell_nbank_else_allnum_diff'
,'als_m1_cell_nbank_orgnum'
,'als_m1_cell_nbank_cons_orgnum'
,'als_m6_cell_nbank_avg_monnum_diff'
,'als_m3_id_cooff_allnum_diff'
,'als_m12_cell_nbank_nsloan_allnum_diff'
,'model_score_01_zr_tongdun_2'
,'als_m6_id_nbank_max_monnum_diff'
,'als_m1_cell_nbank_night_allnum_diff'
,'als_m12_id_max_inteday'
,'als_fst_id_nbank_inteday'
,'als_m3_cell_nbank_else_orgnum'
,'als_m12_cell_nbank_oth_orgnum_diff'
,'value_054_baihang_6'
,'als_m12_cell_nbank_ca_allnum_diff'
,'als_m3_cell_caon_allnum_diff'
,'als_m12_cell_nbank_allnum_diff'
,'value_052_baihang_6'
,'model_score_01_ruizhi_6'
,'als_m3_cell_nbank_allnum_diff'
,'als_m6_cell_oth_orgnum_diff'
,'value_005_bairong_12'
,'als_d15_cell_caon_orgnum'
,'als_m12_id_bank_avg_monnum_diff'
,'model_score_01_tianchuang_8'
,'model_score_01_ronghuijinke_3'
,'als_m3_id_avg_monnum_diff'
,'value_044_bairong_1'
,'als_m3_cell_rel_orgnum_diff'
,'als_m3_id_nbank_nsloan_orgnum_diff'
,'als_m6_cell_nbank_oth_orgnum_diff'
,'value_053_baihang_6'
,'als_m3_id_nbank_night_allnum_diff'
,'als_m6_cell_nbank_max_monnum_diff'
,'als_m12_id_max_monnum_diff'
,'als_m3_id_max_monnum_diff'
,'value_081_bairong_1'
,'model_score_01_v2_duxiaoman_1'
,'als_m12_id_pdl_orgnum_diff'
,'als_m12_cell_nbank_oth_allnum_diff'
,'als_m12_id_cooff_allnum_diff'
,'als_m1_cell_nbank_oth_orgnum_diff'
,'als_m6_id_nbank_ca_orgnum_diff'
]


# In[292]:


cols = [
 'als_m1_cell_nbank_cons_allnum'
,'value_027_bairong_12'
,'als_m12_cell_nbank_selfnum_diff'
,'als_m6_id_coon_orgnum_diff'
,'als_m3_id_coon_allnum_diff'
,'als_m3_id_nbank_else_orgnum_diff'
,'als_m12_id_coon_allnum_diff'
,'als_d15_id_nbank_allnum'
,'als_m1_id_pdl_allnum'
,'als_m12_cell_nbank_else_orgnum_diff'
,'value_058_baihang_6'
,'als_m3_id_nbank_week_orgnum_diff'
,'als_m3_cell_nbank_avg_monnum_diff'
,'create_time_diff_x'
,'als_m6_id_nbank_nsloan_orgnum_diff'
,'als_m3_cell_pdl_allnum'
,'als_m3_id_nbank_max_inteday_diff'
,'value_087_baihang_6'
,'als_m6_cell_nbank_else_orgnum_diff'
,'als_m12_cell_nbank_max_monnum_diff'
,'als_m3_cell_coon_orgnum_diff'
,'als_m1_cell_nbank_oth_allnum_diff'
,'als_m1_id_nbank_oth_allnum'
,'als_m3_cell_nbank_min_monnum_diff'
,'model_score_01_moxingfen_7'
,'ppdi_m3_cell_nbank_fin_orgnum'
,'value_050_bairong_1'
,'als_d15_cell_nbank_cons_orgnum_diff'
,'als_m12_cell_nbank_week_allnum_diff'
,'als_d15_id_nbank_oth_allnum_diff'
,'value_031_bairong_12'
,'als_m6_id_nbank_oth_orgnum_diff'
,'als_d15_cell_nbank_orgnum_diff'
,'als_m6_id_max_inteday'
,'als_m1_cell_nbank_orgnum_diff'
,'als_m3_cell_nbank_cons_allnum_diff'
,'als_m12_id_caon_orgnum_diff'
,'value_025_baihang_6'
,'als_m12_id_avg_monnum_diff'
,'als_m6_cell_avg_monnum_diff'
,'value_021_baihang_6'
,'als_d15_cell_nbank_selfnum'
,'als_m3_id_nbank_cons_orgnum_diff'
,'als_m12_cell_nbank_else_allnum_diff'
,'als_m1_cell_nbank_orgnum'
,'als_m1_cell_nbank_cons_orgnum'
,'als_m6_cell_nbank_avg_monnum_diff'
,'als_m3_id_cooff_allnum_diff'
,'als_m12_cell_nbank_nsloan_allnum_diff'
,'als_m6_id_nbank_max_monnum_diff'
,'als_m1_cell_nbank_night_allnum_diff'
,'als_m12_id_max_inteday'
,'als_fst_id_nbank_inteday'
,'als_m3_cell_nbank_else_orgnum'
,'als_m12_cell_nbank_oth_orgnum_diff'
,'value_054_baihang_6'
,'als_m12_cell_nbank_ca_allnum_diff'
,'als_m3_cell_caon_allnum_diff'
,'als_m12_cell_nbank_allnum_diff'
,'value_052_baihang_6'
,'als_m3_cell_nbank_allnum_diff'
,'als_m6_cell_oth_orgnum_diff'
,'value_005_bairong_12'
,'als_d15_cell_caon_orgnum'
,'als_m12_id_bank_avg_monnum_diff'
,'als_m3_id_avg_monnum_diff'
,'value_044_bairong_1'
,'als_m3_cell_rel_orgnum_diff'
,'als_m3_id_nbank_nsloan_orgnum_diff'
,'als_m6_cell_nbank_oth_orgnum_diff'
,'value_053_baihang_6'
,'als_m3_id_nbank_night_allnum_diff'
,'als_m6_cell_nbank_max_monnum_diff'
,'als_m12_id_max_monnum_diff'
,'als_m3_id_max_monnum_diff'
,'value_081_bairong_1'
,'als_m12_id_pdl_orgnum_diff'
,'als_m12_cell_nbank_oth_allnum_diff'
,'als_m12_id_cooff_allnum_diff'
,'als_m1_cell_nbank_oth_orgnum_diff'
,'als_m6_id_nbank_ca_orgnum_diff'
]


# In[323]:


print(len(cols))


# In[324]:


print(train_selected_woe_psi.shape)
final_data = toad.selection.stepwise(train_selected_woe_psi[['target']+cols], target='target',
                                     estimator='ols',
                                     direction='both', criterion='aic', exclude=None,
                                     intercept=False)
print(final_data.shape)


# In[306]:


# final_data = train_woe.copy()
oot_woe = transer.transform(c.transform(oot[list(final_data.columns)].fillna(-99999)))
final_oot = oot_woe[list(final_data.columns)]


# In[307]:


# 初次建模变量
cols = list(final_data.drop(['target'], axis=1).columns)
print(len(cols))
print(cols)


# # 6.Lr模型训练

# ## 6.1模型技术效果

# In[50]:


# 建模变量
# cols = list(final_data.drop(['target'], axis=1).columns)
print(len(cols))
print(cols)


# In[240]:


df_result_rm = pd.merge(df_result, pd.DataFrame({'varsname':cols}),how='inner',on='varsname')
df_result_rm.head()
df_result_rm.to_excel(r'D:\liuyedao\B卡开发\lr_result_174\B卡_chi_10bins_58同城_入模变量_20231025.xlsx',index=False)


# In[315]:


# cols.remove('model_score_01_tianchuang_8') 
# cols.remove('als_m6_cell_nbank_max_monnum_diff')
cols.append('model_score_01_tianchuang_8')


# In[250]:


for i in ['als_m6_cell_nbank_else_orgnum','als_d15_id_nbank_nsloan_orgnum_diff','ppdi_d15_id_orgnum'
            ,'als_m1_id_caon_allnum_diff','als_m6_cell_nbank_oth_allnum']:
    cols.remove(i)


# In[313]:


print(len(cols),cols)


# In[316]:


clf = LR_model(final_data[cols], final_oot[cols], final_data['target'], final_oot['target'])


# In[317]:


opt_best = {'target': 0.6487565005430443,
            'params': {'gamma': 0.75, 'learning_rate': 0.45, 'max_depth': 6.0, 'min_child_weight': 9.0,
                       'n_estimators': 12, 'reg_alpha': 0, 'reg_lambda': 290}
           }


# In[318]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 


# In[319]:


xgb_model.fit(train[cols] ,train['target'])


# In[320]:


# 对训练集进行预测

from sklearn import metrics

y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[321]:


clf = LR_model(final_oot[cols], final_data[cols], final_oot['target'], final_data['target'])


# In[177]:


clf = LR_model(final_data[cols], final_oot[cols], final_data['target'], final_oot['target'])


# In[178]:


clf = LR_model(final_oot[cols], final_data[cols], final_oot['target'], final_data['target'])


# In[179]:


vif=pd.DataFrame()
X = np.matrix(final_data[cols])
vif['features']=cols
vif['VIF_Factor']=[variance_inflation_factor(np.matrix(X),i) for i in range(X.shape[1])]
vif


# In[180]:


corr = final_data[cols].corr()
corr


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_corr_vif_20231018"+'.xlsx')

corr.to_excel(writer,sheet_name='corr')
vif.to_excel(writer,sheet_name='vif')

writer.save()


# In[ ]:


# 训练集
pred_train = clf.predict(sm.add_constant(final_data[cols]))
# KS/AUC
from toad.metrics import KS,AUC

print('train AUC: ', AUC(pred_train, final_data['target']))
print('train KS: ', KS(pred_train, final_data['target']))


# In[ ]:


# 测试集
pred_oot = clf.predict(sm.add_constant(final_oot[cols]))
# KS/AUC
from toad.metrics import KS,AUC

print('-------------oot结果--------------------')

print('test AUC: ', AUC(pred_oot, final_oot['target']))
print('test KS: ', KS(pred_oot, final_oot['target']))



# ## 6.1转换评分

# In[ ]:


card = toad.ScoreCard(combiner=c,
                      transer = transer,
                      base_odds=35,
                      base_score=700,
                      pdo=60,
                      rate=2,
                      C=1e8
                     )


# In[ ]:


card.fit(final_data[cols], final_data['target'])


# In[ ]:


def scorecard_scale(self):
    scorecard_kedu = pd.DataFrame(
    [
        ["base_odds", self.base_odds,"根据业务经验设置的基础比率"],
        ["base_score", self.base_score,"基础odds对应的分数"],
        ["rate", self.rate, "设置分数的倍率"],
        ["pdo", self.pdo, "表示分数增长pdo时，odds值增长rate倍"],
        ["B", self.factor,"补偿值，计算方式：pdo/ln(rate)"],
        ["A", self.offset,"刻度,计算方式：base_score - B * ln(base_odds)"]
    ],
    columns = ["刻度项", "刻度值","备注"]
    )
    
    return scorecard_kedu


# In[ ]:


scorecard_scale(card)


# In[ ]:


score_card = card.export(to_frame=True).round(0)
print(len(set(score_card.name)))
score_card


# In[ ]:


# (card.offset - card.factor * card.coef_[0] * card.intercept_)/9


# In[ ]:


score_card.to_excel(r'D:\liuyedao\B卡开发\lr_xgb_174\B卡_评分卡_58同城_20231018_v1.xlsx')


# In[ ]:


name = {
    'als_m12_cell_nbank_ca_orgnum_diff': '按手机号查询，近12个月在非银机构-现金类分期申请机构数与上次查询的差',
    'model_score_01_rong360_4': '融360评分',
    'model_score_01_tianchuang_7': '天创信用联合定制分',
    'model_score_01_moxingfen_7': '贷中评分',
    'value_054_bairong_1': '按身份证号查询，近15天在非银机构-持牌消费金融机构申请机构数',
    'value_060_baihang_6': '百行_续侦_证件号近3个月内查询机构数-网络小贷类机构',
    'als_d15_id_nbank_nsloan_orgnum': '按身份证号查询，近15天在非银机构-持牌网络小贷机构申请机构数',
    'model_score_01_ruizhi_6': '睿智联合分',
    'als_fst_id_nbank_inteday': '按身份证号查询，距最早在非银行机构申请的间隔天数',
    'value_026_bairong_12': '近6个月用户利率偏好'
}


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_变量箱子_分数_v2.xlsx")
train_selected['score'] = card.predict(train_selected[cols]).round(0)
train_selected_bins = c.transform(train_selected, labels=True)
 

for col in cols:
    tmp1 = CalWoeIv(train_selected_bins, col, target='target')
    tmp2 = train_selected_bins[[col, 'score']].drop_duplicates(col,keep='first')
    tmp = pd.merge(tmp1, tmp2, how='left',left_on='bins', right_on=col)
    tmp['name'] = name[col]
    tmp = tmp[['varsname','name','bins','total','total_pct','bad_rate','woe']]
    tmp.to_excel(writer,sheet_name=col)
    writer.save()


# In[ ]:


print(len(cols), cols)


# In[ ]:


train['score'] = card.predict(train[cols]).round(0)
train['prob'] = card.predict_proba(train[cols])[:,1]


# In[ ]:


oot['score'] = card.predict(oot[cols]).round(0)
oot['prob'] = card.predict_proba(oot[cols])[:,1]


# In[ ]:


# tmp1 = train[['order_no','lending_time','channel_id','score','prob']+cols]
# tmp2 = oot[['order_no','lending_time','channel_id','score','prob']+cols]
# tmp = pd.concat([tmp1, tmp2], axis=0)
# tmp.info()
# tmp.head()
# tmp.to_csv(r'D:\liuyedao\B卡开发\Lr_result\score_lr_部署测试样例_58同城.csv',index=False)


# ## 6.2模型业务效果

# In[ ]:


# 业务效果-训练集
train['score'] = card.predict(train[cols]).round(0)
pred_data = toad.metrics.KS_bucket(train['score'], train['target'], bucket=10, method='quantile')
pred_data


# In[ ]:


cut_bins = [float('-inf')]+list(pred_data['min'])[1:]+[float('inf')]
# cut_bins = [float('-inf')]+ [711.0, 750.0, 763.0, 797.0] + [float('inf')]
print(cut_bins)


# In[ ]:


# 业务效果-训练集
train['score'] = card.predict(train[cols]).round(0)
train['bins'] = pd.cut(train['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


# 业务效果-oot
oot['score'] = card.predict(oot[cols]).round(0)
oot['bins'] = pd.cut(oot['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


pred_data_train = score_distribute(train, 'bins', target='target')
pred_data_train


# In[ ]:


pred_data_oot = score_distribute(oot, 'bins', target='target')
pred_data_oot


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_分数分布_step_lr.xlsx")
pred_data_train = score_distribute(train, 'bins', target='target')
# pred_data_valid = score_distribute(valid, 'bins', target='target')
pred_data_oot = score_distribute(oot, 'bins', target='target')


pred_data_train.to_excel(writer,sheet_name='pred_data_train')
# pred_data_valid.to_excel(writer,sheet_name='pred_data_valid')
pred_data_oot.to_excel(writer,sheet_name='pred_data_oot')

writer.save()


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_分数分布_quantile_lr.xlsx")
pred_data_train = score_distribute(train, 'bins', target='target')
# pred_data_valid = score_distribute(valid, 'bins', target='target')
pred_data_oot = score_distribute(oot, 'bins', target='target')


pred_data_train.to_excel(writer,sheet_name='pred_data_train')
# pred_data_valid.to_excel(writer,sheet_name='pred_data_valid')
pred_data_oot.to_excel(writer,sheet_name='pred_data_oot')

writer.save()


# In[ ]:


def cal_psi(exp, act):
    psi = []
    for i in range(len(exp)):
        psi_i = (act[i] - exp[i])*np.log(act[i]/exp[i])
        psi.append(psi_i)
    return sum(psi)


# In[ ]:


# print(cal_psi(pred_data_train['total_pct'], pred_data_valid['total_pct'])) 
print(cal_psi(pred_data_train['total_pct'], pred_data_oot['total_pct'])) 




#==============================================================================
# File: B卡建模-Lr-174渠道.py
#==============================================================================

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import toad
import os 
from datetime import datetime
from sklearn.model_selection import train_test_split
from IPython.core.interactiveshell import InteractiveShell
import warnings
import gc
from statistics import mode
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, cross_validate, cross_val_score
import time
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from pypmml import Model

warnings.filterwarnings("ignore")
InteractiveShell.ast_node_interactivity = 'all'
pd.set_option('display.max_row',None)
pd.set_option('display.width',1000)


# In[2]:


os.getcwd()


# In[3]:


# 运行函数脚本
get_ipython().run_line_magic('run', 'function.ipynb')


# # 1.读取数据集

# In[4]:


filepath = r'D:\liuyedao\B卡开发\三方数据匹配'
data = pd.read_csv(filepath + r'\order_20230927.csv') 


# In[53]:


df_base = data.query("channel_id==174 & apply_date<'2023-07-01' & apply_date>='2022-10-01'")
# 删除全部是空值的列 
df_base.dropna(how='all', axis=1, inplace=True)
df_base.dropna(how='all', axis=1, inplace=True)
df_base.reset_index(drop=True, inplace=True)
df_base.info()
df_base.head()


# In[54]:


to_drop = list(df_base.select_dtypes(include='object').columns)
print(to_drop)


# In[55]:


df_base.drop(['operationType', 'swift_number', 'name', 'mobileEncrypt', 'orderNo', 'idCardEncrypt'],
             axis=1,inplace=True)


# In[56]:


# 小于0为异常值，转为空值
for col in df_base.select_dtypes(include='float64').columns[1:]:
    df_base[col] = df_base[col].mask(df_base[col]<0)


# In[57]:


df_behavior = pd.read_csv(r'D:\liuyedao\B卡开发\lr_xgb_174\b卡衍生变量_20231013.csv')


# In[58]:


df_base = pd.merge(df_base, df_behavior, how='left', on='user_id')
print(df_base.shape, df_base.order_no.nunique())


# In[59]:


df_pudao_3 = pd.read_csv(r'D:\liuyedao\B卡开发\lr_xgb_174\order_pudao_3_diff_20231013.csv')
df_bairong_1 = pd.read_csv(r'D:\liuyedao\B卡开发\lr_xgb_174\order_bairong_1_diff_20231013.csv')


# In[60]:


df_base = pd.merge(df_base, df_pudao_3, how='left', on='order_no')
print(df_base.shape, df_base.order_no.nunique())


# In[61]:


df_base = pd.merge(df_base, df_bairong_1, how='left', on='order_no')
print(df_base.shape, df_base.order_no.nunique())


# In[62]:


df_model = df_base.query("apply_date>='2022-10-01' & apply_date<'2023-06-01' & target in [0.0, 1.0]")


# In[63]:


path = r'D:\liuyedao\B卡开发\lr_result_174\\'


# In[64]:


tmp_df_model = toad.detect(df_model)
tmp_df_model.to_excel(path + 'df_explore_20231023.xlsx')


# ## 训练数据集

# In[65]:


df_train = df_base.query("apply_date>='2022-10-01' & apply_date<'2023-05-01' & target in [0.0, 1.0]")
# df_train.groupby(['lending_month','target'])['order_no'].count().unstack()


# In[66]:


print(df_train.target.value_counts())
print(df_train.target.value_counts(normalize=True))


# ## oot测试数据集

# In[67]:


df_oot = df_base.query("apply_date>='2023-05-01' & apply_date<'2023-06-01' & target in [0.0, 1.0]")
print(df_oot.target.value_counts())
print(df_oot.target.value_counts(normalize=True))


# In[68]:


print(df_train.shape, df_oot.shape)


# # 2.数据处理

# In[69]:


train = df_train.copy()
oot = df_oot.copy()

# oot = df_train.copy()
# train = df_oot.copy()


# In[70]:


to_drop = train.columns[0:6].to_list()
print(to_drop)


# In[71]:


train = train.drop(to_drop, axis=1)
oot = oot.drop(to_drop, axis=1)


# In[72]:


to_drop = list(train.select_dtypes(include='object').columns)
print(to_drop)


# In[ ]:


# train = train.drop(to_drop, axis=1)
# oot = oot.drop(to_drop, axis=1)


# In[73]:


train.info()


# In[74]:


oot.info()


# In[75]:


print(train.target.value_counts())
print(oot.target.value_counts())


# In[76]:


print(train.target.value_counts(normalize=True))
print(oot.target.value_counts(normalize=True))


# # 3.数据探索分析

# In[ ]:


# # 小于0为异常值，转为空值
# train = train.mask(train<0)
# oot = oot.mask(oot<0)


# In[77]:


# 删除全部是空值的列
train.dropna(how='all', axis=1, inplace=True)
oot.dropna(how='all', axis=1, inplace=True)


# In[78]:


train_explore = toad.detect(train.drop('target', axis=1))
oot_explore = toad.detect(oot.drop('target', axis=1))

modes = train.drop('target', axis=1).apply(mode)
mode_counts = train.drop('target', axis=1).eq(modes).sum()
train_modes = pd.DataFrame({'mode':modes, 'mode_num':mode_counts}, index=list(train.columns[1:]))

modes = oot.drop('target', axis=1).apply(mode)
mode_counts = oot.drop('target', axis=1).eq(modes).sum()
oot_modes = pd.DataFrame({'mode':modes, 'mode_num':mode_counts}, index=list(oot.columns[1:]))

train_isna = pd.DataFrame(train.drop('target', axis=1).isnull().sum(), columns=['missing_num'])
oot_isna = pd.DataFrame(oot.drop('target', axis=1).isnull().sum(), columns=['missing_num'])


train_iv = toad.quality(train,'target',iv_only=False)
oot_iv = toad.quality(oot,'target',iv_only=False)


# In[79]:


train_df_explore = pd.concat([train_explore, train_modes, train_isna, train_iv.drop('unique',axis=1)], axis=1)

train_df_explore['no_null_num'] = train_df_explore['size'] - train_df_explore['missing_num']
train_df_explore['miss_rate'] = train_df_explore['missing_num'] / train_df_explore['size']
train_df_explore['mode_pct_all'] = train_df_explore['mode_num']/train_df_explore['size']
train_df_explore['mode_pct_notna'] = train_df_explore['mode_num']/train_df_explore['no_null_num']


# In[80]:


oot_df_explore = pd.concat([oot_explore, oot_modes, oot_isna, oot_iv.drop('unique',axis=1)], axis=1)

oot_df_explore['no_null_num'] = oot_df_explore['size'] - oot_df_explore['missing_num']
oot_df_explore['miss_rate'] = oot_df_explore['missing_num'] / oot_df_explore['size']
oot_df_explore['mode_pct_all'] = oot_df_explore['mode_num']/oot_df_explore['size']
oot_df_explore['mode_pct_notna'] = oot_df_explore['mode_num']/oot_df_explore['no_null_num']


# In[81]:


df_iv = pd.merge(train_iv, oot_iv, how='inner',left_index=True, right_index=True,suffixes=['_train','_oot'])
df_iv['diff_iv'] = df_iv['iv_oot']-df_iv['iv_train']
df_iv['rate_iv'] = df_iv['iv_oot']/df_iv['iv_train'] - 1


# In[82]:


writer=pd.ExcelWriter(path + 'B卡_探索性分析_58同城_20231023.xlsx')

train_df_explore.to_excel(writer,sheet_name='train_df_explore')
oot_df_explore.to_excel(writer,sheet_name='oot_df_explore')

# train_df_iv.to_excel(writer,sheet_name='train_df_iv')
# oot_df_iv.to_excel(writer,sheet_name='oot_df_iv')

df_iv.to_excel(writer,sheet_name='df_iv')

writer.save()


# # 4.特征粗筛选

# In[83]:


# 删除缺失率大于0.85/删除枚举值只有一个/删除方差等于0/删除集中度大于0.85

to_drop_missing = list(train_df_explore[train_df_explore.miss_rate>=0.85].index)
print(len(to_drop_missing))
to_drop_unique = list(train_df_explore[train_df_explore.unique==1].index)
print(len(to_drop_unique))
to_drop_std = list(train_df_explore[train_df_explore.std_or_top2==0].index)
print(len(to_drop_std))
to_drop_mode = list(train_df_explore[train_df_explore.mode_pct_notna>=0.85].index)
print(len(to_drop_mode))
to_drop_mode2 = list(train_df_explore[train_df_explore.mode_pct_all>=0.85].index)
print(len(to_drop_mode2))

to_drop_train = list(set(to_drop_missing+to_drop_unique+to_drop_std+to_drop_mode+to_drop_mode2))
print(len(to_drop_train))


# In[84]:


# 删除缺失率大于0.85/删除枚举值只有一个/删除方差等于0/删除集中度大于0.85
to_drop_missing = list(oot_df_explore[oot_df_explore.miss_rate>=0.85].index)
print(len(to_drop_missing))
to_drop_unique = list(oot_df_explore[oot_df_explore.unique==1].index)
print(len(to_drop_unique))
to_drop_std = list(oot_df_explore[oot_df_explore.std_or_top2==0].index)
print(len(to_drop_std))
to_drop_mode = list(oot_df_explore[oot_df_explore.mode_pct_notna>=0.85].index)
print(len(to_drop_mode))
to_drop_mode2 = list(oot_df_explore[oot_df_explore.mode_pct_all>=0.85].index)
print(len(to_drop_mode))

to_drop_oot = list(set(to_drop_missing+to_drop_unique+to_drop_std+to_drop_mode+to_drop_mode2))
print(len(to_drop_oot))


# In[85]:


train_1 = train.drop(to_drop_train, axis=1)
print(train_1.shape)

oot_1 = oot.drop(to_drop_oot, axis=1)
print(oot_1.shape)


# In[197]:


# 共同的变量
sim_cols = list(set(train_1.drop('target',axis=1).columns).intersection(set(oot_1.drop('target',axis=1).columns)))
print(len(sim_cols))

train_2 = train_1[['target']+sim_cols]
oot_2 = oot_1[['target']+sim_cols]
print(train_2.shape, oot_2.shape)


# In[198]:


# psi/iv稳定性筛选
to_drop_iv = list(df_iv.query("iv_train<=0.02").index)
print(len(to_drop_iv))

# psi = toad.metrics.PSI(train_2.drop('target', axis=1), oot_2.drop('target',axis=1))
# to_drop_psi = list(psi[psi>=0.25].index)
# print(len(to_drop_psi))

to_drop = []
for col in list(set(to_drop_iv)):
    if col in train_2.columns:
        to_drop.append(col)
print(len(to_drop))

train_2.drop(to_drop, axis=1, inplace=True)
print(train_2.shape)


# In[ ]:





# In[199]:


# iv值/相关性筛选
train_selected, dropped = toad.selection.select(train_2, target='target', empty=0.85, iv=0.02,
                                                corr=1.0,
                                                return_drop=True, exclude=None)
train_selected.shape


# In[200]:


for i in dropped.keys():
    print("变量筛选维度：{}， 共剔除变量{}".format(i, len(dropped[i])))


# In[201]:


# 人工剔除
to_drop_man = ['model_score_01_baihang_1','model_score_01_q_tianchuang_1','model_score_01_r_tianchuang_1',
           'model_score_01_1_bileizhen_1',
           'model_score_01_2_bileizhen_1','model_score_01_3_bileizhen_1','value_011_moxingfen_7',
           'value_012_pudao_5',
           'model_score_01_pudao_11','model_score_01_pudao_12','model_score_01_pudao_16',
           'model_score_01_2_bairong_8','model_score_01_3_bairong_8','model_score_01_7_bairong_8',
           'model_score_01_8_bairong_8']

to_drop = []
for col in to_drop_man:
    if col in train_selected.columns:
        to_drop.append(col)
print(len(to_drop))

train_selected.drop(to_drop, axis=1, inplace=True)
print(train_selected.shape)


# ## 变量分箱

# In[181]:


# cols = [
#  'als_m12_cell_nbank_ca_orgnum_diff'
# ,'model_score_01_rong360_4'
# ,'model_score_01_moxingfen_7'
# ,'value_054_bairong_1'
# ,'value_060_baihang_6'
# ,'als_d15_id_nbank_nsloan_orgnum'
# ,'model_score_01_ruizhi_6'
# ,'als_fst_id_nbank_inteday'
# ,'value_026_bairong_12'
# ]

# train_selected = train[['target']+cols]


# In[202]:


# 第一次分箱
c = toad.transform.Combiner()
c.fit(train_selected, y='target', method='chi', min_samples=0.05, n_bins=10, empty_separate=True) 
bins_result = c.export()


# In[184]:


adj_bins = {
    'als_m12_cell_nbank_ca_orgnum_diff': [-88888, 1.0, 2.0],
    'model_score_01_rong360_4': [-88888, 0.055878434330225, 0.074964128434658, 0.10986643284559],
    'model_score_01_tianchuang_7': [-88888, 555.0, 595.0],
    'als_m3_cell_bank_max_inteday_diff': [-0.5, 46.0],
    'model_score_01_moxingfen_7': [-88888, 530.0],
    'value_054_bairong_1': [-88888, 3.0, 5.0],
    'value_060_baihang_6': [-88888, 2.0, 3.0, 4.0],
    'als_d15_id_nbank_nsloan_orgnum': [-88888, 3.0, 4.0],
    'model_score_01_ruizhi_6': [-88888, 670.0, 774.0, 842.0],
    'als_fst_id_nbank_inteday': [-88888, 343.0, 357.0],
    'value_026_bairong_12': [-88888, 34.1395]
}

# 更新分箱
c.update(adj_bins) 


# In[216]:


train_selected_bins = c.transform(train_selected.fillna(-99999), labels=True)
oot_selected_bins = c.transform(oot[train_selected.columns].fillna(-99999), labels=True)


# In[204]:


train_selected_bins = c.transform(train_selected, labels=True)
oot_selected_bins = c.transform(oot[train_selected.columns], labels=True)


# In[217]:


print(train_selected_bins.shape,oot_selected_bins.shape)


# In[218]:


bins_dict_train = {}
for col in train_selected_bins.columns[1:]:
    bins_dict_train[col] = regroup(train_selected_bins, col, target='target')
    
df_result_train = pd.concat(list(bins_dict_train.values()), axis=0, ignore_index =True)


# In[219]:


bins_dict_oot = {}
for col in oot_selected_bins.columns[1:]:
    bins_dict_oot[col] = regroup(oot_selected_bins, col, target='target')
    
df_result_oot = pd.concat(list(bins_dict_oot.values()), axis=0, ignore_index =True)


# In[220]:


df_result = pd.merge(df_result_train, df_result_oot, how='left', on=['varsname','bins'],
                     suffixes=['_train','_oot'])


# In[221]:


df_result.to_excel(path + 'B卡_chi_58同城_20231023_all_单调.xlsx',index=False)


# In[ ]:


to_drop_bins1 = []
for col in list(bins_result.keys()):
    if len(bins_result[col])==1:
        to_drop_bins1.append(col)
print(len(to_drop_bins1))
to_drop_bins2 = list(set(df_result[(df_result['iv_train']<0.02)]['varsname']))
print(len(to_drop_bins2))
to_drop_bins = list(set(to_drop_bins1 + to_drop_bins2))
print(len(to_drop_bins))


# In[ ]:


train_selected_1 = train_selected.drop(to_drop_bins, axis=1)
train_selected_1.shape


# In[ ]:


train_selected_bins_1 = train_selected_bins[train_selected_1.columns]
oot_selected_bins_1 = oot_selected_bins[train_selected_bins_1.columns]


# In[ ]:


from toad.plot import bin_plot, badrate_plot


# In[ ]:


# col = list(train_selected_bins_1.drop('target',axis=1).columns)[1]

# bin_plot(train_selected_bins_1, x=col, target='target')
# bin_plot(oot_selected_bins_1, x=col, target='target')


# In[209]:


# 连续变量分箱主函数
def ContinueVarBins(data, col, flag='target', cutbins=[]):
    df = data[[flag, col]]
    df = df[~df[col].isnull()].reset_index(drop=True)
    df['bins'] = pd.cut(df[col], cutbins, duplicates='drop', right=False, precision=4)
    regroup = pd.DataFrame()
    regroup['bins'] = df.groupby(['bins'])[col].max()
    regroup['total'] = df.groupby(['bins'])[flag].count()
    regroup['bad'] = df.groupby(['bins'])[flag].sum()
    regroup['good'] = regroup['total'] - regroup['bad']
    regroup.drop(['total'], axis=1, inplace=True)
    np_regroup = np.array(regroup)
    np_regroup = MergeZero(np_regroup)
    np_regroup = MinPct(np_regroup)
    np_regroup = MonTone(np_regroup)
    regroup = pd.DataFrame(index=np.arange(np_regroup.shape[0]))
    regroup['bins'] = np_regroup[:,0]
    regroup['total'] = np_regroup[:,1] + np_regroup[:,2]
    regroup['bad'] = np_regroup[:,1]
    regroup['good'] = np_regroup[:,2]
    cutoffpoints = list(np_regroup[:,0])
    cutoffpoints = [float('-inf')] + cutoffpoints
    # 最大值分割点，转换最小值分割点
    df['bins_new'] = pd.cut(df[col], cutoffpoints, duplicates='drop', right=True, precision=4)
    tmp = pd.DataFrame()
    tmp['bins'] = df.groupby(['bins_new'])[col].min()
    cutoffpoints = list(tmp['bins'])  
    
    return cutoffpoints


# In[213]:


# 自动调整分箱
adj_bins = {}
for col in list(train_selected.columns)[1:]:
    tmp = np.isnan(bins_result[col])
    cutbins = [bins_result[col][x] for x in range(len(tmp)) if not tmp[x]]
    if len(cutbins)>0:
        cutbins = [float('-inf')] + cutbins + [float('inf')]
        cutoffpoints = ContinueVarBins(train_selected, col, flag='target', cutbins=cutbins)
        tmp_value1 = train_selected[col].min()
        tmp_value2 = cutoffpoints[0]
        if tmp_value1==tmp_value2:
            adj_bins[col] = [-88888] + cutoffpoints[1:]
#             print("分割点有最小值", col)
        else:
            adj_bins[col] = [-88888] + cutoffpoints
    else:
        print("无分割点", col)


# In[214]:


# 更新分箱
c.update(adj_bins)


# In[146]:


# 调整分箱:空值单独一箱
train_selected.fillna(-99999, inplace=True)


# In[215]:


adj_bins


# In[137]:


train_selected.head()


# In[ ]:


# c.export()
# c.load(dict)
# c.transform(dataframe, labels=False)


# ## WOE转换

# In[222]:


transer = toad.transform.WOETransformer()
train_woe = transer.fit_transform(c.transform(train_selected.fillna(-99999)), train_selected['target'],
                                  exclude=['target'])
train_woe.shape


# In[223]:


train_selected_woe, dropped = toad.selection.select(train_woe, target='target', empty=0.85,
                                                    iv=0.02, corr=0.7, return_drop=True, exclude=None)
train_selected_woe.shape


# In[224]:


for i in dropped.keys():
    print("变量筛选维度：{}， 共剔除变量{}".format(i, len(dropped[i])))


# In[225]:


df_result_finally = pd.merge(df_result, pd.DataFrame({'varsname':list(train_selected_woe.columns)}),
                             how='inner', on='varsname')
df_result_finally.to_excel(path + 'B卡_chi_58同城_20231023_all_单调_iv_corr_筛选.xlsx',index=False)


# In[160]:


cols = [
 'model_score_01_3_xinyongsuanli_1'
,'als_m1_cell_nbank_cons_allnum'
,'value_002_moxingfen_7'
,'als_m12_cell_nbank_selfnum_diff'
,'als_m6_id_coon_orgnum_diff'
,'model_score_01_tianchuang_7'
,'als_d15_cell_nbank_cons_allnum_diff'
,'model_score_01_tengxun_1'
,'value_087_baihang_6'
,'als_m12_cell_nbank_max_monnum_diff'
,'value_068_baihang_6'
,'ppdi_m3_cell_nbank_loan_orgnum'
,'als_m3_cell_coon_allnum_diff'
,'als_m12_cell_bank_ret_orgnum'
,'model_score_01_moxingfen_7'
,'als_m3_id_nbank_nsloan_orgnum'
,'value_050_bairong_1'
,'als_m12_cell_nbank_week_allnum_diff'
,'als_d15_id_nbank_nsloan_orgnum'
,'value_031_bairong_12'
,'als_m3_cell_nbank_cons_allnum_diff'
,'als_d15_cell_nbank_selfnum'
,'als_m12_id_nbank_ca_allnum_diff'
,'model_score_01_rong360_4'
,'als_m3_id_cooff_allnum_diff'
,'als_m12_cell_nbank_nsloan_allnum_diff'
,'als_m1_cell_nbank_night_allnum_diff'
,'als_m12_id_max_inteday'
,'value_060_baihang_6'
,'als_m3_cell_caon_allnum_diff'
,'model_score_01_ruizhi_6'
,'als_m6_cell_oth_orgnum_diff'
,'value_019_baihang_6'
,'model_score_01_tianchuang_8'
,'model_score_01_ronghuijinke_3'
,'als_m6_cell_nbank_oth_orgnum_diff'
,'als_m3_id_nbank_night_allnum_diff'
,'value_057_baihang_6'
,'als_m12_cell_avg_monnum_diff'
,'model_score_01_v2_duxiaoman_1'
,'als_m12_id_pdl_orgnum_diff'
,'als_m12_id_cooff_allnum_diff'
,'als_m6_id_nbank_ca_orgnum_diff'
]
train_selected_woe = train_selected_woe[['target']+cols]


# In[226]:


oot_woe = transer.transform(c.transform(oot[train_selected_woe.columns].fillna(-99999)))
oot_selected_woe = oot_woe[list(train_selected_woe.columns)]
oot_selected_woe.shape


# In[227]:


psi = toad.metrics.PSI(train_selected_woe.drop('target',axis=1), oot_selected_woe.drop('target',axis=1))
to_drop_psi = list(psi[psi>0.25].index)
print(len(to_drop_psi))


# In[228]:


train_selected_woe_psi = train_selected_woe.drop(to_drop_psi, axis=1)
oot_selected_woe_psi = oot_selected_woe[train_selected_woe_psi.columns]


# In[229]:


df_result_finally = pd.merge(df_result, pd.DataFrame({'varsname':list(train_selected_woe_psi.columns)}),
                             how='inner', on='varsname')
df_result_finally.to_excel(path + 'B卡_chi_58同城_20231023_all_单调_psi_筛选.xlsx',index=False)


# ## 逐步回归

# In[322]:


cols = [
 'model_score_01_3_xinyongsuanli_1'
,'als_m1_cell_nbank_cons_allnum'
,'value_027_bairong_12'
,'als_m12_cell_nbank_selfnum_diff'
,'als_m6_id_coon_orgnum_diff'
,'als_m3_id_coon_allnum_diff'
,'als_m3_id_nbank_else_orgnum_diff'
,'als_m12_id_coon_allnum_diff'
,'als_d15_id_nbank_allnum'
,'als_m1_id_pdl_allnum'
,'als_m12_cell_nbank_else_orgnum_diff'
,'value_058_baihang_6'
,'model_score_01_tianchuang_7'
,'als_m3_id_nbank_week_orgnum_diff'
,'als_m3_cell_nbank_avg_monnum_diff'
,'create_time_diff_x'
,'model_score_01_tengxun_1'
,'als_m6_id_nbank_nsloan_orgnum_diff'
,'als_m3_cell_pdl_allnum'
,'als_m3_id_nbank_max_inteday_diff'
,'value_087_baihang_6'
,'als_m6_cell_nbank_else_orgnum_diff'
,'als_m12_cell_nbank_max_monnum_diff'
,'als_m3_cell_coon_orgnum_diff'
,'als_m1_cell_nbank_oth_allnum_diff'
,'als_m1_id_nbank_oth_allnum'
,'als_m3_cell_nbank_min_monnum_diff'
,'model_score_01_moxingfen_7'
,'ppdi_m3_cell_nbank_fin_orgnum'
,'value_050_bairong_1'
,'als_d15_cell_nbank_cons_orgnum_diff'
,'als_m12_cell_nbank_week_allnum_diff'
,'model_score_01_hangliezhi_1'
,'als_d15_id_nbank_oth_allnum_diff'
,'value_031_bairong_12'
,'als_m6_id_nbank_oth_orgnum_diff'
,'als_d15_cell_nbank_orgnum_diff'
,'als_m6_id_max_inteday'
,'als_m1_cell_nbank_orgnum_diff'
,'als_m3_cell_nbank_cons_allnum_diff'
,'als_m12_id_caon_orgnum_diff'
,'value_025_baihang_6'
,'als_m12_id_avg_monnum_diff'
,'als_m6_cell_avg_monnum_diff'
,'value_021_baihang_6'
,'als_d15_cell_nbank_selfnum'
,'als_m3_id_nbank_cons_orgnum_diff'
,'als_m12_cell_nbank_else_allnum_diff'
,'als_m1_cell_nbank_orgnum'
,'als_m1_cell_nbank_cons_orgnum'
,'als_m6_cell_nbank_avg_monnum_diff'
,'als_m3_id_cooff_allnum_diff'
,'als_m12_cell_nbank_nsloan_allnum_diff'
,'model_score_01_zr_tongdun_2'
,'als_m6_id_nbank_max_monnum_diff'
,'als_m1_cell_nbank_night_allnum_diff'
,'als_m12_id_max_inteday'
,'als_fst_id_nbank_inteday'
,'als_m3_cell_nbank_else_orgnum'
,'als_m12_cell_nbank_oth_orgnum_diff'
,'value_054_baihang_6'
,'als_m12_cell_nbank_ca_allnum_diff'
,'als_m3_cell_caon_allnum_diff'
,'als_m12_cell_nbank_allnum_diff'
,'value_052_baihang_6'
,'model_score_01_ruizhi_6'
,'als_m3_cell_nbank_allnum_diff'
,'als_m6_cell_oth_orgnum_diff'
,'value_005_bairong_12'
,'als_d15_cell_caon_orgnum'
,'als_m12_id_bank_avg_monnum_diff'
,'model_score_01_tianchuang_8'
,'model_score_01_ronghuijinke_3'
,'als_m3_id_avg_monnum_diff'
,'value_044_bairong_1'
,'als_m3_cell_rel_orgnum_diff'
,'als_m3_id_nbank_nsloan_orgnum_diff'
,'als_m6_cell_nbank_oth_orgnum_diff'
,'value_053_baihang_6'
,'als_m3_id_nbank_night_allnum_diff'
,'als_m6_cell_nbank_max_monnum_diff'
,'als_m12_id_max_monnum_diff'
,'als_m3_id_max_monnum_diff'
,'value_081_bairong_1'
,'model_score_01_v2_duxiaoman_1'
,'als_m12_id_pdl_orgnum_diff'
,'als_m12_cell_nbank_oth_allnum_diff'
,'als_m12_id_cooff_allnum_diff'
,'als_m1_cell_nbank_oth_orgnum_diff'
,'als_m6_id_nbank_ca_orgnum_diff'
]


# In[292]:


cols = [
 'als_m1_cell_nbank_cons_allnum'
,'value_027_bairong_12'
,'als_m12_cell_nbank_selfnum_diff'
,'als_m6_id_coon_orgnum_diff'
,'als_m3_id_coon_allnum_diff'
,'als_m3_id_nbank_else_orgnum_diff'
,'als_m12_id_coon_allnum_diff'
,'als_d15_id_nbank_allnum'
,'als_m1_id_pdl_allnum'
,'als_m12_cell_nbank_else_orgnum_diff'
,'value_058_baihang_6'
,'als_m3_id_nbank_week_orgnum_diff'
,'als_m3_cell_nbank_avg_monnum_diff'
,'create_time_diff_x'
,'als_m6_id_nbank_nsloan_orgnum_diff'
,'als_m3_cell_pdl_allnum'
,'als_m3_id_nbank_max_inteday_diff'
,'value_087_baihang_6'
,'als_m6_cell_nbank_else_orgnum_diff'
,'als_m12_cell_nbank_max_monnum_diff'
,'als_m3_cell_coon_orgnum_diff'
,'als_m1_cell_nbank_oth_allnum_diff'
,'als_m1_id_nbank_oth_allnum'
,'als_m3_cell_nbank_min_monnum_diff'
,'model_score_01_moxingfen_7'
,'ppdi_m3_cell_nbank_fin_orgnum'
,'value_050_bairong_1'
,'als_d15_cell_nbank_cons_orgnum_diff'
,'als_m12_cell_nbank_week_allnum_diff'
,'als_d15_id_nbank_oth_allnum_diff'
,'value_031_bairong_12'
,'als_m6_id_nbank_oth_orgnum_diff'
,'als_d15_cell_nbank_orgnum_diff'
,'als_m6_id_max_inteday'
,'als_m1_cell_nbank_orgnum_diff'
,'als_m3_cell_nbank_cons_allnum_diff'
,'als_m12_id_caon_orgnum_diff'
,'value_025_baihang_6'
,'als_m12_id_avg_monnum_diff'
,'als_m6_cell_avg_monnum_diff'
,'value_021_baihang_6'
,'als_d15_cell_nbank_selfnum'
,'als_m3_id_nbank_cons_orgnum_diff'
,'als_m12_cell_nbank_else_allnum_diff'
,'als_m1_cell_nbank_orgnum'
,'als_m1_cell_nbank_cons_orgnum'
,'als_m6_cell_nbank_avg_monnum_diff'
,'als_m3_id_cooff_allnum_diff'
,'als_m12_cell_nbank_nsloan_allnum_diff'
,'als_m6_id_nbank_max_monnum_diff'
,'als_m1_cell_nbank_night_allnum_diff'
,'als_m12_id_max_inteday'
,'als_fst_id_nbank_inteday'
,'als_m3_cell_nbank_else_orgnum'
,'als_m12_cell_nbank_oth_orgnum_diff'
,'value_054_baihang_6'
,'als_m12_cell_nbank_ca_allnum_diff'
,'als_m3_cell_caon_allnum_diff'
,'als_m12_cell_nbank_allnum_diff'
,'value_052_baihang_6'
,'als_m3_cell_nbank_allnum_diff'
,'als_m6_cell_oth_orgnum_diff'
,'value_005_bairong_12'
,'als_d15_cell_caon_orgnum'
,'als_m12_id_bank_avg_monnum_diff'
,'als_m3_id_avg_monnum_diff'
,'value_044_bairong_1'
,'als_m3_cell_rel_orgnum_diff'
,'als_m3_id_nbank_nsloan_orgnum_diff'
,'als_m6_cell_nbank_oth_orgnum_diff'
,'value_053_baihang_6'
,'als_m3_id_nbank_night_allnum_diff'
,'als_m6_cell_nbank_max_monnum_diff'
,'als_m12_id_max_monnum_diff'
,'als_m3_id_max_monnum_diff'
,'value_081_bairong_1'
,'als_m12_id_pdl_orgnum_diff'
,'als_m12_cell_nbank_oth_allnum_diff'
,'als_m12_id_cooff_allnum_diff'
,'als_m1_cell_nbank_oth_orgnum_diff'
,'als_m6_id_nbank_ca_orgnum_diff'
]


# In[323]:


print(len(cols))


# In[324]:


print(train_selected_woe_psi.shape)
final_data = toad.selection.stepwise(train_selected_woe_psi[['target']+cols], target='target',
                                     estimator='ols',
                                     direction='both', criterion='aic', exclude=None,
                                     intercept=False)
print(final_data.shape)


# In[306]:


# final_data = train_woe.copy()
oot_woe = transer.transform(c.transform(oot[list(final_data.columns)].fillna(-99999)))
final_oot = oot_woe[list(final_data.columns)]


# In[307]:


# 初次建模变量
cols = list(final_data.drop(['target'], axis=1).columns)
print(len(cols))
print(cols)


# # 6.Lr模型训练

# ## 6.1模型技术效果

# In[50]:


# 建模变量
# cols = list(final_data.drop(['target'], axis=1).columns)
print(len(cols))
print(cols)


# In[240]:


df_result_rm = pd.merge(df_result, pd.DataFrame({'varsname':cols}),how='inner',on='varsname')
df_result_rm.head()
df_result_rm.to_excel(r'D:\liuyedao\B卡开发\lr_result_174\B卡_chi_10bins_58同城_入模变量_20231025.xlsx',index=False)


# In[315]:


# cols.remove('model_score_01_tianchuang_8') 
# cols.remove('als_m6_cell_nbank_max_monnum_diff')
cols.append('model_score_01_tianchuang_8')


# In[250]:


for i in ['als_m6_cell_nbank_else_orgnum','als_d15_id_nbank_nsloan_orgnum_diff','ppdi_d15_id_orgnum'
            ,'als_m1_id_caon_allnum_diff','als_m6_cell_nbank_oth_allnum']:
    cols.remove(i)


# In[313]:


print(len(cols),cols)


# In[316]:


clf = LR_model(final_data[cols], final_oot[cols], final_data['target'], final_oot['target'])


# In[317]:


opt_best = {'target': 0.6487565005430443,
            'params': {'gamma': 0.75, 'learning_rate': 0.45, 'max_depth': 6.0, 'min_child_weight': 9.0,
                       'n_estimators': 12, 'reg_alpha': 0, 'reg_lambda': 290}
           }


# In[318]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 


# In[319]:


xgb_model.fit(train[cols] ,train['target'])


# In[320]:


# 对训练集进行预测

from sklearn import metrics

y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[321]:


clf = LR_model(final_oot[cols], final_data[cols], final_oot['target'], final_data['target'])


# In[177]:


clf = LR_model(final_data[cols], final_oot[cols], final_data['target'], final_oot['target'])


# In[178]:


clf = LR_model(final_oot[cols], final_data[cols], final_oot['target'], final_data['target'])


# In[179]:


vif=pd.DataFrame()
X = np.matrix(final_data[cols])
vif['features']=cols
vif['VIF_Factor']=[variance_inflation_factor(np.matrix(X),i) for i in range(X.shape[1])]
vif


# In[180]:


corr = final_data[cols].corr()
corr


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_corr_vif_20231018"+'.xlsx')

corr.to_excel(writer,sheet_name='corr')
vif.to_excel(writer,sheet_name='vif')

writer.save()


# In[ ]:


# 训练集
pred_train = clf.predict(sm.add_constant(final_data[cols]))
# KS/AUC
from toad.metrics import KS,AUC

print('train AUC: ', AUC(pred_train, final_data['target']))
print('train KS: ', KS(pred_train, final_data['target']))


# In[ ]:


# 测试集
pred_oot = clf.predict(sm.add_constant(final_oot[cols]))
# KS/AUC
from toad.metrics import KS,AUC

print('-------------oot结果--------------------')

print('test AUC: ', AUC(pred_oot, final_oot['target']))
print('test KS: ', KS(pred_oot, final_oot['target']))



# ## 6.1转换评分

# In[ ]:


card = toad.ScoreCard(combiner=c,
                      transer = transer,
                      base_odds=35,
                      base_score=700,
                      pdo=60,
                      rate=2,
                      C=1e8
                     )


# In[ ]:


card.fit(final_data[cols], final_data['target'])


# In[ ]:


def scorecard_scale(self):
    scorecard_kedu = pd.DataFrame(
    [
        ["base_odds", self.base_odds,"根据业务经验设置的基础比率"],
        ["base_score", self.base_score,"基础odds对应的分数"],
        ["rate", self.rate, "设置分数的倍率"],
        ["pdo", self.pdo, "表示分数增长pdo时，odds值增长rate倍"],
        ["B", self.factor,"补偿值，计算方式：pdo/ln(rate)"],
        ["A", self.offset,"刻度,计算方式：base_score - B * ln(base_odds)"]
    ],
    columns = ["刻度项", "刻度值","备注"]
    )
    
    return scorecard_kedu


# In[ ]:


scorecard_scale(card)


# In[ ]:


score_card = card.export(to_frame=True).round(0)
print(len(set(score_card.name)))
score_card


# In[ ]:


# (card.offset - card.factor * card.coef_[0] * card.intercept_)/9


# In[ ]:


score_card.to_excel(r'D:\liuyedao\B卡开发\lr_xgb_174\B卡_评分卡_58同城_20231018_v1.xlsx')


# In[ ]:


name = {
    'als_m12_cell_nbank_ca_orgnum_diff': '按手机号查询，近12个月在非银机构-现金类分期申请机构数与上次查询的差',
    'model_score_01_rong360_4': '融360评分',
    'model_score_01_tianchuang_7': '天创信用联合定制分',
    'model_score_01_moxingfen_7': '贷中评分',
    'value_054_bairong_1': '按身份证号查询，近15天在非银机构-持牌消费金融机构申请机构数',
    'value_060_baihang_6': '百行_续侦_证件号近3个月内查询机构数-网络小贷类机构',
    'als_d15_id_nbank_nsloan_orgnum': '按身份证号查询，近15天在非银机构-持牌网络小贷机构申请机构数',
    'model_score_01_ruizhi_6': '睿智联合分',
    'als_fst_id_nbank_inteday': '按身份证号查询，距最早在非银行机构申请的间隔天数',
    'value_026_bairong_12': '近6个月用户利率偏好'
}


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_变量箱子_分数_v2.xlsx")
train_selected['score'] = card.predict(train_selected[cols]).round(0)
train_selected_bins = c.transform(train_selected, labels=True)
 

for col in cols:
    tmp1 = CalWoeIv(train_selected_bins, col, target='target')
    tmp2 = train_selected_bins[[col, 'score']].drop_duplicates(col,keep='first')
    tmp = pd.merge(tmp1, tmp2, how='left',left_on='bins', right_on=col)
    tmp['name'] = name[col]
    tmp = tmp[['varsname','name','bins','total','total_pct','bad_rate','woe']]
    tmp.to_excel(writer,sheet_name=col)
    writer.save()


# In[ ]:


print(len(cols), cols)


# In[ ]:


train['score'] = card.predict(train[cols]).round(0)
train['prob'] = card.predict_proba(train[cols])[:,1]


# In[ ]:


oot['score'] = card.predict(oot[cols]).round(0)
oot['prob'] = card.predict_proba(oot[cols])[:,1]


# In[ ]:


# tmp1 = train[['order_no','lending_time','channel_id','score','prob']+cols]
# tmp2 = oot[['order_no','lending_time','channel_id','score','prob']+cols]
# tmp = pd.concat([tmp1, tmp2], axis=0)
# tmp.info()
# tmp.head()
# tmp.to_csv(r'D:\liuyedao\B卡开发\Lr_result\score_lr_部署测试样例_58同城.csv',index=False)


# ## 6.2模型业务效果

# In[ ]:


# 业务效果-训练集
train['score'] = card.predict(train[cols]).round(0)
pred_data = toad.metrics.KS_bucket(train['score'], train['target'], bucket=10, method='quantile')
pred_data


# In[ ]:


cut_bins = [float('-inf')]+list(pred_data['min'])[1:]+[float('inf')]
# cut_bins = [float('-inf')]+ [711.0, 750.0, 763.0, 797.0] + [float('inf')]
print(cut_bins)


# In[ ]:


# 业务效果-训练集
train['score'] = card.predict(train[cols]).round(0)
train['bins'] = pd.cut(train['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


# 业务效果-oot
oot['score'] = card.predict(oot[cols]).round(0)
oot['bins'] = pd.cut(oot['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


pred_data_train = score_distribute(train, 'bins', target='target')
pred_data_train


# In[ ]:


pred_data_oot = score_distribute(oot, 'bins', target='target')
pred_data_oot


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_分数分布_step_lr.xlsx")
pred_data_train = score_distribute(train, 'bins', target='target')
# pred_data_valid = score_distribute(valid, 'bins', target='target')
pred_data_oot = score_distribute(oot, 'bins', target='target')


pred_data_train.to_excel(writer,sheet_name='pred_data_train')
# pred_data_valid.to_excel(writer,sheet_name='pred_data_valid')
pred_data_oot.to_excel(writer,sheet_name='pred_data_oot')

writer.save()


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_分数分布_quantile_lr.xlsx")
pred_data_train = score_distribute(train, 'bins', target='target')
# pred_data_valid = score_distribute(valid, 'bins', target='target')
pred_data_oot = score_distribute(oot, 'bins', target='target')


pred_data_train.to_excel(writer,sheet_name='pred_data_train')
# pred_data_valid.to_excel(writer,sheet_name='pred_data_valid')
pred_data_oot.to_excel(writer,sheet_name='pred_data_oot')

writer.save()


# In[ ]:


def cal_psi(exp, act):
    psi = []
    for i in range(len(exp)):
        psi_i = (act[i] - exp[i])*np.log(act[i]/exp[i])
        psi.append(psi_i)
    return sum(psi)


# In[ ]:


# print(cal_psi(pred_data_train['total_pct'], pred_data_valid['total_pct'])) 
print(cal_psi(pred_data_train['total_pct'], pred_data_oot['total_pct'])) 




#==============================================================================
# File: B卡建模-LR.py
#==============================================================================

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import toad
import os 
from datetime import datetime
from sklearn.model_selection import train_test_split
from IPython.core.interactiveshell import InteractiveShell
import warnings
import gc

warnings.filterwarnings("ignore")
InteractiveShell.ast_node_interactivity = 'all'
pd.set_option('display.max_row',None)
pd.set_option('display.width',1000)


# In[2]:


os.getcwd()


# In[513]:


# 运行函数脚本
get_ipython().run_line_magic('run', 'function.ipynb')


# # 1.读取数据集

# In[4]:


data_target = pd.read_csv(r'D:\liuyedao\B卡开发\mid_result\B卡_建模数据集_20230919.csv') 
print('数据大小：', data_target.shape)


# In[5]:


data_target.info()


# In[ ]:


data_target.head()


# ## 1.2拆分数据

# In[6]:


to_drop = list(data_target.columns[0:13]) + list(data_target.select_dtypes(include=['object','datetime64']).columns)
to_drop.remove('target')
print(to_drop)


# In[7]:


to_drop = list(set(to_drop))
print(to_drop)


# In[8]:


df_model = data_target.drop(to_drop, axis=1)


# In[9]:


df_model.info()


# In[10]:


df_var = df_model.drop(['target'], axis=1)
df_target = df_model['target']

X = np.array(df_var)
y = np.array(df_target)


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(df_var ,df_target, test_size=0.3, random_state=22, stratify=y)


# In[12]:


df_model.loc[X_train.index,'sample_set'] = 'train'
df_model.loc[X_test.index,'sample_set'] = 'valid'


# In[13]:


train = pd.merge(X_train, y_train, how='inner', left_index=True, right_index=True)
print(train.shape, X_train.shape, y_train.shape)


# In[14]:


valid = pd.merge(X_test, y_test, how='inner', left_index=True, right_index=True)
print(valid.shape, X_test.shape, y_test.shape)


# In[15]:


print(train['target'].value_counts())
print(valid['target'].value_counts())


# In[ ]:


train = train.mask(train<0)
valid = valid.mask(valid<0)


# In[ ]:


# b_cols = list(train.columns[train.columns.str.contains('his|days|cur|cnt|ovdue|period|no|balance|first|last|observe|credi')])
# print(len(b_cols))


# In[246]:


# train = train[['target'] + b_cols]

# valid = valid[['target'] + b_cols]
# train = data_target[data_target['channel_id']==174]
# train = train[train.lending_time<'2023-05-01']
# train['flag'] = train['ever_overdue_days'].apply(lambda x: 1 if x>15 else 0 if x==0 else 0.5)
# train = train[train['flag'].isin([0.0, 1.0])]
# train.groupby(['lending_month','flag'])['order_no'].count().unstack()

oot = data_target[data_target['channel_id']==80004]
oot = oot[oot.lending_time<'2023-05-01']
oot['flag'] = oot['ever_overdue_days'].apply(lambda x: 1 if x>15 else 0 if x==0 else 0.5)
oot = oot[oot['flag'].isin([0.0, 1.0])]
oot.groupby(['lending_month','flag'])['order_no'].count().unstack()


# In[599]:


train = data_target[data_target['channel_id']==174]
train = train[train.lending_time<'2023-05-01']
train['flag'] = train['ever_overdue_days'].apply(lambda x: 1 if x>15 else 0 if x==0 else 0.5)
train = train[train['flag'].isin([0.0, 1.0])]
train = train.reset_index(drop=True)
train.drop(train.columns[0:12].to_list(), axis=1, inplace=True)
train.drop(['value_015_moxingfen_7'], axis=1, inplace=True)
train.rename(columns={'flag':'target'}, inplace=True)
train.mask(train<0, inplace=True)
train.info()


# In[600]:


oot = data_target[data_target['channel_id']==80004]
oot = oot[oot.lending_time<'2023-05-01']
oot['flag'] = oot['ever_overdue_days'].apply(lambda x: 1 if x>15 else 0 if x==0 else 0.5)
oot = oot[oot['flag'].isin([0.0, 1.0])]
oot = oot.reset_index(drop=True)
oot.drop(oot.columns[0:12].to_list(), axis=1, inplace=True)
oot.drop(['value_015_moxingfen_7'], axis=1, inplace=True)
oot.rename(columns={'flag':'target'}, inplace=True)
oot.mask(oot<0, inplace=True)
oot.info()


# In[601]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('target', axis=1) ,train['target'], test_size=0.3, random_state=22, stratify=train['target'])


# In[602]:


train = pd.merge(X_train, y_train, how='inner', left_index=True, right_index=True)
print(train.shape, X_train.shape, y_train.shape)

valid = pd.merge(X_test, y_test, how='inner', left_index=True, right_index=True)
print(valid.shape, X_test.shape, y_test.shape)


# # 2.数据探索分析

# In[525]:


# for name, tmp_df in df_model.groupby(['sample_set']):
#     print(name)
#     tmp_explore = toad.detect(tmp_df)
#     tmp_explore['type_{}'.format(name)] = name
#     df_explore = pd.merge(df_explore, tmp_explore, how='left', left_index=True, right_index=True,suffixes=['',f'_{name}'])
    
# train_df_explore = toad.detect(train)
# oot_df_explore = toad.detect(oot)
train_df_iv = toad.quality(train,'target',iv_only=True)
oot_df_iv = toad.quality(oot,'target',iv_only=True)


# In[66]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\Lr_result\B卡_探索性分析_lr_58同城_20230922.xlsx")

train_df_explore.to_excel(writer,sheet_name='train_df_explore')
oot_df_explore.to_excel(writer,sheet_name='oot_df_explore')

train_df_iv.to_excel(writer,sheet_name='train_df_iv')
oot_df_iv.to_excel(writer,sheet_name='oot_df_iv')

writer.save()


# In[526]:


df_iv = pd.concat([train_df_iv, oot_df_iv],axis=1)
df_iv.head()


# In[465]:


df_iv.to_excel(r'D:\liuyedao\B卡开发\Lr_result\df_iv.xlsx')


# In[527]:


set1 = set(train_df_iv[train_df_iv.iv>0.1].index)
set2 = set(oot_df_iv[oot_df_iv.iv>0.1].index)
to_keep = list(set1.intersection(set2))
print(len(to_keep), to_keep)


# # 3.特征筛选

# In[ ]:


# train['credit_use'] = train['observe_credit_amt']/train['cur_balance_observe']
# valid['credit_use'] = valid['observe_credit_amt']/valid['cur_balance_observe']


# In[352]:


to_keep = list(train.columns[train.columns.str.contains("als| ppdi")]) + ['target']
train = train[to_keep]


# In[528]:


# train = df_model.query("sample_set=='train'").drop('sample_set', axis=1)
# valid = df_model.query("sample_set=='valid'").drop('sample_set', axis=1)
# train.drop(['operationType'], axis=1, inplace=True)
# valid.drop(['operationType'], axis=1, inplace=True)
train_selected, dropped = toad.selection.select(train, target='target', empty=0.90, iv=0.02, corr=0.8, return_drop=True, exclude=None)
train_selected.shape


# In[529]:


for i in dropped.keys():
    print("变量筛选维度：{}， 共剔除变量{}".format(i, len(dropped[i])))


# In[530]:


oot_selected, dropped = toad.selection.select(oot, target='target', empty=0.90, iv=0.02, corr=0.8, return_drop=True, exclude=None)
oot_selected.shape

for i in dropped.keys():
    print("变量筛选维度：{}， 共剔除变量{}".format(i, len(dropped[i])))


# In[531]:


train_df_iv = toad.quality(train_selected,'target',iv_only=True)
oot_df_iv = toad.quality(oot_selected,'target',iv_only=True)
df_iv = pd.merge(train_df_iv, oot_df_iv,how='inner',left_index=True, right_index=True,suffixes=['174','80004'])
df_iv.head()


# In[532]:


df_iv.info()


# In[492]:


to_keep = list(df_iv.index)
print(len(to_keep))


# In[548]:


to_keep = ['model_score_01_107_hulian_5','model_score_01_2_xinyongsuanli_1','model_score_01_baihang_1',
           'model_score_01_fulin_1','model_score_01_hengpu_4','model_score_01_q_tianchuang_1',
           'ppdi_m1_cell_nbank_fin_allnum','ppdi_m1_id_nbank_cons_allnum','ppdi_m1_id_nbank_nloan_allnum',
           'ppdi_m12_id_nbank_cons_allnum','ppdi_m12_id_nbank_cons_orgnum',
           'ppdi_m12_id_nbank_fin_orgnum','ppdi_m12_id_nbank_nloan_allnum','ppdi_m12_id_nbank_nloan_orgnum',
           'ppdi_m3_id_nbank_loan_orgnum','value_014_bairong_15',
           'value_016_bairong_15','value_018_bairong_15']


# In[493]:


df_iv.to_excel(r'D:\liuyedao\B卡开发\Lr_result\df_iv_v2.xlsx')


# In[549]:


train_selected = train_selected[to_keep+['target']]
oot_selected = oot_selected[to_keep+['target']]


# In[180]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\Lr_result\B卡_探索性分析_lr_58同城_20230922_v3.xlsx")
# train_df_explore = toad.detect(train_selected)
# valid_df_explore = toad.detect(valid[train_selected.columns])
train_df_iv = toad.quality(train_selected,'target',iv_only=True)
# valid_df_iv = toad.quality(valid[train_selected.columns],'target',iv_only=True)

train_df_explore.to_excel(writer,sheet_name='train_df_explore')
# valid_df_explore.to_excel(writer,sheet_name='valid_df_explore')
train_df_iv.to_excel(writer,sheet_name='train_df_iv')
# valid_df_iv.to_excel(writer,sheet_name='valid_df_iv')

writer.save()


# In[429]:


df_iv = toad.quality(train_selected,'target',iv_only=True)
df_iv.head(5)


# In[430]:


df_iv['iv'].describe()


# # 4. 变量分箱

# In[533]:


# 第一次分箱
c = toad.transform.Combiner()
c.fit(train_selected, y='target', method='dt', min_samples=2, n_bins=10, empty_separate=True) 
bins_result = c.export()


# In[552]:


train_selected_bins = c.transform(train_selected, labels=True)
train_selected_bins.head(2)


# In[553]:


# valid_selected = valid[train_selected.columns]
# valid_selected_bins = c.transform(valid_selected, labels=True)
# valid_selected_bins.head(2)

oot_selected = oot[train_selected.columns]
oot_selected_bins = c.transform(oot_selected, labels=True)
oot_selected_bins.head(2)


# In[554]:


df_result = pd.DataFrame()
for col in train_selected_bins.columns[1:]:
#     print('------------变量：{}-----------'.format(col))
    tmp = regroup(train_selected_bins, col, target='target')
    df_result = pd.concat([df_result, tmp], axis=0)


# In[555]:


# df_result_valid = pd.DataFrame()
# for col in valid_selected_bins.columns[1:]:
# #     print('------------变量：{}-----------'.format(col))
#     tmp = regroup(valid_selected_bins, col, target='target')
#     df_result_valid = pd.concat([df_result_valid, tmp], axis=0)
    
df_result_valid = pd.DataFrame()
for col in oot_selected_bins.columns[1:]:
#     print('------------变量：{}-----------'.format(col))
    tmp = regroup(oot_selected_bins, col, target='target')
    df_result_valid = pd.concat([df_result_valid, tmp], axis=0)


# In[556]:


# tmp = pd.merge(df_result, df_result_valid, how='left', on=['bins', 'varsname'], suffixes=['_train', '_valid'])

tmp = pd.merge(df_result, df_result_valid, how='left', on=['bins', 'varsname'], suffixes=['_train', '_oot'])


# In[ ]:





# In[557]:


tmp.to_excel(r'D:\liuyedao\B卡开发\Lr_result\B卡_变量分箱_dt_10bins_58同城_20230922_v7.xlsx',index=False)


# In[ ]:


psi = toad.metrics.PSI(train_selected_bins.drop('target',axis=1), oot_selected_bins.drop('target',axis=1))


# In[514]:


# psi = toad.metrics.PSI(train_selected_bins.drop('target',axis=1), oot_selected_bins.drop('target',axis=1))
# psi = psi[psi<0.25]
# psi
# # to_keep = list(psi.index)
# print(len(psi))


# In[515]:


# 自动调整分箱
adj_bins = {}
for col in list(bins_result.keys()):
    tmp = np.isnan(bins_result[col])
    cutbins = [bins_result[col][x] for x in range(len(tmp)) if not tmp[x]]
    if len(cutbins)>0:
        cutbins = [float('-inf')] + cutbins + [float('inf')]
        cutoffpoints = ContinueVarBins(train_selected, col, flag='target', cutbins=cutbins)
        tmp_value1 = train_selected[train_selected[col]>=0][col].min()
        tmp_value2 = cutoffpoints[0]
        if tmp_value1==tmp_value2:
            adj_bins[col] = [-0.5] + cutoffpoints[1:]
        else:
            adj_bins[col] = [-0.5] + cutoffpoints


# In[550]:


adj_bins = {
    'value_018_bairong_15':[-0.5, 815.5],
    'value_016_bairong_15':[-0.5, 3.0465],
    'value_014_bairong_15':[-0.5, 8.58],
    'ppdi_m3_id_nbank_loan_orgnum':[-0.5, 3.5, 5.5],
    'ppdi_m12_id_nbank_nloan_orgnum':[-0.5, 3.5, 5.5],
    'ppdi_m12_id_nbank_nloan_allnum':[-0.5, 6.5, 15.5, 25.5],
    'ppdi_m12_id_nbank_fin_orgnum':[-0.5, 3.5, 4.5, 5.5],
    'ppdi_m12_id_nbank_cons_orgnum':[-0.5, 0.5, 2.5, 3.5],
    'ppdi_m12_id_nbank_cons_allnum':[-0.5, 3.5, 7.5, 11.5],
    'ppdi_m1_id_nbank_nloan_allnum':[-0.5, 3.5, 4.5],
    'ppdi_m1_id_nbank_cons_allnum':[-0.5, 2.5],
    'ppdi_m1_cell_nbank_fin_allnum':[-0.5, 2.5, 3.5],
    'model_score_01_q_tianchuang_1':[-0.5, 448.5, 596.5, 710.5],
    'model_score_01_hengpu_4':[-0.5, 33.5, 44.5, 54.5],
    'model_score_01_fulin_1':[-0.5, 693.5],
    'model_score_01_baihang_1':[-0.5, 740.5, 808.0],
    'model_score_01_2_xinyongsuanli_1':[-0.5, 0.024117, 0.125772],
    'model_score_01_107_hulian_5':[-0.5, 812.5, 829.5]    
}


# In[551]:


# 调整分箱:空值单独一箱
train_selected.fillna(-9999, inplace=True)
# adj_bins = {}
# for col in list(bins_result.keys()):
#     tmp = np.isnan(bins_result[col])
#     cutbins = [bins_result[col][x] for x in range(len(tmp)) if not tmp[x]]
#     if len(cutbins)>0:
#         cutbins = [float('-inf')] + cutbins + [float('inf')]
#         cutoffpoints = ContinueVarBins(train_selected, col, flag='target', cutbins=cutbins)
#         adj_bins[col] = cutoffpoints
# 更新分箱
c.update(adj_bins)


# In[ ]:


# c.export()
# c.load(dict)
# c.transform(dataframe, labels=False)


# # 5.WOE转换

# In[51]:


# train_selected.drop(['model_score_01_zr_tongdun_2'], axis=1, inplace=True)


# In[558]:


exclude = ['target']


# In[576]:


transer = toad.transform.WOETransformer()
train_woe = transer.fit_transform(c.transform(train_selected), train_selected['target'], exclude=exclude)
train_woe.shape


# In[603]:


cols


# In[608]:


transer = toad.transform.WOETransformer()
train_woe = transer.fit_transform(c.transform(train[cols+['target']]), train['target'], exclude=exclude)
valid_woe = transer.transform(c.transform(valid[cols+['target']].fillna(-9999)))

print(train_woe.shape,oot_woe.shape)


# In[577]:


oot_woe = transer.transform(c.transform(oot_selected.fillna(-9999)))
oot_woe.shape


# # 6.逐步回归

# In[560]:


train_selected_woe, dropped = toad.selection.select(train_woe, target='target', empty=0.90, iv=0.0, corr=0.8, return_drop=True, exclude=None)
train_selected_woe.shape


# In[561]:


for i in dropped.keys():
    print("变量筛选维度：{}， 共剔除变量{}".format(i, len(dropped[i])))


# In[571]:


toad.quality(train_woe,'target',iv_only=True)


# In[563]:


iv_df = toad.quality(train_selected_woe,'target',iv_only=True)
iv_df


# In[25]:


# to_drop = list(iv_df[iv_df.iv>=0.5].index)


# In[57]:


# droped_list = CorrSelect(train_selected_woe, iv_df, exclude_list=['target'], threshold=0.7)
# print(droped_list)


# In[564]:


oot_woe = transer.transform(c.transform(oot_selected.fillna(-9999)))
final_oot = oot_woe[list(train_selected_woe.columns)]
final_oot.shape


# In[573]:


toad.quality(final_oot,'target',iv_only=True)


# In[565]:


psi = toad.metrics.PSI(train_selected_woe, final_oot) 


# In[566]:


print(len(psi))
psi


# In[450]:


to_drop = list(psi[psi>0.10].index)


# In[451]:


len(to_drop)


# In[250]:


# cols = ['model_score_01_2_xinyongsuanli_1','als_d7_id_nbank_else_orgnum','als_m3_id_coon_orgnum',
#  'model_score_01_tianchuang_7','model_score_01_hangliezhi_1','model_score_01_124_hulian_5']
final_data = train_woe[cols+['target']]
final_data.shape


# In[452]:


# final_data = toad.selection.stepwise(train_woe[cols+['target']], target='target', estimator='ols', direction='both', criterion='aic', exclude=None)
# final_data.shape

final_data = toad.selection.stepwise(train_selected_woe.drop(to_drop,axis=1), target='target', estimator='ols', direction='both', criterion='aic', exclude=None)
final_data.shape


# In[567]:


final_data = toad.selection.stepwise(train_selected_woe, target='target', estimator='ols', direction='both', criterion='aic', exclude=None)
final_data.shape


# In[59]:


# valid_woe = transer.transform(c.transform(valid.fillna(-9999)))
# final_valid = valid_woe[list(final_data.columns)]
# final_valid.shape


# In[568]:


oot_woe = transer.transform(c.transform(oot.fillna(-9999)))
final_oot = oot_woe[list(final_data.columns)]
final_oot.shape


# In[569]:


# 初次建模变量
cols = list(final_data.drop(['target'], axis=1).columns)
print(cols)


# In[570]:


print(toad.metrics.PSI(final_data[cols], final_oot[cols]))


# # 7.建模和评估

# In[574]:


from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[211]:


# cols.remove('model_score_01_baihang_1')
# cols.remove('value_020_bairong_15') 
# cols.remove('model_score_01_q_tianchuang_1') 
# cols.remove('als_m3_cell_nbank_else_orgnum')
cols.remove('als_d7_id_nbank_else_orgnum')
# als_m3_id_coon_orgnum
# cols.remove('model_score_01_ruizhi_6') 
# cols.remove('value_052_baihang_6') 
# cols.remove('value_027_bairong_14') 


# In[609]:


clf = LR_model(train_woe.drop('target', axis=1), valid_woe.drop('target', axis=1), train_woe['target'], valid_woe['target'])


# In[579]:


clf = LR_model(final_data[cols], final_oot[cols], final_data['target'], final_oot['target'])


# In[580]:


vif=pd.DataFrame()
X = np.matrix(final_data[cols])
vif['features']=cols
vif['VIF_Factor']=[variance_inflation_factor(np.matrix(X),i) for i in range(X.shape[1])]
vif


# In[581]:


corr = final_data[cols].corr()
corr


# In[215]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\Lr_result\B卡_模型相关性_20230919"+'.xlsx')

corr.to_excel(writer,sheet_name='corr')
vif.to_excel(writer,sheet_name='vif')

writer.save()


# In[582]:


# 训练集
pred_train = clf.predict(sm.add_constant(final_data[cols]))
# KS/AUC
from toad.metrics import KS,AUC

print('train AUC: ', AUC(pred_train, final_data['target']))
print('train KS: ', KS(pred_train, final_data['target']))


# In[105]:


# # 验证集
# pred_valid = clf.predict(sm.add_constant(final_valid[cols]))
# # KS/AUC
# from toad.metrics import KS,AUC

# print('-------------valid结果--------------------')

# print('test AUC: ', AUC(pred_valid, final_valid['target']))
# print('test KS: ', KS(pred_valid, final_valid['target']))



# # oot测试

# In[69]:


oot = pd.read_csv(r'D:\liuyedao\B卡开发\mid_result\b卡_建模数据集_oot_20230915.csv')


# In[70]:


oot_woe = transer.transform(c.transform(oot.fillna(-9999)))
final_oot = oot_woe[cols]
final_oot.shape


# In[583]:


# oot测试
pred_oot = clf.predict(sm.add_constant(final_oot[cols]))

# 模型评估指标
AUC = roc_auc_score(oot['target'], pred_oot)
print('AUC:', AUC)
fpr, tpr, thresholds = roc_curve(oot['target'], pred_oot)
KS = max(tpr-fpr)
print('KS:', KS)

# print('oot KS: ', KS(pred_oot, oot['target']))
# print('oot AUC: ', AUC(pred_oot, oot['target']))


# # 9.转换评分

# In[584]:


card = toad.ScoreCard(combiner=c,
                      transer = transer,
                      base_odds=35,
                      base_score=700,
                      pdo=60,
                      rate=2,
                      C=1e8
                     )


# In[585]:


card.fit(final_data[cols], final_data['target'])


# In[586]:


card1 = card.export(to_frame=True).round(0)
card1


# In[222]:


card1.to_excel(r'D:\liuyedao\B卡开发\Lr_result\B卡_评分卡_58同城_20230922_v2.xlsx')


# In[223]:


name={'model_score_01_2_xinyongsuanli_1': '信用算力-万象分'
,'model_score_01_2_bairong_8': '客制化-信用风险识别-线上消费分期-桔子分期机器学习'
,'model_score_01_r_tianchuang_1': '天创信用r分'
,'model_score_01_tianchuang_7': '天创信用联合分'
,'model_score_01_ruizhi_6': 'fico联合建模'
,'model_score_01_hangliezhi_1': '行列秩评分'
,'model_score_01_124_hulian_5': '互联评分'
}


# In[224]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\Lr_result\B卡_变量箱子_分数_v2.xlsx")
train_selected['score'] = card.predict(train_selected[cols]).round(0)
train_selected_bins = c.transform(train_selected, labels=True)
 

for col in cols:
    tmp1 = CalWoeIv(train_selected_bins, col, target='target')
    tmp2 = train_selected_bins[[col, 'score']].drop_duplicates(col,keep='first')
    tmp = pd.merge(tmp1, tmp2, how='left',left_on='bins', right_on=col)
    tmp['name'] = name[col]
    tmp = tmp[['varsname','name','bins','total','total_pct','bad_rate','woe']]
    tmp.to_excel(writer,sheet_name=col)
    writer.save()


# In[225]:


print(len(cols), cols)


# In[226]:


train['score'] = card.predict(train[cols]).round(0)
train['prob'] = card.predict_proba(train[cols])[:,1]


# In[227]:


oot['score'] = card.predict(oot[cols]).round(0)
oot['prob'] = card.predict_proba(oot[cols])[:,1]


# In[229]:


# tmp1 = train[['order_no','lending_time','channel_id','score','prob']+cols]
# tmp2 = oot[['order_no','lending_time','channel_id','score','prob']+cols]
# tmp = pd.concat([tmp1, tmp2], axis=0)
# tmp.info()
# tmp.head()
# tmp.to_csv(r'D:\liuyedao\B卡开发\Lr_result\score_lr_部署测试样例_58同城.csv',index=False)


# # 10.模型业务效果

# In[594]:


# 业务效果-训练集
train['score'] = card.predict(train[cols]).round(0)
pred_data = toad.metrics.KS_bucket(train['score'], train['target'], bucket=5, method='quantile')
pred_data


# In[595]:


cut_bins = [float('-inf')]+list(pred_data['min'])[1:]+[float('inf')]
print(cut_bins)


# In[596]:


# 业务效果-训练集
train['score'] = card.predict(train[cols]).round(0)
train['bins'] = pd.cut(train['score'], bins=cut_bins, include_lowest=True, right=False)


# In[87]:


# # 业务效果-验证集
# valid['score'] = card.predict(valid[cols]).round(0)
# valid['bins'] = pd.cut(valid['score'], bins=cut_bins, include_lowest=True, right=False)


# In[597]:


# 业务效果-oot
oot['score'] = card.predict(oot[cols]).round(0)
oot['bins'] = pd.cut(oot['score'], bins=cut_bins, include_lowest=True, right=False)


# In[598]:


pred_data_oot = score_distribute(oot, 'bins', target='target')
pred_data_oot


# In[236]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\Lr_result\B卡_分数分布_step_lr_58同城.xlsx")
pred_data_train = score_distribute(train, 'bins', target='target')
# pred_data_valid = score_distribute(valid, 'bins', target='target')
pred_data_oot = score_distribute(oot, 'bins', target='target')


pred_data_train.to_excel(writer,sheet_name='pred_data_train')
# pred_data_valid.to_excel(writer,sheet_name='pred_data_valid')
pred_data_oot.to_excel(writer,sheet_name='pred_data_oot')

writer.save()


# In[241]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\Lr_result\B卡_分数分布_quantile_lr.xlsx")
pred_data_train = score_distribute(train, 'bins', target='target')
# pred_data_valid = score_distribute(valid, 'bins', target='target')
pred_data_oot = score_distribute(oot, 'bins', target='target')


pred_data_train.to_excel(writer,sheet_name='pred_data_train')
# pred_data_valid.to_excel(writer,sheet_name='pred_data_valid')
pred_data_oot.to_excel(writer,sheet_name='pred_data_oot')

writer.save()


# In[591]:


def cal_psi(exp, act):
    psi = []
    for i in range(len(exp)):
        psi_i = (act[i] - exp[i])*np.log(act[i]/exp[i])
        psi.append(psi_i)
    return sum(psi)


# In[592]:


# print(cal_psi(pred_data_train['total_pct'], pred_data_valid['total_pct'])) 
print(cal_psi(pred_data_train['total_pct'], pred_data_oot['total_pct'])) 


# # oot验证



#==============================================================================
# File: B卡建模-Lr_Xgb-174渠道.py
#==============================================================================

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import toad
import os 
from datetime import datetime
from sklearn.model_selection import train_test_split
from IPython.core.interactiveshell import InteractiveShell
import warnings
import gc
from statistics import mode
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, cross_validate, cross_val_score
import time
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from pypmml import Model

warnings.filterwarnings("ignore")
InteractiveShell.ast_node_interactivity = 'all'
pd.set_option('display.max_row',None)
pd.set_option('display.width',1000)


# In[2]:


os.getcwd()


# In[3]:


# 运行函数脚本
get_ipython().run_line_magic('run', 'function.ipynb')


# # 1.读取数据集

# In[4]:


filepath = r'D:\liuyedao\B卡开发\三方数据匹配'
data = pd.read_csv(filepath + r'\order_20230927.csv') 


# In[ ]:


usecols = ['order_no','user_id','channel_id','apply_date',
    'target'
    ,'value_026_bairong_12'
,'als_m12_cell_nbank_max_monnum'
,'model_score_01_moxingfen_7'
,'model_score_01_rong360_4'
,'model_score_01_tianchuang_8'
,'value_031_bairong_12'
,'value_027_bairong_12'
,'als_m3_id_nbank_night_orgnum'
,'value_040_baihang_6'
,'als_m6_id_nbank_max_inteday'
]


# In[524]:


df_base = data.query("channel_id==174 & apply_date<'2023-07-01' & apply_date>='2022-10-01'")
# 删除全部是空值的列
df_base.dropna(how='all', axis=1, inplace=True)
df_base.dropna(how='all', axis=1, inplace=True)
df_base.reset_index(drop=True, inplace=True)
df_base.info()
df_base.head()


# In[525]:


to_drop = list(df_base.select_dtypes(include='object').columns)
print(to_drop)


# In[526]:


df_base.drop(['operationType', 'swift_number', 'name', 'mobileEncrypt', 'orderNo', 'idCardEncrypt'],axis=1,inplace=True)


# In[527]:


# 小于0为异常值，转为空值
for col in df_base.select_dtypes(include='float64').columns[1:]:
    df_base[col] = df_base[col].mask(df_base[col]<0)


# In[528]:


df_behavior = pd.read_csv(r'D:\liuyedao\B卡开发\lr_xgb_174\b卡衍生变量_20231013.csv')
df_behavior.head()


# In[529]:


tmp_df_behavior = toad.detect(df_behavior)
tmp_df_behavior.to_excel(r'D:\liuyedao\B卡开发\lr_xgb_174\tmp_df_behavior.xlsx')


# In[530]:


df_base = pd.merge(df_base, df_behavior, how='left', on='user_id')
print(df_base.shape, df_base.order_no.nunique())


# In[531]:


df_pudao_3 = pd.read_csv(r'D:\liuyedao\B卡开发\lr_xgb_174\order_pudao_3_diff_20231013.csv')
df_bairong_1 = pd.read_csv(r'D:\liuyedao\B卡开发\lr_xgb_174\order_bairong_1_diff_20231013.csv')


# In[532]:


tmp_df_pudao_3 = toad.detect(df_pudao_3)
tmp_df_pudao_3.to_excel(r'D:\liuyedao\B卡开发\lr_xgb_174\tmp_df_pudao_3.xlsx')


# In[533]:


tmp_df_bairong_1 = toad.detect(df_bairong_1)
tmp_df_bairong_1.to_excel(r'D:\liuyedao\B卡开发\lr_xgb_174\tmp_df_bairong_1.xlsx')


# In[534]:


df_base = pd.merge(df_base, df_pudao_3, how='left', on='order_no')
print(df_base.shape, df_base.order_no.nunique())


# In[535]:


df_base = pd.merge(df_base, df_bairong_1, how='left', on='order_no')
print(df_base.shape, df_base.order_no.nunique())


# ## 训练数据集

# In[665]:


df_train = df_base.query("apply_date>='2022-10-01' & apply_date<'2023-05-01' & target in [0.0, 1.0]")
# df_train.groupby(['lending_month','target'])['order_no'].count().unstack()


# In[666]:


print(df_train.target.value_counts())
print(df_train.target.value_counts(normalize=True))


# ## oot测试数据集

# In[667]:


df_oot = df_base.query("apply_date>='2023-05-01' & apply_date<'2023-06-01' & target in [0.0, 1.0]")
print(df_oot.target.value_counts())
print(df_oot.target.value_counts(normalize=True))


# In[668]:


print(df_train.shape, df_oot.shape)


# # 2.数据处理

# In[705]:


train = df_train.copy()
oot = df_oot.copy()

# oot = df_train.copy()
# train = df_oot.copy()


# In[706]:


# to_drop = ['order_no','user_id','channel_id','apply_date']
to_drop = train.columns[0:6].to_list()
print(to_drop)


# In[707]:


train = train.drop(to_drop, axis=1)
oot = oot.drop(to_drop, axis=1)


# In[708]:


to_drop = list(train.select_dtypes(include='object').columns)
print(to_drop)


# In[ ]:


# train = train.drop(to_drop, axis=1)
# oot = oot.drop(to_drop, axis=1)


# In[ ]:


# to_drop = list(train.columns[train.columns.str.contains('score')])
# print(to_drop)


# In[ ]:


# train = train.drop(to_drop, axis=1)
# oot = oot.drop(to_drop, axis=1)


# In[ ]:


train.info()


# In[ ]:


oot.info()


# In[ ]:


print(train.target.value_counts())
print(oot.target.value_counts())


# In[ ]:


print(train.target.value_counts(normalize=True))
print(oot.target.value_counts(normalize=True))


# # 3.数据探索分析

# In[ ]:


# # 小于0为异常值，转为空值
# train = train.mask(train<0)
# oot = oot.mask(oot<0)


# In[709]:


# 删除全部是空值的列
train.dropna(how='all', axis=1, inplace=True)
oot.dropna(how='all', axis=1, inplace=True)


# In[698]:


train_explore = toad.detect(train.drop('target', axis=1))
oot_explore = toad.detect(oot.drop('target', axis=1))

modes = train.drop('target', axis=1).apply(mode)
mode_counts = train.drop('target', axis=1).eq(modes).sum()
train_modes = pd.DataFrame({'mode':modes, 'mode_num':mode_counts}, index=list(train.columns[1:]))

modes = oot.drop('target', axis=1).apply(mode)
mode_counts = oot.drop('target', axis=1).eq(modes).sum()
oot_modes = pd.DataFrame({'mode':modes, 'mode_num':mode_counts}, index=list(oot.columns[1:]))

train_isna = pd.DataFrame(train.drop('target', axis=1).isnull().sum(), columns=['missing_num'])
oot_isna = pd.DataFrame(oot.drop('target', axis=1).isnull().sum(), columns=['missing_num'])


train_iv = toad.quality(train,'target',iv_only=False)
oot_iv = toad.quality(oot,'target',iv_only=False)


# In[699]:


train_df_explore = pd.concat([train_explore, train_modes, train_isna, train_iv.drop('unique',axis=1)], axis=1)

train_df_explore['no_null_num'] = train_df_explore['size'] - train_df_explore['missing_num']
train_df_explore['miss_rate'] = train_df_explore['missing_num'] / train_df_explore['size']
train_df_explore['mode_pct_all'] = train_df_explore['mode_num']/train_df_explore['size']
train_df_explore['mode_pct_notna'] = train_df_explore['mode_num']/train_df_explore['no_null_num']


# In[700]:


oot_df_explore = pd.concat([oot_explore, oot_modes, oot_isna, oot_iv.drop('unique',axis=1)], axis=1)

oot_df_explore['no_null_num'] = oot_df_explore['size'] - oot_df_explore['missing_num']
oot_df_explore['miss_rate'] = oot_df_explore['missing_num'] / oot_df_explore['size']
oot_df_explore['mode_pct_all'] = oot_df_explore['mode_num']/oot_df_explore['size']
oot_df_explore['mode_pct_notna'] = oot_df_explore['mode_num']/oot_df_explore['no_null_num']


# In[701]:


df_iv = pd.merge(train_iv, oot_iv, how='inner',left_index=True, right_index=True,suffixes=['_train','_oot'])
df_iv['diff_iv'] = df_iv['iv_oot']-df_iv['iv_train']
df_iv['rate_iv'] = df_iv['iv_oot']/df_iv['iv_train'] - 1


# In[611]:


path = r'D:\liuyedao\B卡开发\lr_xgb_174\231020\\'


# In[612]:


writer=pd.ExcelWriter(path + 'B卡_探索性分析_58同城_20231018.xlsx')

train_df_explore.to_excel(writer,sheet_name='train_df_explore')
oot_df_explore.to_excel(writer,sheet_name='oot_df_explore')

# train_df_iv.to_excel(writer,sheet_name='train_df_iv')
# oot_df_iv.to_excel(writer,sheet_name='oot_df_iv')

df_iv.to_excel(writer,sheet_name='df_iv')

writer.save()


# # 4.特征粗筛选

# In[752]:


# 删除缺失率大于0.85/删除枚举值只有一个/删除方差等于0/删除集中度大于0.85

to_drop_missing = list(train_df_explore[train_df_explore.miss_rate>=0.85].index)
print(len(to_drop_missing))
to_drop_unique = list(train_df_explore[train_df_explore.unique==1].index)
print(len(to_drop_unique))
to_drop_std = list(train_df_explore[train_df_explore.std_or_top2==0].index)
print(len(to_drop_std))
to_drop_mode = list(train_df_explore[train_df_explore.mode_pct_notna>=0.85].index)
print(len(to_drop_mode))
to_drop_mode2 = list(train_df_explore[train_df_explore.mode_pct_all>=0.85].index)
print(len(to_drop_mode2))

to_drop_train = list(set(to_drop_missing+to_drop_unique+to_drop_std+to_drop_mode+to_drop_mode2))
print(len(to_drop_train))


# In[753]:


# 删除缺失率大于0.85/删除枚举值只有一个/删除方差等于0/删除集中度大于0.85
to_drop_missing = list(oot_df_explore[oot_df_explore.miss_rate>=0.85].index)
print(len(to_drop_missing))
to_drop_unique = list(oot_df_explore[oot_df_explore.unique==1].index)
print(len(to_drop_unique))
to_drop_std = list(oot_df_explore[oot_df_explore.std_or_top2==0].index)
print(len(to_drop_std))
to_drop_mode = list(oot_df_explore[oot_df_explore.mode_pct_notna>=0.85].index)
print(len(to_drop_mode))
to_drop_mode2 = list(oot_df_explore[oot_df_explore.mode_pct_all>=0.85].index)
print(len(to_drop_mode))

to_drop_oot = list(set(to_drop_missing+to_drop_unique+to_drop_std+to_drop_mode+to_drop_mode2))
print(len(to_drop_oot))


# In[754]:


train_1 = train.drop(to_drop_train, axis=1)
print(train_1.shape)

oot_1 = oot.drop(to_drop_oot, axis=1)
print(oot_1.shape)


# In[755]:


# 共同的变量
sim_cols = list(set(train_1.drop('target',axis=1).columns).intersection(set(oot_1.drop('target',axis=1).columns)))
print(len(sim_cols))

train_2 = train_1[['target']+sim_cols]
oot_2 = oot_1[['target']+sim_cols]
print(train_2.shape, oot_2.shape)


# In[507]:


# to_exclude = ['value_026_bairong_12'
# ,'als_m12_cell_nbank_max_monnum'
# ,'model_score_01_moxingfen_7'
# ,'model_score_01_rong360_4'
# ,'model_score_01_tianchuang_8'
# ,'value_031_bairong_12'
# ,'value_027_bairong_12'
# ,'als_m3_id_nbank_night_orgnum'
# ,'value_040_baihang_6'
# ,'als_m6_id_nbank_max_inteday'
# ]


# In[756]:


# iv值/相关性筛选
train_selected, dropped = toad.selection.select(train_2, target='target', empty=0.7, iv=0.02,
                                                corr=0.7,
                                                return_drop=True, exclude=None)
train_selected.shape


# In[757]:


for i in dropped.keys():
    print("变量筛选维度：{}， 共剔除变量{}".format(i, len(dropped[i])))


# In[758]:


# 人工剔除
to_drop_man = ['model_score_01_baihang_1','model_score_01_q_tianchuang_1','model_score_01_r_tianchuang_1',
           'model_score_01_1_bileizhen_1',
           'model_score_01_2_bileizhen_1','model_score_01_3_bileizhen_1','value_011_moxingfen_7',
           'value_012_pudao_5',
           'model_score_01_pudao_11','model_score_01_pudao_12','model_score_01_pudao_16',
           'model_score_01_2_bairong_8','model_score_01_3_bairong_8','model_score_01_7_bairong_8',
           'model_score_01_8_bairong_8']

to_drop = []
for col in to_drop_man:
    if col in train_selected.columns:
        to_drop.append(col)
print(len(to_drop))

train_selected.drop(to_drop, axis=1, inplace=True)
print(train_selected.shape)


# ## 变量分箱

# In[759]:


# 第一次分箱
c = toad.transform.Combiner()
c.fit(train_selected, y='target', method='chi', min_samples=0.05, n_bins=10, empty_separate=True) 
bins_result = c.export()


# In[760]:


bins_result


# In[761]:


train_selected_bins = c.transform(train_selected, labels=True)
oot_selected_bins = c.transform(oot[train_selected.columns], labels=True)


# In[762]:


bins_dict_train = {}
for col in train_selected_bins.columns[1:]:
    bins_dict_train[col] = regroup(train_selected_bins, col, target='target')
    
df_result_train = pd.concat(list(bins_dict_train.values()), axis=0, ignore_index =True)


# In[763]:


bins_dict_oot = {}
for col in oot_selected_bins.columns[1:]:
    bins_dict_oot[col] = regroup(oot_selected_bins, col, target='target')
    
df_result_oot = pd.concat(list(bins_dict_oot.values()), axis=0, ignore_index =True)


# In[764]:


df_result = pd.merge(df_result_train, df_result_oot, how='inner', on=['varsname','bins'],suffixes=['_train','_oot'])


# In[728]:


df_result.to_excel(path + 'B卡_chi_58同城_20231020_v2.xlsx',index=False)


# In[765]:


to_drop_bins1 = []
for col in list(bins_result.keys()):
    if len(bins_result[col])==1:
        to_drop_bins1.append(col)
print(len(to_drop_bins1))
to_drop_bins2 = list(set(df_result[(df_result['iv_train']<0.02)]['varsname']))
print(len(to_drop_bins2))
to_drop_bins = list(set(to_drop_bins1 + to_drop_bins2))
print(len(to_drop_bins))


# In[766]:


train_selected_1 = train_selected.drop(to_drop_bins, axis=1)
train_selected_1.shape


# In[732]:


train_selected_bins_1 = train_selected_bins[train_selected_1.columns]
oot_selected_bins_1 = oot_selected_bins[train_selected_bins_1.columns]


# In[632]:


from toad.plot import bin_plot, badrate_plot


# In[635]:


# col = list(train_selected_bins_1.drop('target',axis=1).columns)[1]

# bin_plot(train_selected_bins_1, x=col, target='target')
# bin_plot(oot_selected_bins_1, x=col, target='target')


# In[767]:


# 连续变量分箱主函数
def ContinueVarBins(data, col, flag='target', cutbins=[]):
    df = data[[flag, col]]
    df = df[~df[col].isnull()].reset_index(drop=True)
    df['bins'] = pd.cut(df[col], cutbins, duplicates='drop', right=False, precision=4)
    regroup = pd.DataFrame()
    regroup['bins'] = df.groupby(['bins'])[col].max()
    regroup['total'] = df.groupby(['bins'])[flag].count()
    regroup['bad'] = df.groupby(['bins'])[flag].sum()
    regroup['good'] = regroup['total'] - regroup['bad']
    regroup.drop(['total'], axis=1, inplace=True)
    np_regroup = np.array(regroup)
    np_regroup = MergeZero(np_regroup)
    np_regroup = MinPct(np_regroup)
    np_regroup = MonTone(np_regroup)
    regroup = pd.DataFrame(index=np.arange(np_regroup.shape[0]))
    regroup['bins'] = np_regroup[:,0]
    regroup['total'] = np_regroup[:,1] + np_regroup[:,2]
    regroup['bad'] = np_regroup[:,1]
    regroup['good'] = np_regroup[:,2]
    cutoffpoints = list(np_regroup[:,0])
    cutoffpoints = [float('-inf')] + cutoffpoints
    # 最大值分割点，转换最小值分割点
    df['bins_new'] = pd.cut(df[col], cutoffpoints, duplicates='drop', right=True, precision=4)
    tmp = pd.DataFrame()
    tmp['bins'] = df.groupby(['bins_new'])[col].min()
    cutoffpoints = list(tmp['bins'])  
    
    return cutoffpoints


# In[768]:


# 自动调整分箱
adj_bins = {}
for col in list(train_selected_1.columns)[1:]:
    tmp = np.isnan(bins_result[col])
    cutbins = [bins_result[col][x] for x in range(len(tmp)) if not tmp[x]]
    if len(cutbins)>0:
        cutbins = [float('-inf')] + cutbins + [float('inf')]
        cutoffpoints = ContinueVarBins(train_selected, col, flag='target', cutbins=cutbins)
        tmp_value1 = train_selected[col].min()
        tmp_value2 = cutoffpoints[0]
        if tmp_value1==tmp_value2:
            adj_bins[col] = [-99998] + cutoffpoints[1:]
        else:
            adj_bins[col] = [-99998] + cutoffpoints


# In[769]:


# 调整分箱:空值单独一箱
train_selected_1.fillna(-99999, inplace=True)

# 更新分箱
c.update(adj_bins)


# In[ ]:


# c.export()
# c.load(dict)
# c.transform(dataframe, labels=False)


# ## WOE转换

# In[770]:


transer = toad.transform.WOETransformer()
train_woe = transer.fit_transform(c.transform(train_selected_1), train_selected_1['target'],
                                  exclude=['target'])
train_woe.shape


# In[771]:


train_selected_woe, dropped = toad.selection.select(train_woe, target='target', empty=0.70,
                                                    iv=0.02, corr=0.6, return_drop=True, exclude=None)
train_selected_woe.shape


# In[772]:


for i in dropped.keys():
    print("变量筛选维度：{}， 共剔除变量{}".format(i, len(dropped[i])))


# In[773]:


oot_woe = transer.transform(c.transform(oot))
oot_selected_woe = oot_woe[list(train_selected_woe.columns)]
oot_selected_woe.shape


# In[779]:


psi = toad.metrics.PSI(train_selected_woe.drop('target',axis=1), oot_selected_woe.drop('target',axis=1))
to_drop_psi = list(psi[psi>0.25].index)
print(len(to_drop_psi))


# In[775]:


train_selected_woe_psi = train_selected_woe.drop(to_drop_psi, axis=1)
oot_selected_woe_psi = oot_selected_woe[train_selected_woe_psi.columns]


# ## 逐步回归

# In[780]:


print(train_selected_woe.shape)
final_data = toad.selection.stepwise(train_selected_woe, target='target', estimator='ols',
                                     direction='both', criterion='aic', exclude=None)
print(final_data.shape)


# In[777]:


final_oot = oot_selected_woe_psi[final_data.columns]


# In[778]:


# 初次建模变量
cols = list(final_data.drop(['target'], axis=1).columns)
print(len(cols))
print(cols)


# # 5.xgb模型训练

# ## 5.1 Xgb建模

# In[ ]:


# cols.remove('model_score_01_rong360_4')
# cols.append('model_score_01_tianchuang_7')
# cols.append('model_score_01_rong360_4')


# In[ ]:


opt_best = {'target': 0.6487565005430443,
            'params': {'gamma': 0.75, 'learning_rate': 0.45, 'max_depth': 6.0, 'min_child_weight': 9.0,
                       'n_estimators': 9, 'reg_alpha': 0, 'reg_lambda': 300}
           }


# In[684]:


opt_best = {'target': 0.6487565005430443,
            'params': {'gamma': 0.75, 'learning_rate': 0.45, 'max_depth': 6.0, 'min_child_weight': 9.0,
                       'n_estimators': 10, 'reg_alpha': 0, 'reg_lambda': 290}
           }


# In[ ]:





# In[685]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 


# In[657]:


len(cols)


# In[686]:


cols = ['als_m12_cell_nbank_max_monnum_diff'
,'als_m3_id_nbank_min_monnum_diff'
,'model_score_01_tianchuang_8'
,'ppdi_m1_id_nbank_orgnum_diff'
,'value_026_bairong_12'
,'als_m6_id_nbank_max_inteday'
,'model_score_01_rong360_4'
,'als_m12_cell_nbank_max_monnum'
,'value_031_bairong_12'
,'value_027_bairong_12'
,'model_score_01_moxingfen_7'
]


# In[746]:


xgb_model.fit(train[cols] ,train['target'])


# In[747]:


# 对训练集进行预测

from sklearn import metrics

y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[748]:


xgb_model.fit(oot[cols] ,oot['target'])


# In[749]:


# 对训练集进行预测

from sklearn import metrics

y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[74]:


len(xgb_model.get_booster().get_score(importance_type='gain').keys())


# In[689]:


importance_type = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),orient='index')
importance_type.columns = ['value']
importance_type = importance_type.sort_values('value', ascending=False)
importance_type


# In[ ]:


cols = list(importance_type.head(11).index)
# cols = list(importance_type[importance_type.value>1.7].index)
print(cols, len(cols))


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
# xgb_model.fit(X_train, y_train)
xgb_model.fit(train[cols] ,train['target'])


# In[ ]:


# 对训练集进行预测
y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# # 对测试集进行预测
# y_pred_test = xgb_model.predict_proba(valid[cols])[:,1]
# fpr_test, tpr_test, thresholds_oot = metrics.roc_curve(valid['target'], y_pred_test, pos_label=1)
# roc_auc_test = metrics.auc(fpr_test, tpr_test)
# ks_test = max(tpr_test-fpr_test)
# print('test AUC: ', roc_auc_test)
# print('test KS: ', ks_test)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[ ]:


importance_type = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),orient='index')
importance_type.columns = ['value']
importance_type = importance_type.sort_values('value', ascending=False)
importance_type


# In[ ]:


import pickle

# 保存模型
pickle.dump(xgb_model, open(r"D:\liuyedao\B卡开发\lr_xgb_174_20231017.pkl", "wb"))


# In[ ]:


plt.plot(fpr, tpr, color='darkorange', lw=2, label='trian ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr_oot, tpr_oot, color='blue', lw=2, label='test ROC curve (area = %0.2f)' % roc_auc_oot)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="best")
plt.show()


# ### 贝叶斯调参

# In[ ]:


import time 
def xgb_cv(X, y, learning_rate, n_estimators, max_depth, min_child_weight, gamma, reg_alpha, reg_lambda):
    xgb_model = xgb.XGBClassifier(booster='gbtree',
                              learning_rate=learning_rate,
                              n_estimators = int(n_estimators),
                              max_depth = int(max_depth),
                              min_child_weight = int(min_child_weight),
                              gamma=gamma,
                              subsample=1,
                              colsample_bytree=1,
                              objective = "binary:logistic",
                              nthread = 1,
                              n_jobs = -1,
                              random_state = 1,
                              scale_pos_weight = 1,
                              reg_alpha = int(reg_alpha),
                              reg_lambda = int(reg_lambda))   
    cv = KFold(n_splits=5, shuffle=True, random_state=11)
    valid_loss = cross_validate(xgb_model, X, y, scoring='roc_auc',cv=cv,n_jobs=-1,error_score='raise')
    return np.mean(valid_loss['test_score'])

def bayes_opt_xgb(X, y, init_points, n_iter):
    def xgb_cross_valid(learning_rate, n_estimators, max_depth,
                        min_child_weight, gamma, reg_alpha, reg_lambda):
        target = xgb_cv(X, y, learning_rate, n_estimators, max_depth,
                     min_child_weight, gamma, reg_alpha, reg_lambda)
        return target 
    
    optimizer = BayesianOptimization(xgb_cross_valid,
                                {
                                    'max_depth':(6, 6),
                                    'min_child_weight':(9, 9),
                                    'n_estimators':(12, 12),
                                    'learning_rate':(0, 1),
                                    'gamma':(0, 1),
                                    'reg_alpha':(0, 0),
                                    'reg_lambda':(290, 290)
                                })
    
    start_time = time.time()
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    end_time = time.time()
    print("花费时间：", (end_time-start_time)/60)
    opt_best = optimizer.max
    print("final result:" ,opt_best)
    
    return opt_best


# In[ ]:


opt_best = bayes_opt_xgb(train[cols], train['target'], 7, 50)


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
# xgb_model.fit(X_train, y_train)
xgb_model.fit(train[cols] ,train['target'])


# In[ ]:


# 对训练集进行预测
y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# # 对测试集进行预测
# y_pred_test = xgb_model.predict_proba(valid[cols])[:,1]
# fpr_test, tpr_test, thresholds_oot = metrics.roc_curve(valid['target'], y_pred_test, pos_label=1)
# roc_auc_test = metrics.auc(fpr_test, tpr_test)
# ks_test = max(tpr_test-fpr_test)
# print('test AUC: ', roc_auc_test)
# print('test KS: ', ks_test)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[ ]:


importance_type = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),orient='index')
importance_type.columns = ['value']
importance_type = importance_type.sort_values('value', ascending=False)
importance_type


# In[ ]:


importance_type.shape


# ### 网格调参

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
# xgb_model.fit(X_train, y_train)
# xgb_model.fit(oot[cols_new] ,oot['target'])


# In[ ]:


param_test1 = {'max_depth':[2,3,4], 'min_child_weight':[5,6,7,8,9]}

gsearch = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test1, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch.fit(train[cols], train['target'])

print('gsearch1.best_params_', gsearch.best_params_)
print('gsearch1.best_score_', gsearch.best_score_)


# In[ ]:


train.shape


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate= opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = 4,
                          min_child_weight = 8,
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 

param_test3 = {'n_estimators':[i for i in range(5,35)]}

gsearch3 = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test3, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch3.fit(train[cols], train['target'])

print('gsearch3.best_params_', gsearch3.best_params_)
print('gsearch3.best_score_', gsearch3.best_score_)


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = 12,
                          max_depth = 6,
                          min_child_weight =9,
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 

param_test2 = {'learning_rate':[i/20.0 for i in range(0,20)]}
gsearch2 = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test2, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch2.fit(train[cols], train['target'])

print('gsearch2.best_params_', gsearch2.best_params_)
print('gsearch2.best_score_', gsearch2.best_score_)


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate= 0.65,
                          n_estimators = 12,
                          max_depth = 6,
                          min_child_weight = 9,
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
param_test4 = {'gamma':[i/10.0 for i in range(10)]}

gsearch4 = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test4, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch4.fit(train[cols], train['target'])

print('gsearch4.best_params_', gsearch4.best_params_)
print('gsearch4.best_score_', gsearch4.best_score_)


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate= 0.45,
                          n_estimators = 20,
                          max_depth = 4,
                          min_child_weight = 8,
                          gamma= 0.2,
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = 0,
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
param_test5 = {'reg_lambda':[100,150, 200, 250, 300, 350, 400, 450, 500]}

gsearch5 = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test5, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch5.fit(train[cols], train['target'])

print('gsearch5.best_params_', gsearch5.best_params_)
print('gsearch5.best_score_', gsearch5.best_score_)


# In[ ]:


# xgb_model = xgb.XGBClassifier(booster='gbtree',
#                           learning_rate= 0.05,
#                           n_estimators = 200,
#                           max_depth = 6,
#                           min_child_weight = 2,
#                           gamma= 0.9,
#                           objective = "binary:logistic",
#                           nthread = 1,
#                           n_jobs = -1,
#                           random_state = 1,
#                           scale_pos_weight = 1,
#                           reg_alpha = 10,
#                           reg_lambda = int(opt_best['params']['reg_lambda'])) 
# param_test6 = {'reg_lambda':[0, 0.001, 0.1, 1, 10, 100,300,500]}

# gsearch6 = GridSearchCV(
#     estimator = xgb_model,
#     param_grid=param_test6, 
#     scoring='roc_auc', 
#     n_jobs=-1, 
#     cv=5)
# gsearch6.fit(train[cols], train['target'])

# print('gsearch6.best_params_', gsearch6.best_params_)
# print('gsearch6.best_score_', gsearch6.best_score_)


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate= 0.45,
                          n_estimators = 12,
                          max_depth = 6,
                          min_child_weight = 9,
                          gamma= 0.75,
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = 0,
                          reg_lambda = 290) 

xgb_model.fit(train[cols] ,train['target'])


# ## 5.2 Xgb模型技术效果

# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          subsample=1,
                          colsample_bytree=1,
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
# xgb_model.fit(X_train, y_train)
xgb_model.fit(oot[cols] ,oot['target'])


# In[ ]:


sorted(xgb_model.get_booster().get_score(importance_type='gain').items(),key = lambda x:x[1],reverse=True)


# In[ ]:


plot_importance(xgb_model, importance_type='gain')


# In[ ]:


# # 对训练集进行预测
# pred_train = xgb_model.predict_proba(X_train)[:,1]
# fpr, tpr, thresholds = metrics.roc_curve(y_train, pred_train, pos_label=1)
# roc_auc = metrics.auc(fpr, tpr)
# ks = max(tpr-fpr)
# print('train KS: ', ks)
# print('train AUC: ', roc_auc)


# In[ ]:


# 对训练集进行预测 train_selected[cols] ,train_selected['target']
pred_train = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], pred_train, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)

print('train AUC: ', roc_auc)
print('train KS: ', ks)


# In[ ]:


# # 对测试集进行预测 X_train, X_test, y_train, y_test 
# pred_test = xgb_model.predict_proba(valid[cols])[:,1]
# fpr_test, tpr_test, thresholds_test = metrics.roc_curve(valid['target'], pred_test, pos_label=1)
# roc_auc_test = metrics.auc(fpr_test, tpr_test)
# ks_test = max(tpr_test-fpr_test)

# print('test AUC: ', roc_auc_test)
# print('test KS: ', ks_test)


# In[ ]:


# 对oot测试集进行预测
pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)

print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# ## 5.3 模型业务效果

# In[ ]:


def Prob2Score(prob, base_odds=35, base_score=700, pdo=60, rate=2) :
    # 将概率转化成分数且为正整数
    y = np.log((1 - prob) / prob)
    factor = pdo/np.log(rate)
    offset = base_score - factor * np.log(base_odds)
    score = offset +  factor * (y)
    
    return score


# In[ ]:


print(len(cols))


# In[ ]:


# 业务效果-训练集
train['prob'] = xgb_model.predict_proba(train[cols])[:,1]
train['score'] = train['prob'].apply(lambda x: Prob2Score(x)).round(0)
pred_data = toad.metrics.KS_bucket(train['score'], train['target'], bucket=10, method='step')
pred_data


# In[ ]:


cut_bins = [float('-inf'), 670.0, 689.0, 718.0, 727.0, float('inf')]
print(cut_bins)


# In[ ]:


# 业务效果-训练集
train['prob'] = xgb_model.predict_proba(train[cols])[:,1]
train['score'] = train['prob'].apply(lambda x: Prob2Score(x)).round(0)
# train_ = pd.merge(X_train, y_train, how='inner', left_index=True, right_index=True)
train['bins'] = pd.cut(train['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


# # 业务效果-验证集
# valid['prob'] = xgb_model.predict_proba(valid[cols])[:,1]
# valid['score'] = valid['prob'].apply(lambda x: Prob2Score(x)).round(0)
# # test = pd.merge(X_test, y_test, how='inner', left_index=True, right_index=True)
# valid['bins'] = pd.cut(valid['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


# 业务效果-oot
oot['prob'] = xgb_model.predict_proba(oot[cols])[:,1]
oot['score'] = oot['prob'].apply(lambda x: Prob2Score(x)).round(0)
oot['bins'] = pd.cut(oot['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


pred_data_train = score_distribute(train, 'bins', target='target')
# pred_data_test = score_distribute(valid, 'bins', target='target')
pred_data_oot = score_distribute(oot, 'bins', target='target')


# In[ ]:


pred_data_oot


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_分数分布_xgb_58同城_20231017_v1"+'.xlsx')
pred_data_train.to_excel(writer,sheet_name='train')
# pred_data_test.to_excel(writer,sheet_name='test')
pred_data_oot.to_excel(writer,sheet_name='oot')
writer.save()


# In[ ]:


# 等频分箱
# 业务效果-训练集
train['prob'] = xgb_model.predict_proba(train[cols])[:,1]
train['score'] = train['prob'].apply(lambda x: Prob2Score(x)).round(0)
pred_data = toad.metrics.KS_bucket(train['score'], train['target'], bucket=10, method='quantile')
pred_data


# In[ ]:


cut_bins = [float('-inf')]+list(pred_data['min'])[1:]+[float('inf')]
print(cut_bins)


# In[ ]:


# 业务效果-训练集
train['prob'] = xgb_model.predict_proba(train[cols])[:,1]
train['score'] = train['prob'].apply(lambda x: Prob2Score(x)).round(0)
train['bins'] = pd.cut(train['score'], bins=cut_bins, include_lowest=True, right=False)

# # 业务效果-验证集
# valid['prob'] = xgb_model.predict_proba(valid[cols])[:,1]
# valid['score'] = valid['prob'].apply(lambda x: Prob2Score(x)).round(0)
# valid['bins'] = pd.cut(valid['score'], bins=cut_bins, include_lowest=True, right=False)

# 业务效果
oot['prob'] = xgb_model.predict_proba(oot[cols])[:,1]
oot['score'] = oot['prob'].apply(lambda x: Prob2Score(x)).round(0)
oot['bins'] = pd.cut(oot['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


pred_data_train = score_distribute(train, 'bins', target='target')

# pred_data_test = score_distribute(valid, 'bins', target='target')

pred_data_oot = score_distribute(oot, 'bins', target='target')


# In[ ]:


pred_data_oot


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_分数分布_xgb_quantile_58同城_20231017"+'.xlsx')
pred_data_train.to_excel(writer,sheet_name='train')
# pred_data_test.to_excel(writer,sheet_name='test')
pred_data_oot.to_excel(writer,sheet_name='oot')
writer.save()


# In[ ]:


def cal_psi(exp, act):
    psi = []
    for i in range(len(exp)):
        psi_i = (act[i] - exp[i])*np.log(act[i]/exp[i])
        psi.append(psi_i)
    return sum(psi)


# In[ ]:


# print(cal_psi(pred_data_train['total_pct'], pred_data_test['total_pct'])) 


# In[ ]:


print(cal_psi(pred_data_train['total_pct'], pred_data_oot['total_pct'])) 


# In[ ]:


print(cal_psi(pred_data_test['total_pct'], pred_data_oot['total_pct'])) 


# # 6.Lr模型训练

# ## 变量分箱

# In[425]:


# 第一次分箱
c = toad.transform.Combiner()
c.fit(train_selected, y='target', method='chi',min_samples=2, n_bins=10, empty_separate=True) 
bins_result = c.export()


# In[426]:


# 自动调整分箱
adj_bins = {}
for col in list(bins_result.keys()):
    tmp = np.isnan(bins_result[col])
    cutbins = [bins_result[col][x] for x in range(len(tmp)) if not tmp[x]]
    if len(cutbins)>0:
        cutbins = [float('-inf')] + cutbins + [float('inf')]
        cutoffpoints = ContinueVarBins(train_selected, col, flag='target', cutbins=cutbins)
        tmp_value1 = train_selected[train_selected[col]>=0][col].min()
        tmp_value2 = cutoffpoints[0]
        if tmp_value1==tmp_value2:
            adj_bins[col] = [-0.5] + cutoffpoints[1:]
        else:
            adj_bins[col] = [-0.5] + cutoffpoints


# In[427]:


# 调整分箱:空值单独一箱
# train_selected.fillna(-9999, inplace=True)

# 更新分箱
c.update(adj_bins)


# In[187]:


# adj_bins = {
#     'als_m12_cell_nbank_ca_orgnum_diff': [-0.5, 2.0, 774.0, 842.0],
#     'value_054_bairong_1': [-0.5, 3.0, 5.0],
#     'model_score_01_ruizhi_6': [-0.5, 670.0, 774.0, 842.0],
#     'als_m1_cell_nbank_oth_orgnum_diff': [-0.5, 5.0, 10.0],
#     'value_060_baihang_6': [-0.5, 2.0, 3.0, 4.0],
# #     'als_m12_cell_nbank_max_monnum_diff': [-0.5, 3.0, 7.0],
#     'als_m12_cell_nbank_max_monnum_diff': [-0.5, 3.0],
# #     'value_026_bairong_12': [-0.5, 34.1395],
#     'model_score_01_tianchuang_7': [-0.5, 555.0, 585.0, 596.0, 607.0],
#     'als_fst_id_nbank_inteday': [-0.5, 343.0, 357.0],
#     'als_d15_id_nbank_nsloan_orgnum': [-0.5, 3.0, 4.0],
#     'model_score_01_rong360_4': [-0.5, 0.055878434330225, 0.074964128434658, 0.10986643284559],
#      'model_score_01_moxingfen_7': [-0.5, 555.0, 578.0],
#     'als_m3_cell_bank_max_inteday_diff': [-0.5, 46.0]
# #     'model_score_01_moxingfen_7': [-0.5, 549.0, 580.0],
# #     'model_score_01_tianchuang_8': [-0.5, 552.0, 570.0, 594.0],


# }

# # 更新分箱
# c.update(adj_bins) 


# In[428]:


adj_bins = {
    'als_m12_cell_nbank_ca_orgnum_diff': [-0.5, 1.0, 2.0],
    'model_score_01_rong360_4': [-0.5, 0.055878434330225, 0.074964128434658, 0.10986643284559],
    'model_score_01_tianchuang_7': [-0.5, 555.0, 595.0],
    'als_m3_cell_bank_max_inteday_diff': [-0.5, 46.0],
    'model_score_01_moxingfen_7': [-0.5, 530.0],
    'value_054_bairong_1': [-0.5, 3.0, 5.0],
    'value_060_baihang_6': [-0.5, 2.0, 3.0, 4.0],
    'als_d15_id_nbank_nsloan_orgnum': [-0.5, 3.0, 4.0],
    'model_score_01_ruizhi_6': [-0.5, 670.0, 774.0, 842.0],
    'als_fst_id_nbank_inteday': [-0.5, 343.0, 357.0],
    'value_026_bairong_12': [-0.5, 34.1395]
}

# 更新分箱
c.update(adj_bins) 


# In[327]:


train_selected_bins = c.transform(train_selected, labels=True)
train_selected_bins.head(2)


# In[194]:


bins_dict_train = {}
for col in train_selected_bins.columns[1:]:
    bins_dict_train[col] = regroup(train_selected_bins, col, target='target')
    
df_result_train = pd.concat(list(bins_dict_train.values()), axis=0, ignore_index =True)


# In[195]:


df_result_train.head()


# In[196]:


# valid_selected = valid[train_selected.columns]
# valid_selected_bins = c.transform(valid_selected, labels=True)
# valid_selected_bins.head(2)

oot_selected = oot[train_selected.columns]
oot_selected_bins = c.transform(oot_selected, labels=True)

bins_dict_oot = {}
for col in oot_selected_bins.columns[1:]:
    bins_dict_oot[col] = regroup(oot_selected_bins, col, target='target')
    
df_result_oot = pd.concat(list(bins_dict_oot.values()), axis=0, ignore_index =True)


# In[197]:


df_result_oot.head()


# In[198]:


df_result_bins = pd.merge(df_result_train, df_result_oot, how='inner', on=['varsname','bins'],suffixes=['_train','_oot'])


# In[199]:


df_result_bins.head()


# In[ ]:





# In[200]:


df_result_bins.to_excel(r'D:\liuyedao\B卡开发\lr_xgb_174\B卡_chi_10bins_58同城_20231018_v1.xlsx',index=False)


# In[201]:


psi = toad.metrics.PSI(train_selected_bins.drop('target',axis=1), oot_selected_bins.drop('target',axis=1))
to_drop_psi = list(psi[psi>=0.25].index)
print(len(to_drop_psi))


# In[ ]:


train_selected.drop(to_drop_psi, axis=1,inplace=True)


# In[ ]:


# c.export()
# c.load(dict)
# c.transform(dataframe, labels=False)


# ## WOE转换

# In[429]:


transer = toad.transform.WOETransformer()
train_woe = transer.fit_transform(c.transform(train_selected), train_selected['target'], exclude=['target'])
train_woe.shape


# In[ ]:


# oot_woe = transer.transform(c.transform(oot))
# oot_woe.shape


# In[435]:


train_woe.drop('score', axis=1, inplace=True)
train_selected_woe, dropped = toad.selection.select(train_woe, target='target', empty=0.85, iv=0.02, corr=0.7, return_drop=True, exclude=None)
train_selected_woe.shape


# In[329]:


for i in dropped.keys():
    print("变量筛选维度：{}， 共剔除变量{}".format(i, len(dropped[i])))


# In[ ]:


toad.quality(train_selected_woe,'target',iv_only=True).head(5)


# In[205]:


to_drop_man = ['als_m3_id_nbank_min_monnum_diff','ppdi_d15_id_orgnum',
          'als_m12_id_bank_night_allnum','ppdi_m12_id_nbank_nloan_orgnum',
         'als_m12_cell_nbank_cons_orgnum']

train_selected_woe.drop(to_drop_man, axis=1,inplace=True)


# In[ ]:


need 
for col in list(set(df_result_train.varsname)):
    tmp = df_result_train[df_result_train.varsname==col]
    tmp_train = list(tmp['bad_rate_train'])[1:]
    tmp_oot = list(tmp['bad_rate_oot'])[1:]
    if_montone_train = [tmp_train[i]<tmp_train[i+1] for i in range(len(tmp_train)-1)]
    if_montone_oot = [tmp_oot[i]<tmp_oot[i+1] for i in range(len(tmp_oot)-1)]
    if len(set(if_montone))==1  :
        tmp_list.append(col)
    


# ## 逐步回归

# In[350]:


def Stepwise_Pvalue_Selection(X,y,threshold = 0.01,verbose=False):
    """逐步回归选择变量
    前向后向结合
    threshold:显著性检验p-value
    """
    included = []
    while True:
        changed = False
        excluded = [x for x in X.columns if x not in included]
        new_pvalue = pd.Series(index = excluded)
        for new_column in excluded:
            model = sm.Logit(y,sm.add_constant(X[included+[new_column]])).fit()
            new_pvalue[new_column] = model.pvalues[new_column]
        best_pvalue = new_pvalue.min()
        if best_pvalue < threshold:
            best_feature = new_pvalue.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add {:30} with p-value {:.6}'.format(best_feature,best_pvalue))

        model = sm.Logit(y,sm.add_constant(X[included])).fit()

        p_values = model.pvalues.iloc[1:]
        worst_pvalue = p_values.max()
        if worst_pvalue >= threshold:
            changed = True
            worst_feature = p_values.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature,worst_pvalue))
        if not changed:
            break

    return included


# In[420]:


included = Stepwise_Pvalue_Selection(train_selected_woe.drop('target',axis=1),train_selected_woe['target'],
                                     threshold = 0.01,verbose=True)


# In[415]:


len(included)


# In[353]:


included


# In[449]:


print(train_selected_woe.shape)
final_data = toad.selection.stepwise(train_selected_woe, target='target', estimator='ols', direction='both',
                                     criterion='aic', exclude=None, p_enter=0.05)
print(final_data.shape)


# In[436]:


help(toad.selection.stepwise)


# In[ ]:


# # cols = ['model_score_01_ruizhi_6','als_m1_cell_nbank_oth_orgnum_diff','ppdi_m3_id_nbank_nloan_allnum'
# #         ,'value_060_baihang_6','als_m12_cell_nbank_max_monnum_diff','model_score_01_tianchuang_7'
# #         ,'als_fst_id_nbank_inteday','als_d15_id_nbank_nsloan_orgnum','model_score_01_rong360_4'
# #         ,'model_score_01_moxingfen_7','als_m3_cell_bank_max_inteday_diff']
# cols = ['model_score_01_ruizhi_6', 'als_m1_cell_nbank_oth_orgnum_diff', 'value_060_baihang_6',
#         'als_m12_cell_nbank_max_monnum_diff', 'model_score_01_tianchuang_7',
#         'als_fst_id_nbank_inteday', 'als_d15_id_nbank_nsloan_orgnum', 'model_score_01_rong360_4',
#         'model_score_01_moxingfen_7', 'als_m3_cell_bank_max_inteday_diff','value_026_bairong_12']
# final_data  = train_woe[['target']+cols]


# In[238]:


len(list(adj_bins.keys()))


# In[476]:


need_cols = ['target']+list(adj_bins.keys())
final_data = train_woe[need_cols]
# other_ks = ['model_score_01_ruizhi_6','model_score_01_v1_duxiaoman_1','model_score_01_tianchuang_7',
#             'model_score_01_tianchuang_8','als_m12_id_nbank_min_monnum','model_score_01_3_xinyongsuanli_1',
#             'als_m12_id_nbank_cons_allnum','value_025_bairong_14',
#             'als_m12_id_caon_allnum','ppdi_m6_cell_nbank_week_allnum','als_m1_id_pdl_allnum_diff',
#             'model_score_01_fulin_1','ppdi_m3_id_nbank_nloan_allnum',
#             'als_m12_cell_af_allnum','als_m12_id_rel_allnum_diff',
#             'ppdi_d15_id_orgnum','ppdi_d7_id_bank_orgnum','ppdi_d15_id_nbank_nloan_allnum',
#             'als_m12_id_nbank_week_orgnum','model_score_01_v2_duxiaoman_1',
#             'als_m3_id_nbank_min_monnum_diff','ppdi_m1_id_bank_orgnum']
# final_data = train_selected_woe[['target']+ included]


# In[477]:


# 测试集
# oot_woe = transer.transform(c.transform(oot))
final_oot = oot_woe[list(final_data.columns)]
final_oot.shape


# In[478]:


cols = list(final_data.drop(['target'], axis=1).columns)
print(len(cols),cols)


# In[423]:


cols = list(final_data.drop(['target'], axis=1).columns)
print(len(cols),cols)


# In[107]:


cols = list(final_data.drop(['target'], axis=1).columns)
print(len(cols),cols)


# ## 6.1模型技术效果

# In[ ]:


# 建模变量
# cols = list(final_data.drop(['target'], axis=1).columns)
print(len(cols))
print(cols)


# In[230]:


df_result_rm = pd.merge(df_result_bins,pd.DataFrame({'varsname':cols}),how='inner',on='varsname')
df_result_rm.head()
df_result_rm.to_excel(r'D:\liuyedao\B卡开发\lr_xgb_174\B卡_chi_10bins_58同城_入模变量_v8.xlsx',index=False)


# In[494]:


# cols.remove('model_score_01_tianchuang_7')
cols.append('model_score_01_tianchuang_7')


# In[751]:


clf = LR_model(final_oot[cols], final_data[cols], final_oot['target'], final_data['target'])


# In[750]:


clf = LR_model(final_data[cols], final_oot[cols], final_data['target'], final_oot['target'])


# In[244]:


vif=pd.DataFrame()
X = np.matrix(final_data[cols])
vif['features']=cols
vif['VIF_Factor']=[variance_inflation_factor(np.matrix(X),i) for i in range(X.shape[1])]
vif


# In[245]:


corr = final_data[cols].corr()
corr


# In[246]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_corr_vif_20231018"+'.xlsx')

corr.to_excel(writer,sheet_name='corr')
vif.to_excel(writer,sheet_name='vif')

writer.save()


# In[247]:


# 训练集
pred_train = clf.predict(sm.add_constant(final_data[cols]))
# KS/AUC
from toad.metrics import KS,AUC

print('train AUC: ', AUC(pred_train, final_data['target']))
print('train KS: ', KS(pred_train, final_data['target']))


# In[248]:


# 测试集
pred_oot = clf.predict(sm.add_constant(final_oot[cols]))
# KS/AUC
from toad.metrics import KS,AUC

print('-------------oot结果--------------------')

print('test AUC: ', AUC(pred_oot, final_oot['target']))
print('test KS: ', KS(pred_oot, final_oot['target']))



# ## 6.1转换评分

# In[496]:


card = toad.ScoreCard(combiner=c,
                      transer = transer,
                      base_odds=35,
                      base_score=700,
                      pdo=60,
                      rate=2,
                      C=1e8
                     )


# In[497]:


card.fit(final_data[cols], final_data['target'])


# In[498]:


def scorecard_scale(self):
    scorecard_kedu = pd.DataFrame(
    [
        ["base_odds", self.base_odds,"根据业务经验设置的基础比率"],
        ["base_score", self.base_score,"基础odds对应的分数"],
        ["rate", self.rate, "设置分数的倍率"],
        ["pdo", self.pdo, "表示分数增长pdo时，odds值增长rate倍"],
        ["B", self.factor,"补偿值，计算方式：pdo/ln(rate)"],
        ["A", self.offset,"刻度,计算方式：base_score - B * ln(base_odds)"]
    ],
    columns = ["刻度项", "刻度值","备注"]
    )
    
    return scorecard_kedu


# In[499]:


scorecard_scale(card)


# In[500]:


score_card = card.export(to_frame=True).round(0)
print(len(set(score_card.name)))
score_card


# In[168]:


# (card.offset - card.factor * card.coef_[0] * card.intercept_)/9


# In[254]:


score_card.to_excel(r'D:\liuyedao\B卡开发\lr_xgb_174\B卡_评分卡_58同城_20231018_v1.xlsx')


# In[255]:


name = {
    'als_m12_cell_nbank_ca_orgnum_diff': '按手机号查询，近12个月在非银机构-现金类分期申请机构数与上次查询的差',
    'model_score_01_rong360_4': '融360评分',
    'model_score_01_tianchuang_7': '天创信用联合定制分',
    'model_score_01_moxingfen_7': '贷中评分',
    'value_054_bairong_1': '按身份证号查询，近15天在非银机构-持牌消费金融机构申请机构数',
    'value_060_baihang_6': '百行_续侦_证件号近3个月内查询机构数-网络小贷类机构',
    'als_d15_id_nbank_nsloan_orgnum': '按身份证号查询，近15天在非银机构-持牌网络小贷机构申请机构数',
    'model_score_01_ruizhi_6': '睿智联合分',
    'als_fst_id_nbank_inteday': '按身份证号查询，距最早在非银行机构申请的间隔天数',
    'value_026_bairong_12': '近6个月用户利率偏好'
}


# In[256]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_变量箱子_分数_v2.xlsx")
train_selected['score'] = card.predict(train_selected[cols]).round(0)
train_selected_bins = c.transform(train_selected, labels=True)
 

for col in cols:
    tmp1 = CalWoeIv(train_selected_bins, col, target='target')
    tmp2 = train_selected_bins[[col, 'score']].drop_duplicates(col,keep='first')
    tmp = pd.merge(tmp1, tmp2, how='left',left_on='bins', right_on=col)
    tmp['name'] = name[col]
    tmp = tmp[['varsname','name','bins','total','total_pct','bad_rate','woe']]
    tmp.to_excel(writer,sheet_name=col)
    writer.save()


# In[257]:


print(len(cols), cols)


# In[ ]:


train['score'] = card.predict(train[cols]).round(0)
train['prob'] = card.predict_proba(train[cols])[:,1]


# In[ ]:


oot['score'] = card.predict(oot[cols]).round(0)
oot['prob'] = card.predict_proba(oot[cols])[:,1]


# In[ ]:


# tmp1 = train[['order_no','lending_time','channel_id','score','prob']+cols]
# tmp2 = oot[['order_no','lending_time','channel_id','score','prob']+cols]
# tmp = pd.concat([tmp1, tmp2], axis=0)
# tmp.info()
# tmp.head()
# tmp.to_csv(r'D:\liuyedao\B卡开发\Lr_result\score_lr_部署测试样例_58同城.csv',index=False)


# ## 6.2模型业务效果

# In[501]:


# 业务效果-训练集
train['score'] = card.predict(train[cols]).round(0)
pred_data = toad.metrics.KS_bucket(train['score'], train['target'], bucket=10, method='quantile')
pred_data


# In[502]:


cut_bins = [float('-inf')]+list(pred_data['min'])[1:]+[float('inf')]
# cut_bins = [float('-inf')]+ [711.0, 750.0, 763.0, 797.0] + [float('inf')]
print(cut_bins)


# In[503]:


# 业务效果-训练集
train['score'] = card.predict(train[cols]).round(0)
train['bins'] = pd.cut(train['score'], bins=cut_bins, include_lowest=True, right=False)


# In[504]:


# 业务效果-oot
oot['score'] = card.predict(oot[cols]).round(0)
oot['bins'] = pd.cut(oot['score'], bins=cut_bins, include_lowest=True, right=False)


# In[505]:


pred_data_train = score_distribute(train, 'bins', target='target')
pred_data_train


# In[506]:


pred_data_oot = score_distribute(oot, 'bins', target='target')
pred_data_oot


# In[308]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_分数分布_step_lr.xlsx")
pred_data_train = score_distribute(train, 'bins', target='target')
# pred_data_valid = score_distribute(valid, 'bins', target='target')
pred_data_oot = score_distribute(oot, 'bins', target='target')


pred_data_train.to_excel(writer,sheet_name='pred_data_train')
# pred_data_valid.to_excel(writer,sheet_name='pred_data_valid')
pred_data_oot.to_excel(writer,sheet_name='pred_data_oot')

writer.save()


# In[315]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_分数分布_quantile_lr.xlsx")
pred_data_train = score_distribute(train, 'bins', target='target')
# pred_data_valid = score_distribute(valid, 'bins', target='target')
pred_data_oot = score_distribute(oot, 'bins', target='target')


pred_data_train.to_excel(writer,sheet_name='pred_data_train')
# pred_data_valid.to_excel(writer,sheet_name='pred_data_valid')
pred_data_oot.to_excel(writer,sheet_name='pred_data_oot')

writer.save()


# In[300]:


def cal_psi(exp, act):
    psi = []
    for i in range(len(exp)):
        psi_i = (act[i] - exp[i])*np.log(act[i]/exp[i])
        psi.append(psi_i)
    return sum(psi)


# In[301]:


# print(cal_psi(pred_data_train['total_pct'], pred_data_valid['total_pct'])) 
print(cal_psi(pred_data_train['total_pct'], pred_data_oot['total_pct'])) 




#==============================================================================
# File: B卡建模-xgboost-174渠道.py
#==============================================================================

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import toad
import xgboost as xgb
from xgboost import plot_importance
from sklearn import metrics
from sklearn.model_selection import train_test_split
from datetime import datetime
import os 
import warnings
warnings.filterwarnings("ignore")

from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, cross_validate, cross_val_score
import time
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from pypmml import Model


# In[2]:


os.getcwd()


# In[3]:


# 运行函数脚本
get_ipython().run_line_magic('run', 'function.ipynb')


# # 1.读取训练集

# In[4]:


filepath = r'D:\liuyedao\B卡开发\三方数据匹配'
data = pd.read_csv(filepath + r'\order_20230927.csv') 


# In[5]:


data.info()
data.head()


# In[6]:


data.channel_id.value_counts(dropna=False)


# In[7]:


data.target.value_counts(dropna=False)


# In[8]:


data.lending_month.value_counts(dropna=False)


# In[9]:


df_base = data[data['channel_id']==174]
df_base.reset_index(drop=True, inplace=True)
df_base.info()
df_base.head()


# In[10]:


df_base.groupby(['lending_month','target'])['order_no'].count().unstack()


# In[ ]:





# ## 建模数据集

# In[230]:


df_train = df_base.query("apply_date<'2023-05-01' & target in [0.0, 1.0]")
df_train.groupby(['lending_month','target'])['order_no'].count().unstack()


# In[231]:


print(df_train.target.sum(),df_train.target.mean())


# In[ ]:


df_train.info()
df_train.head(2)


# In[ ]:


df_train.to_csv(r'D:\liuyedao\B卡开发\mid_result\b卡_建模数据集_train_20231008_174.csv',index=False)


# ## oot测试数据集

# In[232]:


df_oot = df_base.query("lending_month=='2023-05' & target in [0.0, 1.0]")
print(df_oot.target.sum(),df_oot.target.mean())


# In[ ]:


df_oot.to_csv(r'D:\liuyedao\B卡开发\mid_result\b卡_建模数据集_oot_20231008_174.csv',index=False)


# ### 建模数据集

# In[190]:


train = df_base.query("apply_date<'2023-06-01' & target in [0.0, 1.0]")
train.groupby(['lending_month','target'])['order_no'].count().unstack()


# In[191]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('target', axis=1) ,train['target'],
                                                    test_size=0.3, random_state=22, stratify=train['target'])
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[192]:


train = pd.merge(X_train, y_train, how='inner', left_index=True, right_index=True)
oot = pd.merge(X_test, y_test, how='inner', left_index=True, right_index=True)
print(train.shape, valid.shape)


# In[193]:


train['target'].value_counts(dropna=False)
# valid['target'].value_counts(dropna=False)
oot['target'].value_counts(dropna=False)


# In[194]:


train['target'].value_counts(dropna=False,normalize=True)
# valid['target'].value_counts(dropna=False,normalize=True)
oot['target'].value_counts(dropna=False,normalize=True)


# # 2.数据处理

# In[ ]:


df_train.shape


# In[ ]:


df_oot.shape


# In[276]:


train = df_train.copy()
oot = df_oot.copy()


# In[234]:


to_drop = train.columns[0:6].to_list()
print(to_drop)


# In[235]:


train = train.drop(to_drop, axis=1)
oot = oot.drop(to_drop, axis=1)


# In[236]:


to_drop = list(train.select_dtypes(include='object').columns)
print(to_drop)


# In[237]:


train = train.drop(to_drop, axis=1)
oot = oot.drop(to_drop, axis=1)


# In[238]:


train.info()


# In[239]:


oot.info()


# In[240]:


print(train.target.value_counts())
print(oot.target.value_counts())


# In[241]:


print(train.target.value_counts(normalize=True))
print(oot.target.value_counts(normalize=True))


# # 3.样本数据探查

# In[242]:


train_df_iv = toad.quality(train,'target',iv_only=True)
oot_df_iv = toad.quality(oot,'target',iv_only=True)


# In[243]:


train_df_iv.head()


# In[244]:


oot_df_iv.head()


# In[245]:


df_iv = pd.merge(train_df_iv, oot_df_iv, how='inner',left_index=True, right_index=True)
df_iv.shape


# In[ ]:


# df_iv.to_excel(r'D:\liuyedao\B卡开发\xgb_result\df_iv_174_20231008.xlsx')


# In[210]:


df_iv['rate'] = df_iv['iv_y']/df_iv['iv_x'] - 1


# In[211]:


to_keep_iv = list(df_iv.query("rate<0.20 & rate>-0.2 & iv_x>0.01 & iv_y>0.01").index)
print(len(to_keep_iv))


# In[259]:


to_keep_iv = list(df_iv.query("iv_x<0.1").index)
print(len(to_keep_iv))


# # 4.特征筛选

# In[ ]:


# train.drop(['model_score_01_zr_tongdun_2','model_score_01_zx_tongdun_2'], axis=1, inplace=True)
# train.shape


# In[ ]:


# train.mask(train<0, inplace=True)
# oot.mask(oot<0, inplace=True)


# In[247]:


train_selected, dropped = toad.selection.select(train, target='target', empty=0.85, iv=0.0, corr=0.70, return_drop=True, exclude=None)
print(train_selected.shape)


# In[248]:


for i in dropped.keys():
    print("变量筛选维度：{}， 共剔除变量{}".format(i, len(dropped[i])))


# In[152]:


psi = toad.metrics.PSI(train.drop('target',axis=1), oot.drop('target',axis=1))
to_keep_psi = list(psi[psi<0.25].index)
print(len(to_keep_psi))


# In[249]:


# 人工剔除
to_drop = ['model_score_01_baihang_1','model_score_01_ronghuijinke_2','model_score_01_q_tianchuang_1',
           'model_score_01_r_tianchuang_1',
           ,'model_score_01_1_bileizhen_1',
           'model_score_01_2_bileizhen_1','model_score_01_3_bileizhen_1','model_score_01_101_hulian_5',
           'model_score_01_103_hulian_5','model_score_01_105_hulian_5','model_score_01_107_hulian_5',
           'model_score_01_121_hulian_5','model_score_01_122_hulian_5','model_score_01_124_hulian_5',
           'model_score_01_125_hulian_5','model_score_01_127_hulian_5','model_score_01_130_hulian_5',
           'model_score_01_131_hulian_5','model_score_01_132_hulian_5','model_score_01_133_hulian_5',
           'model_score_01_136_hulian_5','model_score_01_137_hulian_5','value_011_moxingfen_7',
           'value_012_moxingfen_7','value_013_moxingfen_7','value_014_moxingfen_7','value_012_pudao_5',
           'model_score_01_pudao_11','model_score_01_pudao_12','model_score_01_pudao_16',
           'model_score_01_2_bairong_8','model_score_01_3_bairong_8','model_score_01_7_bairong_8',
           'model_score_01_8_bairong_8']

for col in train_selected.columns:
    if col in to_drop:
        train_selected.drop(col, axis=1, inplace=True)
print(train_selected.shape)


# In[269]:


cols = list(set(to_keep_iv).intersection(set(train_selected.columns)))
print(len(cols))


# In[271]:


train_selected = toad.selection.stepwise(train_selected[['target']+cols].fillna(-999), target='target', estimator='ols', direction='both', criterion='aic', exclude=None)
train_selected.shape


# In[273]:


cols = ['model_score_01_tianchuang_7', 'model_score_01_tianchuang_8', 'model_score_01_v2_duxiaoman_1', 'als_m12_cell_nbank_cons_orgnum', 'als_fst_id_nbank_inteday', 'als_m12_id_nbank_night_orgnum', 'value_027_bairong_12', 'value_031_bairong_12', 'value_025_bairong_14', 'model_score_01_rong360_4', 'value_011_bairong_12', 'als_m3_cell_nbank_max_inteday']


# # 4.模型训练和评估

# In[155]:


cols = list(train_selected.drop('target', axis=1).columns)
print(len(cols))


# In[274]:


opt_best = {'target': 0.6487565005430443,
            'params': {'gamma': 0.8, 'learning_rate': 0.45, 'max_depth': 3.0, 'min_child_weight': 6.0,
                       'n_estimators': 200, 'reg_alpha': 0, 'reg_lambda': 100}
           }


# In[277]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
# xgb_model.fit(X_train, y_train)
xgb_model.fit(train[cols] ,train['target'])


# In[ ]:


# sorted(xgb_model.get_booster().get_score(importance_type='gain').items(),key = lambda x:x[1],reverse=True)


# In[263]:


len(xgb_model.get_booster().get_score(importance_type='gain').keys())


# In[278]:


importance_type = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),orient='index')
importance_type.columns = ['value']
importance_type = importance_type.sort_values('value', ascending=False)
importance_type.head(30)


# In[280]:


# 对训练集进行预测
y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# # 对测试集进行预测
# y_pred_test = xgb_model.predict_proba(valid[cols])[:,1]
# fpr_test, tpr_test, thresholds_oot = metrics.roc_curve(valid['target'], y_pred_test, pos_label=1)
# roc_auc_test = metrics.auc(fpr_test, tpr_test)
# ks_test = max(tpr_test-fpr_test)
# print('test AUC: ', roc_auc_test)
# print('test KS: ', ks_test)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[266]:


# plot_importance(xgb_model, importance_type='gain')
cols = list(importance_type.head(20).index)
# cols = list(importance_type[importance_type.value>3.0].index)
print(cols, len(cols))


# In[267]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
# xgb_model.fit(X_train, y_train)
xgb_model.fit(train_selected[cols] ,train_selected['target'])


# In[268]:


# 对训练集进行预测
y_pred = xgb_model.predict_proba(train_selected[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train_selected['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# # 对测试集进行预测
# y_pred_test = xgb_model.predict_proba(valid[cols])[:,1]
# fpr_test, tpr_test, thresholds_oot = metrics.roc_curve(valid['target'], y_pred_test, pos_label=1)
# roc_auc_test = metrics.auc(fpr_test, tpr_test)
# ks_test = max(tpr_test-fpr_test)
# print('test AUC: ', roc_auc_test)
# print('test KS: ', ks_test)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[281]:


plt.plot(fpr, tpr, color='darkorange', lw=2, label='trian ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr_oot, tpr_oot, color='blue', lw=2, label='test ROC curve (area = %0.2f)' % roc_auc_oot)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="best")
plt.show()


# ## 4.1模型调参

# In[282]:


def xgb_cv(X, y, learning_rate, n_estimators, max_depth, min_child_weight, gamma, reg_alpha, reg_lambda):
    xgb_model = xgb.XGBClassifier(booster='gbtree',
                              learning_rate=learning_rate,
                              n_estimators = int(n_estimators),
                              max_depth = int(max_depth),
                              min_child_weight = int(min_child_weight),
                              gamma=gamma,
                              subsample=1,
                              colsample_bytree=1,
                              objective = "binary:logistic",
                              nthread = 1,
                              n_jobs = -1,
                              random_state = 1,
                              scale_pos_weight = 1,
                              reg_alpha = int(reg_alpha),
                              reg_lambda = int(reg_lambda))   
    cv = KFold(n_splits=5, shuffle=True, random_state=11)
    valid_loss = cross_validate(xgb_model, X, y, scoring='roc_auc',cv=cv,n_jobs=-1,error_score='raise')
    return np.mean(valid_loss['test_score'])

def bayes_opt_xgb(X, y, n_iter):
    def xgb_cross_valid(learning_rate, n_estimators, max_depth,
                        min_child_weight, gamma, reg_alpha, reg_lambda):
        target = xgb_cv(X, y, learning_rate, n_estimators, max_depth,
                     min_child_weight, gamma, reg_alpha, reg_lambda)
        return target 
    
    optimizer = BayesianOptimization(xgb_cross_valid,
                                {
                                    'max_depth':(3, 10),
                                    'min_child_weight':(1, 20),
                                    'n_estimators':(50, 500),
                                    'learning_rate':(0, 1),
                                    'gamma':(0, 1),
                                    'reg_alpha':(0, 300),
                                    'reg_lambda':(0, 300)
                                })
    
    start_time = time.time()
    optimizer.maximize(init_points=7, n_iter=n_iter)
    end_time = time.time()
    print("花费时间：", (end_time-start_time)/60)
    opt_best = optimizer.max
    print("final result:" ,opt_best)
    
    return opt_best


# In[283]:


opt_best = bayes_opt_xgb(train[cols], train['target'], 50)


# In[284]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
# xgb_model.fit(X_train, y_train)
xgb_model.fit(train[cols] ,train['target'])


# In[285]:


# 对训练集进行预测
y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# # 对测试集进行预测
# y_pred_test = xgb_model.predict_proba(valid[cols])[:,1]
# fpr_test, tpr_test, thresholds_oot = metrics.roc_curve(valid['target'], y_pred_test, pos_label=1)
# roc_auc_test = metrics.auc(fpr_test, tpr_test)
# ks_test = max(tpr_test-fpr_test)
# print('test AUC: ', roc_auc_test)
# print('test KS: ', ks_test)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[286]:


from sklearn.model_selection import GridSearchCV


# In[287]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
# xgb_model.fit(X_train, y_train)
# xgb_model.fit(oot[cols_new] ,oot['target'])


# In[289]:


param_test1 = {'max_depth':[3,4,5,6,7,8,9], 'min_child_weight':[1,2,3,4,5,6]}

gsearch = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test1, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch.fit(train[cols], train['target'])

print('gsearch1.best_params_', gsearch.best_params_)
print('gsearch1.best_score_', gsearch.best_score_)


# In[291]:


train.shape


# In[290]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = 6,
                          min_child_weight =2,
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 

param_test2 = {'learning_rate':[i/20.0 for i in range(0,20)]}
gsearch2 = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test2, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch2.fit(train[cols], train['target'])

print('gsearch2.best_params_', gsearch2.best_params_)
print('gsearch2.best_score_', gsearch2.best_score_)


# In[293]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate= 0.05,
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = 6,
                          min_child_weight = 2,
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 

param_test3 = {'n_estimators':[100, 200, 300, 400, 500, 600]}

gsearch3 = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test3, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch3.fit(train[cols], train['target'])

print('gsearch3.best_params_', gsearch3.best_params_)
print('gsearch3.best_score_', gsearch3.best_score_)


# In[294]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate= 0.05,
                          n_estimators = 200,
                          max_depth = 6,
                          min_child_weight = 2,
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
param_test4 = {'gamma':[i/10.0 for i in range(10)]}

gsearch4 = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test4, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch4.fit(train[cols], train['target'])

print('gsearch4.best_params_', gsearch4.best_params_)
print('gsearch4.best_score_', gsearch4.best_score_)


# In[296]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate= 0.05,
                          n_estimators = 200,
                          max_depth = 6,
                          min_child_weight = 2,
                          gamma= 0.9,
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
param_test5 = {'reg_alpha':[0, 0.001, 0.1, 1, 10, 100]}

gsearch5 = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test5, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch5.fit(train[cols], train['target'])

print('gsearch5.best_params_', gsearch5.best_params_)
print('gsearch5.best_score_', gsearch5.best_score_)


# In[298]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate= 0.05,
                          n_estimators = 200,
                          max_depth = 6,
                          min_child_weight = 2,
                          gamma= 0.9,
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = 10,
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
param_test6 = {'reg_lambda':[0, 0.001, 0.1, 1, 10, 100]}

gsearch6 = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test6, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch6.fit(train[cols], train['target'])

print('gsearch6.best_params_', gsearch6.best_params_)
print('gsearch6.best_score_', gsearch6.best_score_)


# In[300]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate= 0.05,
                          n_estimators = 200,
                          max_depth = 6,
                          min_child_weight = 2,
                          gamma= 0.9,
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = 10,
                          reg_lambda = 100) 

xgb_model.fit(train[cols] ,train['target'])


# ## 4.2 模型效果评估

# In[ ]:


cols = list(train_selected.drop('target', axis=1).columns)
print(len(cols))


# In[ ]:


# opt_best = {'target': 0.6547141005835768, 'params': {'gamma': 0.8308208247682911, 'learning_rate': 0.2804997959155717, 'max_depth': 4.229976816558083, 'min_child_weight': 9.232634969126048, 'n_estimators': 448.71023663952815, 'reg_alpha': 2.8937408854371682, 'reg_lambda': 87.19537423984058, 'subsample': 0.8932601124914011}}
# opt_best = {'target': 0.6588812578259035, 'params': {'gamma': 0.5219039246581273, 'learning_rate': 0.14533762037842712, 'max_depth': 9.19440465374945, 'min_child_weight': 1.7208760081426773, 'n_estimators': 318.95417117029183, 'reg_alpha': 15.889516912458845, 'reg_lambda': 63.65061115921122}}


# In[70]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          subsample=1,
                          colsample_bytree=1,
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
# xgb_model.fit(X_train, y_train)
xgb_model.fit(oot[cols] ,oot['target'])


# In[ ]:


sorted(xgb_model.get_booster().get_score(importance_type='gain').items(),key = lambda x:x[1],reverse=True)


# In[ ]:


plot_importance(xgb_model, importance_type='gain')


# In[ ]:


# # 对训练集进行预测
# pred_train = xgb_model.predict_proba(X_train)[:,1]
# fpr, tpr, thresholds = metrics.roc_curve(y_train, pred_train, pos_label=1)
# roc_auc = metrics.auc(fpr, tpr)
# ks = max(tpr-fpr)
# print('train KS: ', ks)
# print('train AUC: ', roc_auc)


# In[301]:


# 对训练集进行预测 train_selected[cols] ,train_selected['target']
pred_train = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], pred_train, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)

print('train AUC: ', roc_auc)
print('train KS: ', ks)


# In[ ]:


# # 对测试集进行预测 X_train, X_test, y_train, y_test 
# pred_test = xgb_model.predict_proba(X_test[cols])[:,1]
# fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_test, pos_label=1)
# roc_auc = metrics.auc(fpr, tpr)
# ks = max(tpr-fpr)
# print('valid AUC: ', roc_auc_test)
# print('valid KS: ', ks_test)


# In[ ]:


# 对测试集进行预测 X_train, X_test, y_train, y_test 
pred_test = xgb_model.predict_proba(valid[cols])[:,1]
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(valid['target'], pred_test, pos_label=1)
roc_auc_test = metrics.auc(fpr_test, tpr_test)
ks_test = max(tpr_test-fpr_test)

print('test AUC: ', roc_auc_test)
print('test KS: ', ks_test)


# In[302]:


# 对oot测试集进行预测
pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)

print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[186]:


len(cols)


# # 5.模型业务效果

# In[ ]:


def Prob2Score(prob, base_odds=35, base_score=700, pdo=60, rate=2) :
    # 将概率转化成分数且为正整数
    y = np.log((1 - prob) / prob)
    factor = pdo/np.log(rate)
    offset = base_score - factor * np.log(base_odds)
    score = offset +  factor * (y)
    
    return score


# In[ ]:


# 业务效果-训练集
train['prob'] = xgb_model.predict_proba(train[cols])[:,1]
train['score'] = train['prob'].apply(lambda x: Prob2Score(x)).round(0)
pred_data = toad.metrics.KS_bucket(train['score'], train['target'], bucket=10, method='step')
pred_data


# In[ ]:


cut_bins = [float('-inf')]+list(pred_data['min'])[1:]+[float('inf')]
print(cut_bins)


# In[ ]:


# 业务效果-训练集
train['prob'] = xgb_model.predict_proba(train[cols])[:,1]
train['score'] = train['prob'].apply(lambda x: Prob2Score(x)).round(0)
# train_ = pd.merge(X_train, y_train, how='inner', left_index=True, right_index=True)
train['bins'] = pd.cut(train['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


# # 业务效果-验证集
# valid['prob'] = xgb_model.predict_proba(valid[cols])[:,1]
# valid['score'] = valid['prob'].apply(lambda x: Prob2Score(x)).round(0)
# # test = pd.merge(X_test, y_test, how='inner', left_index=True, right_index=True)
# valid['bins'] = pd.cut(valid['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


# 业务效果-oot
oot['prob'] = xgb_model.predict_proba(oot[cols])[:,1]
oot['score'] = oot['prob'].apply(lambda x: Prob2Score(x)).round(0)
oot['bins'] = pd.cut(oot['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


pred_data_train = score_distribute(train, 'bins', target='target')
# pred_data_test = score_distribute(valid, 'bins', target='target')
pred_data_oot = score_distribute(oot, 'bins', target='target')


# In[ ]:


pred_data_oot


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\mid_result\B卡_分数分布_xgb_step_58同城_20230922"+'.xlsx')
pred_data_train.to_excel(writer,sheet_name='train')
# pred_data_test.to_excel(writer,sheet_name='test')
pred_data_oot.to_excel(writer,sheet_name='oot')
writer.save()


# In[ ]:


# 等频分箱
# 业务效果-训练集
train['prob'] = xgb_model.predict_proba(train[cols])[:,1]
train['score'] = train['prob'].apply(lambda x: Prob2Score(x)).round(0)
pred_data = toad.metrics.KS_bucket(train['score'], train['target'], bucket=10, method='quantile')
pred_data


# In[ ]:





# In[ ]:


cut_bins = [float('-inf')]+list(pred_data['min'])[1:]+[float('inf')]
print(cut_bins)


# In[ ]:


# 业务效果-训练集
train['prob'] = xgb_model.predict_proba(train[cols])[:,1]
train['score'] = train['prob'].apply(lambda x: Prob2Score(x)).round(0)
train['bins'] = pd.cut(train['score'], bins=cut_bins, include_lowest=True, right=False)

# # 业务效果-验证集
# valid['prob'] = xgb_model.predict_proba(valid[cols])[:,1]
# valid['score'] = valid['prob'].apply(lambda x: Prob2Score(x)).round(0)
# valid['bins'] = pd.cut(valid['score'], bins=cut_bins, include_lowest=True, right=False)

# 业务效果
oot['prob'] = xgb_model.predict_proba(oot[cols])[:,1]
oot['score'] = oot['prob'].apply(lambda x: Prob2Score(x)).round(0)
oot['bins'] = pd.cut(oot['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


pred_data_train = score_distribute(train, 'bins', target='target')

# pred_data_test = score_distribute(valid, 'bins', target='target')

pred_data_oot = score_distribute(oot, 'bins', target='target')


# In[ ]:


pred_data_oot


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\mid_result\B卡_分数分布_xgb_quantile_58同城_20230922"+'.xlsx')
pred_data_train.to_excel(writer,sheet_name='train')
# pred_data_test.to_excel(writer,sheet_name='test')
pred_data_oot.to_excel(writer,sheet_name='oot')
writer.save()


# In[ ]:


def cal_psi(exp, act):
    psi = []
    for i in range(len(exp)):
        psi_i = (act[i] - exp[i])*np.log(act[i]/exp[i])
        psi.append(psi_i)
    return sum(psi)


# In[ ]:


# print(cal_psi(pred_data_train['total_pct'], pred_data_test['total_pct'])) 


# In[ ]:


print(cal_psi(pred_data_train['total_pct'], pred_data_oot['total_pct'])) 


# In[ ]:


print(cal_psi(pred_data_test['total_pct'], pred_data_oot['total_pct'])) 


# # 6.oot时间外样本测试

# In[ ]:


oot['lending_month'] = oot['lending_time'].apply(lambda x: str(x)[0:7])


# In[ ]:


for i in list(oot['lending_month'].unique()):
    print(i)
    tmp = oot[oot['lending_month']==i]
    tmp['prob'] = xgb_model.predict_proba(tmp[cols])[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(tmp['target'], tmp['prob'], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    ks = max(tpr-fpr)
    print('train AUC: ', auc)
    print('train KS: ', ks)
    print('-----------------------------------')


# # 7.模型保存

# In[ ]:


import pickle


# In[ ]:


# 保存模型
pickle.dump(xgb_model, open(r"D:\liuyedao\B卡开发\保存模型\B卡_xgb_model_20230916.pkl", "wb"))


# In[ ]:


# # 读取模型
# xgb_model_pkl = pickle.load(open(r"D:\liuyedao\B卡开发\保存模型\B卡_xgb_model.pkl", "rb"))


# In[ ]:


pipline_xgb_model = PMMLPipeline([("classfier",xgb_model)])
pipline_xgb_model.fit(X_train[cols], y_train)


# In[ ]:


sklearn2pmml(pipline_xgb_model, r'D:\liuyedao\B卡开发\保存模型\B卡_xgb_model_20230916.pmml', with_repr=True)


# In[ ]:


print(len(cols))


# In[ ]:


data_target['prob'] = xgb_model.predict_proba(data_target[cols])[:,1]
data_target['score'] = data_target['prob'].apply(lambda x: Prob2Score(x)).round(0)
tmp = data_target[['order_no','lending_time','channel_id','prob','score']+cols]
tmp.info()
tmp.head()
tmp.to_csv(r'D:\liuyedao\B卡开发\xgb_result\score_xgb_部署测试样例.csv',index=False)


# # 8单变量分析 

# In[ ]:


train_selected.info()


# In[ ]:


import toad
c = toad.transform.Combiner()
c.fit(train_selected, y='target', method='dt', min_samples=2, n_bins=10, empty_separate=True) 


# In[ ]:


train_selected_bins = c.transform(train_selected, labels=True)

valid_selected = valid[train_selected.columns]
valid_selected_bins = c.transform(valid_selected, labels=True)


# In[ ]:


df_result = pd.DataFrame()
for col in train_selected_bins.columns[:-1]:
    tmp = regroup(train_selected_bins, col, target='target')
    df_result = pd.concat([df_result, tmp], axis=0)
    
df_result_valid = pd.DataFrame()
for col in valid_selected_bins.columns[:-1]:
    tmp = regroup(valid_selected_bins, col, target='target')
    df_result_valid = pd.concat([df_result_valid, tmp], axis=0)
    
tmp = pd.merge(df_result, df_result_valid, how='left', on=['bins', 'varsname'], suffixes=['_train', '_valid'])
tmp.to_excel(r'D:\liuyedao\B卡开发\xgb_result\B卡_变量分箱_dt_bins10_20230916_xgb_v2.xlsx',index=False)




#==============================================================================
# File: B卡建模_Xgb_174渠道.py
#==============================================================================

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import toad
import os 
from datetime import datetime
from sklearn.model_selection import train_test_split
from IPython.core.interactiveshell import InteractiveShell
import warnings
import gc
from statistics import mode
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, cross_validate, cross_val_score
import time
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from pypmml import Model

warnings.filterwarnings("ignore")
InteractiveShell.ast_node_interactivity = 'all'
pd.set_option('display.max_row',None)
pd.set_option('display.width',1000)


# In[2]:


os.getcwd()


# In[3]:


# 运行函数脚本
get_ipython().run_line_magic('run', 'function.ipynb')


# # 1.读取数据集

# In[4]:


filepath = r'D:\liuyedao\B卡开发\三方数据匹配'
data = pd.read_csv(filepath + r'\order_20230927.csv') 


# In[5]:


usecols = ['order_no','user_id','channel_id','apply_date',
    'target'
    ,'value_026_bairong_12'
,'als_m12_cell_nbank_max_monnum'
,'model_score_01_moxingfen_7'
,'model_score_01_rong360_4'
,'model_score_01_tianchuang_8'
,'value_031_bairong_12'
,'value_027_bairong_12'
,'als_m3_id_nbank_night_orgnum'
,'value_040_baihang_6'
,'als_m6_id_nbank_max_inteday'
]


# In[6]:


df_base = data.query("channel_id==174 & apply_date<'2023-07-01' & apply_date>='2022-10-01'")[usecols]
# 删除全部是空值的列
df_base.dropna(how='all', axis=1, inplace=True)
df_base.dropna(how='all', axis=1, inplace=True)
df_base.reset_index(drop=True, inplace=True)
df_base.info()
df_base.head()


# In[ ]:


to_drop = list(df_base.select_dtypes(include='object').columns)
print(to_drop)


# In[ ]:


df_base.drop(['operationType', 'swift_number', 'name', 'mobileEncrypt', 'orderNo', 'idCardEncrypt'],
             axis=1,inplace=True)


# In[7]:


# 小于0为异常值，转为空值
for col in df_base.select_dtypes(include='float64').columns[1:]:
    df_base[col] = df_base[col].mask(df_base[col]<0)


# In[8]:


df_behavior = pd.read_csv(r'D:\liuyedao\B卡开发\lr_xgb_174\b卡衍生变量_20231013.csv')
df_behavior.head()


# In[9]:


df_base = pd.merge(df_base, df_behavior, how='left', on='user_id')
print(df_base.shape, df_base.order_no.nunique())


# In[10]:


df_pudao_3 = pd.read_csv(r'D:\liuyedao\B卡开发\lr_xgb_174\order_pudao_3_diff_20231013.csv')
df_bairong_1 = pd.read_csv(r'D:\liuyedao\B卡开发\lr_xgb_174\order_bairong_1_diff_20231013.csv')


# In[11]:


df_base = pd.merge(df_base, df_pudao_3, how='left', on='order_no')
print(df_base.shape, df_base.order_no.nunique())


# In[12]:


df_base = pd.merge(df_base, df_bairong_1, how='left', on='order_no')
print(df_base.shape, df_base.order_no.nunique())


# ## 训练数据集

# In[13]:


df_train = df_base.query("apply_date>='2022-10-01' & apply_date<'2023-05-01' & target in [0.0, 1.0]")
# df_train.groupby(['lending_month','target'])['order_no'].count().unstack()


# In[14]:


print(df_train.target.value_counts())
print(df_train.target.value_counts(normalize=True))


# ## oot测试数据集

# In[15]:


df_oot = df_base.query("apply_date>='2023-05-01' & apply_date<'2023-06-01' & target in [0.0, 1.0]")
print(df_oot.target.value_counts())
print(df_oot.target.value_counts(normalize=True))


# In[16]:


print(df_train.shape, df_oot.shape)


# # 2.数据处理

# In[17]:


train = df_train.copy()
oot = df_oot.copy()

# oot = df_train.copy()
# train = df_oot.copy()


# In[18]:


# to_drop = ['order_no','user_id','channel_id','apply_date']
to_drop = train.columns[0:4].to_list()
print(to_drop)


# In[19]:


train = train.drop(to_drop, axis=1)
oot = oot.drop(to_drop, axis=1)


# In[20]:


to_drop = list(train.select_dtypes(include='object').columns)
print(to_drop)


# In[ ]:


# train = train.drop(to_drop, axis=1)
# oot = oot.drop(to_drop, axis=1)


# In[ ]:


# to_drop = list(train.columns[train.columns.str.contains('score')])
# print(to_drop)


# In[ ]:


# train = train.drop(to_drop, axis=1)
# oot = oot.drop(to_drop, axis=1)


# In[ ]:


train.info()


# In[ ]:


oot.info()


# In[ ]:


print(train.target.value_counts())
print(oot.target.value_counts())


# In[ ]:


print(train.target.value_counts(normalize=True))
print(oot.target.value_counts(normalize=True))


# # 3.数据探索分析

# In[ ]:


# # 小于0为异常值，转为空值
# train = train.mask(train<0)
# oot = oot.mask(oot<0)


# In[21]:


# 删除全部是空值的列
train.dropna(how='all', axis=1, inplace=True)
oot.dropna(how='all', axis=1, inplace=True)


# In[22]:


train_explore = toad.detect(train.drop('target', axis=1))
oot_explore = toad.detect(oot.drop('target', axis=1))

modes = train.drop('target', axis=1).apply(mode)
mode_counts = train.drop('target', axis=1).eq(modes).sum()
train_modes = pd.DataFrame({'mode':modes, 'mode_num':mode_counts}, index=list(train.columns[1:]))

modes = oot.drop('target', axis=1).apply(mode)
mode_counts = oot.drop('target', axis=1).eq(modes).sum()
oot_modes = pd.DataFrame({'mode':modes, 'mode_num':mode_counts}, index=list(oot.columns[1:]))

train_isna = pd.DataFrame(train.drop('target', axis=1).isnull().sum(), columns=['missing_num'])
oot_isna = pd.DataFrame(oot.drop('target', axis=1).isnull().sum(), columns=['missing_num'])


train_iv = toad.quality(train,'target',iv_only=False)
oot_iv = toad.quality(oot,'target',iv_only=False)


# In[23]:


train_df_explore = pd.concat([train_explore, train_modes, train_isna, train_iv.drop('unique',axis=1)], axis=1)

train_df_explore['no_null_num'] = train_df_explore['size'] - train_df_explore['missing_num']
train_df_explore['miss_rate'] = train_df_explore['missing_num'] / train_df_explore['size']
train_df_explore['mode_pct_all'] = train_df_explore['mode_num']/train_df_explore['size']
train_df_explore['mode_pct_notna'] = train_df_explore['mode_num']/train_df_explore['no_null_num']


# In[ ]:


oot_df_explore = pd.concat([oot_explore, oot_modes, oot_isna, oot_iv.drop('unique',axis=1)], axis=1)

oot_df_explore['no_null_num'] = oot_df_explore['size'] - oot_df_explore['missing_num']
oot_df_explore['miss_rate'] = oot_df_explore['missing_num'] / oot_df_explore['size']
oot_df_explore['mode_pct_all'] = oot_df_explore['mode_num']/oot_df_explore['size']
oot_df_explore['mode_pct_notna'] = oot_df_explore['mode_num']/oot_df_explore['no_null_num']


# In[ ]:


df_iv = pd.merge(train_iv, oot_iv, how='inner',left_index=True, right_index=True,suffixes=['_train','_oot'])
df_iv['diff_iv'] = df_iv['iv_oot']-df_iv['iv_train']
df_iv['rate_iv'] = df_iv['iv_oot']/df_iv['iv_train'] - 1


# In[ ]:


path = r'D:\liuyedao\B卡开发\xgb_result\20231023\\'


# In[ ]:


writer=pd.ExcelWriter(path + 'B卡_探索性分析_58同城_20231023.xlsx')

train_df_explore.to_excel(writer,sheet_name='train_df_explore')
oot_df_explore.to_excel(writer,sheet_name='oot_df_explore')

# train_df_iv.to_excel(writer,sheet_name='train_df_iv')
# oot_df_iv.to_excel(writer,sheet_name='oot_df_iv')

df_iv.to_excel(writer,sheet_name='df_iv')

writer.save()


# # 4.特征粗筛选

# In[ ]:


# 删除缺失率大于0.85/删除枚举值只有一个/删除方差等于0/删除集中度大于0.85

to_drop_missing = list(train_df_explore[train_df_explore.miss_rate>=0.85].index)
print(len(to_drop_missing))
to_drop_unique = list(train_df_explore[train_df_explore.unique==1].index)
print(len(to_drop_unique))
to_drop_std = list(train_df_explore[train_df_explore.std_or_top2==0].index)
print(len(to_drop_std))
to_drop_mode = list(train_df_explore[train_df_explore.mode_pct_notna>=0.85].index)
print(len(to_drop_mode))
to_drop_mode2 = list(train_df_explore[train_df_explore.mode_pct_all>=0.85].index)
print(len(to_drop_mode2))

to_drop_train = list(set(to_drop_missing+to_drop_unique+to_drop_std+to_drop_mode+to_drop_mode2))
print(len(to_drop_train))


# In[ ]:


# 删除缺失率大于0.85/删除枚举值只有一个/删除方差等于0/删除集中度大于0.85
to_drop_missing = list(oot_df_explore[oot_df_explore.miss_rate>=0.85].index)
print(len(to_drop_missing))
to_drop_unique = list(oot_df_explore[oot_df_explore.unique==1].index)
print(len(to_drop_unique))
to_drop_std = list(oot_df_explore[oot_df_explore.std_or_top2==0].index)
print(len(to_drop_std))
to_drop_mode = list(oot_df_explore[oot_df_explore.mode_pct_notna>=0.85].index)
print(len(to_drop_mode))
to_drop_mode2 = list(oot_df_explore[oot_df_explore.mode_pct_all>=0.85].index)
print(len(to_drop_mode))

to_drop_oot = list(set(to_drop_missing+to_drop_unique+to_drop_std+to_drop_mode+to_drop_mode2))
print(len(to_drop_oot))


# In[ ]:


train_1 = train.drop(to_drop_train, axis=1)
print(train_1.shape)

oot_1 = oot.drop(to_drop_oot, axis=1)
print(oot_1.shape)


# In[ ]:


# 共同的变量
sim_cols = list(set(train_1.drop('target',axis=1).columns).intersection(set(oot_1.drop('target',axis=1).columns)))
print(len(sim_cols))

train_2 = train_1[['target']+sim_cols]
oot_2 = oot_1[['target']+sim_cols]
print(train_2.shape, oot_2.shape)


# In[ ]:


# psi/iv稳定性筛选
to_drop_iv = list(df_iv.query("iv_train<=0.02 | iv_oot<=0.02").index)
print(len(to_drop_iv))

psi = toad.metrics.PSI(train_2.drop('target', axis=1), oot_2.drop('target',axis=1))
to_drop_psi = list(psi[psi>=0.25].index)
print(len(to_drop_psi))

to_drop = []
for col in list(set(to_drop_iv+to_drop_psi)):
    if col in train_2.columns:
        to_drop.append(col)
print(len(to_drop))

train_2.drop(to_drop, axis=1, inplace=True)
print(train_2.shape)


# In[ ]:


to_exclude = ['value_026_bairong_12'
,'als_m12_cell_nbank_max_monnum'
,'model_score_01_moxingfen_7'
,'model_score_01_rong360_4'
,'model_score_01_tianchuang_8'
,'value_031_bairong_12'
,'value_027_bairong_12'
,'als_m3_id_nbank_night_orgnum'
,'value_040_baihang_6'
,'als_m6_id_nbank_max_inteday'
]


# In[ ]:


# iv值/相关性筛选
train_selected, dropped = toad.selection.select(train_2, target='target', empty=0.85, iv=0.02,
                                                corr=0.7,
                                                return_drop=True, exclude=to_exclude)
train_selected.shape


# In[ ]:


for i in dropped.keys():
    print("变量筛选维度：{}， 共剔除变量{}".format(i, len(dropped[i])))


# In[ ]:


# 人工剔除
to_drop_man = ['model_score_01_baihang_1','model_score_01_q_tianchuang_1','model_score_01_r_tianchuang_1',
           'model_score_01_1_bileizhen_1',
           'model_score_01_2_bileizhen_1','model_score_01_3_bileizhen_1','value_011_moxingfen_7',
           'value_012_pudao_5',
           'model_score_01_pudao_11','model_score_01_pudao_12','model_score_01_pudao_16',
           'model_score_01_2_bairong_8','model_score_01_3_bairong_8','model_score_01_7_bairong_8',
           'model_score_01_8_bairong_8']

to_drop = []
for col in to_drop_man:
    if col in train_selected.columns:
        to_drop.append(col)
print(len(to_drop))

train_selected.drop(to_drop, axis=1, inplace=True)
print(train_selected.shape)


# ## 变量分箱

# In[ ]:


# 第一次分箱
c = toad.transform.Combiner()
c.fit(train_selected, y='target', method='chi', min_samples=2, n_bins=10, empty_separate=True) 
bins_result = c.export()


# In[ ]:


bins_result 


# In[ ]:


train_selected_bins = c.transform(train_selected, labels=True)
oot_selected_bins = c.transform(oot[train_selected.columns], labels=True)


# In[ ]:


bins_dict_train = {}
for col in train_selected_bins.columns[1:]:
    bins_dict_train[col] = regroup(train_selected_bins, col, target='target')
    
df_result_train = pd.concat(list(bins_dict_train.values()), axis=0, ignore_index =True)


# In[ ]:


bins_dict_oot = {}
for col in oot_selected_bins.columns[1:]:
    bins_dict_oot[col] = regroup(oot_selected_bins, col, target='target')
    
df_result_oot = pd.concat(list(bins_dict_oot.values()), axis=0, ignore_index =True)


# In[ ]:


df_result = pd.merge(df_result_train, df_result_oot, how='inner', on=['varsname','bins'],suffixes=['_train','_oot'])


# In[ ]:


df_result.to_excel(path + 'B卡_chi_58同城_20231020_v2.xlsx',index=False)


# In[ ]:


to_drop_bins1 = []
for col in list(bins_result.keys()):
    if len(bins_result[col])==1:
        to_drop_bins1.append(col)
print(len(to_drop_bins1))
to_drop_bins2 = list(set(df_result[(df_result['iv_train']<0.02)]['varsname']))
print(len(to_drop_bins2))
to_drop_bins = list(set(to_drop_bins1 + to_drop_bins2))
print(len(to_drop_bins))


# In[ ]:


train_selected_1 = train_selected.drop(to_drop_bins, axis=1)
train_selected_1.shape


# In[ ]:


train_selected_bins_1 = train_selected_bins[train_selected_1.columns]
oot_selected_bins_1 = oot_selected_bins[train_selected_bins_1.columns]


# In[ ]:


from toad.plot import bin_plot, badrate_plot


# In[ ]:


# col = list(train_selected_bins_1.drop('target',axis=1).columns)[1]

# bin_plot(train_selected_bins_1, x=col, target='target')
# bin_plot(oot_selected_bins_1, x=col, target='target')


# In[ ]:


# 连续变量分箱主函数
def ContinueVarBins(data, col, flag='target', cutbins=[]):
    df = data[[flag, col]]
    df = df[~df[col].isnull()].reset_index(drop=True)
    df['bins'] = pd.cut(df[col], cutbins, duplicates='drop', right=False, precision=4)
    regroup = pd.DataFrame()
    regroup['bins'] = df.groupby(['bins'])[col].max()
    regroup['total'] = df.groupby(['bins'])[flag].count()
    regroup['bad'] = df.groupby(['bins'])[flag].sum()
    regroup['good'] = regroup['total'] - regroup['bad']
    regroup.drop(['total'], axis=1, inplace=True)
    np_regroup = np.array(regroup)
    np_regroup = MergeZero(np_regroup)
    np_regroup = MinPct(np_regroup)
    np_regroup = MonTone(np_regroup)
    regroup = pd.DataFrame(index=np.arange(np_regroup.shape[0]))
    regroup['bins'] = np_regroup[:,0]
    regroup['total'] = np_regroup[:,1] + np_regroup[:,2]
    regroup['bad'] = np_regroup[:,1]
    regroup['good'] = np_regroup[:,2]
    cutoffpoints = list(np_regroup[:,0])
    cutoffpoints = [float('-inf')] + cutoffpoints
    # 最大值分割点，转换最小值分割点
    df['bins_new'] = pd.cut(df[col], cutoffpoints, duplicates='drop', right=True, precision=4)
    tmp = pd.DataFrame()
    tmp['bins'] = df.groupby(['bins_new'])[col].min()
    cutoffpoints = list(tmp['bins'])  
    
    return cutoffpoints


# In[ ]:


# 自动调整分箱
adj_bins = {}
for col in list(train_selected_1.columns)[1:]:
    tmp = np.isnan(bins_result[col])
    cutbins = [bins_result[col][x] for x in range(len(tmp)) if not tmp[x]]
    if len(cutbins)>0:
        cutbins = [float('-inf')] + cutbins + [float('inf')]
        cutoffpoints = ContinueVarBins(train_selected, col, flag='target', cutbins=cutbins)
        tmp_value1 = train_selected[col].min()
        tmp_value2 = cutoffpoints[0]
        if tmp_value1==tmp_value2:
            adj_bins[col] = [-99998] + cutoffpoints[1:]
        else:
            adj_bins[col] = [-99998] + cutoffpoints


# In[ ]:


# 调整分箱:空值单独一箱
train_selected_1.fillna(-99999, inplace=True)

# 更新分箱
c.update(adj_bins)


# In[ ]:


# c.export()
# c.load(dict)
# c.transform(dataframe, labels=False)


# ## WOE转换

# In[ ]:


transer = toad.transform.WOETransformer()
train_woe = transer.fit_transform(c.transform(train_selected), train_selected['target'],
                                  exclude=['target'])
train_woe.shape


# In[ ]:


train_selected_woe, dropped = toad.selection.select(train_woe, target='target', empty=0.85,
                                                    iv=0.02, corr=0.7, return_drop=True, exclude=to_exclude)
train_selected_woe.shape


# In[ ]:


for i in dropped.keys():
    print("变量筛选维度：{}， 共剔除变量{}".format(i, len(dropped[i])))


# In[ ]:


oot_woe = transer.transform(c.transform(oot))
oot_selected_woe = oot_woe[list(train_selected_woe.columns)]
oot_selected_woe.shape


# In[ ]:


psi = toad.metrics.PSI(train_selected_woe.drop('target',axis=1), oot_selected_woe.drop('target',axis=1))
to_drop_psi = list(psi[psi>0.05].index)
print(len(to_drop_psi))


# In[ ]:


train_selected_woe_psi = train_selected_woe.drop(to_drop_psi, axis=1)
oot_selected_woe_psi = oot_selected_woe[train_selected_woe_psi.columns]


# ## 逐步回归

# In[ ]:


print(train_woe.shape)
final_data = toad.selection.stepwise(train_woe, target='target', estimator='ols',
                                     direction='both', criterion='aic', exclude=to_exclude)
print(final_data.shape)


# In[ ]:


final_oot = oot_selected_woe_psi[final_data.columns]


# In[ ]:


# 初次建模变量
cols = list(final_data.drop(['target'], axis=1).columns)
print(len(cols))
print(cols)


# # 5.xgb模型训练

# ## 5.1 Xgb建模

# In[24]:


# cols.remove('model_score_01_rong360_4')
# cols.append('model_score_01_tianchuang_7')
# cols.append('model_score_01_rong360_4')
cols = [
 'als_m12_cell_nbank_max_monnum_diff'
,'als_m3_id_nbank_min_monnum_diff'
,'model_score_01_tianchuang_8'
,'ppdi_m1_id_nbank_orgnum_diff'
,'value_026_bairong_12'
,'als_m6_id_nbank_max_inteday'
,'model_score_01_rong360_4'
,'als_m12_cell_nbank_max_monnum'
,'value_031_bairong_12'
,'value_027_bairong_12'
,'model_score_01_moxingfen_7'
]


# In[ ]:


opt_best = {'target': 0.6487565005430443,
            'params': {'gamma': 0.75, 'learning_rate': 0.45, 'max_depth': 6.0, 'min_child_weight': 9.0,
                       'n_estimators': 9, 'reg_alpha': 0, 'reg_lambda': 300}
           }


# In[25]:


opt_best = {'target': 0.6487565005430443,
            'params': {'gamma': 0.75, 'learning_rate': 0.45, 'max_depth': 6.0, 'min_child_weight': 9.0,
                       'n_estimators': 12, 'reg_alpha': 0, 'reg_lambda': 290}
           }


# In[26]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 


# In[27]:


xgb_model.fit(train[cols] ,train['target'])


# In[28]:


# 对训练集进行预测

from sklearn import metrics

y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[ ]:


importance_type = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),orient='index')
importance_type.columns = ['value']
importance_type = importance_type.sort_values('value', ascending=False)
importance_type


# In[29]:


xgb_model.fit(oot[cols] ,oot['target'])


# In[30]:


# 对训练集进行预测

from sklearn import metrics

y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[ ]:


importance_type = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),orient='index')
importance_type.columns = ['value']
importance_type = importance_type.sort_values('value', ascending=False)
importance_types


# In[ ]:


cols = ['als_m12_cell_nbank_max_monnum_diff'
,'als_m3_id_nbank_min_monnum_diff'
,'model_score_01_tianchuang_8'
,'ppdi_m1_id_nbank_orgnum_diff'
,'value_026_bairong_12'
,'als_m6_id_nbank_max_inteday'
,'model_score_01_rong360_4'
,'als_m12_cell_nbank_max_monnum'
,'value_031_bairong_12'
,'value_027_bairong_12'
,'model_score_01_moxingfen_7'
]


# In[ ]:


xgb_model.fit(train[cols] ,train['target'])


# In[ ]:


# 对训练集进行预测

from sklearn import metrics

y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[ ]:


importance_type = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),orient='index')
importance_type.columns = ['value']
importance_type = importance_type.sort_values('value', ascending=False)
importance_type


# In[ ]:


xgb_model.fit(oot[cols] ,oot['target'])


# In[ ]:


# 对训练集进行预测
y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[ ]:


importance_type = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),orient='index')
importance_type.columns = ['value']
importance_type = importance_type.sort_values('value', ascending=False)
importance_type


# In[ ]:


len(xgb_model.get_booster().get_score(importance_type='gain').keys())


# In[ ]:


importance_type = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),orient='index')
importance_type.columns = ['value']
importance_type = importance_type.sort_values('value', ascending=False)
importance_type


# In[ ]:


# cols = list(importance_type.head(11).index)
cols = list(importance_type[importance_type.value>1.5].index)
print(cols, len(cols))


# In[ ]:


cols = [
 'als_m12_cell_nbank_ca_orgnum_diff'
,'model_score_01_rong360_4'
,'model_score_01_moxingfen_7'
,'value_054_bairong_1'
,'value_060_baihang_6'
,'als_d15_id_nbank_nsloan_orgnum'
,'model_score_01_ruizhi_6'
,'als_fst_id_nbank_inteday'
,'value_026_bairong_12'
]


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 


# In[ ]:


xgb_model.fit(train[cols] ,train['target'])


# In[ ]:


# 对训练集进行预测
y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[ ]:


importance_type = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),orient='index')
importance_type.columns = ['value']
importance_type = importance_type.sort_values('value', ascending=False)
importance_type


# In[ ]:


import pickle

# 保存模型
pickle.dump(xgb_model, open(r"D:\liuyedao\B卡开发\lr_xgb_174_20231017.pkl", "wb"))


# In[ ]:


plt.plot(fpr, tpr, color='darkorange', lw=2, label='trian ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr_oot, tpr_oot, color='blue', lw=2, label='test ROC curve (area = %0.2f)' % roc_auc_oot)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="best")
plt.show()


# ### 贝叶斯调参

# In[ ]:


import time 
def xgb_cv(X, y, learning_rate, n_estimators, max_depth, min_child_weight, gamma, reg_alpha, reg_lambda):
    xgb_model = xgb.XGBClassifier(booster='gbtree',
                              learning_rate=learning_rate,
                              n_estimators = int(n_estimators),
                              max_depth = int(max_depth),
                              min_child_weight = int(min_child_weight),
                              gamma=gamma,
                              subsample=1,
                              colsample_bytree=1,
                              objective = "binary:logistic",
                              nthread = 1,
                              n_jobs = -1,
                              random_state = 1,
                              scale_pos_weight = 1,
                              reg_alpha = int(reg_alpha),
                              reg_lambda = int(reg_lambda))   
    cv = KFold(n_splits=5, shuffle=True, random_state=11)
    valid_loss = cross_validate(xgb_model, X, y, scoring='roc_auc',cv=cv,n_jobs=-1,error_score='raise')
    return np.mean(valid_loss['test_score'])

def bayes_opt_xgb(X, y, init_points, n_iter):
    def xgb_cross_valid(learning_rate, n_estimators, max_depth,
                        min_child_weight, gamma, reg_alpha, reg_lambda):
        target = xgb_cv(X, y, learning_rate, n_estimators, max_depth,
                     min_child_weight, gamma, reg_alpha, reg_lambda)
        return target 
    
    optimizer = BayesianOptimization(xgb_cross_valid,
                                {
                                    'max_depth':(3, 6),
                                    'min_child_weight':(6, 9),
                                    'n_estimators':(9, 15),
                                    'learning_rate':(0, 1),
                                    'gamma':(0, 1),
                                    'reg_alpha':(0, 0),
                                    'reg_lambda':(250, 300)
                                })
    
    start_time = time.time()
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    end_time = time.time()
    print("花费时间：", (end_time-start_time)/60)
    opt_best = optimizer.max
    print("final result:" ,opt_best)
    
    return opt_best


# In[ ]:


opt_best = bayes_opt_xgb(train[cols], train['target'], 7, 50)


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 


# In[ ]:


xgb_model.fit(train[cols] ,train['target'])


# In[ ]:


# 对训练集进行预测
y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[ ]:


xgb_model.fit(oot[cols] ,oot['target'])


# In[ ]:


# 对训练集进行预测
y_pred = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)
print('train AUC: ', roc_auc)
print('train KS: ', ks)

# 对测试集进行预测
y_pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], y_pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)
print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[ ]:


importance_type = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),orient='index')
importance_type.columns = ['value']
importance_type = importance_type.sort_values('value', ascending=False)
importance_type


# In[ ]:


importance_type.shape


# ### 网格调参

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=1.0,
                          n_estimators = 9,
                          max_depth = 5,
                          min_child_weight = 6,
                          gamma=1.0,
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = 0,
                          reg_lambda = 281) 
# xgb_model.fit(X_train, y_train)
# xgb_model.fit(oot[cols_new] ,oot['target'])


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
# xgb_model.fit(X_train, y_train)
# xgb_model.fit(oot[cols_new] ,oot['target'])


# In[ ]:


param_test1 = {'max_depth':[2,3,4], 'min_child_weight':[5,6,7,8,9]}

gsearch = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test1, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch.fit(train[cols], train['target'])

print('gsearch1.best_params_', gsearch.best_params_)
print('gsearch1.best_score_', gsearch.best_score_)


# In[ ]:


train.shape


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate= opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = 4,
                          min_child_weight = 8,
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 

param_test3 = {'n_estimators':[i for i in range(5,35)]}

gsearch3 = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test3, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch3.fit(train[cols], train['target'])

print('gsearch3.best_params_', gsearch3.best_params_)
print('gsearch3.best_score_', gsearch3.best_score_)


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = 12,
                          max_depth = 6,
                          min_child_weight =9,
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 

param_test2 = {'learning_rate':[i/20.0 for i in range(0,20)]}
gsearch2 = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test2, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch2.fit(train[cols], train['target'])

print('gsearch2.best_params_', gsearch2.best_params_)
print('gsearch2.best_score_', gsearch2.best_score_)


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate= 0.65,
                          n_estimators = 12,
                          max_depth = 6,
                          min_child_weight = 9,
                          gamma=opt_best['params']['gamma'],
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
param_test4 = {'gamma':[i/10.0 for i in range(10)]}

gsearch4 = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test4, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch4.fit(train[cols], train['target'])

print('gsearch4.best_params_', gsearch4.best_params_)
print('gsearch4.best_score_', gsearch4.best_score_)


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate= 1.0,
                          n_estimators = 9,
                          max_depth = 5,
                          min_child_weight = 6,
                          gamma= 1.0,
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = 0,
                          reg_lambda = 281) 
param_test5 = {'reg_lambda': [i for i in range(250,351)]}

gsearch5 = GridSearchCV(
    estimator = xgb_model,
    param_grid=param_test5, 
    scoring='roc_auc', 
    n_jobs=-1, 
    cv=5)
gsearch5.fit(train[cols], train['target'])

print('gsearch5.best_params_', gsearch5.best_params_)
print('gsearch5.best_score_', gsearch5.best_score_)


# In[ ]:


# xgb_model = xgb.XGBClassifier(booster='gbtree',
#                           learning_rate= 0.05,
#                           n_estimators = 200,
#                           max_depth = 6,
#                           min_child_weight = 2,
#                           gamma= 0.9,
#                           objective = "binary:logistic",
#                           nthread = 1,
#                           n_jobs = -1,
#                           random_state = 1,
#                           scale_pos_weight = 1,
#                           reg_alpha = 10,
#                           reg_lambda = int(opt_best['params']['reg_lambda'])) 
# param_test6 = {'reg_lambda':[0, 0.001, 0.1, 1, 10, 100,300,500]}

# gsearch6 = GridSearchCV(
#     estimator = xgb_model,
#     param_grid=param_test6, 
#     scoring='roc_auc', 
#     n_jobs=-1, 
#     cv=5)
# gsearch6.fit(train[cols], train['target'])

# print('gsearch6.best_params_', gsearch6.best_params_)
# print('gsearch6.best_score_', gsearch6.best_score_)


# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate= 1.0,
                          n_estimators = 9,
                          max_depth = 5,
                          min_child_weight = 6,
                          gamma= 1.0,
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = 0,
                          reg_lambda = 301) 


# In[ ]:


xgb_model.fit(train[cols] ,train['target'])


# In[ ]:


# 对训练集进行预测 train_selected[cols] ,train_selected['target']
pred_train = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], pred_train, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)

print('train AUC: ', roc_auc)
print('train KS: ', ks)

# 对oot测试集进行预测
pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)

print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# In[ ]:


xgb_model.fit(oot[cols] ,oot['target'])


# In[ ]:


# 对训练集进行预测 train_selected[cols] ,train_selected['target']
pred_train = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], pred_train, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)

print('train AUC: ', roc_auc)
print('train KS: ', ks)

# 对oot测试集进行预测
pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)

print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# ## 5.2 Xgb模型技术效果

# In[ ]:


xgb_model = xgb.XGBClassifier(booster='gbtree',
                          learning_rate=opt_best['params']['learning_rate'],
                          n_estimators = int(opt_best['params']['n_estimators']),
                          max_depth = int(opt_best['params']['max_depth']),
                          min_child_weight = int(opt_best['params']['min_child_weight']),
                          gamma=opt_best['params']['gamma'],
                          subsample=1,
                          colsample_bytree=1,
                          objective = "binary:logistic",
                          nthread = 1,
                          n_jobs = -1,
                          random_state = 1,
                          scale_pos_weight = 1,
                          reg_alpha = int(opt_best['params']['reg_alpha']),
                          reg_lambda = int(opt_best['params']['reg_lambda'])) 
# xgb_model.fit(X_train, y_train)
xgb_model.fit(oot[cols] ,oot['target'])


# In[ ]:


sorted(xgb_model.get_booster().get_score(importance_type='gain').items(),key = lambda x:x[1],reverse=True)


# In[ ]:


plot_importance(xgb_model, importance_type='gain')


# In[ ]:


# # 对训练集进行预测
# pred_train = xgb_model.predict_proba(X_train)[:,1]
# fpr, tpr, thresholds = metrics.roc_curve(y_train, pred_train, pos_label=1)
# roc_auc = metrics.auc(fpr, tpr)
# ks = max(tpr-fpr)
# print('train KS: ', ks)
# print('train AUC: ', roc_auc)


# In[ ]:


# 对训练集进行预测 train_selected[cols] ,train_selected['target']
pred_train = xgb_model.predict_proba(train[cols])[:,1]
fpr, tpr, thresholds = metrics.roc_curve(train['target'], pred_train, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
ks = max(tpr-fpr)

print('train AUC: ', roc_auc)
print('train KS: ', ks)


# In[ ]:


# # 对测试集进行预测 X_train, X_test, y_train, y_test 
# pred_test = xgb_model.predict_proba(valid[cols])[:,1]
# fpr_test, tpr_test, thresholds_test = metrics.roc_curve(valid['target'], pred_test, pos_label=1)
# roc_auc_test = metrics.auc(fpr_test, tpr_test)
# ks_test = max(tpr_test-fpr_test)

# print('test AUC: ', roc_auc_test)
# print('test KS: ', ks_test)


# In[ ]:


# 对oot测试集进行预测
pred_oot = xgb_model.predict_proba(oot[cols])[:,1]
fpr_oot, tpr_oot, thresholds_oot = metrics.roc_curve(oot['target'], pred_oot, pos_label=1)
roc_auc_oot = metrics.auc(fpr_oot, tpr_oot)
ks_oot = max(tpr_oot-fpr_oot)

print('oot AUC: ', roc_auc_oot)
print('oot KS: ', ks_oot)


# ## 5.3 模型业务效果

# In[ ]:


def Prob2Score(prob, base_odds=35, base_score=700, pdo=60, rate=2) :
    # 将概率转化成分数且为正整数
    y = np.log((1 - prob) / prob)
    factor = pdo/np.log(rate)
    offset = base_score - factor * np.log(base_odds)
    score = offset +  factor * (y)
    
    return score


# In[ ]:


print(len(cols))


# In[ ]:


# 业务效果-训练集
train['prob'] = xgb_model.predict_proba(train[cols])[:,1]
train['score'] = train['prob'].apply(lambda x: Prob2Score(x)).round(0)
pred_data = toad.metrics.KS_bucket(train['score'], train['target'], bucket=10, method='step')
pred_data


# In[ ]:


cut_bins = [float('-inf'), 670.0, 689.0, 718.0, 727.0, float('inf')]
print(cut_bins)


# In[ ]:


# 业务效果-训练集
train['prob'] = xgb_model.predict_proba(train[cols])[:,1]
train['score'] = train['prob'].apply(lambda x: Prob2Score(x)).round(0)
# train_ = pd.merge(X_train, y_train, how='inner', left_index=True, right_index=True)
train['bins'] = pd.cut(train['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


# # 业务效果-验证集
# valid['prob'] = xgb_model.predict_proba(valid[cols])[:,1]
# valid['score'] = valid['prob'].apply(lambda x: Prob2Score(x)).round(0)
# # test = pd.merge(X_test, y_test, how='inner', left_index=True, right_index=True)
# valid['bins'] = pd.cut(valid['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


# 业务效果-oot
oot['prob'] = xgb_model.predict_proba(oot[cols])[:,1]
oot['score'] = oot['prob'].apply(lambda x: Prob2Score(x)).round(0)
oot['bins'] = pd.cut(oot['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


pred_data_train = score_distribute(train, 'bins', target='target')
# pred_data_test = score_distribute(valid, 'bins', target='target')
pred_data_oot = score_distribute(oot, 'bins', target='target')


# In[ ]:


pred_data_oot


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_分数分布_xgb_58同城_20231017_v1"+'.xlsx')
pred_data_train.to_excel(writer,sheet_name='train')
# pred_data_test.to_excel(writer,sheet_name='test')
pred_data_oot.to_excel(writer,sheet_name='oot')
writer.save()


# In[ ]:


# 等频分箱
# 业务效果-训练集
train['prob'] = xgb_model.predict_proba(train[cols])[:,1]
train['score'] = train['prob'].apply(lambda x: Prob2Score(x)).round(0)
pred_data = toad.metrics.KS_bucket(train['score'], train['target'], bucket=10, method='quantile')
pred_data


# In[ ]:


cut_bins = [float('-inf')]+list(pred_data['min'])[1:]+[float('inf')]
print(cut_bins)


# In[ ]:


# 业务效果-训练集
train['prob'] = xgb_model.predict_proba(train[cols])[:,1]
train['score'] = train['prob'].apply(lambda x: Prob2Score(x)).round(0)
train['bins'] = pd.cut(train['score'], bins=cut_bins, include_lowest=True, right=False)

# # 业务效果-验证集
# valid['prob'] = xgb_model.predict_proba(valid[cols])[:,1]
# valid['score'] = valid['prob'].apply(lambda x: Prob2Score(x)).round(0)
# valid['bins'] = pd.cut(valid['score'], bins=cut_bins, include_lowest=True, right=False)

# 业务效果
oot['prob'] = xgb_model.predict_proba(oot[cols])[:,1]
oot['score'] = oot['prob'].apply(lambda x: Prob2Score(x)).round(0)
oot['bins'] = pd.cut(oot['score'], bins=cut_bins, include_lowest=True, right=False)


# In[ ]:


pred_data_train = score_distribute(train, 'bins', target='target')

# pred_data_test = score_distribute(valid, 'bins', target='target')

pred_data_oot = score_distribute(oot, 'bins', target='target')


# In[ ]:


pred_data_oot


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\lr_xgb_174\B卡_分数分布_xgb_quantile_58同城_20231017"+'.xlsx')
pred_data_train.to_excel(writer,sheet_name='train')
# pred_data_test.to_excel(writer,sheet_name='test')
pred_data_oot.to_excel(writer,sheet_name='oot')
writer.save()


# In[ ]:


def cal_psi(exp, act):
    psi = []
    for i in range(len(exp)):
        psi_i = (act[i] - exp[i])*np.log(act[i]/exp[i])
        psi.append(psi_i)
    return sum(psi)


# In[ ]:


# print(cal_psi(pred_data_train['total_pct'], pred_data_test['total_pct'])) 


# In[ ]:


print(cal_psi(pred_data_train['total_pct'], pred_data_oot['total_pct'])) 


# In[ ]:


print(cal_psi(pred_data_test['total_pct'], pred_data_oot['total_pct'])) 




#==============================================================================
# File: B卡建设_数据分析-5个渠道16717420680048005.py
#==============================================================================

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from datetime import datetime
import re
from IPython.core.interactiveshell import InteractiveShell
import warnings
import gc
import os

warnings.filterwarnings("ignore")
InteractiveShell.ast_node_interactivity = 'all'
pd.set_option('display.max_row',None)
pd.set_option('display.width',1000)


# In[2]:


os.getcwd()


# In[3]:


# 运行函数脚本
get_ipython().run_line_magic('run', 'function.ipynb')


# # 1.基础数据表

# ## 1.1授信表

# In[ ]:


# 授信表
cols_auth = ['user_id','order_no','id_no_des','auth_status','apply_date','apply_time',
             'channel_id','auth_credit_amount']
df_auth_167 = pd.read_csv(r'D:\juzi\0711\167\dwd_beforeloan_auth_examine_fd_167.csv',usecols=cols_auth)
df_auth_other = pd.read_csv(r'D:\juzi\0711\other\dwd_beforeloan_auth_examine_fd_other.csv',usecols=cols_auth)
df_auth_8005 = pd.read_csv(r'D:\liuyedao\转转渠道\dwd_beforeloan_auth_examine_fd_80005.csv',usecols=cols_auth)


# In[ ]:


df_auth_other = df_auth_other[df_auth_other['channel_id'].isin([167, 174, 206, 80004, 80005])]


# In[ ]:


df_auth = pd.concat([df_auth_167, df_auth_other, df_auth_8005],axis=0, ignore_index=True)


# In[ ]:


del df_auth_167, df_auth_other, df_auth_8005
gc.collect()


# ## 1.2.提现表

# In[ ]:


# 提现表 
cols_order = ['user_id','order_no','id_no_des','order_status','apply_date','apply_time',
              'channel_id','loan_amount']
df_order_167 = pd.read_csv(r'D:\juzi\0711\167\dwd_beforeloan_order_examine_fd_167.csv',usecols=cols_order)
df_order_other = pd.read_csv(r'D:\juzi\0711\other\dwd_beforeloan_order_examine_fd_other.csv',usecols=cols_order)
df_order_8005 = pd.read_csv(r'D:\liuyedao\转转渠道\dwd_beforeloan_order_examine_fd_80005.csv',usecols=cols_order)


# In[ ]:


df_order_other = df_order_other[df_order_other['channel_id'].isin([167,174,206,80004,80005])]


# In[ ]:


df_order = pd.concat([df_order_167, df_order_other, df_order_8005],axis=0)


# In[ ]:


df_order['apply_month'] = df_order['apply_date'].str[0:7]
tmp_order = df_order.groupby(['apply_month','channel_id'])['order_no'].count().unstack()


# In[ ]:


tmp_order.head()


# In[ ]:


df_order.info()


# In[ ]:


tmp_order = df_order.query("order_status==6 & apply_date>'2022-05-31'")
tmp_order['rank'] = tmp_order.groupby(['user_id'])['apply_time'].rank(ascending=True, method='dense')
tmp_order.head(3)


# In[ ]:


tmp_order.shape


# In[ ]:


tmp_order.query("user_id==134612852")


# In[ ]:


tmp_order = tmp_order.query(" rank>1")
tmp_order = tmp_order.reset_index()
tmp_order.shape


# In[ ]:


tmp_order.user_id.nunique()


# In[ ]:


tmp_order[['order_no','rank']].to_csv(r'D:\liuyedao\B卡开发\统计数据\老客户名单_v2.csv')


# In[ ]:


tmp_order = df_order.query("order_status==6").groupby(['user_id','channel_id'])['order_no'].count()
tmp_order = tmp_order.reset_index()
tmp_order = pd.pivot_table(tmp_order,values='user_id',index='order_no',columns='channel_id',aggfunc='count',margins=True,fill_value=0)
tmp_order.to_excel(r'D:\liuyedao\B卡开发\统计数据\各渠道订单次数.xlsx')
tmp_order


# ## 1.3.还款计划表

# In[8]:


cols_repay = ['order_no','user_id','id_no_des','channel_id','lending_time','loan_amount','loan_rate',
              'total_periods','period', 'repay_date','period_settle_date','repid_time',
              'is_settle_period','overdue_day','actual_overdue_days','principal','already_repaid_principal',
              'repay_amount']

df_repay_167 = pd.read_csv(r'D:\juzi\0711\167\dwd_cap_repay_plan_fd_167.csv',usecols=cols_repay)
df_repay_174 = pd.read_csv(r'D:\juzi\0711\other\dwd_cap_repay_plan_fd_174.csv',usecols=cols_repay)
df_repay_206 = pd.read_csv(r'D:\juzi\0711\other\dwd_cap_repay_plan_fd_206.csv',usecols=cols_repay)
df_repay_other = pd.read_csv(r'D:\juzi\0711\other\dwd_cap_repay_plan_fd_other.csv',usecols=cols_repay)
df_repay_8005 = pd.read_csv(r'D:\liuyedao\转转渠道\dwd_cap_repay_plan_fd_80005.csv',usecols=cols_repay)


# In[ ]:


df_repay_other = df_repay_other[df_repay_other['channel_id'].isin([167, 174, 206, 80004, 80005])]


# In[9]:


cols_repay = ['order_no','user_id','id_no_des','channel_id','lending_date','loan_amount','loan_rate',
              'total_periods','period', 'repay_date','period_settle_date','repid_time',
              'is_settle_period','overdue_day','actual_overdue_days','principal','already_repaid_principal']
df_repay = pd.read_csv(r'D:\juzi\0925\dwd_cap_repay_plan_fd.csv',usecols=cols_repay)


# In[5]:


df_repay.channel_id.value_counts()


# In[6]:


print(df_repay.lending_date.max(), df_repay.lending_date.min())


# In[10]:


tmp_order = df_repay.sort_values(['order_no','lending_date']).drop_duplicates(subset=['order_no'], keep='first')
tmp_order['rank'] = tmp_order.groupby(['user_id'])['lending_date'].rank(ascending=True, method='dense')
tmp_order.head(3)


# In[11]:


tmp_order = tmp_order.query(" rank>1")
tmp_order = tmp_order.reset_index()
tmp_order.shape
tmp_order.head()


# In[ ]:


df_repay = pd.concat([df_repay_167,df_repay_174,df_repay_206,df_repay_other,df_repay_8005],axis=0)
df_repay.channel_id.value_counts()


# In[ ]:


del df_repay_167,df_repay_174,df_repay_206,df_repay_other,df_repay_8005
gc.collect()


# In[ ]:


df_repay.info()
df_repay.shape 


# In[12]:


df_base = pd.merge(df_repay, tmp_order[['order_no', 'rank']], how='inner', on='order_no')
df_base.shape


# In[13]:


print(df_base.order_no.nunique())


# In[19]:


df_base.query("user_id==148567945").sort_values(['order_no','rank','period']).to_csv(r'D:\liuyedao\B卡开发\test.csv',index=False)


# In[ ]:


df_base.to_csv(r'D:\liuyedao\B卡开发\mid_result\复借客户还款计划.csv', index=False)


# In[ ]:


# tmp_repay_base = df_repay.drop_duplicates(subset=['order_no'], keep='first')
tmp_repay = tmp_repay_base.groupby(['user_id', 'channel_id'])['order_no'].count()
tmp_repay = tmp_repay.reset_index()
tmp_repay = tmp_repay.query(" order_no>1")
# tmp_repay = pd.pivot_table(tmp_repay,values='user_id',index='order_no',
#                            columns='channel_id',aggfunc='count',margins=True,fill_value=0)
# tmp_repay.to_excel(r'D:\liuyedao\B卡开发\统计数据\各渠道借据次数.xlsx')
# tmp_repay


# In[ ]:


tmp_repay_base = df_repay.drop_duplicates(subset=['order_no'], keep='first')


# In[ ]:


tmp_repay_base_new = tmp_repay_base[tmp_repay_base['user_id'].isin(tmp_repay['user_id'])]
tmp_repay_base_new.shape


# In[ ]:


tmp_repay_base_new['lending_time'] = pd.to_datetime(tmp_repay_base_new['lending_time'].str[0:10], format='%Y-%m-%d')
tmp_repay_base_new = tmp_repay_base_new.sort_values(['user_id', 'lending_time'])
top2_loans = tmp_repay_base_new.groupby('user_id').head(2)

top2_loans['time_diff'] = top2_loans.groupby('user_id')['lending_time'].diff()
top2_loans['time_diff_days'] = top2_loans['time_diff'].dt.days


# In[ ]:


top2_loans.head()


# In[ ]:


top2_loans = top2_loans.query("time_diff_days==time_diff_days")
print(top2_loans.shape, top2_loans.order_no.nunique(), top2_loans.user_id.nunique())


# In[ ]:


top2_loans.head()


# In[ ]:


top2_loans = top2_loans.query("lending_time>'2022-05-31' & lending_time<'2023-02-01'")


# In[ ]:


top2_loans.shape


# In[ ]:


top2_loans['inter_day'] = pd.cut(top2_loans['time_diff_days'], [-1,0,30,60,90,120,150,180,float('inf')])


# In[ ]:


top2_loans['inter_day'].value_counts(sort=False, normalize=True)


# In[ ]:


# tmp_repay = tmp_repay_base['user_id'].drop_duplicates()
# tmp_repay = tmp_repay.groupby(['user_id', 'channel_id'])['order_no'].count()
# tmp_repay = tmp_repay.reset_index()
# tmp_repay.to_excel(r'D:\liuyedao\B卡开发\统计数据\各渠道借据次数.xlsx')


# In[ ]:


tmp_repay = tmp_repay.merge()


# In[ ]:


tmp_repay = df_repay.groupby(['user_id', 'channel_id'])['overdue_day'].max()
tmp_repay = tmp_repay.reset_index()
tmp_repay['bins'] = pd.cut(tmp_repay['overdue_day'],bins=[float('-inf'),0,3,5,7,15,30,60,90,120,float('inf')])
tmp_repay = pd.pivot_table(tmp_repay,values='user_id',index='bins',
                           columns='channel_id',aggfunc='count',margins=True,fill_value=0)
tmp_repay.to_excel(r'D:\liuyedao\B卡开发\统计数据\各渠道逾期天数.xlsx')
tmp_repay


# In[ ]:


tmp_repay = df_repay.drop_duplicates(subset=['order_no'], keep='first')
tmp_repay = tmp_repay.groupby(['total_periods', 'channel_id'])['order_no'].count()
tmp_repay = tmp_repay.reset_index()
tmp_repay = pd.pivot_table(tmp_repay,values='order_no',index='channel_id',
                           columns='total_periods',aggfunc='sum',margins=True,fill_value=0)
tmp_repay.to_excel(r'D:\liuyedao\B卡开发\统计数据\各渠道借据期限分布.xlsx')
tmp_repay


# # 4. 标签定义

# In[ ]:


df_repay.shape


# In[ ]:


df_base.shape


# In[ ]:


df_base.info()


# In[ ]:


# def define_sample_target(df, observe_date, perform_date):
#     df_order_base = df.query("mob_date==@observe_date")
#     df_auth_base = df_order_base.groupby(['user_id','channel_id', 'mob_date']).agg({'ever_overdue_days':'max',
#                                                                                       'ovd_status_ever':'max',
#                                                                                      'curr_overdue_days':'max',
#                                                                                      'ovd_status_curr':'max',
#                                                                                     'mob':'max',
#                                                                                     'loan_bal':'sum'
#                                                                                     })
#     df_auth_base = df_auth_base.reset_index()
#     df_auth_base = df_auth_base.query("loan_bal>0 & mob>=3 & curr_overdue_days<=0")
    
#     df_perform = df.query("mob_date>@observe_date &mob_date<=@perform_date & mob>=3 & lending_time<@observe_date")
#     df_perform = df_perform.groupby(['user_id','channel_id']).agg({'ever_overdue_days':'max',
#                                                                   'ovd_status_ever':'max',
#                                                                  'curr_overdue_days':'max',
#                                                                  'ovd_status_curr':'max',
#                                                                 'mob':'max',
#                                                                 'loan_bal':'sum'
#                                                                    })
#     df_perform = df_perform.reset_index()
    
#     base_cols = ['user_id','channel_id', 'mob_date']
#     perform_cols = ['user_id','channel_id','curr_overdue_days','ovd_status_curr']
#     df_return = df_auth_base[base_cols].merge(df_perform[perform_cols], how='left',on=['user_id','channel_id'])
    
#     return df_return


# In[ ]:


# observe_date = '2022-10-31'
# perform_date = (datetime.strptime(observe_date,'%Y-%m-%d') + relativedelta(months=6)).date()
# print(observe_date, perform_date)

# df_flag_1031 = define_sample_target(df_base, observe_date, perform_date)
# print(df_flag_1031.shape, df_flag_1031.user_id.nunique())


# In[ ]:


# observe_date = '2023-01-31'
# perform_date = (datetime.strptime(observe_date,'%Y-%m-%d') + relativedelta(months=6)).date()
# print(observe_date, perform_date)

# df_flag_0131 = define_sample_target(df_base, observe_date, perform_date)
# print(df_flag_0131.shape, df_flag_0131.user_id.nunique())


# In[ ]:


# df_flag_0131.head()


# In[ ]:


# df_flag_1031['target'] = df_flag_1031['ovd_status_curr'].apply(lambda x: 1 if x>=2 else 0 if x==0 else 0.5)
# df_flag_0131['target'] = df_flag_0131['ovd_status_curr'].apply(lambda x: 1 if x>=2 else 0 if x==0 else 0.5)
# df_flag = pd.concat([df_flag_1031, df_flag_0131], axis=0)


# In[ ]:


# tmp = pd.pivot_table(df_flag_1031, values='user_id',index='channel_id',columns='target',aggfunc='count',margins=True)
# tmp['bad_rate'] = tmp[1.0]/tmp['All'] 
# tmp


# In[ ]:


# tmp = pd.pivot_table(df_flag_0131, values='user_id',index='channel_id',columns='target',aggfunc='count',margins=True)
# tmp['bad_rate'] = tmp[1.0]/tmp['All'] 
# tmp


# In[ ]:


# tmp = pd.pivot_table(df_flag, values='user_id',index='channel_id',columns='target',aggfunc='count',margins=True)
# tmp['bad_rate'] = tmp[1.0]/tmp['All'] 
# tmp


# In[ ]:


# observe_date = '2023-04-30'
# perform_date = (datetime.strptime(observe_date,'%Y-%m-%d') + relativedelta(months=6)).date()
# print(observe_date, perform_date)

# df_flag_0430 = define_sample_target(df_base, observe_date, perform_date)
# print(df_flag_0430.shape, df_flag_0430.user_id.nunique())


# In[ ]:


# df_flag_0430 = df_flag_0430[~df_flag_0430['user_id'].isin(df_flag['user_id'])]
# print(df_flag_0430.shape, df_flag_0430.user_id.nunique())


# In[ ]:


# df_flag_0430['target'] = df_flag_0430['ovd_status_curr'].apply(lambda x: 1 if x>=2 else 0 if x==0 else 0.5)


# In[ ]:


# tmp = pd.pivot_table(df_flag_0430, values='user_id',index='channel_id',columns='target',aggfunc='count',margins=True)
# tmp['bad_rate'] = tmp[1.0]/tmp['All'] 
# tmp


# In[24]:


def defin_target(t1, t2, mob, n1):
    if t1>n1:
        target = 1
    elif t1==0 and t2==mob:
        target = 0
    elif t1>0 and t1<=n1 and t2==mob:
        target = 0.5
    elif t2<mob:
        target = -1
    else:
        target = np.nan
    return target


# In[25]:


df.info()


# In[26]:


df_flag_mob6 = df.query("mob_date<'2023-09-01' & mob<=6")

usecols = ['order_no','user_id','id_no_des', 'lending_time', 'channel_id']
df_flag_mob6 = df_flag_mob6.groupby(usecols).agg({'ever_overdue_days':'max',
                                          'ovd_status_ever':'max',
                                         'curr_overdue_days':'max',
                                         'ovd_status_curr':'max',
                                        'mob':'max'
                                       })
df_flag_mob6 = df_flag_mob6.reset_index()
df_flag_mob6['target'] = [*map(lambda t1,t2: defin_target(t1, t2, 6, 1), df_flag_mob6['ovd_status_ever'],df_flag_mob6['mob'])]


# In[27]:


df_flag_mob6.shape


# In[28]:


df_flag_mob6['target'].value_counts()


# In[29]:


df_flag_mob6.groupby(['channel_id','target'])['order_no'].count().unstack()


# In[30]:


df_flag_mob6['lending_month'] = df_flag_mob6['lending_time'].apply(lambda x:str(x)[0:7])


# In[31]:


df_flag_mob6.groupby(['lending_month','target'])['order_no'].count().unstack()


# In[32]:


df_flag_mob6.query("channel_id==174").groupby(['lending_month','target'])['order_no'].count().unstack()


# In[33]:


df_flag_mob6.query("channel_id==80004").groupby(['lending_month','target'])['order_no'].count().unstack()


# In[41]:


df_flag_mob3 = df.query("mob_date<'2023-09-01' & mob<=3")
usecols = ['order_no','user_id','id_no_des', 'lending_time', 'channel_id']
df_flag_mob3 = df_flag_mob3.groupby(usecols).agg({'ever_overdue_days':'max',
                                          'ovd_status_ever':'max',
                                         'curr_overdue_days':'max',
                                         'ovd_status_curr':'max',
                                        'mob':'max'
                                       })
df_flag_mob3 = df_flag_mob3.reset_index()
df_flag_mob3['target'] = [*map(lambda t1,t2: defin_target(t1, t2, 3, 1), df_flag_mob3['ovd_status_ever'],df_flag_mob3['mob'])]


# In[42]:


df_flag_mob3.groupby(['channel_id','target'])['order_no'].count().unstack()


# In[43]:


df_flag_mob3['lending_month'] = df_flag_mob3['lending_time'].apply(lambda x:str(x)[0:7])


# In[44]:


df_flag_mob3.groupby(['lending_month','target'])['order_no'].count().unstack()


# In[45]:


df_flag_mob3.query("channel_id==174").groupby(['lending_month','target'])['order_no'].count().unstack()


# In[46]:


df_flag_mob3.query("channel_id==80004").groupby(['lending_month','target'])['order_no'].count().unstack()


# In[47]:


df_flag_mob3.info()
df_flag_mob3.to_csv(r'D:\liuyedao\B卡开发\mid_result\B卡_order_target_mob3_train_oot_20230927.csv',index=False)


# In[ ]:


# df_flag_mob6['smaple_set'] = 'train'
# df_flag_mob3['smaple_set'] = 'oot'
# df_target = pd.concat([df_flag_mob6, df_flag_mob3], axis=0)
# df_target.shape


# In[ ]:


# df_target.to_csv(r'D:\liuyedao\B卡开发\mid_result\B卡_order_target_train_oot_20230913.csv',index=False)


# In[ ]:


df_target.groupby(['lending_month','target'])['order_no'].count().unstack()


# In[ ]:


df_target.groupby(['smaple_set','target'])['order_no'].count().unstack()


# In[ ]:


df_target.groupby(['lending_month','smaple_set'])['order_no'].count().unstack()


# In[ ]:


df_target.groupby(['channel_id','smaple_set'])['order_no'].count().unstack()


# In[ ]:


df_target.query("channel_id==80005").groupby(['lending_month','target'])['order_no'].count().unstack()


# ## 数据抽样

# In[ ]:


from random import sample

channel_samples = {}
for channel in df_target['channel_id'].unique():
    channel_df = df_target[df_target['channel_id']==channel]
    samples = []
    for month in channel_df['lending_month'].unique():
        month_df = channel_df[channel_df['lending_month']==month]
        bad = month_df[month_df['target']==1.0]
        num_bad = len(bad)
        num_good = num_bad*10
        good = month_df[month_df['target']==0.0]
        samples.extend(bad.to_dict('records'))

        selects = sample(list(good.index), min(len(good), num_good))
        samples.extend(good.loc[selects].to_dict('records'))

    channel_samples[channel] = pd.DataFrame(samples)

resul = pd.concat(list(channel_samples.values()),ignore_index=True)


# In[ ]:


resul.info()


# In[ ]:


resul.to_csv(r'D:\liuyedao\B卡开发\mid_result\B卡_order_target_train_oot_sample_20230914.csv',index=False)


# In[ ]:


resul.groupby(['smaple_set','target'])['order_no'].count().unstack()


# In[ ]:


resul.groupby(['lending_month','target'])['order_no'].count().unstack()


# In[ ]:


resul.groupby(['channel_id','smaple_set','target'])['order_no'].count().unstack()


# In[ ]:


resul.groupby(['lending_month','smaple_set'])['order_no'].count().unstack()


# In[ ]:





# # 5.三大分析

# In[15]:


from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar

def calc_curr_overdue_days(x):
    if x['repay_date'] >= x['mob_date']:
        due_days = 0
    elif x['is_settle_period'] == 0 and x['repay_date'] < x['mob_date']:
        due_days = (x['mob_date'] - x['repay_date']).days
    elif x['is_settle_period'] == 1 and x['period_settle_date'] >= x['mob_date'] and x['repay_date'] < x['mob_date']:
        due_days = (x['mob_date'] - x['repay_date']).days
    elif x['is_settle_period'] == 1 and x['period_settle_date'] < x['mob_date'] and x['repay_date'] < x['mob_date']:
        due_days = 0
    else:
        due_days = 0 
    return due_days

def calc_ever_overdue_days(x):
    if x['repay_date'] >= x['mob_date']:
        due_days = 0
    elif x['is_settle_period'] == 0 and x['repay_date'] < x['mob_date']:
        due_days = (x['mob_date'] - x['repay_date']).days
    elif x['is_settle_period'] == 1 and x['period_settle_date'] >= x['mob_date'] and x['repay_date'] < x['mob_date']:
        due_days = (x['mob_date'] - x['repay_date']).days
    elif x['is_settle_period'] == 1 and x['period_settle_date'] < x['mob_date'] and x['repay_date'] < x['mob_date']:
        due_days = (x['period_settle_date'] - x['repay_date']).days
    else:
        due_days = 0 
    return due_days


def calculate_month_difference(start_date, end_date):
#     start_date = datetime.strptime(start_date, "%Y-%m-%d")
#     end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # 取得两个月末日期
    start_month_end = (start_date + relativedelta(day=31)).replace(day=1)
    end_month_end = (end_date + relativedelta(day=31)).replace(day=1)

    # 计算月份数
    month_difference = (end_month_end.year - start_month_end.year) * 12 + (end_month_end.month - start_month_end.month)

    return month_difference


def get_month_end_dates(year, month_start, month_end):
    month_end_dates = []
    
    for month in range(month_start, month_end + 1):
        _, last_day = calendar.monthrange(year, month)
        month_end_date = datetime(year, month, last_day)
        month_end_dates.append(month_end_date)
    
    return month_end_dates



def calc_ovedue_status(x):
    if x<=0:
        status=0
    elif x>0 and x<=30:
        status=1
    elif x>30 and x<=60:
        status=2
    elif x>60 and x<=90:
        status=3
    elif x>90 and x<=120:
        status=4
    elif x>120:
        status=5
    else:
        status=np.nan
    return status


# In[16]:


time1 = get_month_end_dates(2022, 5, 12)
time2 = get_month_end_dates(2023, 1, 9)


# In[17]:


mob_month_end = pd.DataFrame({'mob_date':[date.strftime("%Y-%m-%d") for date in (time1+time2)]})
mob_month_end['keys'] = 1  
print(list(mob_month_end['mob_date']))


# In[18]:


def three_big_analysis_base_data(df_base, date):
    # 格式转换
    df_base['repay_date'] = pd.to_datetime(df_base['repay_date'],format='%Y-%m-%d')
    df_base['period_settle_date'] = pd.to_datetime(df_base['period_settle_date'],format='%Y-%m-%d')
    df_base['lending_time'] = pd.to_datetime(df_base['lending_time'].str[0:10],format='%Y-%m-%d')
    df_base['repid_time'] = pd.to_datetime(df_base['repid_time'].str[0:10],format='%Y-%m-%d')
    df_base['mob_date'] = pd.to_datetime(df_base['mob_date'],format='%Y-%m-%d')
    # 筛选符合要求的数据
    df_base = df_base.query("mob_date>=lending_time & mob_date< @date ")
    df_base = df_base.reset_index(drop=True)
    # 计算账龄
    df_base['mob'] = [*map(lambda t1,t2: calculate_month_difference(t1, t2), df_base['lending_time'], df_base['mob_date'])]
    # 计算逾期天数
    df_base['ever_overdue_days'] = df_base.apply(calc_ever_overdue_days, axis=1)
    df_base['curr_overdue_days'] = df_base.apply(calc_curr_overdue_days, axis=1)
    #计算到期应还本金
    df_base['due_prin_amt'] = df_base.apply(lambda x: x['principal'] if x['repay_date'] <= x['mob_date'] else 0, axis=1)
    #计算到期已还本金
    df_base['act_prin_amt'] = df_base.apply(lambda x: x['already_repaid_principal'] if x['repay_date'] <= x['mob_date'] and x['repid_time'] <= x['mob_date'] else 0, axis=1)

    #计算逾期状态
    df_base['ovd_status_ever'] = df_base['ever_overdue_days'].apply(lambda x: calc_ovedue_status(x))
    df_base['ovd_status_curr'] = df_base['curr_overdue_days'].apply(lambda x: calc_ovedue_status(x))

    usecols = ['order_no','user_id','id_no_des', 'channel_id', 'lending_time', 'loan_amount','mob','mob_date']
    df_vintage = df_base.groupby(usecols).agg({
                                                'ever_overdue_days':'max',
                                                'curr_overdue_days':'max',
                                                'ovd_status_ever':'max',
                                                'ovd_status_curr':'max',
                                                'due_prin_amt':'sum',
                                                'act_prin_amt':'sum'
                                            })
    df_vintage = df_vintage.reset_index()
    df_vintage['loan_bal'] = df_vintage['loan_amount'] - df_vintage['act_prin_amt'] #剩余本金
    
    del df_base
    gc.collect()

    return df_vintage


# In[19]:


df_base['lending_time'] = df_base['lending_date']


# In[20]:


print(df_base.lending_time.min(), df_base.lending_time.max())


# In[21]:


# df_base = df_repay.query("lending_time>='2022-04-01'&lending_time<'2023-07-01'")
# df_base['keys']=1
# df_base_copy = df_base.copy()
# df_base = df_base.query("lending_time>='2022-06-01'&lending_time<'2023-08-01'")
df_base['keys']=1


# In[22]:


df_dict = {}
for i in [174, 80004, 80005]:
    print(i)
    tmp = df_base.query("channel_id==@i").merge(mob_month_end, on='keys')
    tmp = three_big_analysis_base_data(tmp, '2023-09-01')
    df_dict[i] = tmp
    del tmp
    gc.collect()

df = pd.concat(list(df_dict.values()), axis=0)


# In[23]:


df.shape


# In[ ]:


df.info(show_counts=True)


# In[ ]:


df1 = df[df['order_no'].isin(df_base['order_no'])]
df1.shape


# In[ ]:


df1= df.copy()
df1.shape


# In[ ]:


df['ovd_status_curr'] = df['curr_overdue_days'].apply(lambda x: calc_ovedue_status(x))


# In[ ]:


# 缺少ovd_status_curr
df_copy = df.copy()


# In[ ]:


df_base = df.copy()


# ## 3.1滚动率分析

# In[ ]:


def roll_rate_analysis(df_roll, mob_date_month, is_auth=1):
    roll_curr = pd.DataFrame(columns=[0,1,2,3,4,5,'All','mob_date_month'])
    roll_curr_rate = pd.DataFrame(columns=[0,1,2,3,4,5,'All','mob_date_month'])
    for i in range(len(mob_date_month)-1):
        date1 = mob_date_month[i]
        date2 = mob_date_month[i+1]
        df_roll_part1 = df_roll.query("lending_time<= @date1 & mob_date<= @date1")
        df_roll_part2 = df_roll.query("lending_time<= @date1 & mob_date<= @date2 & mob_date> @date1")
        
        if is_auth==1:
            df_roll_part1 = df_roll_part1.groupby(['user_id']).agg({'curr_overdue_days':'max',
                                                                    'ovd_status_curr':'max'})
            df_roll_part1 = df_roll_part1.reset_index()
            df_roll_part2 = df_roll_part2.groupby(['user_id']).agg({'curr_overdue_days':'max',
                                                                    'ovd_status_curr':'max'})
            df_roll_part2 = df_roll_part2.reset_index()
            kyes = 'user_id'
        else:
            df_roll_part1 = df_roll_part1.groupby(['order_no']).agg({'curr_overdue_days':'max',
                                                                     'ovd_status_curr':'max'})
            df_roll_part1 = df_roll_part1.reset_index()
            df_roll_part2 = df_roll_part2.groupby(['order_no']).agg({'curr_overdue_days':'max',
                                                                     'ovd_status_curr':'max'})
            df_roll_part2 = df_roll_part2.reset_index()
            kyes = 'order_no'

        df_roll_v1 = pd.merge(df_roll_part1, df_roll_part2, how='left',on=kyes)

        tmp_curr = pd.pivot_table(df_roll_v1, values=kyes,index=['ovd_status_curr_x'],
                                  columns=['ovd_status_curr_y'],aggfunc='count',margins=True)
        tmp_curr_rate = tmp_curr.copy()
        for k in list(tmp_curr.columns):
            tmp_curr_rate[k] = tmp_curr[k]/tmp_curr["All"]
        for j in tmp_curr.index:
            if j==0:
                tmp_curr_rate.loc[j, 'to_bad_rate'] = tmp_curr_rate.loc[j, list(tmp_curr.columns)[1:-1]].sum()
                tmp_curr_rate.loc[j, 'status'] = "M0-M0+"
            elif j==1:
                tmp_curr_rate.loc[j, 'to_bad_rate'] = tmp_curr_rate.loc[j, list(tmp_curr.columns)[2:-1]].sum()
                tmp_curr_rate.loc[j, 'status'] = "M1-M1+"
            elif j==2:
                tmp_curr_rate.loc[j, 'to_bad_rate'] = tmp_curr_rate.loc[j, list(tmp_curr.columns)[3:-1]].sum()
                tmp_curr_rate.loc[j, 'status'] = "M2-M2+"
            elif j==3:
                tmp_curr_rate.loc[j, 'to_bad_rate'] = tmp_curr_rate.loc[j, list(tmp_curr.columns)[4:-1]].sum()
                tmp_curr_rate.loc[j, 'status'] = "M3-M3+"
            elif j==4:
                tmp_curr_rate.loc[j, 'to_bad_rate'] = tmp_curr_rate.loc[j, list(tmp_curr.columns)[5:-1]].sum()
                tmp_curr_rate.loc[j, 'status'] = "M4-M4+"
            elif j==5:
                tmp_curr_rate.loc[j, 'to_bad_rate'] = tmp_curr_rate.loc[j, list(tmp_curr.columns)[5:-1]].sum()
                tmp_curr_rate.loc[j, 'status'] = "M5-M5"                
            else:
                pass
        tmp_curr['mob_date_month'] = mob_date_month[i]
        tmp_curr_rate['mob_date_month'] = mob_date_month[i]

        roll_curr = pd.concat([roll_curr, tmp_curr])
        roll_curr_rate = pd.concat([roll_curr_rate, tmp_curr_rate])
        
    roll_rate_month = pd.pivot_table(roll_curr_rate, values='to_bad_rate', index=['mob_date_month'],
                                    columns=['status'],aggfunc='mean',margins=True)
           
    return roll_curr, roll_curr_rate,roll_rate_month
  


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\统计数据\B卡_各渠道_滚动率分析_20230904_v2.xlsx")
df_roll = pd.DataFrame()
for i in [167, 174, 206, 80004, 80005]:
    print(i)
    tmp = df_base.query("channel_id==@i")
    tmp = tmp.reset_index(drop=True)
#     date_min = tmp['lending_time'].max().month
#     date_max = tmp['lending_time'].max().month
    mob_date_month = get_month_end_dates(2022, 8, 12)+get_month_end_dates(2023, 1, 6)
    roll_curr_total, roll_curr_rate_total,roll_rate_month = roll_rate_analysis(tmp, mob_date_month, is_auth=0)
    roll_rate_month['channel_id'] = i
    df_roll = pd.concat([df_roll, roll_rate_month], axis=0)
    
    roll_rate_month.to_excel(writer,sheet_name='roll_rate_month_{}'.format(i))
    roll_curr_rate_total.to_excel(writer,sheet_name='roll_curr_rate_total_{}'.format(i))
    roll_curr_total.to_excel(writer,sheet_name='roll_curr_total_{}'.format(i))
    writer.save() 
    del tmp
    gc.collect()
     


# ## 3.2迁徙率分析

# In[ ]:


df_base.shape


# In[ ]:


def migration_rate_analysis(df, ovd_status='ovd_status_curr', is_auth=1):
    if is_auth==1:
        # 汇总聚合到用户层
        df_qx = df.groupby(['user_id','mob_date']).agg({'ovd_status_curr':'max',
                                                            'curr_overdue_days':'max',
                                                            'ever_overdue_days':'max',
                                                            'ovd_status_ever':'max',
                                                            'act_prin_amt':'sum',
                                                           'loan_amount':'sum',
                                                           'loan_bal':'sum'
                                                          })
        df_qx = df_qx.reset_index()
        values= 'user_id'
    else:
        df_qx = df.copy()
        values= 'order_no'
    df_qx_tmp = df_qx.groupby(['mob_date', ovd_status]).agg({values:'count','loan_bal':'sum'})
    df_qx_tmp = df_qx_tmp.reset_index() 
    tmp_curr_cnt = pd.pivot_table(df_qx_tmp, values=values,index=['mob_date'], columns=[ovd_status],aggfunc='sum',margins=True)
    tmp_curr_cnt_rate = pd.DataFrame(columns=['M0-M1','M1-M2','M2-M3','M3-M4','M4-M5'])
    for i in range(1,len(tmp_curr_cnt.index)-2):
        tmp_curr_cnt_rate.loc[tmp_curr_cnt.index[i],'M0-M1'] = tmp_curr_cnt.loc[tmp_curr_cnt.index[i+1], 1]/tmp_curr_cnt.loc[tmp_curr_cnt.index[i], 0]
        tmp_curr_cnt_rate.loc[tmp_curr_cnt.index[i],'M1-M2'] = tmp_curr_cnt.loc[tmp_curr_cnt.index[i+1], 2]/tmp_curr_cnt.loc[tmp_curr_cnt.index[i], 1]
        tmp_curr_cnt_rate.loc[tmp_curr_cnt.index[i],'M2-M3'] = tmp_curr_cnt.loc[tmp_curr_cnt.index[i+1], 3]/tmp_curr_cnt.loc[tmp_curr_cnt.index[i], 2]
        tmp_curr_cnt_rate.loc[tmp_curr_cnt.index[i],'M3-M4'] = tmp_curr_cnt.loc[tmp_curr_cnt.index[i+1], 4]/tmp_curr_cnt.loc[tmp_curr_cnt.index[i], 3]
        tmp_curr_cnt_rate.loc[tmp_curr_cnt.index[i],'M4-M5'] = tmp_curr_cnt.loc[tmp_curr_cnt.index[i+1], 5]/tmp_curr_cnt.loc[tmp_curr_cnt.index[i], 4]
        
    tmp_curr_amt = pd.pivot_table(df_qx_tmp, values='loan_bal',index=['mob_date'], columns=[ovd_status],aggfunc='sum',margins=True)
    tmp_curr_amt_rate = pd.DataFrame(columns=['M0-M1','M1-M2','M2-M3','M3-M4','M4-M5'])
    for i in range(1,len(tmp_curr_amt.index)-2):
        tmp_curr_amt_rate.loc[tmp_curr_amt.index[i],'M0-M1'] = tmp_curr_amt.loc[tmp_curr_amt.index[i+1], 1]/tmp_curr_amt.loc[tmp_curr_amt.index[i], 0]
        tmp_curr_amt_rate.loc[tmp_curr_amt.index[i],'M1-M2'] = tmp_curr_amt.loc[tmp_curr_amt.index[i+1], 2]/tmp_curr_amt.loc[tmp_curr_amt.index[i], 1]
        tmp_curr_amt_rate.loc[tmp_curr_amt.index[i],'M2-M3'] = tmp_curr_amt.loc[tmp_curr_amt.index[i+1], 3]/tmp_curr_amt.loc[tmp_curr_amt.index[i], 2]
        tmp_curr_amt_rate.loc[tmp_curr_amt.index[i],'M3-M4'] = tmp_curr_amt.loc[tmp_curr_amt.index[i+1], 4]/tmp_curr_amt.loc[tmp_curr_amt.index[i], 3]
        tmp_curr_amt_rate.loc[tmp_curr_amt.index[i],'M4-M5'] = tmp_curr_amt.loc[tmp_curr_amt.index[i+1], 5]/tmp_curr_amt.loc[tmp_curr_amt.index[i], 4]    
    
    return tmp_curr_cnt,tmp_curr_cnt_rate,tmp_curr_amt,tmp_curr_amt_rate


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\统计数据\B卡_各渠道_迁徙率分析_20230904.xlsx")
df_migration = pd.DataFrame()
for i in [167, 174, 206, 80004, 80005]:
    print(i)
    tmp = df_base.query("channel_id==@i")
    tmp = tmp.reset_index(drop=True)
    tmp_cnt,tmp_cnt_rate,tmp_amt,tmp_amt_rate = migration_rate_analysis(tmp, ovd_status='ovd_status_curr',is_auth=0)
    tmp_amt_rate['channel_id'] = i
    tmp_cnt.to_excel(writer,sheet_name='tmp_cnt_{}'.format(i))
    tmp_cnt_rate.to_excel(writer,sheet_name='tmp_cnt_rate_{}'.format(i))
    tmp_amt.to_excel(writer,sheet_name='tmp_amt_{}'.format(i))
    tmp_amt_rate.to_excel(writer,sheet_name='tmp_amt_rate_{}'.format(i))
    writer.save() 
    gc.collect()


# ## 3.3账龄分析

# In[ ]:


def vintage_analysis(df1):
#     df1 = df1.groupby(['user_id','mob']).agg({'lending_time':'min',
#                                              'ovd_flag_ever_30':'max',
#                                              'ovd_flag_ever_15':'max'})
#     df1 = df1.reset_index()
    df1['lending_month'] = df1['lending_time'].apply(lambda x:str(x)[0:7])
    tmp = df1.groupby(['lending_month', 'mob']).agg({'order_no':'count',
                                                     'ovd_flag_ever_30':'sum',
                                                     'ovd_flag_ever_15':'sum'})
    tmp = tmp.reset_index()
    tmp_total = pd.pivot_table(tmp, values='order_no', index=['lending_month'],columns=['mob'],
                               aggfunc='sum',margins=True)
    tmp_30 = pd.pivot_table(tmp, values='ovd_flag_ever_30', index=['lending_month'],columns=['mob'],
                            aggfunc='sum',margins=True)
    tmp_15 = pd.pivot_table(tmp, values='ovd_flag_ever_15', index=['lending_month'],columns=['mob'],
                            aggfunc='sum',margins=True)

    tmp_30_pct = tmp_30/tmp_total
    tmp_15_pct = tmp_15/tmp_total   
    
    return tmp_total,tmp_30,tmp_15,tmp_30_pct,tmp_15_pct


# In[ ]:


#标记逾期30天以上标识
df_base['ovd_flag_ever_30'] = df_base['ever_overdue_days'].apply(lambda x: 1 if x > 30 else 0)
#标记逾期15天以上标识
df_base['ovd_flag_ever_15'] = df_base['ever_overdue_days'].apply(lambda x: 1 if x > 15 else 0)


# In[ ]:


writer=pd.ExcelWriter(r"D:\liuyedao\B卡开发\统计数据\B卡_各渠道_账龄分析_20230904.xlsx")

for i in [167, 174, 206, 80004, 80005]:
    print(i)
    tmp = df_base.query("channel_id==@i")
    tmp = tmp.reset_index(drop=True)
    tmp_total,tmp_30,tmp_15,tmp_30_pct,tmp_15_pct = vintage_analysis(tmp)
    tmp_30_pct['channel_id'] = i
    tmp_15_pct['channel_id'] = i
    
    tmp_15.to_excel(writer,sheet_name='tmp_15_{}'.format(i))
    tmp_30.to_excel(writer,sheet_name='tmp_30_{}'.format(i))
    tmp_total.to_excel(writer,sheet_name='tmp_total_{}'.format(i))
    
    tmp_15_pct.to_excel(writer,sheet_name='tmp_15_pct_{}'.format(i))
    tmp_30_pct.to_excel(writer,sheet_name='tmp_30_pct_{}'.format(i))

    writer.save() 
    gc.collect()


# # 特征变量开发

# In[ ]:


get_ipython().run_line_magic('run', 'function.ipynb')


# In[ ]:


resul.info()


# In[ ]:


train_list = list(resul.query("smaple_set=='train'")['order_no'])
oot_list = list(resul.query("smaple_set=='oot'")['order_no'])

df_repay_train = df_repay[df_repay['order_no'].isin(train_list)]
df_repay_train = df_repay_train.reset_index(drop=True)

df_repay_oot = df_repay[df_repay['order_no'].isin(oot_list)]
df_repay_oot = df_repay_oot.reset_index(drop=True)


print(df_repay_train.shape, df_repay_oot.shape)


# In[ ]:


df_repay_train = df_repay[df_repay['user_id'].isin(tmp_order['user_id'])]
df_repay_train.shape


# In[ ]:


df_repay_train.to_csv(r'D:\liuyedao\B卡开发\mid_result\复借客户建模的还款计划表.csv',index=False)


# In[ ]:


resul.info()


# In[ ]:


print(resul.order_no.nunique(), resul.user_id.nunique())


# In[ ]:


resul['lending_month'].unique()


# In[ ]:


df_repay_train = df_repay[df_repay['user_id'].isin(resul['user_id'])]
print(df_repay_train.order_no.nunique(), df_repay_train.user_id.nunique())


# In[ ]:


df_repay_train.to_csv(r'D:\liuyedao\B卡开发\mid_result\复借客户建模的还款计划表_samples.csv',index=False)


# In[ ]:


df_repay_train.shape


# In[ ]:


df_repay_train.info()


# In[ ]:


df_repay_train['lengding_month'] = df_repay_train['lending_time'].apply(lambda x: str(x)[0:7])


# In[ ]:


df_repay_train.info()


# In[ ]:


df_auth_train = df_auth[df_auth['user_id'].isin(tmp_order['user_id'])]
df_auth_train.shape


# In[ ]:


df_auth_train.to_csv(r'D:\liuyedao\B卡开发\mid_result\复借客户建模的授信信息表.csv',index=False)


# In[ ]:


df_repay_train['repay_date'] = pd.to_datetime(df_repay_train['repay_date'],format='%Y-%m-%d')
df_repay_train['period_settle_date'] = pd.to_datetime(df_repay_train['period_settle_date'],format='%Y-%m-%d')
df_repay_train['lending_time'] = pd.to_datetime(df_repay_train['lending_time'].str[0:10],format='%Y-%m-%d')

# df_repay_oot['repay_date'] = pd.to_datetime(df_repay_oot['repay_date'],format='%Y-%m-%d')
# df_repay_oot['period_settle_date'] = pd.to_datetime(df_repay_oot['period_settle_date'],format='%Y-%m-%d')
# df_repay_oot['lending_time'] = pd.to_datetime(df_repay_oot['lending_time'].str[0:10],format='%Y-%m-%d')


# In[4]:


# 提取脚本中的每个函数名
import ast 
function_names = []

with open('行为评分卡特征变量开发v2.py',encoding='utf-8') as f:
    tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_names.append(node.name)
print(function_names)


# In[5]:


tmp_df = pd.DataFrame({'name':function_names})
tmp_df.head()


# In[7]:


tmp_df.to_excel(r'特征名.xlsx')


# In[ ]:


function_names = function_names[:-2]


# In[ ]:


function_names[-1]


# In[ ]:


def apply_multiple_indicator(df, function_list, observe_date):
    """
    df: DataFrame
    group_coulumn:需要分组的列
    indictor_function：指标加工函数名称列表
    observe_date：观察点日期
    **kwagrgs：额外参数
    """  
    result_dict = {}
    tmp = df.copy()
    tmp['period_settle_date'].fillna(observe_date + relativedelta(days=1), inplace=True)
    
    for i,function in enumerate(function_list):
        result_dict[function] = tmp.groupby('user_id').apply(lambda x: eval(function)(x, observe_date)).to_dict()
            
    result_df = pd.DataFrame.from_dict(result_dict, orient='index').T
    result_df['user_id'] = result_df.index
    result_df['observe_date'] = observe_date.date()
    del tmp
    gc.collect()    
    
    return result_df


# In[ ]:


def data_clean(df_repay, resul, lending_month):
    
    user_id = resul[resul['lending_month']==lending_month]['user_id']
    df_repay_train = df_repay[df_repay['user_id'].isin(user_id)]
    
    df_repay_train['repay_date'] = pd.to_datetime(df_repay_train['repay_date'],format='%Y-%m-%d')
    df_repay_train['period_settle_date'] = pd.to_datetime(df_repay_train['period_settle_date'],format='%Y-%m-%d')
    df_repay_train['lending_time'] = pd.to_datetime(df_repay_train['lending_time'].str[0:10],format='%Y-%m-%d')
    
    return df_repay_train


# In[ ]:


observe_date.strftime(format='%Y-%m')


# In[ ]:


print(get_month_end_dates(2022, 6, 12) + get_month_end_dates(2023, 1, 4))


# In[ ]:


# lending_month = '2022-11'
# df_repay_train = data_clean(df_repay, resul, lending_month)

lending_month = '2022-09'
df_repay_train = data_clean(df_repay, resul, lending_month)

print(df_repay_train.shape, df_repay_train.order_no.nunique(), df_repay_train.user_id.nunique())


# In[ ]:


get_ipython().run_line_magic('run', 'function.ipynb')


# In[ ]:


import time 

star_time = time.time()
# output = {}
observe_dates = get_month_end_dates(2022, 8, 8)
print(observe_dates)

for observe_date in observe_dates:
    print('-----------------观察点：{}，开始变量加工--------------------'.format(observe_date))
    
    output[observe_date.date()] = apply_multiple_indicator(df_repay_train, function_names, observe_date)

    
    end_time = time.time()
    total_time = (end_time -star_time)/60
    print(total_time)
    print('-----------------观察点：{}，结束变量加工--------------------'.format(observe_date))
    



# In[ ]:


train_result = pd.concat(list(output.values()), axis=0, ignore_index=False)
train_result.shape


# In[ ]:


list(output.keys())


# In[ ]:


train_result.head()


# In[ ]:


train_result.to_csv(r"D:\liuyedao\B卡开发\mid_result\B卡_建模_行为特征变量_20230914.csv",index=False)


# In[ ]:


print(train_result.shape,  train_result.user_id.nunique())


# In[ ]:


train_result_new.to_csv(r"D:\liuyedao\B卡开发\mid_result\B卡_行为特征变量_train.csv",index=False)


# In[ ]:


observe_dates_oot = get_month_end_dates(2023, 1, 3)
print(observe_dates_oot)


# In[ ]:


star_time = time.time()
print(star_time)

out_put_oot = {}
for observe_date in observe_dates_oot:
    print('---------------观察点：{}，开始变量加工---------------'.format(observe_date))
    result_dict_oot = {}
    tmp = df_repay_oot.copy()
    tmp['period_settle_date'].fillna(observe_date + relativedelta(days=1), inplace=True)
    
    for i,function in enumerate(function_names):
        result_dict_oot[function] = tmp.groupby('user_id').apply(lambda x: eval(function)(x, observe_date)).to_dict()
        
    
    result_df_oot = pd.DataFrame.from_dict(result_dict_oot, orient='index').T
    result_df_oot = result_df_oot.reset_index()
    result_df_oot['observe_date'] = observe_date.date()
    
    del tmp
    gc.collect()
    
    out_put_oot[observe_date.date()] = result_df_oot
    
    end_time = time.time()
    print(end_time)
    total_time = (end_time -star_time)/60
    print(total_time)
    print('-----------------观察点：{}，结束变量加工--------------------'.format(observe_date))

    
oot_result = pd.concat(list(out_put_oot.values()), axis=0, ignore_index=True)
oot_result.to_csv(r"D:\liuyedao\B卡开发\mid_result\B卡_行为特征变量_oot.csv",index=False)

end_time = time.time()
print(end_time)
total_time = (end_time -star_time)/60
print(total_time)


# In[ ]:


oot_result.head()


# In[ ]:


oot_result.rename(columns={'index':'user_id'},inplace=True)
oot_result.to_csv(r"D:\liuyedao\B卡开发\mid_result\B卡_行为特征变量_oot.csv",index=False)


# ### 授信信息特征变量

# In[ ]:





# In[ ]:


import time 

star_time = time.time()

observe_dates_train = get_month_end_dates(2022, 6, 12)
print(observe_dates_train)

for observe_date in observe_dates_train:
    print('---------------{}---------------'.format(observe_date))
    result_df = {}
    tmp = df_repay_train.copy()
    tmp['period_settle_date'].fillna(observe_date + relativedelta(days=1), inplace=True)
    
    for i,function in enumerate(function_names):
#         print('========{}：{}========'.format(i, function))
        result_df[function] = tmp.groupby('user_id').apply(lambda x: eval(function)(x, observe_date)).to_dict()
        
    
    result_df = pd.DataFrame.from_dict(result_df, orient='index').T
    result_df['observe_date'] = observe_date.date()
    
    del tmp
    gc.collect()
    
    print('-----------------观察点：{}，结束变量加工--------------------'.format(observe_date))
    output[observe_date] = result_df

train_result = pd.concat(list(output.values()), axis=0, ignore_index=True)
train_result.to_csv(r"D:\liuyedao\B卡开发\mid_result\B卡_行为特征变量_train.csv",index=False)
end_time = time.time()
total_time = (end_time -star_time)/60
print(total_time)




#==============================================================================
# File: category_label_encoder.py
#==============================================================================

#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: category_label_encoder.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-09-21
'''

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
#
#
# def category_to_labelencoder(data, labelencoder=[]):
#     label_encoder_dict = {}
#     le = LabelEncoder()
#     for col in labelencoder:
#         print('{} in process!!!'.format(col))
#         data[col] = le.fit_transform(data[col].values)
#         number = [i for i in range(0, len(le.classes_))]
#         key = list(le.inverse_transform(number))
#         label_encoder_dict[col] = dict(zip(key, number))
#     return label_encoder_dict
#
#
# def category_to_labelencoder_apply(data, labelencoder_dict={}):
#     for col, mapping in labelencoder_dict.items():
#         print('{} in process!!!'.format(col))
#         data[col] = data[col].map(mapping).fillna(-1)
#         data[col] = data[col].astype(int)
#
#
# #####训练
# fruit_data = pd.DataFrame({
#     'fruit': ['apple', 'orange', 'pear', 'orange', 'red'],
#     'color': ['red', 'orange', 'green', 'green', 'red'],
#     'weight': [5, 6, 3, 4, 2]
# })
# print(fruit_data)
#
# labelencoder_cols = ['fruit', 'color']
#
# label_encoder_dict = category_to_labelencoder(fruit_data, labelencoder_cols)
# print(fruit_data)
#
# #####应用
# test_data = pd.DataFrame({
#     'fruit': ['apple', 'orange', 'pear', 'orange', 'red'],
#     'color': ['aaa', 'orange', 'green', 'green', 'red'],
#     'weight': [5, 6, 3, 4, 2]
# })
# print(test_data)
#
# category_to_labelencoder_apply(test_data, label_encoder_dict)
# print(test_data)


print('########################'*5)

import pandas as pd

df = pd.DataFrame([
    ['green', 'Chevrolet', 2017],
    ['blue', 'BMW', 2015],
    ['yellow', 'Lexus', 2018],
])
df.columns = ['color', 'make', 'year']
df.to_csv('df_source.csv')

df_processed = pd.get_dummies(df, prefix_sep="_", columns=df.columns[:-1])
print(df_processed.head(10))
df_processed.to_csv('df_processed.csv')

aa = ['aa', 'bb', 'cc']
print(aa)
aa.remove('aa')
print(aa)

df = pd.DataFrame([
    ['green;yellow;aaa', 'Chevrolet', 2017],
    ['blue;green', 'BMW', 2015],
    ['yellow', 'Lexus', 2018],
])
df.columns = ['color', 'make', 'year']
print(df)

print(df.set_index(["make", "year"])["color"].str.split(";", expand=True))
df = df.set_index(["make", "year"])["color"].str.split(";", expand=True).reset_index()
print(df)
del df['make']
print(pd.get_dummies(df,prefix='', prefix_sep='').groupby(level=0, axis=1).max())


df = pd.DataFrame([
    ['green;yellow;aaa', 'Chevrolet;Lexus', 2017],
    ['blue;green', 'BMW', 2015],
    ['yellow', 'Lexus;BMW', 2018],
])
df.columns = ['color', 'make', 'year']
print(df)

print(df.set_index(["year"])["color"].str.split(";", expand=True))
df = df.set_index(["year"])["color"].str.split(";", expand=True).reset_index()
print(df)
print(pd.get_dummies(df,prefix='', prefix_sep='').groupby(level=0, axis=1).max())
exit(0)
# df = df.set_index(["make", "year"])["color"].str.split(";", expand=True).stack().reset_index(drop=True, level=-1).reset_index().rename(columns={0: "color"})
# print(df)
# del df['make']
# print(df)
# print(pd.get_dummies(df,prefix='', prefix_sep='').groupby(level=0, axis=1).max().set_index("year"))

# df_processed = pd.get_dummies(df, prefix_sep="_", columns=['color'])
# print(df_processed.head(10))
# df_processed.to_csv('df_processed_get_dummies.csv')


df = pd.DataFrame([
    [101, 'roof', 'garage', 'basement'],
    [102, 'basement', 'garage', 'painting'],
])
df.columns = ['no', 'point1', 'point2', 'point3']
print(df)

#print(pd.get_dummies(df))
print(pd.get_dummies(df,prefix='', prefix_sep='').groupby(level=0, axis=1).max().set_index("no"))



#==============================================================================
# File: category_woe_encoder.py
#==============================================================================

#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: category_woe_encoder.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-09-21
'''

import json
import operator
import sys
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def category_2_woe(df, category_cols=[], target='target'):
    """
    每个类别都会转成woe值。缺失值不转，即还是为缺失值。在考虑到未来如果有新类别，给予other对应woe为0
    Args:
        df (DataFrame):
        category_cols (list): 类别变量list
        target:

    Returns:

    """
    var_value_woe = {}
    for i in category_cols:
        bin_g = df.groupby(by=i)[target].agg([('total_cnt', 'count'), ('bad_cnt', 'sum')])
        bin_g['good_cnt'] = bin_g['total_cnt'] - bin_g['bad_cnt']
        bad_count = sum(bin_g['bad_cnt'])
        good_count = sum(bin_g['good_cnt'])
        # bin_g['bad_rate'] = bin_g['bad_cnt'] / sum(bin_g['bad_cnt'])
        bin_g['bad_rate'] = bin_g['bad_cnt'].map(lambda x: 1 / bad_count if x == 0 else x / bad_count)
        # bin_g['good_rate'] = bin_g['good_cnt'] / sum(bin_g['good_cnt'])
        bin_g['good_rate'] = bin_g['good_cnt'].map(lambda x: 1 / good_count if x == 0 else x / good_count)
        # bin_g['good_rate'].replace({0: 0.0000000001}, inplace=True)  # good_rate为0的情况下，woe算出来是-inf。即将0使用一个极小数替换
        # bin_g['woe'] = bin_g.apply(lambda x: 0.0 if x['bad_rate'] == 0 else np.log(x['good_rate'] / x['bad_rate']),
        #                            axis=1)
        bin_g['woe'] = bin_g.apply(lambda x: np.log(x['good_rate'] / x['bad_rate']), axis=1)

        value_woe = bin_g['woe'].to_dict()
        value_woe['other'] = 0  # 未来有新类别的情况下，woe值给予0
        var_value_woe[i] = value_woe

    return var_value_woe


def bin_to_woe(df, var_bin_woe_dict):
    """
    根据传进来的var_bin_woe_dict对原始值进行映射。
    如在var_bin_woe_dict没有的类别（数据集中新出现的类别，归为到other这类）同时var_bin_woe_dict中得有other该类别对应的woe值
    如果var_bin_woe_dict中没有other该类别对应的woe值，即数据集中新出现的类别归为缺失值，即新出现的类别没有woe值
    Args:
        df:
        var_bin_woe_dict (dict):

    Returns:

    """

    for feature, bin_woe in var_bin_woe_dict.items():
        df[feature] = df[feature].map(
            lambda x: x if (x in bin_woe.keys() or x is np.nan or pd.isna(x)) else 'other')
        df[feature] = df[feature].map(bin_woe)

    return df


def data_to_bin(df, bins_dict={}):
    """
    原始数据根据bins_dict进行分箱
    Args:
        df:
        bins_dict: 分箱字典, 形如{'D157': [-999, 1.0, 2.0, 3.0, 5.0, inf]}

    Returns:

    """

    if not isinstance(bins_dict, dict):
        assert '请传入类似 {\'D157\': [-999, 1.0, 2.0, 3.0, 5.0, inf]}'

    data_with_bins = Parallel(n_jobs=-1)(
        delayed(pd.cut)(df[col], bins=bins, right=False, retbins=True) for col, bins in bins_dict.items())
    data_bin = pd.DataFrame([i[0].astype(str) for i in data_with_bins]).T
    b_dict = dict([(i[0].name, i[1].tolist()) for i in data_with_bins])
    if not operator.eq(bins_dict, b_dict):
        assert '传入的分箱和应用后的分箱不对等，请联系开发者'

    return data_bin


def transform(df, var_bin_woe_dict, bins_dict={}):
    """

    Args:
        df:
        var_bin_woe_dict (dict): 形如{"Sex": {"female": -1.5298770033401874, "male": 0.9838327092415774}, "Embarked": {"C": -0.694264203516269, "S": 0.1977338357888416, "other": -0.030202603851420356}}
        bins_dict:

    Returns:
        df (DataFrame): 转换woe后的数据集
    """

    df_ = df.copy()
    if bins_dict:
        df_ = data_to_bin(df, bins_dict=bins_dict)
    return bin_to_woe(df_, var_bin_woe_dict)


def category_2_woe_save(var_value_woe, path=None):
    if path is None:
        path = sys.path[0]

    with open(path + 'category_var_value_woe.json', 'w') as f:
        json.dump(var_value_woe, f)


def category_2_woe_load(path=None):
    with open(path + 'category_var_value_woe.json', 'r') as f:
        var_value_woe = json.load(f)
    return var_value_woe


#########################################测试代码

if __name__ == "__main__":
    from autobmt.utils import select_features_dtypes

    #####读取数据
    to_model_data_path = os.path.join('..','tests','tutorial_data.csv')
    cust_id = 'APP_ID_C'
    target = 'target'  # 目标变量
    data_type = 'type'
    all_data = pd.read_csv(to_model_data_path)
    train_data = all_data[all_data['type'] == 'train']

    n_cols, c_cols, d_cols = select_features_dtypes(train_data, exclude=[cust_id, target, data_type])
    print('数值特征个数: {}'.format(len(n_cols)))
    print('字符特征个数: {}'.format(len(c_cols)))
    print('日期特征个数: {}'.format(len(d_cols)))

    category_2_woe_save_path = os.path.join('..','tests')

    print("类别变量数据处理前", all_data[c_cols])
    if c_cols:
        print('类别变量数据处理')
        # train_data.loc[:, category_cols] = train_data.loc[:, category_cols].fillna('miss')
        # test_data.loc[:, category_cols] = test_data.loc[:, category_cols].fillna('miss')

        var_value_woe = category_2_woe(train_data, c_cols, target=target)
        category_2_woe_save(var_value_woe, '{}'.format(category_2_woe_save_path))
        # var_value_woe = category_2_woe_load('{}'.format(output_dir))
        train_data = transform(train_data, var_value_woe)
        all_data = transform(all_data, var_value_woe)

    print("类别变量数据处理后", all_data[c_cols])



#==============================================================================
# File: Ch05特征工程提取有效的风险特征_20211018.py
#==============================================================================

#------------------------------------------------------------------------------
"""
功能说明：
    本代码是第5章特征工程提取有效的风险特征配套代码。
算法流程：
    1、特征组合多项式特征
    2、非负矩阵分解
    3、featuretools包
    4、TSFresh包
输入数据：
    使用代码自带数据，无需额外的外部数据
输出数据：
    各代码段输出相应结果变量
版本历史：
    20211018：定稿提交出版
"""

#------------------------------------------------------------------------------
# 特征组合多项式特征

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X=np.arange(9).reshape(3,3)

poly=PolynomialFeatures(2) #二阶多项式
poly.fit_transform(X)

poly=PolynomialFeatures(degree=3, interaction_only=True) #三阶多项式仅保留交叉项
poly.fit_transform(X)


#------------------------------------------------------------------------------
# 非负矩阵分解

import numpy as np
from sklearn.decomposition import NMF
from sklearn.datasets import load_iris

X, _ = load_iris(True)

#定义模型
nmf = NMF(n_components=2,  # n_components即前文矩阵分解中的k，如果不设定该参数则默认保留全部特征
          init=None,  # W和H的初始化方法，包括'random','nndsvd'(默认),'nndsvda','nndsvdar','custom'.
          solver='cd',  #取值：'cd'、'mu'
          beta_loss='frobenius',  #取值：{'frobenius','kullback-leibler','itakura-saito'}，一般保持默认
          tol=1e-4,  # 停止迭代的极限条件
          max_iter=1000,  #最大迭代次数
          random_state=None,
          alpha=0.,  #正则化参数
          l1_ratio=0.,  #正则化参数
          verbose=0,  #冗长模式
          shuffle=False  #针对"cd solver"
          )

#模型参数
print('params:', nmf.get_params())  #获取构造函数参数的值，也可以通过nmf.attr得到

#模型拟合
nmf.fit(X)
W = nmf.fit_transform(X)
nmf.inverse_transform(W)
H = nmf.components_  # H矩阵
X_= np.dot(W,H)

print('reconstruction_err_', nmf.reconstruction_err_)  #损失函数值
print('n_iter_', nmf.n_iter_)  #迭代次数


#------------------------------------------------------------------------------
# 使用featuretools包进行特征衍生

#导入相关包
import featuretools as ft


#查看自带的数据情况
es = ft.demo.load_mock_customer(return_entityset=True)
es.plot()


#数据载入
data=ft.demo.load_mock_customer()
customers_df=data["customers"]
sessions_df=data["sessions"]
transactions_df=data["transactions"]


#创建实体和实体间关联关系
dataframes={"customers": (customers_df,"customer_id"),
          "sessions": (sessions_df,"session_id","session_start"),
          "transactions":(transactions_df,"transaction_id","transaction_time")
        }

relationships=[("sessions","session_id","transactions","session_id"),
               ("customers","customer_id","sessions","customer_id")
        ]


#运行DFS衍生特征
feature_matrix_customers, features_defs=ft.dfs(
        dataframes=dataframes,
        relationships=relationships,
        target_dataframe_name="customers")


#查看衍生的变量
feature_matrix_customers_columnslst=list(feature_matrix_customers.columns)


#------------------------------------------------------------------------------
# 使用tsfresh包进行特征衍生

#导入相关包
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures

#下载和读入数据
download_robot_execution_failures()  #下载数据
timeseries, y = load_robot_execution_failures() #加载数据
timeseries.head()
y.head()

#显示数据前几行
print(timeseries.head())
print(y.head())

#显示时间序列
import matplotlib.pyplot as plt
timeseries[timeseries['id'] == 3].plot(subplots=True, sharex=True, figsize=(10,10))
y[3] #True正常
plt.show()
timeseries[timeseries['id'] == 21].plot(subplots=True, sharex=True, figsize=(10,10))
y[21] #False有故障
plt.show()

#特征提取
from tsfresh import extract_features
extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
#特征选择，基于上一步特征提取的结果，注意不允许出现NaN值，所以需要使用impute先填充
from tsfresh.utilities.dataframe_functions import impute
impute(extracted_features) #缺失值都用0填充
from tsfresh import select_features
features_filtered = select_features(extracted_features, y)

#特征提取+特征选择
from tsfresh import extract_relevant_features
features_filtered_direct = extract_relevant_features(timeseries, y,column_id='id', column_sort='time')


#==============================================================================
# File: Ch07评分卡模型开发_20211018.py
#==============================================================================

#------------------------------------------------------------------------------
"""
功能说明：
    本代码是第7章评分卡模型开发配套代码
算法流程：
    1、使用scikit-learn包的LogisticRegression类建立逻辑回归模型
    2、使用statsmodels包的Logit类建立逻辑回归模型
    3、使用scorecardpy包建模
    4、使用toad包建模
输入数据：
    使用代码自带数据，无需额外的外部数据
输出数据：
    各代码段输出相应结果变量
版本历史：
    20211018：定稿提交出版
"""

#------------------------------------------------------------------------------
# 使用scikit-learn的LogisticRegression类建立逻辑回归模型

#导入相关包和模块
import pandas as pd

from sklearn.datasets import load_breast_cancer #乳腺癌数据，Target：二分类
from sklearn.linear_model import LogisticRegression #分类
from sklearn.model_selection import train_test_split #数据集划分

#准备数据，Target二分类
ds_cancer = load_breast_cancer()
data=pd.DataFrame(ds_cancer.data).add_prefix('X')
target = pd.DataFrame(ds_cancer.target,columns=['y'])
X_train,X_test,y_train,y_test =train_test_split(data,target,test_size=0.3)

#定义分类器
clf=LogisticRegression(fit_intercept=True, random_state=123) #模型带截距项
clf.get_params()

#模型拟合
clf.fit(X_train,y_train)

#获取模型拟合系数
clf.coef_
clf.intercept_

'''
注：上述代码仅用于演示LogisticRegression类的使用，计算模型拟合结果后还需要进行模型结果的评估和验证，
必要时需要迭代地进行变量选择。
'''


#------------------------------------------------------------------------------
# 使用statsmodels包的Logit类建立逻辑回归模型

#导入包
import statsmodels.api as sm  #回归类模型

#变量筛选通过相关性
X_train_corr=X_train.corr()[X_train.corr()>0.9] #计算变量相关性发现X0和X1、X2,X20和X22、X23相关性很高
X_train1=X_train.drop(['X0','X2','X3','X10','X12','X13','X20','X22','X23'],axis=1)

#加上常数项
X_train1=sm.add_constant(X_train1)

#拟合模型
model = sm.Logit(y_train, X_train1)
results = model.fit()

#模型结果
print(results.summary())
print(results.params)

'''
注：上述代码仅用于演示Logit类的使用，计算出模型拟合结果后还需要进行进一步的统计检验，
统计检验显示，多个变量的p值>0.05，故变量不显著，所以需要迭代地将不显著的变量去除。
'''


#------------------------------------------------------------------------------
# 使用scorecardpy包建模

"""
功能说明：
	本程序使用scorecardpy进行评分卡建模
算法流程：
	依次读入数据、变量筛选、数据分区、变量分箱、分箱调整、变量转换WOE、训练模型、模型评估、模型验证、评分标尺
输入数据：
	本程序不需要额外输入数据，sc.germancredit自带建模数据
输出数据：
	评分卡模型结果
版本历史：
"""

#导入相关包
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve

import scorecardpy as sc


#1.读入数据

#读入数据
data = sc.germancredit()

#数据信息
data.info()
data.describe()


#2.变量筛选
data_s = sc.var_filter(data, 
                       y="creditability",
                       iv_limit=0.02, 
                       missing_limit=0.95, 
                       identical_limit=0.95, 
                       var_rm=None, 
                       var_kp=None, 
                       return_rm_reason=False, 
                       positive='bad|1')


#3.数据分区
train, test = sc.split_df(data_s, 'creditability', ratio=0.7, seed=123).values()


#4.变量分箱
#自动分箱
bins = sc.woebin(train, y="creditability")
#细分箱结果报告
sc.woebin_plot(bins)


#5.分箱调整
#交互式输入cut后分箱
#breaks_adj = sc.woebin_adj(train, "creditability", bins) 
#也可以手动设置
breaks_adj = {'age.in.years': [22, 35, 40,60],
              'other.debtors.or.guarantors': ["none", "co-applicant%,%guarantor"]}
bins_adj = sc.woebin(train, y="creditability", breaks_list=breaks_adj)


#6.变量转换WOE
train_woe = sc.woebin_ply(train, bins_adj)
test_woe = sc.woebin_ply(test, bins_adj)


#7.训练模型

#处理数据
X_train = train_woe.loc[:,train_woe.columns != 'creditability']
y_train = train_woe.loc[:,'creditability']
X_test = test_woe.loc[:,train_woe.columns != 'creditability']
y_test = test_woe.loc[:,'creditability']

#定义分类器
lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)
lr.get_params()

#拟合模型
lr.fit(X_train, y_train)

#拟合的参数
lr.coef_
lr.intercept_


#8.模型评估
# predicted proability for train
y_train_pred = lr.predict_proba(X_train)[:,1]
# 绘制KS和ROC、PR曲线
train_perf = sc.perf_eva(y_train, y_train_pred, plot_type=["ks", "roc","pr","lift"], title = "train")
plot_roc_curve(lr,X_train,y_train) 
plot_precision_recall_curve(lr,X_train,y_train)


#9.模型验证
# predicted proability for test
y_test_pred = lr.predict_proba(X_test)[:,1]
# 绘制KS和ROC、PR曲线
test_perf = sc.perf_eva(y_test, y_test_pred, plot_type=["ks", "roc","pr","lift"], title = "test")
plot_roc_curve(lr,X_test,y_test) 
plot_precision_recall_curve(lr,X_test,y_test) 


#10.评分标尺
card = sc.scorecard(bins_adj, 
                    lr, 
                    X_train.columns,
                    points0=600, 
                    odds0=1/19, 
                    pdo=50, 
                    basepoints_eq0=True)

#使用评分标尺打分
train_score = sc.scorecard_ply(train, card, print_step=0)
test_score = sc.scorecard_ply(test, card, print_step=0)

#比较train/test分数分布是否一致，计算分值分布PSI
sc.perf_psi(
        score = {'train':train_score, 'test':test_score},
        label = {'train':y_train, 'test':y_test})



#------------------------------------------------------------------------------
# 使用toad包建模

"""
功能说明：
    本程序使用toad进行评分卡建模
算法流程：
    依次进行读入数据、样本分区、数据EDA报告、特征分析、特征预筛选、特征分箱、调整合并分箱、特征选择、模型训练、模型评估、模型验证、评分标尺
输入数据：
    数据下载为https://www.kaggle.com/c/GiveMeSomeCredit/data
输出数据：
    评分模型结果
版本历史：
"""

#导入相关包
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import toad
from toad.plot import badrate_plot, proportion_plot, bin_plot
from toad.metrics import KS, F1, AUC


#1. 读入数据

#读入数据
data = pd.read_csv(r'D:\cs-training.csv')

#数据描述
data.info()
data.describe()
data.head()


#2. 样本分区
Xtr,Xts,Ytr,Yts = train_test_split(data.drop('SeriousDlqin2yrs',axis=1),
                                   data['SeriousDlqin2yrs'],
                                   test_size=0.25,
                                   random_state=450)
data_tr = pd.concat([Xtr,Ytr],axis=1)
data_tr['type'] = 'train'
data_ts = pd.concat([Xts,Yts],axis=1)
data_ts['type'] = 'test'


#3. 数据EDA报告
toad.detector.detect(data_tr).to_excel(r'D:\数据EDA结果.xlsx')


#4. 特征分析计算特征IV、gini、entropy、unique
quality = toad.quality(data,'SeriousDlqin2yrs')
quality.head(6)


#5. 特征预筛选
selected_train, drop_lst= toad.selection.select(data_tr,target = 'SeriousDlqin2yrs', 
                                               empty = 0.5, 
                                               iv = 0.05, 
                                               corr = 0.7, 
                                               return_drop=True, 
                                               exclude='type')
selected_test = data_ts[selected_train.columns]
selected_train.shape
drop_lst  #删除的额变量


#6. 特征分箱，必须基于train数据集来做

# 初始化一个combiner类
combiner = toad.transform.Combiner()

# 训练数据并指定分箱方法，需要分箱的变量共7个
combiner.fit(selected_train,
             y='SeriousDlqin2yrs',
             method='chi',
             min_samples =  0.05,
             exclude='type')

# 以字典形式保存分箱结果
bins = combiner.export()

#查看每个特征的分箱结果
print('DebtRatio分箱cut:',bins['DebtRatio'])
print('MonthlyIncome分箱cut:',bins['MonthlyIncome'])
print('NumberOfOpenCreditLinesAndLoans分箱cut:',bins['NumberOfOpenCreditLinesAndLoans'])
print('NumberOfTimes90DaysLate分箱cut:',bins['NumberOfTimes90DaysLate'])
print('NumberRealEstateLoansOrLines分箱cut:',bins['NumberRealEstateLoansOrLines'])
print('RevolvingUtilizationOfUnsecuredLines分箱cut:',bins['RevolvingUtilizationOfUnsecuredLines'])
print('age分箱cut:',bins['age'])

#使用combiner.transform方法对数据进行分箱转换
selected_train_bin = combiner.transform(selected_train)

#画分箱图，bin_plot双轴图同时绘制分箱占比和分箱badrate
proportion_plot(selected_train_bin['DebtRatio'])
proportion_plot(selected_train_bin['MonthlyIncome'])
proportion_plot(selected_train_bin['NumberOfOpenCreditLinesAndLoans'])
proportion_plot(selected_train_bin['NumberOfTimes90DaysLate'])
proportion_plot(selected_train_bin['NumberRealEstateLoansOrLines'])
proportion_plot(selected_train_bin['RevolvingUtilizationOfUnsecuredLines'])
proportion_plot(selected_train_bin['age'])
badrate_plot(selected_train_bin, target = 'SeriousDlqin2yrs', x = 'type',by = 'DebtRatio')
badrate_plot(selected_train_bin, target = 'SeriousDlqin2yrs', x = 'type',by = 'MonthlyIncome')
badrate_plot(selected_train_bin, target = 'SeriousDlqin2yrs', x = 'type',by = 'NumberOfOpenCreditLinesAndLoans')
badrate_plot(selected_train_bin, target = 'SeriousDlqin2yrs', x = 'type',by = 'NumberOfTimes90DaysLate')
badrate_plot(selected_train_bin, target = 'SeriousDlqin2yrs', x = 'type',by = 'NumberRealEstateLoansOrLines')
badrate_plot(selected_train_bin, target = 'SeriousDlqin2yrs', x = 'type',by = 'RevolvingUtilizationOfUnsecuredLines')
badrate_plot(selected_train_bin, target = 'SeriousDlqin2yrs', x = 'type',by = 'age')
bin_plot(selected_train_bin,x='DebtRatio',target='SeriousDlqin2yrs') 
bin_plot(selected_train_bin,x='MonthlyIncome',target='SeriousDlqin2yrs') 
bin_plot(selected_train_bin,x='NumberOfOpenCreditLinesAndLoans',target='SeriousDlqin2yrs')
bin_plot(selected_train_bin,x='NumberOfTimes90DaysLate',target='SeriousDlqin2yrs')
bin_plot(selected_train_bin,x='NumberRealEstateLoansOrLines',target='SeriousDlqin2yrs')
bin_plot(selected_train_bin,x='RevolvingUtilizationOfUnsecuredLines',target='SeriousDlqin2yrs')
bin_plot(selected_train_bin,x='age',target='SeriousDlqin2yrs')


#7. 调整合并分箱

#定义调整分箱#调整分箱cutpoint
bins_adj=bins
bins_adj["age"]=[22, 35, 45, 60]
bins_adj["NumberOfOpenCreditLinesAndLoans"]=[2]
bins_adj["DebtRatio"]=[0.02,0.4,0.5,2] 

#定义分箱combiner
combiner2 = toad.transform.Combiner() #定义分箱combiner
combiner2.set_rules(bins_adj) #设置需要施加的分箱

#应用调整分箱
selected_train_binadj = combiner2.transform(selected_train)

#画分箱坏账率badrate图
proportion_plot(selected_train_binadj['DebtRatio'])
proportion_plot(selected_train_binadj['MonthlyIncome'])
proportion_plot(selected_train_binadj['NumberOfOpenCreditLinesAndLoans'])
proportion_plot(selected_train_binadj['NumberOfTimes90DaysLate'])
proportion_plot(selected_train_binadj['NumberRealEstateLoansOrLines'])
proportion_plot(selected_train_binadj['RevolvingUtilizationOfUnsecuredLines'])
proportion_plot(selected_train_binadj['age'])
badrate_plot(selected_train_binadj, target = 'SeriousDlqin2yrs', x = 'type',by = 'DebtRatio')
badrate_plot(selected_train_binadj, target = 'SeriousDlqin2yrs', x = 'type',by = 'MonthlyIncome')
badrate_plot(selected_train_binadj, target = 'SeriousDlqin2yrs', x = 'type',by = 'NumberOfOpenCreditLinesAndLoans')
badrate_plot(selected_train_binadj, target = 'SeriousDlqin2yrs', x = 'type',by = 'NumberOfTimes90DaysLate')
badrate_plot(selected_train_binadj, target = 'SeriousDlqin2yrs', x = 'type',by = 'NumberRealEstateLoansOrLines')
badrate_plot(selected_train_binadj, target = 'SeriousDlqin2yrs', x = 'type',by = 'RevolvingUtilizationOfUnsecuredLines')
badrate_plot(selected_train_binadj, target = 'SeriousDlqin2yrs', x = 'type',by = 'age')


#8. 转换WOE值

#设置分箱号
combiner.set_rules(bins_adj)

#将特征的值转化为分箱的箱号。
selected_train_binadj = combiner.transform(selected_train)
selected_test_binadj = combiner.transform(selected_test)

#定义WOE转换器
WOETransformer = toad.transform.WOETransformer()

#对WOE的值进行转化，映射到原数据集上。对训练集用fit_transform，测试集用transform
data_tr_woe = WOETransformer.fit_transform(selected_train_binadj, 
                                           selected_train_binadj['SeriousDlqin2yrs'], 
                                           exclude=['SeriousDlqin2yrs','type'])
data_ts_woe = WOETransformer.transform(selected_test_binadj)



#9. 特征选择，使用stepwise选择变量
train_final = toad.selection.stepwise(data_tr_woe.drop('type',axis=1),
                                     target = 'SeriousDlqin2yrs',
                                     direction = 'both', 
                                     criterion = 'aic')

test_final = data_ts_woe[train_final.columns]
print(train_final.shape) #7个特征减少为5个特征。


#10. 模型训练

#准备数据
Xtr = train_final.drop('SeriousDlqin2yrs',axis=1)
Ytr = train_final['SeriousDlqin2yrs']

#逻辑回归模型拟合
lr = LogisticRegression()
lr.fit(Xtr, Ytr)

#打印模型拟合的参数
lr.coef_
lr.intercept_


#11. 模型评估

#在训练集上的模型表现
EYtr_proba = lr.predict_proba(Xtr)[:,1]
EYtr = lr.predict(Xtr)

print('train F1:', F1(EYtr_proba,Ytr))
print('train KS:', KS(EYtr_proba,Ytr))
print('train AUC:', AUC(EYtr_proba,Ytr))

#分值排序性
tr_bucket = toad.metrics.KS_bucket(EYtr_proba,Ytr,bucket=10,method='quantile')  #等频分段
tr_bucket


#12. 模型验证

#在测试集上的模型表现
Xts = test_final.drop('SeriousDlqin2yrs',axis=1)
Yts = test_final['SeriousDlqin2yrs']

EYts_proba = lr.predict_proba(Xts)[:,1]
EYts = lr.predict(Xts)

print('test F1:', F1(EYts_proba,Yts))
print('test KS:', KS(EYts_proba,Yts))
print('test AUC:', AUC(EYts_proba,Yts))

#比较train、test变量稳定性分布是否有显著差异，基于分箱之后的数据
psi = toad.metrics.PSI(train_final,test_final)
psi.sort_values(0,ascending=False)


#13. 分值转换scaling
scorecard = toad.scorecard.ScoreCard(combiner = combiner, transer = WOETransformer , C = 0.1)
scorecard.fit(Xtr, Ytr)
scorecard.export(to_frame = True,)


#==============================================================================
# File: Ch09评分卡模型部署_20211018.py
#==============================================================================

#------------------------------------------------------------------------------
"""
功能说明：
    本代码是第9章评分卡模型部署配套代码。
算法流程：  
    - 训练模型并将模型持久化为PKL文件
    - 本地加载模型PKL文件
    - 训练模型并将模型持久化为PMML文件
    - 本地加载模型PMML文件
    - 在服务器部署模型PMML，然后在客户端调用打分服务
输入数据：
    使用代码自带数据，无需额外的外部数据
输出数据：
    各代码段输出相应结果变量
版本历史：
    20211018：定稿提交出版
"""

#------------------------------------------------------------------------------
# 训练模型并将模型持久化为PKL文件

#导入相关包
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn2pmml import PMMLPipeline

#读入数据
iris = load_iris()	
X_train=pd.DataFrame(iris.data,columns=['sepal_length','sepal_width','petal_length', 'petal_width'])
y_train=pd.DataFrame(iris.target,columns=['series'])

#训练模型pipeline
clf = tree.DecisionTreeClassifier(max_depth=2) #定义分类器
pipeline = PMMLPipeline([("classifier", clf)]) #定义pipeline
pipeline.fit(X_train, y_train) #此处使用带columns变量名称的dataframe进行模型训练

#方法1：使用pickle包将模型保存为pkl
import pickle
with open("D:\\mdl.pkl", "wb") as f:
    pickle.dump(pipeline, f)
    
#方法2：使用joblib包将模型导出为pkl
#from sklearn.externals import joblib  #高版本sklearn不再支持joblib
import joblib
joblib.dump(pipeline, "d:\\mdl.pkl", compress = 9)


#------------------------------------------------------------------------------
# 本地加载和使用模型PKL文件

#使用pickle包读取pickle
with open('D:\\mdl.pkl', 'rb') as f:
    mdl_in = pickle.load(f)
y_pred=mdl_in.predict(iris.data)

#使用joblib包读取pickle
mdl_in=joblib.load("d:\\mdl.pkl")
y_pred=mdl_in.predict(iris.data)


#------------------------------------------------------------------------------
# 将模型持久化为PMML文件

# 方法一：使用sklearn2pmml包导出模型PMML文件

#导入相关包
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn2pmml import PMMLPipeline

#读入数据
iris = load_iris()
X_train=pd.DataFrame(iris.data,columns=['sepal_length','sepal_width','petal_length', 'petal_width'])
y_train=pd.DataFrame(iris.target,columns=['series'])

#训练模型pipeline
clf = tree.DecisionTreeClassifier(max_depth=2) #定义分类器
pipeline = PMMLPipeline([("classifier", clf)]) #定义pipeline
pipeline.fit(X_train, y_train) #此处使用带columns变量名称的dataframe进行模型训练

#模型导出为PMML
from sklearn2pmml import sklearn2pmml
sklearn2pmml(pipeline, "d:\\DecisionTree_Iris_sklearn2pmml.pmml", with_repr = True) #生成PMML时带变量名


# 方法二：使用nyoka包导出模型PMML文件

#导入相关包
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml import PMMLPipeline

#读入数据
iris = load_iris()
features = iris.feature_names
target = 'Species'

#创建pipeline并训练模型
clf_pipeline=PMMLPipeline([('clf',DecisionTreeClassifier(max_depth=2))])
clf_pipeline.fit(iris.data, iris.target) #此处训练模型时用的是数组不带变量名称

#使用nyoka将模型导出为pmml
from nyoka import skl_to_pmml
skl_to_pmml(clf_pipeline, features, target, "d:\\DecisionTree_iris_nyoka.pmml") #生成PMML时带变量名


#------------------------------------------------------------------------------
# 本地加载和使用PMML模型文件

#加载pmml
from pypmml import Model
model = Model.fromFile("d:\\DecisionTree_Iris_sklearn2pmml.pmml")

#使用PMML的模型打分，整个数据集
y_train_pred=model.predict(X_train) #注：此处待打分的DataFrame是否带变量名称须与训练模型PMML时保持一致

#使用PMML的模型打分，单条记录
model.predict({'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2})
model.predict('[{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]')
model.predict('{"columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"], "data": [[5.1, 3.5, 1.4, 0.2]]}')
model.predict(pd.Series({'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2}))


#------------------------------------------------------------------------------
# 下面代码使用FastAPI包实现在服务器部署模型


# （1）首先将下面代码保存在服务器端，命名为main.py，然后在服务器端执行命令行：先定位到main.py目录，然后执行：uvicorn main:app –-reload

#导入相关包和模块
from fastapi import FastAPI
from pypmml import Model

#定义FastAPI对象
app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int, x: str=''):
    
    #读取模型PMML
    mdl = Model.fromFile("d:\\DecisionTree_Iris_sklearn2pmml.pmml")
    
    #将读入的字符串x输入predict函数得到预测结果
    y_predict=mdl.predict(x)

    #将计算结果返回给客户端
    return {"item_id": item_id, "x":x, "y_predict": y_predict}


# （2）客户端执行如下代码，在服务器模式下执行时将127.0.0.1替换为服务器IP地址
URL_str='http://127.0.0.1:8000/items/5?x='+'[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}]'
res=requests.get(URL_str)
returnjson=res.text
print(returnjson)




#------------------------------------------------------------------------------
# 下面代码使用Flask包实现在服务器部署模型

# （1）首先将下面代码保存在服务器端，命名为main.py，然后在服务器端执行命令行：python main.py

#导入相关包和模块
import numpy as np
import pandas as pd

from pypmml import Model

from flask import Flask
from flask import request
from flask import jsonify


#导入模型
model = Model.fromFile("d:\\DecisionTree_Iris_sklearn2pmml.pmml")
 
app = Flask(__name__)
 
@app.route('/',methods=['POST','GET'])
def scoring():
    text=request.args.get('inputdata')
    if text:
        temp =  [float(x) for x in text.split(',')]
        temp = pd.DataFrame(data=np.array(temp).reshape((1, -1)),columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
        ouputdata = model.predict(temp)	#outputdata是DataFrame格式
        return jsonify(dict(ouputdata.iloc[0])) #进行json化
        
if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(host='127.0.0.1',port=5003)  # 127.0.0.1 #指的是本地ip
    
    
# （2）客户端执行如下代码，在服务器模式下执行时将127.0.0.1替换为服务器IP地址

import requests

base = 'http://127.0.0.1:5003/?inputdata=5.1,3.5,1.4,2'
response = requests.get(base)
print(response.text)
answer = response.json()
print('预测结果',answer)


#==============================================================================
# File: Ch13评分卡模型可解释性_20211018.py
#==============================================================================

#------------------------------------------------------------------------------
"""
功能说明：
    本代码是第13章评分卡模型可解释性配套代码。
算法流程：  
    - PDP与ICE
    - 变量重要性方法：XGBoost和LightGBM的plot_importance
    - SKlearn模型解释工具treeinterpreter包
    - 特征随机置换Permutation Importance，使用eli5包
    - LIME
    - SHAP
输入数据：
    使用代码自带数据，无需额外的外部数据
输出数据：
    各代码段输出相应结果变量
版本历史：
    20211018：定稿提交出版
"""


#------------------------------------------------------------------------------
# 导入相关包和模块

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing

from sklearn.tree import DecisionTreeClassifier #分类
from sklearn.tree import DecisionTreeRegressor #回归
from sklearn.ensemble import RandomForestRegressor #随机森林
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#------------------------------------------------------------------------------
# PDP与ICE
"""
PDP方法有两个工具包可用：
—— sklearn.inspection
—— pdpbox
"""

#读入数据
from sklearn.datasets import fetch_california_housing

cal_housing=fetch_california_housing()
X=pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y=cal_housing.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


#训练模型
from sklearn.ensemble import GradientBoostingRegressor
gbdt=GradientBoostingRegressor()
gbdt.fit(X_train,y_train)


#方法一：使用sklearn.inspection进行PDP分析
from sklearn.inspection import plot_partial_dependence
fig,ax=plt.subplots(figsize=(12,4))
plot_partial_dependence(gbdt,
                        X_train,
                        ['MedInc','AveOccup','HouseAge'],
                        method="brute",
                        ax=ax)

#也可以输出三维图形，考察两个变量间的交互性
fig,ax=plt.subplots(figsize=(9,6))
plot_partial_dependence(gbdt,
                        X_train,
                        [('HouseAge','AveOccup')],
                        grid_resolution=50,
                        method="brute",
                        ax=ax)


#方法二：使用pdpbox包
from pdpbox import pdb
pdp_MedInc=pdp.pdp_isolate(model=gbdt,
                           dataset=X_train,
                           model_features=X_train.columns.tolist(),
                           feature='MedInc',
                           num_grid_points=30)

pdb.pdp_plot(pdp_MedInc,
             'MedInc',
             center=False
             )

#使用pdpbox包绘制单实例ICE图
pdb.pdp_plot(pdp_MedInc,
             'MedInc',
             center=False,
             plot_lines=True,
             frac_to_plot=10,
             plot_pts_dist=True)




#------------------------------------------------------------------------------
#变量重要性方法：XGBoost和LightGBM的plot_importance

#导入包
from sklearn.datasets import load_boston

import xgboost as xgb
import lightgbm as lgb


#读取数据
ds=load_boston()
df=pd.DataFrame(data=ds.data)
df=df.add_prefix('X')
df=df.join(pd.DataFrame(ds.target,columns=['y']))


#定义xgb预测器
clf=xgb.XGBRegressor()
clf.get_params()

#拟合模型
clf.fit(df.iloc[:,0:13],df.iloc[:,-1])

#模型评估
clf.score(df.iloc[:,0:13],df.iloc[:,-1])

#打印变量重要性
xgb.plot_importance(clf,importance_type='gain')


#定义ligbm预测器
lgbdata=lgb.Dataset(df.iloc[:,0:13],df.iloc[:,-1])
# 将参数写成字典下形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression', # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 31,   # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9, # 建树的特征选择比例
    'bagging_fraction': 0.8, # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}
clf_lgb=lgb.train(params,lgbdata)

#绘制模型重要性
clf_lgb.feature_importance()
plt.bar(height=clf_lgb.feature_importance(),x=df.iloc[:,0:13].columns)




#------------------------------------------------------------------------------
# SKlearn模型解释工具treeinterpreter包

#导入treeinterpreter包
from treeinterpreter import treeinterpreter as ti

#加载数据
ds=load_boston()

#定义分类器
rf=RandomForestRegressor(random_state=123)
#拟合模型
rf.fit(ds.data,ds.target)

#取出一个样本
spl=ds.data[0].reshape(1,-1)
#使用模型打分
rf.predict(spl)

#使用treeinterpreter解释，prediction是预测值，bias是全体样本Y平均值
prediction,bias,contributions=ti.predict(rf,spl)

#各变量的contributions
df_contributions=pd.DataFrame(data=np.hstack([ds.feature_names.reshape(-1,1),
                                            contributions.reshape(-1,1)]),
                            columns=['Feature','contribution'])
df_contributions.sort_values(by=['contribution'],ascending=False)

#验证计算逻辑
print(ds.target.mean()) #全体样本目标真实值平均值
print(rf.predict(ds.data).mean()) #rf预测值平均值
print(rf.predict(spl))
print(prediction)
print(bias)
print(prediction-np.sum(contributions))  #prediction是bias和每个变量贡献的总和




#------------------------------------------------------------------------------
#特征随机置换Permutation Importance，使用eli5包

#导入包
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor #随机森林

import eli5
from eli5.sklearn import PermutationImportance


#加载数据
ds=load_boston()

#定义分类器
rf=RandomForestRegressor(random_state=123)
#拟合模型
rf.fit(pd.DataFrame(ds.data,columns=ds.feature_names),ds.target)
rf.feature_importances_


#计算置换变量值重要性
perm=PermutationImportance(rf).fit(pd.DataFrame(ds.data,columns=ds.feature_names),ds.target)
df_perm=pd.DataFrame(data=np.hstack([ds.feature_names.reshape(-1,1),
                                     perm.feature_importances_.reshape(-1,1).round(4),
                                     perm.feature_importances_std_.reshape(-1,1).round(4)]),
                    columns=['Feature','mean','std'])
df_perm.sort_values(by=['mean'],ascending=False,inplace=True)


#查看置换变量重要性绘图，本段代码只能在notebook中查看
eli5.show_weights(perm,feature_names=ds.feature_names) 




#------------------------------------------------------------------------------
#LIME

#导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor #随机森林
from sklearn.model_selection import train_test_split

import lime
import lime.lime_tabular


#加载数据
ds=load_boston()

#定义分类器
rf=RandomForestRegressor(random_state=123)
#拟合模型
rf.fit(ds.data,ds.target)


#取值水平数小于10的视作分类变量
categorical_features=np.argwhere(np.array([len(set(ds.data[:,1])) for i in range(ds.data.shape[1])])<=10).flatten()


#创建解释器
explainer=lime.lime_tabular.LimeTabularExplainer(
        ds.data,
        feature_names=ds.feature_names,
        class_names=['house_price'],
        categorical_features=None,
        verbose=True,
        mode='regression'
        )

#选取一个样本
spl=ds.data[0]
#生成模型解释结果
exp=explainer.explain_instance(
        spl,
        rf.predict,
        num_features=5
        )
#输出各变量的贡献
exp.as_list()
#进行可视化，本段代码只能在Jupyter notebook中查看
exp.show_in_notebook(show_table=True)




#------------------------------------------------------------------------------
# SHAP

#导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

import shap

#初始化图形环境
shap.initjs()

#加载数据
ds=load_boston()
#取出一个样本作为下文单样本SHAP的实例
spl=ds.data[0].reshape(1,-1)

#定义基准分类器
rf=RandomForestRegressor(random_state=123)
#拟合模型
rf.fit(ds.data,ds.target)


#定义shap树解释器
explainer=shap.TreeExplainer(rf,data=ds.data)

#训练集上全体样本预测均值作为基准值
explainer.expected_value  #22.28338



#------------------------------------
#该单样本上各变量的SHAP值
splshapvalues=explainer.shap_values(spl).round(4)

df_splshapvalues=pd.DataFrame(data=np.hstack([ds.feature_names.reshape(-1,1),
                                            splshapvalues.reshape(-1,1),
                                            abs(splshapvalues).reshape(-1,1)]),
                            columns=['Feature','shap','shapabs'])
df_splshapvalues.sort_values(by=['shapabs'],ascending=False,inplace=True)  #按SHAP绝对值降序排列
df_splshapvalues.drop(['shapabs'],axis=1,inplace=True) #df_splshapvalues是纵表存储
df_splshapvaluescol=pd.DataFrame(data=splshapvalues,columns=ds.feature_names) #df_splshapvaluescol是横表存储

df_splshapvalues #显示单样本各变量shap值


#该单样本上验证计算逻辑
ds.target.mean()  #全体样本真实值均值 22.533
rf.predict(ds.data).mean() #全体样本rf预测值均值 22.535

explainer.expected_value  #Shap计算基准值即全体样本rf预测值均值22.28338
rf.predict(spl) #给定样本预测值 25.421
rf.predict(spl)-splshapvalues.sum() #22.2835 约等于explainer.expected_value


#查看单样本shap值绘图，本段代码只能在notebook中查看
shap.force_plot(explainer.expected_value,
                splshapvalues,
                features=spl,
                feature_names=ds.feature_names)



#------------------------------------
#样本集上各变量的SHAP值
shapvalues=explainer.shap_values(ds.data)

#查看样本集上shap值绘图，本段代码只能在Jupyter notebook中查看
shap.force_plot(explainer.expected_value,
                shapvalues,
                features=ds.data,
                feature_names=ds.feature_names)

#绘制决策路径图，本段代码只能在Jupyter notebook中查看
shap.decision_plot(explainer.expected_value,
                   shapvalues[:12],
                   ds.feature_names)

#绘制特征依赖图，本段代码只能在Jupyter notebook中查看
shap.dependence_plot(ds.feature_names.tolist().index('LSTAT'),
                     shapvalues,
                     ds.data)

#全局特征重要性，本段代码只能在Jupyter notebook中查看
shap.summary_plot(shapvalues,
                  ds.data,
                  feature_names=ds.feature_names,
                  max_display=5)
#以柱状图方式展示各变量SHAP绝对值平均值
shap.summary_plot(shapvalues,
                  feature_names=ds.feature_names,
                  plot_type='bar',
                  max_display=5)


#==============================================================================
# File: Ch15从评分卡模型到高维机器学习模型_20211018.py
#==============================================================================

#------------------------------------------------------------------------------
"""
功能说明：
    本代码是第15章从评分卡模型到高维机器学习模型配套代码。
算法流程：
    - 使用XGBoost建立预测模型
    - 使用LightGBM建立预测模型
输入数据：
    使用代码自带数据，无需额外的外部数据
输出数据：
    各代码段输出相应结果变量
版本历史：
    20211018：定稿提交出版
"""

#------------------------------------------------------------------------------
# 使用XGBoost建立预测模型

#导入库包
import pandas as pd

from sklearn.datasets import load_breast_cancer #乳腺癌数据，Target：二分类
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve

import xgboost
from xgboost import XGBClassifier

#准备数据
ds_cancer = load_breast_cancer()
data = pd.DataFrame(data=ds_cancer.data,columns=ds_cancer.feature_names)
target = pd.DataFrame(data=ds_cancer.target,columns=['target'])

#数据分区
X_train,X_test,y_train,y_test =train_test_split(data,target,test_size=0.3)

#定义XGBoost模型
xgb = XGBClassifier(n_estimators=3, max_depth=2)

#显示模型参数
xgb.get_params()

#模型拟合
xgb.fit(X_train, y_train)

#获得模型对象的属性和方法
xgb.score(X_train, y_train)
xgb.feature_importances_ 

#模型预测
y_train_predict=xgb.predict(X_train)
y_train_predict_proba=xgb.predict_proba(X_train)

#模型评估
accuracy_score(y_train,y_train_predict) #Accuracy指标基于真实标签vs预测标签
roc_auc_score(y_train, y_train_predict_proba[:,1]) #AUC指标基于真实标签vs预测概率
plot_roc_curve(xgb,X_train,y_train) #绘制ROC曲线
plot_precision_recall_curve(xgb,X_train,y_train) #绘制PR曲线

#打印变量重要性
xgboost.plot_importance(xgb) 





#------------------------------------------------------------------------------
# 使用LightGBM建立预测模型

#导入库包
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer #乳腺癌数据，Target：二分类
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve

import lightgbm as lgb
from lightgbm import LGBMClassifier


# 准备数据
ds_cancer = load_breast_cancer()
data = pd.DataFrame(data=ds_cancer.data,columns=ds_cancer.feature_names)
target = pd.DataFrame(data=ds_cancer.target,columns=['target'])

#数据分区
X_train,X_test,y_train,y_test =train_test_split(data,target,test_size=0.3)

#定义分类器
lgbm = LGBMClassifier(boosting_type="gbdt", class_weight=None, colsample_bytree=0.7, 
                                 isunbalance=True, learning_rate=0.01, max_bin=15, 
                                 max_depth=1, min_child_samples=100, min_child_weight=1, 
                                 min_split_gain=0.04, n_estimators=100, num_leaves=32, 
                                 objective="binary", random_state=27, subsample=0.8, subsample_freq=1)

#显示模型对象参数
lgbm.get_params()

#拟合模型
lgbm.fit(X_train,y_train)


#获得模型对象的属性
lgbm.classes_
lgbm.feature_importances_
lgbm.n_classes_
lgbm.n_features_
lgbm.objective_


#模型预测
y_train_predict=lgbm.predict(X_train)
y_train_predict_proba=lgbm.predict_proba(X_train)


#模型评估
fpr,tpr,pct = roc_curve(y_train, y_train_predict_proba[:,1]) #ROC曲线计算FPR和TPR序列值
ks=abs(fpr-tpr).max() #KS指标
plt.plot(tpr,"b-",fpr,"r-") #KS曲线
accuracy_score(y_train,y_train_predict) #Accuracy指标基于真实标签vs预测标签
roc_auc_score(y_train, y_train_predict_proba[:,1]) #AUC指标基于真实标签vs预测概率
plot_precision_recall_curve(lgbm,X_train,y_train) #绘制PR曲线
plot_roc_curve(lgbm,X_train,y_train) #绘制ROC曲线

#调用lightGBM函数绘制相关图
lgb.create_tree_digraph(lgbm,tree_index=1)
lgb.plot_importance(lgbm)
lgb.plot_tree(lgbm,tree_index=1,figsize=(12,9))


#==============================================================================
# End of batch 1
#==============================================================================
