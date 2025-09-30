# Auto-merged batch 2/4
# Total files in this batch: 59



#==============================================================================
# File: ch2_00_german_credit.py
#==============================================================================

# -*- coding: utf-8 -*- 

import scorecardpy as sc

# 加载数据集
german_credit_data = sc.germancredit()
# 打印前5行, 前4列和最后一列
print(german_credit_data.iloc[:5, list(range(-1, 4))])


#==============================================================================
# File: ch2_01_train_test_split.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.model_selection import train_test_split

# 导入添加month列的数据
model_data = data_utils.get_data()
# 选取OOT样本  
oot_set = model_data[model_data['month'] == '2020-05']
# 划分训练集和测试集
train_valid_set = model_data[model_data['month'] != '2020-05']
X = train_valid_set[data_utils.x_cols]
Y = train_valid_set['creditability']
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=88)
model_data.loc[oot_set.index, 'sample_set'] = 'oot'
model_data.loc[X_train.index, 'sample_set'] = 'train'
model_data.loc[X_valid.index, 'sample_set'] = 'valid'



#==============================================================================
# File: ch2_02_toad_eda_detect.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import toad
from utils import data_utils

# 加载数据集
german_credit_data = data_utils.get_data()
detect_res = toad.detector.detect(german_credit_data)
# 打印前5行, 前4列

print("前5行, 前4列:")
print(detect_res.iloc[:5, :4])
print("前5行, 第5至9列:")
# 打印前5行, 第5至9列
print(detect_res.iloc[:5, 4:9])
# 打印前5行, 第10至14列
print("前5行, 第10至14列:")
print(detect_res.iloc[:5, 9:])




#==============================================================================
# File: ch2_03_missrate_by_month.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")
from utils import data_utils

def missrate_by_month(x_with_month, month_col, x_cols):
    """
    按月统计缺失率
    :param x_cols: x变量列名
    :param month_col: 月份时间列名
    :param x_with_month: 包含月份的数据
    :return:
    """
    df = x_with_month.groupby(month_col)[x_cols].apply(lambda x: x.isna().sum() / len(x))
    df = df.T
    df['miss_rate_std'] = df.std(axis=1)
    return df

def main():
    """
    主函数
    """
    # 导入添加month列的数据
    model_data = data_utils.get_data()
    miss_rate_by_month = missrate_by_month(model_data, month_col='month', x_cols=data_utils.numeric_cols)
    print("按月统计缺失率结果: \n", miss_rate_by_month)

if __name__ == "__main__":
    main()




#==============================================================================
# File: ch2_04_preprocess_missing_value.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import numpy as np
import pandas as pd
from utils import data_utils
from sklearn.impute import SimpleImputer

# 导入数值型样例数据
data = data_utils.get_data()
# 缺失值处理
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imped_data = imp.fit_transform(data[data_utils.numeric_cols])
imped_df = pd.DataFrame(imped_data, columns=data_utils.numeric_cols)
print("缺失值填充结果: \n", imped_df)



#==============================================================================
# File: ch2_05_preprocess_value_scaler.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import pandas as pd
from utils import data_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# 导入数值型样例数据
data = data_utils.get_data()
# max-min标准化
X_MinMaxScaler = MinMaxScaler().fit_transform(data[data_utils.numeric_cols])
max_min_df = pd.DataFrame(X_MinMaxScaler, columns=data_utils.numeric_cols)
print("max-min标准化结果: \n", max_min_df)
# z-score标准化
X_StandardScaler = StandardScaler().fit_transform(data[data_utils.numeric_cols])
standard_df = pd.DataFrame(X_StandardScaler, columns=data_utils.numeric_cols)
print("z-score标准化结果: \n", standard_df)



#==============================================================================
# File: ch2_06_preprocess_value_bining.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import toad
from toad.plot import bin_plot
from utils import data_utils

german_credit_data = data_utils.get_data()
# 利用toad库等频分箱
# 初始化分箱对象
c = toad.transform.Combiner()
c.fit(german_credit_data[data_utils.x_cols],
      y=german_credit_data[data_utils.label], n_bins=6, method='quantile', empty_separate=True)
# 特征age.in.years分箱结果画图
data_binned = c.transform(german_credit_data, labels=True)
bin_plot(data_binned, x='age.in.years', target=data_utils.label)



#==============================================================================
# File: ch2_07_ordinal_encoder_based_sklearn.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import pandas as pd
from utils import data_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

def label_encode(x):
    """
    将原始分类变量用数字编码
    :param str x: 需要编码的原始变量
    :returns: x_encoded 数字编码后的变量
    """
    le = LabelEncoder()
    x_encoded = le.fit_transform(x.astype(str))
    class_ = le.classes_
    return class_, pd.DataFrame(x_encoded, columns=x.columns)

def ordinal_encode(x):
    """
    将原始分类变量用数字编码
    :param str x: 需要编码的原始变量，shape为[m,n]
    :returns: x_encoded 数字编码后的变量
    """
    enc = OrdinalEncoder()
    x_encoded = enc.fit_transform(x.astype(str))
    return pd.DataFrame(x_encoded).values


def main():
    """
    主函数
    """
    # 加载数据
    german_credit_data = data_utils.get_data()
    # 以特征purpose为例，进行类别编码
    class_, label_encode_x = label_encode(german_credit_data[['purpose']])
    print("特征'purpose'的类别编码结果: \n", label_encode_x)
    print("特征'purpose'编码顺序为: \n", class_)
    # 以特征purpose、credit.history为例，进行类别编码
    ordinal_encode_x = ordinal_encode(german_credit_data[['purpose', 'credit.history']])
    print("特征'purpose'和'credit.history'的类别编码结果: \n", ordinal_encode_x)


if __name__ == "__main__":
    main()




#==============================================================================
# File: ch2_08_ordinal_encode_based_category_encoders.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from category_encoders.ordinal import OrdinalEncoder

# 加载数据
german_credit_data = data_utils.get_data()
# 初始化OrdinalEncoder类
encoder = OrdinalEncoder(cols=['purpose', 'personal.status.and.sex'],
                         handle_unknown='value',
                         handle_missing='value')
# 将 handle_unknown设为"value"，即测试集中的未知特征值将被标记为-1
# 将 handle_missing设为"value"，即测试集中的缺失值将被标记为-2
# 当设为"error"，即报错；当设为"return_nan"，即未知值/缺失值被标记为nan
result = encoder.fit_transform(german_credit_data)
category_mapping = encoder.category_mapping
print("类别编码结果: \n", result)
print("类别编码映射关系: \n", category_mapping)



#==============================================================================
# File: ch2_09_one_hot_based_sklearn.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import pandas as pd
from utils import data_utils
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


def one_hot_encode(x):
    """
    将原始类别变量进行one-hot编码
    :param str x: 需要编码的原始变量
    :returns: x_oht one-hot编码后的变量
    """
    # 首先将类别值进行数值化
    re = OrdinalEncoder()
    x_encoded = re.fit_transform(x.astype(str))
    x_encoded = pd.DataFrame(x_encoded).values
    # 在对数值化后的类别变量进行one-hot编码
    ohe = OneHotEncoder(handle_unknown='ignore')
    x_oht = ohe.fit_transform(x_encoded).toarray()
    return x_oht

def main():
    """
    主函数
    """
    # 加载数据
    german_credit_data = data_utils.get_data()
    # 以特征purpose为例，进行one-hot编码
    label_encode_x = one_hot_encode(german_credit_data[['purpose']])
    label_encode_df = pd.DataFrame(label_encode_x)
    print("特征purpose的one-hot编码结果: \n", label_encode_df)


if __name__ == "__main__":
    main()


#==============================================================================
# File: ch2_10_one_hot_based_category_encoders.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from category_encoders.one_hot import OneHotEncoder


# 加载数据
german_credit_data = data_utils.get_data()
# 初始化OneHotEncoder类
encoder = OneHotEncoder(cols=['purpose', 'personal.status.and.sex'],
                        handle_unknown='indicator',
                        handle_missing='indicator',
                        use_cat_names=True)
# 转换数据集
result = encoder.fit_transform(german_credit_data)
print("one-hot编码结果: \n", result)


#==============================================================================
# File: ch2_11_target_encoder.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from category_encoders.target_encoder import TargetEncoder


# 加载数据
german_credit_data = data_utils.get_data()
y = german_credit_data['creditability']
x = german_credit_data[['purpose', 'personal.status.and.sex']]
# 目标编码
enc = TargetEncoder(cols=x.columns)
result = enc.fit_transform(x, y)
print("目标编码结果: \n", result)



#==============================================================================
# File: ch2_12_woe_encoder.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from category_encoders.woe import WOEEncoder

# 加载数据
german_credit_data = data_utils.get_data()
y = german_credit_data['creditability']
x = german_credit_data[['purpose', 'personal.status.and.sex']]

# WOE编码
encoder = WOEEncoder(cols=x.columns)
result = encoder.fit_transform(x, y)
print("WOE编码结果: \n", result)



#==============================================================================
# File: ch2_13_fs_variation.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from scipy.stats import variation

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
# 计算各个特征的变异系数
x_var = variation(x, nan_policy='omit')
result = dict(zip(x.columns ,x_var))
print("变异系数结果: \n", result)


#==============================================================================
# File: ch2_14_fs_corr_pandas.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils


# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
# 利用pandas库计算相关系数
# pearson相关系数
pearson_corr = x.corr(method='pearson')
print("pandas库计算 pearson相关系数: \n", pearson_corr)
# spearman相关系数
spearman_corr = x.corr(method='spearman')  
print("pandas库计算 spearman相关系数: \n", spearman_corr)
# kendall相关系数
kendall_corr = x.corr(method='kendall')  
print("pandas库计算 kendall相关系数: \n", kendall_corr)



#==============================================================================
# File: ch2_15_fs_corr_scipy.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from scipy.stats import pearsonr


# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
x1, x2 = x.loc[:, 'age.in.years'], x.loc[:, 'credit.history',]
r, p_value = pearsonr(x1, x2)
print("scipy库计算 特征'age.in.years'和'credit.history'的pearson相关系数 \n", 
    "pearson相关系数: %s, \n" % r, "p_value: %s" % p_value)



#==============================================================================
# File: ch2_16_fs_vif.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
vif = [variance_inflation_factor(x.values, ix) for ix in range(x.shape[1])]
print("各特征的vif值计算结果: \n", dict(zip(x.columns, vif)))

# 筛选阈值小于10的特征
selected_cols = x.iloc[:, [f < 10 for f in vif]].columns.tolist()
print("设置vif阈值为10, 筛选得到%s个特征: \n" % len(selected_cols), selected_cols)



#==============================================================================
# File: ch2_17_fs_iv.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
import toad
sys.path.append("./")
sys.path.append("../")

from utils import data_utils

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
# 利用toad库quality()方法计算IV
var_iv = toad.quality(all_x_y,
                      target='creditability',
                      method='quantile',
                      n_bins=6,
                      iv_only=True)

selected_cols = var_iv[var_iv.iv > 0.1].index.tolist()
print("各特征的iv值计算结果: \n", var_iv)
print("设置iv阈值为0.1, 筛选得到%s个特征: \n" % len(selected_cols), selected_cols)



#==============================================================================
# File: ch2_18_fs_chi.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
y = all_x_y.pop(data_utils.label)
# 选择K个最好的特征，返回选择特征后的数据
fs_chi = SelectKBest(chi2, k=5)
fs_chi.fit(all_x_y, y)
x_new = fs_chi.transform(all_x_y)

selected_cols = all_x_y.columns[fs_chi.get_support()].tolist()
print("卡方检验筛选得到%s个特征: \n" % len(selected_cols), selected_cols)



#==============================================================================
# File: ch2_19_fs_stepwise.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
import toad
sys.path.append("./")
sys.path.append("../")

from utils import data_utils

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
final_data = toad.selection.stepwise(all_x_y,
                                     target=data_utils.label,
                                     estimator='lr',
                                     direction='both',
                                     criterion='aic',
                                     return_drop=False)
selected_cols = final_data.columns
print("通过stepwise筛选得到%s个特征: \n" % len(selected_cols), selected_cols)



#==============================================================================
# File: ch2_20_fs_rfe.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
import toad
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
y = all_x_y.pop(data_utils.label)
x = all_x_y
# 递归特征消除法，返回特征选择后的数据
# 参数estimator为基模型
# 参数n_features_to_select为选择的特征个数
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
x_new = rfe.fit_transform(x, y)

selected_cols = x.columns[rfe.get_support()].tolist()
print("通过递归特征消除法筛选得到%s个特征: \n" % len(selected_cols), selected_cols)



#==============================================================================
# File: ch2_21_fs_l1_norm.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
import toad
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
y = all_x_y.pop(data_utils.label)
x = all_x_y
# 带L1惩罚项的逻辑回归作为基模型的特征选择
LR = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
sf = SelectFromModel(LR)
x_new = sf.fit_transform(x, y)

selected_cols = x.columns[sf.get_support()].tolist()
print("基于L1范数筛选得到%s个特征: \n" % len(selected_cols), selected_cols)



#==============================================================================
# File: ch2_22_fs_select_from_model.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
import toad
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
y = all_x_y.pop(data_utils.label)
x = all_x_y
# GBDT作为基模型的特征选择
sf = SelectFromModel(GradientBoostingClassifier())
x_new = sf.fit_transform(x, y)

selected_cols = x.columns[sf.get_support()].tolist()
print("基于树模型筛选得到%s个特征: \n" % len(selected_cols), selected_cols)



#==============================================================================
# File: ch2_23_fs_psi.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
import toad
sys.path.append("./")
sys.path.append("../")

from utils import data_utils

# 加载数据
all_x_y = data_utils.get_all_x_y()
# 定义分箱方法
Combiner = toad.transform.Combiner()
Combiner.fit(all_x_y,
             y=data_utils.label,
             n_bins=6,
             method='quantile',
             empty_separate=True)
# 计算psi
var_psi = toad.metrics.PSI(all_x_y.iloc[:500, :],
                           all_x_y.iloc[500:, :],
                           combiner=Combiner)
var_psi_df = var_psi.to_frame(name='psi')

selected_cols = var_psi[var_psi_df.psi < 0.1].index.tolist()
print("各特征的psi值计算结果: \n", var_psi_df)
print("设置psi阈值为0.1, 筛选得到%s个特征: \n" % len(selected_cols), selected_cols)



#==============================================================================
# File: ch2_24_fs_badrate_by_month.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import toad
import pandas as pd
from utils import data_utils


# 导入添加month列的数据
model_data = data_utils.get_data()

x = model_data[data_utils.x_cols]
y = model_data[data_utils.label]

# 分箱
Combiner = toad.transform.Combiner()
x_cat = Combiner.fit_transform(x, y, n_bins=6, method='quantile', empty_separate=True)

# 合并标签和month
x_cat_with_month = x_cat.merge(model_data[['month', 'creditability']], left_index=True, right_index=True)

# 单个特征对比逾期率
feature_col = 'age.in.years'
x_cat_one = x_cat_with_month[[feature_col, 'month', 'creditability']]
feature_var = x_cat_one.pivot_table(index=feature_col,
                                columns='month',
                                values='creditability',
                                aggfunc=['mean'])
print("特征'age.in.years'的按月分箱逾期率统计结果: \n", feature_var)


# 计算特征按月逾期率波动值
def variation_by_month(df, time_col, columns, label_col):
    variation_dict = {}
    for col in columns:
        feature_v = df.pivot_table(
            index=col, columns=time_col, values=label_col, aggfunc=['mean'])
        variation_dict[col] = feature_v.rank().std(axis=1).mean()

    return pd.DataFrame([variation_dict], index=['variation']).T


var_badrate = variation_by_month(x_cat_with_month, 'month', data_utils.x_cols, 'creditability')
print("各特征按月逾期率的标准差: \n", var_badrate)

selected_cols = var_badrate[var_badrate['variation'] < 0.8].index.tolist()
print("设置标准差阈值为0.8, 筛选得到%s个特征: \n" % len(selected_cols), selected_cols)



#==============================================================================
# File: ch2_25_feature_extraction_pca.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
import toad
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.decomposition import PCA


# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
pca = PCA(n_components=0.9)
x_new = pca.fit_transform(x)
x_new_df = pd.DataFrame(x_new)
print("利用sklearn进行PCA特征提取, 保留90%信息后结果: \n", x_new_df)



#==============================================================================
# File: ch2_26_feature_extraction_lda.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
y = all_x_y[data_utils.label]
lda = LinearDiscriminantAnalysis(n_components=1)
x_new = lda.fit_transform(x, y)
x_new_df = pd.DataFrame(x_new)
print("利用sklearn进行LDA特征提取结果: \n", x_new_df)



#==============================================================================
# File: ch2_27_feature_extraction_lle.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.manifold import LocallyLinearEmbedding

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
lle = LocallyLinearEmbedding(n_neighbors=5, n_components=10)
x_new = lle.fit_transform(x)
x_new_df = pd.DataFrame(x_new)
print("利用sklearn进行LLE特征提取结果: \n", x_new_df)



#==============================================================================
# File: ch2_28_feature_extraction_mds.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.manifold import MDS


# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
mds = MDS(n_components=10)
x_new = mds.fit_transform(x)
x_new_df = pd.DataFrame(x_new)
print("利用sklearn进行MDS特征提取结果: \n", x_new_df)



#==============================================================================
# File: ch2_29_p_to_score.py
#==============================================================================

# -*- coding: utf-8 -*- 
import sys
import numpy as np
import pandas as pd
sys.path.append("./")
sys.path.append("../")

def p_to_score(p, pdo, base, odds):
    """ 
    逾期概率转换分数 
    :param p: 逾期概率 
    :param pdo: points double odds. default = 60 
    :param base: base points. default = 600 
    :param odds: odds. default = 1.0/15.0 
    :returns: 模型分数 
    """
    B = pdo / np.log(2)
    A = base + B * np.log(odds)
    score = A - B * np.log(p / (1 - p))
    return round(score, 0)

pros = pd.Series(np.random.rand(100))
pros_score = p_to_score(pros, pdo=60.0, base=600, odds=1.0 / 15.0)
print("随机产生100个概率并转化为score结果: \n", dict(zip(pros, pros_score)))



#==============================================================================
# File: ch2_30_validation_curve.py
#==============================================================================

# -*- coding: utf-8 -*- 
# 绘制验证曲线

import sys
import numpy as np
import pandas as pd
sys.path.append("./")
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve

X, y = load_digits(return_X_y=True)

param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(
    SVC(), X, y, param_name="gamma", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel(r"$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()



#==============================================================================
# File: ch2_31_model_deployment_pickle.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
import numpy as np
import pandas as pd
sys.path.append("./")
sys.path.append("../")


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


def load_model_from_pkl(path):
    """
    从路径path加载模型
    :param path: 保存的目标路径
    """
    import pickle
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model




#==============================================================================
# File: ch2_32_model_deployment_pmml.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

# PMML方式保存和读取模型
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn_pandas import DataFrameMapper
from pypmml import Model
from xgboost.sklearn import XGBClassifier
from utils import data_utils
from chapter2.ch2_31_model_deployment_pickle import load_model_from_pkl


# 以xgb模型为例，方式1：
# sklearn接口的xgboost，可使用sklearn2pmml生成pmml文件
def save_model_as_pmml(x, y, save_file_path):
    """
    保存模型到路径save_file_path
    :param x: 训练数据特征
    :param y: 训练数据标签
    :param save_file_path: 保存的目标路径
    """
    # 设置pmml的pipeline
    xgb = XGBClassifier(random_state=88)
    mapper = DataFrameMapper([([i], None) for i in x.columns])
    pipeline = PMMLPipeline([('mapper', mapper), ('classifier', xgb)])
    # 模型训练
    pipeline.fit(x, y)
    # 模型结果保存
    sklearn2pmml(pipeline, pmml=save_file_path, with_repr=True)


# PMML格式读取
def load_model_from_pmml(load_file_path):
    """
    从路径load_file_path加载模型
    :param load_file_path: pmml文件路径
    """
    model = Model.fromFile(load_file_path)
    return model


train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)
save_model_as_pmml(train_x, train_y, 'data/model/xgb_model.pmml')
model = load_model_from_pmml('data/model/xgb_model.pmml')
pre = model.predict(test_x)
print(pre.head())

# 方式2：
# 原生xgboost.core库生成的XGBoost模型，不能使用sklearn2pmml生成pmml文件，只能通过jpmml-xgboost包，将已有的.bin或.model
# 格式模型文件转为pmml文件

# step1.获取到xgb模型文件
xgb_model = load_model_from_pkl("data/model/xgb_model.pkl")


# step2.生成fmap文件
def create_feature_map(file_name, features):
    outfile = open(file_name, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))


create_feature_map('data/model/xgb_model.fmap', xgb_model.feature_names)

# step3.jpmml-xgboost的环境配置及pmml转换：
# step3.1. 下载jpmml-xgboost
# step3.2. 命令行切换到jpmml-xgboost的项目文件夹，输入代码编译
# mvn clean install
# 该步执行完后，jpmml-xgboost的项目文件夹下会多出一个target文件夹，里面包含生成好的jar包
# step3.3. jar包转换为pmml文件
# java -jar jpmml-xgboost_path/target/jpmml-xgboost-executable-1.5-SNAPSHOT.jar  --X-nan-as-missing False
# --model-input data/model/xgb.model --fmap-input data/model/xgb.fmap --target-name target
# --pmml-output data/model/xgb_pmml.pmml



#==============================================================================
# File: ch2_33_lr.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
import numpy as np
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders.woe import WOEEncoder

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)

# woe特征处理
encoder = WOEEncoder(cols=train_x.columns)
train_x = encoder.fit_transform(train_x, train_y)
test_x = encoder.transform(test_x)

# 利用梯度下降法训练逻辑回归模型
lr = SGDClassifier(loss="log",
                   penalty="l2",
                   learning_rate='optimal',
                   max_iter=100,
                   tol=0.001,
                   epsilon=0.1,
                   random_state=1)
clf = make_pipeline(StandardScaler(), lr)
clf.fit(train_x, train_y)
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("梯度下降法训练逻辑回归模型 AUC: ", auc_score)

# 利用牛顿法训练逻辑回归模型
lr = LogisticRegression(penalty="l2",
                        solver='lbfgs',
                        max_iter=100,
                        tol=0.001,
                        random_state=1)
clf = make_pipeline(StandardScaler(), lr)
clf.fit(train_x, train_y)
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("牛顿法训练逻辑回归模型 AUC: ", auc_score)



#==============================================================================
# File: ch2_34_svm.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
import numpy as np
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils import data_utils
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from category_encoders.woe import WOEEncoder

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)

# woe特征处理
encoder = WOEEncoder(cols=train_x.columns)
train_x = encoder.fit_transform(train_x, train_y)
test_x = encoder.transform(test_x)

# 线性SVM, Linear Support Vector Classification
line_svm = LinearSVC(penalty='l2',
                     loss='hinge',
                     C=0.2,
                     tol=0.001)
clf = make_pipeline(StandardScaler(), line_svm)
clf.fit(train_x, train_y)
acc_score = accuracy_score(test_y, clf.predict(test_x))
print("线性SVM模型 ACC: ", acc_score)


# 支持核函数的SVM, C-Support Vector Classification
svm = SVC(C=0.2,
          kernel='rbf',
          tol=0.001,
          probability=True)
clf = make_pipeline(StandardScaler(), svm)
clf.fit(train_x, train_y)
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("支持核函数SVM模型 AUC: ", auc_score)



#==============================================================================
# File: ch2_35_decision_tree.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")


from sklearn.tree import DecisionTreeClassifier
from utils import data_utils
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)
# 导入数值型样例数据
clf = DecisionTreeClassifier(criterion='gini',
                             max_depth=8,
                             min_samples_leaf=15,
                             random_state=88)
clf.fit(train_x, train_y)
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("决策树模型 AUC: ", auc_score)



#==============================================================================
# File: ch2_36_randomforest.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from sklearn.ensemble import RandomForestClassifier
from utils import data_utils
from sklearn.metrics import roc_auc_score

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)
clf = RandomForestClassifier(n_estimators=200,
                             criterion='gini',
                             max_depth=6,
                             min_samples_leaf=15,
                             bootstrap=True,
                             oob_score=True,
                             random_state=88)
clf.fit(train_x, train_y)
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("随机森林模型 AUC: ", auc_score)



#==============================================================================
# File: ch2_37_gbdt.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from sklearn.ensemble import GradientBoostingClassifier
from utils import data_utils
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)
clf = GradientBoostingClassifier(n_estimators=100,
                                 learning_rate=0.1,
                                 subsample=0.9,
                                 max_depth=4,
                                 min_samples_leaf=20,
                                 random_state=88)
clf.fit(train_x, train_y)
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("GBDT模型 AUC: ", auc_score)



#==============================================================================
# File: ch2_38_xgboost.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import bayes_opt as bo
import sklearn.model_selection as sk_ms
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from utils import data_utils
import shap
from chapter2.ch2_31_model_deployment_pickle import save_model_as_pkl


# 确定最优树的颗数
def xgb_cv(param, x, y, num_boost_round=10000):
    dtrain = xgb.DMatrix(x, label=y)
    cv_res = xgb.cv(param, dtrain, num_boost_round=num_boost_round, early_stopping_rounds=30)
    num_boost_round = cv_res.shape[0]
    return num_boost_round

def train_xgb(params, x_train, y_train, x_test=None, y_test=None, num_boost_round=10000, early_stopping_rounds=30, verbose_eval=50):
    """
    训练xgb模型
    """
    dtrain = xgb.DMatrix(x_train, label=y_train)
    if x_test is None:
        num_boost_round = xgb_cv(params, x_train, y_train)
        early_stopping_rounds = None
        eval_sets = ()
    else:
        dtest = xgb.DMatrix(x_test, label=y_test)
        eval_sets = [(dtest, 'test')]
    model = xgb.train(params, dtrain, num_boost_round, evals=eval_sets, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)
    return model


def xgboost_grid_search(params_space, x_train, y_train, x_test=None, y_test=None, num_boost_round=10000):
    """
    网格调参, 确定其他参数
    """
    # 设置训练参数
    if x_test is None:
        x_train, x_test, y_train, y_test = sk_ms.train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    score_list = []
    test_params = list(ParameterGrid(params_space))
    for params_try in test_params:
        params_try['eval_metric'] = "auc"
        params_try['random_state'] = 1
        clf_obj = train_xgb(params_try, x_train, y_train, x_test, y_test, num_boost_round=num_boost_round,
                            early_stopping_rounds=30, verbose_eval=0)
        score_list.append(roc_auc_score(y_test, clf_obj.predict(xgb.DMatrix(x_test))))
    result = pd.DataFrame(dict(zip(score_list, test_params))).T
    print(result)
    # 取测试集上效果最好的参数组合
    params = test_params[np.array(score_list).argmax()]
    return params


def xgboost_bayesian_optimization(params_space, x_train, y_train, x_test=None, y_test=None, num_boost_round=10000, nfold=5, init_points=2, n_iter=5, verbose_eval=0, early_stopping_rounds=30):
    """
    贝叶斯调参, 确定其他参数
    """
    # 设置需要调节的参数及效果评价指标
    def xgboost_cv_for_bo(eta, gamma, max_depth, min_child_weight,
                          subsample, colsample_bytree):
        params = {
            'eval_metric': 'auc',
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eta': eta,
            'gamma': gamma,
            'max_depth': int(max_depth),
            'min_child_weight': int(min_child_weight),
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'seed': 1
        }
        if x_test is None:
            dtrain = xgb.DMatrix(x_train, label=y_train)
            xgb_cross = xgb.cv(params,
                               dtrain,
                               nfold=nfold,
                               metrics='auc',
                               early_stopping_rounds=early_stopping_rounds,
                               num_boost_round=num_boost_round)
            test_auc = xgb_cross['test-auc-mean'].iloc[-1]
        else:
            clf_obj = train_xgb(params, x_train, y_train, x_test, y_test, num_boost_round=num_boost_round,
                                early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)
            test_auc = roc_auc_score(y_test, clf_obj.predict(xgb.DMatrix(x_test)))
        return test_auc

    # 指定需要调节参数的取值范围
    xgb_bo_obj = bo.BayesianOptimization(xgboost_cv_for_bo, params_space, random_state=1)
    xgb_bo_obj.maximize(init_points=init_points, n_iter=n_iter)
    best_params = xgb_bo_obj.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    best_params['eval_metric'] = 'auc'
    best_params['booster'] = 'gbtree'
    best_params['objective'] = 'binary:logistic'
    best_params['seed'] = 1
    return best_params


# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)

# 经验参数
exp_params = {
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eta': 0.1,
    'gamma': 0.01,
    'max_depth': 4,
    'min_child_weight': 1,
    'subsample': 1,
    'colsample_bytree': 1,
    'seed': 1
}
final_xgb_model = train_xgb(exp_params, train_x, train_y, test_x, test_y)
auc_score = roc_auc_score(test_y, final_xgb_model.predict(xgb.DMatrix(test_x)))
print("经验参数模型AUC: ", auc_score)

# 随机搜索调参
choose_tuner = 'bayesian'  # bayesian grid_search
if choose_tuner == 'grid_search':
    params_test = {
        'learning_rate': [0.1, 0.15],
        'gamma': [0.01, 0],
        'max_depth': [4, 3],
        'min_child_weight': [1, 2],
        'subsample': [0.95, 1],
        'colsample_bytree': [1]
    }
    optimal_params = xgboost_grid_search(params_test, train_x, train_y, test_x, test_y)
elif choose_tuner == 'bayesian':
    # 贝叶斯调参
    params_test = {'eta': (0.05, 0.2),
                   'gamma': (0.005, 0.05),
                   'max_depth': (3, 5),
                   'min_child_weight': (0, 3),
                   'subsample': (0.9, 1.0),
                   'colsample_bytree': (0.9, 1.0)}
    optimal_params = xgboost_bayesian_optimization(params_test, train_x, train_y, test_x, test_y, init_points=5, n_iter=8)

print("随机搜索调参最优参数: ", optimal_params)

final_xgb_model = train_xgb(optimal_params, train_x, train_y, test_x, test_y)
auc_score = roc_auc_score(test_y, final_xgb_model.predict(xgb.DMatrix(test_x)))
print("随机搜索调参模型AUC: ", auc_score)

# 保存模型
save_model_as_pkl(final_xgb_model, "./data/xgb_model.pkl")

# SHAP计算
explainer = shap.TreeExplainer(final_xgb_model)
shap_values = explainer.shap_values(train_x)
# SHAP可视化
shap.summary_plot(shap_values, train_x, max_display=5)



#==============================================================================
# File: ch2_39_lightgbm.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import lightgbm as lgb
from utils import data_utils
from sklearn.metrics import roc_auc_score

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)
clf = lgb.LGBMClassifier(objective='binary',
                         boosting_type='gbdt',
                         max_depth=3,
                         n_estimators=1000,
                         subsample=1,
                         colsample_bytree=1)
lgb_model = clf.fit(train_x, train_y, eval_set=[(test_x, test_y)], eval_metric='auc', early_stopping_rounds=30)
auc_score = roc_auc_score(test_y, lgb_model.predict_proba(test_x)[:, 1])
print("LightGBM模型 AUC: ", auc_score)



#==============================================================================
# File: ch2_40_DNN_credit_data.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

# https://keras.io

from utils import data_utils
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, models, callbacks

# 加载数据集
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(transform_method='standard')

# 设置随机数种子
tf.random.set_seed(1)
# 设置早停
callback = callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min')
# 构建DNN模型结构
model = models.Sequential()
model.add(layers.Flatten(input_shape=(train_x.shape[1], 1)))
model.add(layers.Dense(32, activation=tf.nn.relu))
model.add(layers.Dropout(0.3, seed=1))
model.add(layers.Dense(16, activation=tf.nn.relu))
model.add(layers.Dense(1, activation=tf.nn.sigmoid))
# 显示模型的结构
model.summary()
# 设置模型训练参数
model.compile(optimizer='SGD',
              metrics=[tf.metrics.AUC()],
              loss='binary_crossentropy')
# 模型训练
model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=16, epochs=240, callbacks=[callback], verbose=2)

# 效果评估
auc_score = roc_auc_score(train_y, model.predict(train_x))
print("训练集AUC", auc_score)
auc_score = roc_auc_score(test_y, model.predict(test_x))
print("测试集AUC", auc_score)



#==============================================================================
# File: ch2_41_CNN_credit_data.py
#==============================================================================

# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from utils import data_utils
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, models, callbacks

# 加载数据集
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(transform_method='standard')

# 数据预处理
train_x = train_x.to_numpy().reshape((train_x.shape[0], train_x.shape[1], 1))
test_x = test_x.to_numpy().reshape((test_x.shape[0], test_x.shape[1], 1))
train_y = train_y.values.reshape((train_y.shape[0], 1))
test_y = test_y.values.reshape((test_y.shape[0], 1))

# 设置随机数种子，保证每次运行结果一致
tf.random.set_seed(1)
callback = callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min')

# 构建CNN模型结构
model = models.Sequential()
model.add(layers.Conv1D(filters=16, kernel_size=4, activation='relu', input_shape=(train_x.shape[1], 1)))
model.add(layers.Conv1D(filters=8, kernel_size=1, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.3, seed=1))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# 显示模型的结构
model.summary()
# 设置模型训练参数
model.compile(optimizer='SGD',
              metrics=[tf.metrics.AUC()],
              loss='binary_crossentropy')
# 模型训练
model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=16, epochs=240, callbacks=[callback], verbose=2)

# 测试集效果评估
auc_score = roc_auc_score(train_y, model.predict(train_x))
print("训练集AUC", auc_score)
auc_score = roc_auc_score(test_y, model.predict(test_x))
print("测试集AUC", auc_score)



#==============================================================================
# File: ch3_00_order_data_preprocess.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import numpy as np
import pandas as pd
from utils.data_utils import stamp_to_date
from utils.data_utils import date_to_week


def data_preprocess(data, time_col, back_time, dtypes_dict):
    """
    数据预处理函数
    :param data: 待处理的数据
    :param time_col: 回溯依据的时间列名称
    :param back_time: 特征计算时间，datetime.datetime时间格式
    :param dtypes_dict: 指定列字段类型的字典，如{'col1':int}
    :return: 清洗完成的数据
    """
    # 删除time_col为空的行
    data = data[~data[time_col].isin(['nan', np.nan, 'NAN', 'null', 'NULL', 'Null'])]
    # 将时间列的时间戳转为日期格式
    data[time_col] = data[time_col].apply(stamp_to_date)
    # 过滤订单创建时间在back_time之后的数据，避免特征穿越
    data = data[data[time_col] <= back_time]
    # 删除整条缺失的数据
    data.dropna(how='all', inplace=True)
    # 空字符串替换为np.nan
    data.replace('', np.nan, inplace=True)
    # 单个字段缺失填充为0
    data.fillna(0, inplace=True)
    # 去重
    data.drop_duplicates(keep='first', inplace=True)
    # 字段格式转换
    data = data.astype(dtypes_dict)
    # 补充字段
    data['create_time_week'] = data[time_col].apply(date_to_week)
    data['is_weekend'] = data['create_time_week'].apply(lambda x: 1 if x > 5 else 0)

    return data


if __name__ == '__main__':
    # 原始数据读入
    orders = pd.read_excel('data/order_data.xlsx')
    # 取一个用户的历史订单数据
    raw_data = pd.DataFrame(eval(orders['data'][1]))
    # 数据预处理
    data_processed = data_preprocess(raw_data, time_col='create_time',
                                     back_time='2020-12-14',
                                     dtypes_dict={'has_overdue': int,
                                                  'application_term': float,
                                                  'application_amount': float})
    print(data_processed.shape)



#==============================================================================
# File: ch3_01_order_fea_gen_manual.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import pandas as pd
import datetime as dt
from dateutil.parser import parse
from chapter3.ch3_00_order_data_preprocess import data_preprocess


def calculate_age(born_day, back_time=None):
    """
    根据出生日期解析年龄
    :param born_day: 出生日期
    :param back_time: 回溯时间，默认当前日期
    :return: 年龄
    """
    if back_time is None:
        today = dt.date.today()
    else:
        today = back_time
    if isinstance(born_day, str):
        born_day = parse(born_day)
    if isinstance(today, str):
        today = parse(today)
    return today.year - born_day.year - ((today.month, today.day) < (born_day.month, born_day.day))


def gen_order_feature_manual(data, time_col, back_time, dtypes_dict, fea_prefix='f'):
    """
    根据业务逻辑生成特征
    :param data: 业务订单原始数据
    :param time_col: 回溯依据的时间列名称
    :param back_time: 回溯时间点
    :param dtypes_dict: 指定列字段类型的字典，如{'col1':int}
    :param fea_prefix: 特征前缀
    :return: features，根据业务逻辑生成的特征
    """
    # 数据预处理函数，见文件ch3_01_order_data_preprocess.py
    data_processed = data_preprocess(data, time_col, back_time, dtypes_dict=dtypes_dict)
    features = {}
    # 从生日解析年龄
    features['%s_age' % fea_prefix] = calculate_age(data_processed.get('birthday')[0], back_time)
    # 用户历史订单数
    features['%s_history_order_num' % fea_prefix] = data_processed.shape[0]
    # 用户历史逾期次数
    features['%s_overdue_num' % fea_prefix] = data_processed['has_overdue'].sum()
    # 用户历史最大逾期天数
    features['%s_max_overdue_days' % fea_prefix] = data_processed['overdue_days'].max()
    # 用户历史平均逾期天数
    features['%s_mean_overdue_days' % fea_prefix] = data_processed['overdue_days'].mean()

    return features


if __name__ == '__main__':
    # 原始数据读入
    orders = pd.read_excel('data/order_data.xlsx')
    # 取一个用户的历史订单数据
    raw_data = pd.DataFrame(eval(orders['data'][1]))
    back_time_value = orders['back_time'][1]
    cols_dtypes_dict = {'has_overdue': int, 'application_term': float, 'application_amount': float}

    # 根据业务逻辑生成用户历史订单特征
    features_manual = gen_order_feature_manual(raw_data, 'create_time', back_time_value, cols_dtypes_dict)
    print(features_manual)

    # 批量生成特征
    feature_dict = {}
    for i, row in orders.iterrows():
        feature_dict[i] = gen_order_feature_manual(pd.DataFrame(eval(row['data'])), 'create_time', row['back_time'],
                                                   cols_dtypes_dict, fea_prefix='orderv1')
    feature_df = pd.DataFrame(feature_dict).T
    # feature_df.to_excel('data/features_manual.xlsx', index=True)



#==============================================================================
# File: ch3_02_order_fea_gen_rfm_auto.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys

sys.path.append("./")
sys.path.append("../")

# 根据业务逻辑自动生成用户历史订单特征
import pandas as pd
import numpy as np
from dateutil.parser import parse
from utils.data_utils import stamp_to_date
from chapter3.ch3_00_order_data_preprocess import data_preprocess

func_trans = {'sum': np.sum,
              'mean': np.mean,
              'cnt': np.size,
              'max': np.max,
              'min': np.min,
              'std': np.std,
              }


def apply_func(f, *args):
    return f(*args)


def rfm_cut(data, time_col, back_time, type_dict, comp_dict, time_arr, fea_prefix='f'):
    """
    基于RFM思想切分数据，生成特征
    :param DataFrame data: 待切分的数据，时间列为create_time(timestamp)，距今天数列为gap_days
    :param str time_col: 回溯依据的时间列名称
    :param datetime.datetime back_time: 回溯时间点，datetime.datetime时间格式
    :param dict type_dict: 类别变量，以及其对应的取值类别，用于划分数据，类别列名必须在data中
    :param dict comp_dict: 指定计算字段以及对该字段采用的计算方法, 计算变量名必须在data中
    :param list time_arr: 切分时间列表(近N天)
    :param fea_prefix: 特征前缀
    :return dict: 特征
    """
    data[time_col] = data[time_col].apply(stamp_to_date)
    # 业务时间距back_time天数
    data['gap_days'] = data[time_col].apply(lambda x: (back_time - x).days)

    res_feas = {}
    for col_time in time_arr:
        for col_comp in comp_dict.keys():
            for type_k, type_v in type_dict.items():
                # 按类别和时间维度切分,筛选数据
                for item in type_v:
                    data_cut = data[(data['gap_days'] < col_time) & (data[type_k] == item)]
                    for func_k in comp_dict[col_comp]:
                        func_v = func_trans.get(func_k, np.size)
                        # 对筛选出的数据, 在各统计指标上做聚合操作生成特征
                        fea_name = '%s_%s_%s_%s_%s' % (
                            fea_prefix, col_time, '%s_%s' % (type_k, item), col_comp, func_k)
                        if data_cut.empty:
                            res_feas[fea_name] = np.nan
                        else:
                            res_feas[fea_name] = apply_func(func_v, data_cut[col_comp])
    return res_feas


def gen_order_feature_auto(raw_data, time_col, back_time, dtypes_dict, type_dict, comp_dict, time_arr,
                           fea_prefix='f'):
    """
    基于RFM切分，自动生成订单特征
    :param pd.DataFrame raw_data: 原始数据
    :param str time_col: 回溯依据的时间列名称
    :param str back_time: 回溯时间点，字符串格式
    :param dict dtypes_dict: 指定列字段类型的字典，如{'col1':int}
    :param list time_arr: 切分时间列表(近N天)
    :param dict type_dict: 类别变量，以及其对应的取值类别，用于划分数据，类别列名必须在data中
    :param dict comp_dict: 指定计算字段以及对该字段采用的计算方法,计算变量名必须在data中
    :param fea_prefix: 特征前缀
    :return: res_feas 最终生成的特征
    """
    if raw_data.empty:
        return {}
    back_time = parse(str(back_time))

    order_df = data_preprocess(raw_data, time_col=time_col, back_time=back_time, dtypes_dict=dtypes_dict)
    if order_df.empty:
        return {}

    # 特征衍生：使用rfm切分
    res_feas = rfm_cut(order_df, time_col, back_time, type_dict, comp_dict, time_arr, fea_prefix)
    return res_feas


if __name__ == '__main__':
    # 原始数据读入
    orders = pd.read_excel('data/order_data.xlsx')
    # 取一个用户的历史订单数据
    raw_orders = pd.DataFrame(eval(orders['data'][1]))

    # 设置自动特征的参数
    # 类别字段及其取值
    type_dict_param = {
        'has_overdue': [0, 1],
        'is_weekend': [0, 1]
    }
    # 计算字段及其计算函数
    comp_dict_param = {
        'order_no': ['cnt'],
        'application_amount': ['sum', 'mean', 'max', 'min']
    }
    time_cut = [30, 90, 180, 365]

    cols_dtypes_dict = {'has_overdue': int, 'application_term': float, 'application_amount': float}

    # 根据业务逻辑生成用户历史订单特征
    features_auto = gen_order_feature_auto(raw_orders, 'create_time', '2020-12-14', cols_dtypes_dict,
                                           type_dict_param, comp_dict_param, time_cut)
    print("特征维度: ", len(features_auto.keys()))
    print(features_auto)

    # 批量生成特征
    feature_dict = {}
    for i, row in orders.iterrows():
        feature_dict[i] = gen_order_feature_auto(pd.DataFrame(eval(row['data'])), 'create_time', row['back_time'],
                                                 cols_dtypes_dict, type_dict_param, comp_dict_param, time_cut,
                                                 'order_auto')
    feature_df_auto = pd.DataFrame(feature_dict).T
    # feature_df_auto.to_excel('data/features_auto.xlsx', index=True)



#==============================================================================
# File: ch3_03_tsfresh_orders.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 时间序列特征挖掘
import pandas as pd
from tsfresh.feature_extraction import extract_features

if __name__ == '__main__':
    # 读取数据
    orders = pd.read_excel('data/order_data.xlsx')
    orders_new = []
    for i in range(len(orders)):
        sub_data = pd.DataFrame.from_records(eval(orders['data'][i]))
        sub_data['uid'] = orders['uid'][i]
        orders_new.append(sub_data)
    orders_new_df = pd.concat(orders_new)
    # 数据格式
    orders_new_df['application_amount'] = orders_new_df['application_amount'].astype(float)
    orders_new_df['has_overdue'] = orders_new_df['has_overdue'].astype(float)

    # 调用extract_features生成时间序列特征:order_feas
    order_feas = extract_features(orders_new_df[['uid', 'create_time', 'application_amount', 'has_overdue']], column_id="uid", column_sort="create_time")
    print("时间序列挖掘特征数: \n", order_feas.shape[1])
    print("时间序列特征挖掘结果: \n", order_feas.head())



#==============================================================================
# File: ch3_04_feature_evaluation.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import variation
sys.path.append("./")
sys.path.append("../")

def cover_ratio(x):
    """
    计算特征覆盖度
    :param x: 特征向量
    :return: cover_ratio, 特征覆盖度
    """
    len_x = len(x)
    len_nan = sum(pd.isnull(x))
    ratio = 1 - len_nan / float(len_x)
    return ratio


def get_datestamps(begin_date, end_date):
    """
    返回[begin_date,end_date]之间日期的时间戳
    :param begin_date: 开始时间
    :param end_date: 结束时间
    :return: [begin_date,end_date]日期的时间戳
    """
    date_arr = [int(time.mktime(x.timetuple())) for x in list(pd.date_range(start=begin_date, end=end_date))]
    return date_arr


if __name__ == '__main__':
    # 模拟生成几个特征
    fea_1 = [-1, -1, -1, 0, 1, 1, 1]  # 特征均值为0
    fea_2 = [1, 1, 1, 1, 1, 1, 1]  # 所有特征均为唯一指
    fea_3 = [1, 2, 3, 4, 5, 6, 7]  # 与时间正相关
    fea_4 = [7, 6, 5, 4, 3, 2, 1]  # 与时间负相关
    fea_5 = [1, 2, 1, 2, np.nan, 2, np.nan]  # 与时间无线性关系

    x_all = pd.DataFrame([fea_1, fea_2, fea_3, fea_4, fea_5]).T
    x_all.columns = ['fea_1', 'fea_2', 'fea_3', 'fea_4', 'fea_5']

    # 特征覆盖度
    fea_cover = x_all.apply(cover_ratio).to_frame('cover_ratio')
    print("特征覆盖度: ", fea_cover)

    # 特征离散度
    fea_variation = variation(fea_2)
    print("特征离散度: ", fea_variation)

    # 计算时间相关性
    x_all['tm_col'] = get_datestamps('2020-10-01', '2020-10-07')

    # 计算三个特征与时间的Peason系数
    fea_time_corr = x_all.loc[:, ['fea_3', 'fea_4', 'fea_5', 'tm_col']].corr().loc[:, ['tm_col']]

    print("构造的特征为: \n", x_all)
    print("特征与时间的Peason系数计算结果: \n", fea_time_corr)



#==============================================================================
# File: ch3_05_gbdt_construct_feature.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 使用GBDT算法做特征衍生
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier


def gbdt_fea_gen(train_data, label, n_estimators=100):
    # 训练GBDT模型
    gbc_model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=1)
    gbc_model.fit(train_data, label)

    # 得到样本元素落在叶节点中的位置
    train_leaf_fea = gbc_model.apply(train_data).reshape(-1, n_estimators)

    # 借用编码将位置信息转化为0，1
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(train_leaf_fea)
    return gbc_model, one_hot_encoder


def gbdt_fea_appy(data, model, encoder):
    # 获得GBDT特征
    new_feature_train = encoder.transform(model.apply(data).reshape(-1, model.n_estimators)).toarray()
    # new_feas为生成的新特征
    new_fea = pd.DataFrame(new_feature_train)
    new_fea.index = data.index
    new_fea.columns = ['fea_%s' % i for i in range(1, new_fea.shape[1] + 1)]
    return new_fea


if __name__ == '__main__':
    # 读取原始特征数据
    all_x_y = pd.read_excel('data/order_feas.xlsx')
    all_x_y.set_index('order_no', inplace=True)
    # 生成训练数据
    x_train = all_x_y.drop(columns='label')
    x_train.fillna(0, inplace=True)
    y = all_x_y['label']
    # 获取特征
    gbr, encode = gbdt_fea_gen(x_train, y, n_estimators=100)
    new_features = gbdt_fea_appy(x_train, gbr, encode)
    print("使用GBDT算法衍生特征结果: \n", new_features.head())



#==============================================================================
# File: ch3_06_cluster_alg.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 使用聚类算法衍生特征
import pandas as pd
from sklearn.cluster import KMeans


def cluster_fea_gen(data, selected_cols, n_clusters):
    """
    使用聚类算法生成特征
    :param data: 用作输入的x,y
    :param selected_cols: 选取用来做聚类的特征列
    :param n_clusters: 聚类类别数
    :return: 聚类算法生成的特征
    """
    x_cluster_feas = data.loc[:, selected_cols]
    # 拟合聚类模型
    clf = KMeans(n_clusters=n_clusters, random_state=1)
    clf.fit(x_cluster_feas)
    return clf


def cluster_fea_apply(data, selected_cols, clf):
    """
    使用聚类算法生成特征
    :param data: 用作输入的x,y
    :param selected_cols: 选取用来做聚类的特征列
    :param clf: 聚类模型
    :return: 聚类算法生成的特征
    """
    # 对原数据表进行类别标记
    data['group'] = clf.predict(data[selected_cols])

    # 距质心距离特征的计算
    centers_df = pd.DataFrame(clf.cluster_centers_)
    centers_df.columns = [x + '_center' for x in selected_cols]

    for item in selected_cols:
        data[item + '_center'] = data['group'].apply(
            lambda x: centers_df.iloc[x, :][item + '_center'])
        data[item + '_distance'] = data[item] - data[item + '_center']

    fea_cols = ['group']
    fea_cols.extend([x + '_distance' for x in selected_cols])

    return data.loc[:, fea_cols]


if __name__ == '__main__':
    # 数据读取
    all_x_y = pd.read_excel('data/order_feas.xlsx')
    all_x_y.set_index('order_no', inplace=True)
    # 取以下几个特征做聚类
    chose_cols = ['orderv1_age', 'orderv1_90_workday_application_amount_mean', 'orderv1_history_order_num',
                  'orderv1_max_overdue_days']
    all_x_y.fillna(0, inplace=True)

    # 生成聚类特征
    model = cluster_fea_gen(all_x_y, chose_cols, n_clusters=5)
    fea_cluster = cluster_fea_apply(all_x_y, chose_cols, model)
    print("使用聚类算法衍生特征数: \n", fea_cluster.shape[1])
    print("使用聚类算法衍生特征结果: \n", fea_cluster.head())



#==============================================================================
# File: ch3_07_jieba_demo.py
#==============================================================================

# -*- coding: utf-8 -*-

# 结巴分词使用示例
from utils.text_utils import cut_words

text_demo = "通过资料审核与电话沟通用户审批通过借款金额10000元操作人小明审批时间2020年10月5日 经过电话核实用户确认所有资料均为本人提交提交时间2020年11月5日用户当前未逾期"
segs = cut_words(text_demo)
print("原文: ", text_demo)
print("切词后的结果:", list(segs))



#==============================================================================
# File: ch3_08_bag_of_words.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 文本特征挖掘：词袋模型示例
import pandas as pd
from utils.text_utils import sentences_prepare
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def gen_count_doc_vec(text):
    """
    基于词频统计生成文本的向量表示
    :param text: 输入文本
    :return: 生成的文本向量表示
    """
    cv = CountVectorizer(binary=True)
    document_vec = cv.fit_transform(text)
    return pd.DataFrame(document_vec.toarray())


def gen_tfidf_doc_vec(text):
    """
    基于TfidfVectorizer生成文本向量表示
    :param text: 输入文本
    :return: 生成的文本向量表示
    """
    cv = TfidfVectorizer()
    document_vec = cv.fit_transform(text)
    return pd.DataFrame(document_vec.toarray())


def gen_hash_doc_vec(text, n_features=8):
    """
    基于HashingVectorizer生成文本向量表示
    :param text: 输入文本
    :param n_features: 指定输出特征的维数
    :return: 生成的文本向量表示
    """
    cv = HashingVectorizer(n_features=n_features)
    document_vec = cv.fit_transform(text)
    return pd.DataFrame(document_vec.toarray())


def gen_ngram_doc_vec(text):
    ngram_cv = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                               token_pattern=r'\b\w+\b', min_df=1)
    document_vec = ngram_cv.fit_transform(text)
    return pd.DataFrame(document_vec.toarray())


if __name__ == '__main__':
    sentences = sentences_prepare()
    # 词袋模型应用示例
    # 取前三条文本用于展示
    texts = sentences[0:5]
    fea_vec_count = gen_count_doc_vec(texts)
    print("CountVectorizer词向量:")
    print(fea_vec_count)

    fea_vec_tfidf = gen_tfidf_doc_vec(texts)
    print("TfidfVectorizer词向量:")
    print(fea_vec_tfidf)

    fea_vec_hash = gen_hash_doc_vec(texts, n_features=8)
    print("HashingVectorizer词向量:")
    print(fea_vec_hash)

    fea_vec_ngram = gen_ngram_doc_vec(texts)
    print("CountVectorizer词向量(ngram):")
    print(fea_vec_ngram)



#==============================================================================
# File: ch3_09_word2vec.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 文本特征挖掘：word2vec
import numpy as np
import pandas as pd
from utils.text_utils import sentences_prepare
from gensim.models import word2vec


def sent2vec(words, w2v_model):
    """
    转换成句向量
    :param words: 词列表
    :param w2v_model: word2vec模型
    :return:
    """
    if words == '':
        return np.array([0] * model.wv.vector_size)

    vector_list = []
    for w in words:
        try:
            vector_list.append(w2v_model.wv[w])
        except:
            continue
    vector_list = np.array(vector_list)
    v = vector_list.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


if __name__ == '__main__':
    # 加载语料
    sentences = sentences_prepare()

    # 获取词向量
    model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=2, workers=2)
    fea_vec = pd.DataFrame([sent2vec(x, model).tolist() for x in sentences])
    fea_vec.columns = ['fea_%s' % i for i in range(model.wv.vector_size)]
    print('词向量维度：', fea_vec.shape)



#==============================================================================
# File: ch3_10_fasttext_vec.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 文本特征挖掘：fasttext
import pandas as pd
from utils.text_utils import sentences_prepare
import fasttext

if __name__ == '__main__':
    # 加载语料
    sentences = sentences_prepare()

    # 预处理过后的文本写入文件unsupervised_train_data
    with open('data/text_data/unsupervised_train_data.txt', 'w') as f:
        for sentence in sentences:
            f.write(sentence)
            f.write('\n')

    # 获取fasttext词向量
    model = fasttext.train_unsupervised('data/text_data/unsupervised_train_data.txt', model='skipgram', dim=8)
    fea_vec = pd.DataFrame([model.get_sentence_vector(x).tolist() for x in sentences])
    fea_vec.columns = ['fea_%s' % i for i in range(model.get_dimension())]
    print('词向量维度：', fea_vec.shape)



#==============================================================================
# File: ch3_11_text_classifier_bayes.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 文本分类算法：朴素贝叶斯
import pandas as pd
from utils.text_utils import sentences_prepare_x_y
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score


def get_model(x, y):
    # 训练朴素贝叶斯分类器
    clf = GaussianNB()
    bayes_model = clf.fit(x, y)
    return bayes_model


def text_sample_split(texts, y, rate=0.75):
    # 文本向量化
    cv = TfidfVectorizer(binary=True)
    sentence_vec = cv.fit_transform(texts)

    # 划分训练集和测试集
    split_size = int(len(texts) * rate)
    x_train = sentence_vec[:split_size].toarray()
    y_train = y[:split_size]
    x_test = sentence_vec[split_size:].toarray()
    y_test = y[split_size:]
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    # 加载语料
    sentences, target = sentences_prepare_x_y()
    print("文本数目: %s" % len(sentences))
    # 训练模型
    x_train, y_train, x_test, y_test = text_sample_split(pd.Series(sentences), pd.Series(target))
    model = get_model(x_train, y_train)
    # 预测
    y_pred = model.predict_proba(x_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred)
    print("AUC结果: ", auc_score)



#==============================================================================
# File: ch3_12_text_classifier_fasttext.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 文本分类算法：fasttext
import fasttext
import pandas as pd
from utils.text_utils import sentences_prepare_with_y
from sklearn.metrics import roc_auc_score


def process_sentences(train_path, test_path, rate=0.8):
    sentences = sentences_prepare_with_y()
    # 预处理之后的数据写入文件train_data.txt
    num = int(len(sentences) * rate)
    train_out = open(train_path, 'w')
    test_out = open(test_path, 'w')
    for sentence in sentences[:num]:
        train_out.write(sentence)
        train_out.write("\n")
    for sentence in sentences[num:]:
        test_out.write(sentence)
        test_out.write("\n")
    print("预处理之后的数据已写入文件train_data.txt, test_data.txt")
    print("train文本数目: %s, test文本数目: %s" % (num, len(sentences) - num))


if __name__ == '__main__':
    # 处理文本数据
    process_sentences(train_path='data/train_data.txt', test_path='data/test_data.txt', rate=0.8)

    # 训练、保存模型
    classifier = fasttext.train_supervised('data/train_data.txt', label='__label__', wordNgrams=3, loss='softmax')
    classifier.save_model('data/fasttext_demo.model')

    # 加载模型
    classifier = fasttext.load_model('data/fasttext_demo.model')
    texts = "系列 票房 不差 口碑 生化危机 资深 玩家 张艳 告诉 玩家 很难 承认 一系列 电影 " \
            "电影 原著 面目全非 女主角 爱丽丝 游戏 角色 电影 渐渐 脱离 游戏 打着 游戏 名号 发展 票房 " \
            "号召力 观众 影响力 电影 系列 具备 剧情 世界观 游戏 生硬 强加 角色 背景 "
    print("当前文本所属类别: ", classifier.predict(texts))

    # 测试集
    test_data = pd.read_csv('data/test_data.txt', header=None)
    texts_new = test_data[1].tolist()
    y_true = [1 if x.strip() == '__label__sports' else 0 for x in test_data[0].tolist()]

    # 预测效果评估
    result = classifier.predict(texts_new)
    y_pre = []
    for i in range(len(result[0])):
        if result[0][i][0] == '__label__sports':
            y_pre.append(result[1][i][0])
        else:
            y_pre.append(1 - result[1][i][0])
    auc_score = roc_auc_score(y_true, y_pre)
    print("测试集AUC为: ", auc_score)



#==============================================================================
# File: ch3_13_random_walk.py
#==============================================================================

# -*- coding: utf-8 -*-
"""
使用DeepWalk算法生成特征(可以直接在shell命令窗口中运行deepwalk命令)
"""

import os
import sys
import pandas as pd
sys.path.append("./")
sys.path.append("../")

size = 8
os.system(
    "deepwalk --input data/graph_data/graph_demo.adjlist "
    f"--output data/graph_data/graph_demo.embeddings --representation-size {size}")

fea_vec = pd.read_csv('data/graph_data/graph_demo.embeddings', sep=' ', skiprows=1, index_col=0,
                      names=['fea_%s' % i for i in range(size)]).sort_index()
print('词向量维度：', fea_vec.shape)
print('词向量结果：', fea_vec)



#==============================================================================
# File: ch3_14_node2vec.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 使用Node2Vec算法生成特征
import networkx as nx
import pandas as pd
from node2vec import Node2Vec
import matplotlib.pyplot as plt


def adj_to_graph(adj_table):
    # 根据邻接表生成图G
    graph = nx.Graph()
    # 添加边
    for i in range(0, len(adj_table)):
        node_edgs = adj_table[i]
        for j in range(0, len(node_edgs)):
            graph.add_edge(node_edgs[0], node_edgs[j])
    return graph


def gen_node2vec_fea(graph, dimensions=8):
    # 生成随机游走序列
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=30, num_walks=100, workers=4)
    # 向量化
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return model.wv.vectors


if __name__ == '__main__':
    # 数据读取
    adj_tbl = []
    with open('data/graph_data/graph_demo.adjlist') as f:
        for line in f.readlines():
            adj_tbl.append(line.replace('\n', '').split(' '))
    G = adj_to_graph(adj_tbl)
    # 使用networkx展示图结构
    nx.draw(G, with_labels=True)
    plt.show()
    feas = gen_node2vec_fea(G, dimensions=8)
    print(pd.DataFrame(feas))



#==============================================================================
# File: ch3_15_gcn_order.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# GCN关系网络节点预测
import pickle
import os
import itertools
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import namedtuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cpu_type = "cuda" if torch.cuda.is_available() else "cpu"


def numpy_to_tensor(x):
    return torch.from_numpy(x).to(cpu_type)


def build_adjacency(adj_dict):
    """
    根据邻接表创建邻接矩阵
    :param adj_dict: 输入的邻接表
    :return: 邻接矩阵
    """
    edge_index = []
    node_counts = len(adj_dict)
    for src, dst in adj_dict.items():
        edge_index.extend([src, v] for v in dst)
        edge_index.extend([v, src] for v in dst)
    # 去重
    edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
    edge_index = np.asarray(edge_index)
    # 构建邻接矩阵，相接的节点值为1
    adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                               (edge_index[:, 0], edge_index[:, 1])),
                              shape=(node_counts, node_counts), dtype="double")
    return adjacency


def read_data(path_of_data):
    """
    数据读取
    :param path_of_data: 文件路径
    :return:
    """
    out = pickle.load(open(path_of_data, "rb"), encoding="latin1")
    out = out.toarray() if hasattr(out, "toarray") else out
    return out


def data_preprocess():
    print("Start data preprocess.")
    filenames = ["order.{}".format(name) for name in ['x', 'y', 'graph']]
    # 图有2000个节点，每个节点有104维特征，y值为0或1，graph用字典表示，字典key为节点编号，value为关联的节点编号list
    root_path = 'data/graph_data'
    x, y, graph = [read_data(os.path.join(root_path, name)) for name in filenames]

    # 划分train，validation和test节点编号
    train_index = list(range(0, 700))
    val_index = list(range(700, 1000))
    test_index = list(range(1000, 2000))

    num_nodes = x.shape[0]
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    train_mask[train_index] = True
    val_mask[val_index] = True
    test_mask[test_index] = True

    adjacency = build_adjacency(graph)
    print("特征维度: ", x.shape)
    print("标签长度: ", y.shape)
    print("邻接矩阵维度: ", adjacency.shape)
    # 构建带字段名的元组
    Data = namedtuple('Data', ['x', 'y', 'adjacency',
                               'train_mask', 'val_mask', 'test_mask'])
    return Data(x=x, y=y, adjacency=adjacency,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def adj_norm(adjacency):
    """
    正则化：公式L=D^-0.5 * (A+I) * D^-0.5
    :param torch.sparse.FloatTensor adjacency:
    :return:
    """
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """
        # 图卷积层定义
        :param int input_dim: 输入特征维度
        :param int output_dim: 输出特征维度
        :param bool use_bias: 偏置
        :return:
        """
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, fea_input):
        """
        :param torch.sparse.FloatTensor adjacency : 邻接矩阵
        :param torch.Tensor fea_input: 输入特征
        :return:
        """
        support = torch.mm(fea_input, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output


class GcnNet(nn.Module):
    def __init__(self, input_dim):
        """
        模型定义
        :param int input_dim: 输入特征维度
        """
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConv(input_dim, 16)
        self.gcn2 = GraphConv(16, 2)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        lg = self.gcn2(adjacency, h)
        return lg


def model_predict(model, tensor, tensor_adj, mask):
    model.eval()
    with torch.no_grad():
        lg = model(tensor_adj, tensor)
        lg_mask = lg[mask]
        y_pred = lg_mask.max(1)[1]
    return y_pred


def cal_accuracy(y_true, y_pred):
    accuracy = torch.eq(y_pred, y_true).double().mean()
    return accuracy


def model_train(tensor_x, tensor_y, tensor_adjacency, train_mask, val_mask, epochs, learning_rate,
                weight_decay):
    # 模型定义：Model, Loss, Optimizer
    model = GcnNet(tensor_x.shape[1]).to(cpu_type)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    loss_list = []
    test_accuracy_list = []
    model.train()
    train_y = tensor_y[train_mask].long()

    for epoch in range(epochs):
        # 前向传播
        lg = model(tensor_adjacency, tensor_x)
        train_mask_logits = lg[train_mask]
        loss = nn.CrossEntropyLoss().to(cpu_type)(train_mask_logits, train_y)
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()
        # 准确率
        train_accuracy = cal_accuracy(tensor_y[train_mask],
                                      model_predict(model, tensor_x, tensor_adjacency, train_mask))
        test_accuracy = cal_accuracy(tensor_y[val_mask],
                                     model_predict(model, tensor_x, tensor_adjacency, val_mask))

        loss_list.append(loss.item())
        test_accuracy_list.append(test_accuracy.item())
        if epoch % 10 == 1:
            print("epoch {:04d}: loss {:.4f}, train accuracy {:.4}, test accuracy {:.4f}".format(
                epoch, loss.item(), train_accuracy.item(), test_accuracy.item()))
    return model, loss_list, test_accuracy_list


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    # 坐标系ax1画曲线1
    ax1 = fig.add_subplot(111)  # 指的是将plot界面分成1行1列，此子图占据从左到右从上到下的1位置
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)  # c为颜色
    plt.ylabel('Loss')

    # 坐标系ax2画曲线2
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)  # 其本质就是添加坐标系，设置共享ax1的x轴，ax2背景透明
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()  # 开启右边的y坐标

    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


if __name__ == '__main__':
    # 数据预处理
    dataset = data_preprocess()

    # x、y规范化
    node_feature = (dataset.x - dataset.x.mean()) / dataset.x.std()
    tensor_x_all = numpy_to_tensor(node_feature).to(torch.float32)
    tensor_y_all = numpy_to_tensor(dataset.y)

    tensor_train_mask = numpy_to_tensor(dataset.train_mask)
    tensor_val_mask = numpy_to_tensor(dataset.val_mask)
    tensor_test_mask = numpy_to_tensor(dataset.test_mask)

    # 邻接矩阵规范化
    normed_adj = adj_norm(dataset.adjacency)

    indices = torch.from_numpy(np.asarray([normed_adj.row,
                                           normed_adj.col]).astype('int64')).long()
    values = torch.from_numpy(normed_adj.data.astype(np.float32))

    tensor_adjacency_all = torch.sparse.FloatTensor(indices, values,
                                                    (node_feature.shape[0], node_feature.shape[0])).to(cpu_type)

    # 训练模型并做预测
    gcn_model, loss_arr, test_accuracy_arr = model_train(tensor_x_all, tensor_y_all, tensor_adjacency_all,
                                                         tensor_train_mask,
                                                         tensor_val_mask, epochs=300,
                                                         learning_rate=0.04, weight_decay=5e-4)
    y_predict = model_predict(gcn_model, tensor_x_all, tensor_adjacency_all, tensor_test_mask)
    test_acc = cal_accuracy(tensor_y_all[tensor_test_mask], y_predict)
    print(test_acc.item())

    plot_loss_with_acc(loss_arr, test_accuracy_arr)



#==============================================================================
# File: ch4_00_rules_for_iv.py
#==============================================================================

# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import toad
import numpy as np
import pandas as pd
from utils import data_utils
from toad.plot import bin_plot
from matplotlib import pyplot as plt


def cal_iv(x, y):
    """ 
    IV计算函数  
    :param x: feature 
    :param y: label 
    :return: 
    """
    crtab = pd.crosstab(x, y, margins=True)
    crtab.columns = ['good', 'bad', 'total']
    crtab['factor_per'] = crtab['total'] / len(y)
    crtab['bad_per'] = crtab['bad'] / crtab['total']
    crtab['p'] = crtab['bad'] / crtab.loc['All', 'bad']
    crtab['q'] = crtab['good'] / crtab.loc['All', 'good']
    crtab['woe'] = np.log(crtab['p'] / crtab['q'])
    crtab2 = crtab[abs(crtab.woe) != np.inf]

    crtab['IV'] = sum(
        (crtab2['p'] - crtab2['q']) * np.log(crtab2['p'] / crtab2['q']))
    crtab.reset_index(inplace=True)
    crtab['varname'] = crtab.columns[0]
    crtab.rename(columns={crtab.columns[0]: 'var_level'}, inplace=True)
    crtab.var_level = crtab.var_level.apply(str)
    return crtab


german_credit_data = data_utils.get_data()

# 生成分箱初始化对象  
bin_transformer = toad.transform.Combiner()

# 采用等距分箱训练  
bin_transformer.fit(german_credit_data,
                    y='creditability',
                    n_bins=6,
                    method='step',
                    empty_separate=True)

# 分箱数据  
trans_data = bin_transformer.transform(german_credit_data, labels=True)

# 查看Credit amount分箱结果  
bin_plot(trans_data, x='credit.amount', target='creditability')
plt.show()

# 查看Credit amount分箱数据  
cal_iv(trans_data['credit.amount'], trans_data['creditability'])

# 构建单规则
german_credit_data['credit.amount.rule'] = np.where(german_credit_data['credit.amount'] > 12366.0, 1, 0)



#==============================================================================
# End of batch 2
#==============================================================================
