#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 入力データとして
# 電荷0の水素原子を含まない原子ペアの非結合相互作用と
# 電荷0の水素原子を含む原子ペアの非結合相互作用
# ねじれ項についてのデータである
# Tensorflow中の電荷データを事前に配列として作成する。


# In[ ]:


#　コードの最初に実行時の環境変数を集め実行ごとの修正を行いやすくする
#　出力時のファイル名
file_date        = "191004_0"
#　使用するGPUカードの番号
DEVICES = "0"
#　参照構造をQM計算での最安定構造とするか、DDEの中央値とするかの設定
reference_index = "min"
#　LJパラメータをフィッティングするか否か
lj_opt = False
#　Iterationの設定
iteration = 100000

# 入出力ファイル名とハイパーパラメータの設定
import numpy as np
c_charge_reg     = np.float32(0.1)
c_charge_sum     = np.int32(1000000)
c_torsion        = np.int32(3)
c_SCEE           = np.int(100)
# In[ ]:


# For using .py
# ライブラリのインポート
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=DEVICES
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import math
import time

np.random.seed(seed=100)

# In[ ]:

"""
# For using .ipynb
# ライブラリのインポート
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import math
import time
import sys
"""

# In[ ]:


# データの数値の型を設定
tf_float_type = tf.float32
tf_int_type   = tf.int32
df_float_type = "float32"
epsilon       = 1.0


# In[ ]:




# In[ ]:


# 事前に用意しておいた電荷ファイルから原子電荷を読み込む。
# 電荷ファイルは d:/cycwin/home/MI-Lab/python/accomplish/get_charge_from_prepfile.py を用いて計算している
dict_charge = {}
with open("../../0GB.charge.txt") as charge_inp:
    line_num = 0
    for line in charge_inp:
        line_num += 1
        if 3 <= line_num:
            atom_name = line.split()[0]
            charge = float(line.split()[1])
            if charge != 0. :
                dict_charge[atom_name] = charge


# In[ ]:


# 電荷の分布ファイルを読み込む
df_charge_distribution = pd.read_csv("../../charge_distribution.csv",index_col=0)
df_charge_distribution = df_charge_distribution.astype(df_float_type)


# In[ ]:


# 今回は電荷をパラメータフィッティングするので、静電相互作用の関数を定義する
# 電荷と距離で電荷の単位はクーロン、距離の単位は Å、出力はエネルギーで単位は kcal/mol である

def Electrostatic(qi,qj,r,epsilon=1.):
    E = np.float32(332.05) * ( qi * qj ) / ( r * epsilon )
    return E
def Electrostatic14(qi,qj,r,scee , epsilon=1.):
    E = np.float32(332.05) * ( qi * qj ) / ( r * epsilon * scee )
    return E

# Torsion と　二面角パラメータからそのTorsionのエネルギーを計算する関数の定義
# 入力のTorsionとGammaはともに単位がRadian
def Etorsion_var(V1,V2,V3,V4,V5,V6,gamma1,gamma2,gamma3,gamma4,gamma5,gamma6,phi):
    et1 = V1 * ( 1. + np.cos(1. * phi - gamma1 ) )
    et2 = V2 * ( 1. + np.cos(2. * phi - gamma2 ) )
    et3 = V3 * ( 1. + np.cos(3. * phi - gamma3 ) )
    et4 = V4 * ( 1. + np.cos(4. * phi - gamma4 ) )
    et5 = V5 * ( 1. + np.cos(5. * phi - gamma5 ) )
    et6 = V6 * ( 1. + np.cos(6. * phi - gamma6 ) )
    et  = et1 + et2 + et3 + et4 + et5 + et6
    return et


# In[ ]:


# フィッティング前の相対エネルギーデータの読み込み
# CSVファイルには、相対DFTエネルギー、相対MMエネルギー、全ねじれ項の相対エネルギー、全静電項の相対エネルギー
# フィッティングする変数（電荷および二面角）に依存して変わるねじれ項と静電項の和
# 相対DFTエネルギー　-　相対MMエネルギーの変数に依存しない部分としての　y_target
# 非結合項の原子ペアの原子間距離、14結合項の原子ペアの原子間距離
# 全二面角
# 非結合項の原子ペアの静電相互作用エネルギー、14結合項の原子ペアの静電相互作用エネルギー
# 二面角相対エネルギー　の順に並んでいる
# エネルギーは全て参照構造との差分となっており、参照構造は min_index で与えられる16740番目の構造である。
# 列数は、y_target までが6列、非結合項の原子ペア数は105列、14結合項の原子ペア数は23列、ねじれ項が66列

# 列数の確認は完了

if reference_index == "min":
    INPUT_FILENAME = "../../delta_energy_profile_edft_opt_reduce_datasets_bf.csv"
    print(INPUT_FILENAME)
elif reference_index == "median":
    INPUT_FILENAME = "../../delta_energy_profile_edft_opt_reduce_datasets_median_ref_bf.csv"
    print(INPUT_FILENAME)
else:
    print("Input reference type")
    exit
df_delta_edat_bf_fit       = pd.read_csv(INPUT_FILENAME,index_col=0)
df_delta_edat_bf_fit.drop(index = 17774,inplace=True)
df_delta_edat_bf_fit_train = df_delta_edat_bf_fit.sample(n=int(len(df_delta_edat_bf_fit)*0.8))
df_delta_edat_bf_fit_train = df_delta_edat_bf_fit_train.sort_index()
print(df_delta_edat_bf_fit.shape)


# In[ ]:


# フィッティング後のCSVから学習データ、テストデータを分割し
# 再度グラフ化などをする際に必要な、学習・テストデータのインデックスの保存
test_index  = list(set([df_delta_edat_bf_fit.index][0]) - set([df_delta_edat_bf_fit_train.index][0]))
train_index = set([df_delta_edat_bf_fit_train.index][0])
df_delta_edat_bf_fit_test = df_delta_edat_bf_fit.loc[test_index , :]
df_delta_edat_bf_fit_test = df_delta_edat_bf_fit_test.sort_index()
train_test_set_index_file = "train_test_set_index_" + file_date + ".dat"
with open(train_test_set_index_file,"w") as out:
    print(train_index,file=out)
    print(test_index,file=out)


# In[ ]:


dict_num_columns = {"R":0,"R14":0,"Torsion":0,"Deel":0,"De14eel":0,"Detor":0}
dict_info_columns_num = {"R":{},"R14":{},"Torsion":{},"Deel":{},"De14eel":{},"Detor":{}}
for column_name in df_delta_edat_bf_fit.columns:
    if column_name[:2] == "R_":
        if dict_num_columns["R"] == 0:
            R_first_column_num = df_delta_edat_bf_fit.columns.get_loc(column_name)
            dict_info_columns_num["R"]["first"] = R_first_column_num
        dict_num_columns["R"] += 1

    elif column_name[:4] == "R14_":
        if dict_num_columns["R14"] == 0:
            R14_first_column_num = df_delta_edat_bf_fit.columns.get_loc(column_name)
            dict_info_columns_num["R14"]["first"] = R14_first_column_num
        dict_num_columns["R14"] += 1

    elif column_name[:4] == "Tor_":
        if dict_num_columns["Torsion"] == 0:
            Torsion_first_column_num = df_delta_edat_bf_fit.columns.get_loc(column_name)
            dict_info_columns_num["Torsion"]["first"] = Torsion_first_column_num
        dict_num_columns["Torsion"] += 1

    elif column_name[:5] == "Deel_":
        if dict_num_columns["Deel"] == 0:
            Deel_first_column_num = df_delta_edat_bf_fit.columns.get_loc(column_name)
            dict_info_columns_num["Deel"]["first"] = Deel_first_column_num
        dict_num_columns["Deel"] += 1

    elif column_name[:8] == "De14eel_":
        if dict_num_columns["De14eel"] == 0:
            De14eel_first_column_num = df_delta_edat_bf_fit.columns.get_loc(column_name)
            dict_info_columns_num["De14eel"]["first"] = De14eel_first_column_num
        dict_num_columns["De14eel"] += 1

    elif column_name[:6] == "Detor_":
        if dict_num_columns["Detor"] == 0:
            Detor_first_column_num = df_delta_edat_bf_fit.columns.get_loc(column_name)
            dict_info_columns_num["Detor"]["first"] = Detor_first_column_num
        dict_num_columns["Detor"] += 1

    else:
        column_num = df_delta_edat_bf_fit.columns.get_loc(column_name)
        print("Column Name:: %15s // Column Number:: %3d" % (column_name,column_num))

for column in dict_num_columns:
    first_num = locals()["%s_first_column_num" % column]
    last_num = first_num + dict_num_columns[column]
    dict_info_columns_num[column]["last"] = last_num
    print("Column Name:: %15s // Column Number:: %3d - %3d // Number of Data:: %3d"          % (column , first_num , last_num , dict_num_columns[column]))


# In[ ]:


#　二面角の列名から二面角を構成する4原子の原子タイプの部分だけ抜き出して、リスト list_dihed_type に収容
#　同じ原子タイプの二面角パラメータは同じにするためにこの情報が必要となる
dihed_num = 66
list_dihed_type = []
for i in range(dihed_num):
    dihed_type = df_delta_edat_bf_fit.columns[i+dict_info_columns_num["Torsion"]["first"]][4:12]
    if dihed_type not in list_dihed_type:
        list_dihed_type.append(dihed_type)


# In[ ]:


# GLYCAM_06j-1 の二面角パラメータ
# フィッティングする二面角により値を変更する
# すべての二面角パラメータをフィッティングするのでAmberのParmファイルから直接読み込むスクリプトに変更予定
df_params_torsion = pd.DataFrame(np.zeros((20,12)),index=list_dihed_type)
df_params_torsion = df_params_torsion.astype(df_float_type)
df_params_torsion.columns = ["V1","V2","V3","V4","V5","V6","gamma1","gamma2","gamma3","gamma4","gamma5","gamma6"]
df_params_torsion.loc["CgCgCgCg","V1"] = 0.45
df_params_torsion.loc["CgCgCgH1","V3"] = 0.15
df_params_torsion.loc["CgCgOsCg","V3"] = 0.16
df_params_torsion.loc["CgOsCgH1","V3"] = 0.27
df_params_torsion.loc["H1CgCgH1","V3"] = 0.17
df_params_torsion.loc["H1CgOhHo","V3"] = 0.18
df_params_torsion.loc["H2CgCgCg","V3"] = 0.15
df_params_torsion.loc["H2CgCgH1","V3"] = 0.17
df_params_torsion.loc["H2CgCgOh","V3"] = 0.05
df_params_torsion.loc["H2CgOsCg","V2"] = 0.60
df_params_torsion.loc["H2CgOsCg","V3"] = 0.10
df_params_torsion.loc["HoOhCgCg","V3"] = 0.18
df_params_torsion.loc["HoOhCgOs","V3"] = 0.18
df_params_torsion.loc["HoOhCgH2","V3"] = 0.18
df_params_torsion.loc["OhCgCgCg","V3"] = 0.10
df_params_torsion.loc["OhCgCgH1","V3"] = 0.05
df_params_torsion.loc["OhCgCgOh","V1"] = -0.10
df_params_torsion.loc["OhCgCgOh","V2"] = 0.95
df_params_torsion.loc["OhCgCgOh","V3"] = 0.55
df_params_torsion.loc["OhCgOsCg","V1"] = 1.08
df_params_torsion.loc["OhCgOsCg","V2"] = 1.38
df_params_torsion.loc["OhCgOsCg","V3"] = 0.96
df_params_torsion.loc["OsCgCgCg","V1"] = -0.27
df_params_torsion.loc["OsCgCgH1","V3"] = 0.05
df_params_torsion.loc["OsCgCgOh","V1"] = -1.10
df_params_torsion.loc["OsCgCgOh","V2"] = 0.25


# In[ ]:


#　二面角の列名から、二面角を構成する4原子の原子名の部分だけを抜き出して、リスト list_dihed に収容
#　さらに二面角を構成する4原子がどの原子タイプに属するかを辞書 dict_dihed_atoms_types に収録
Torf = dict_info_columns_num["Torsion"]["first"]
Torl = dict_info_columns_num["Torsion"]["last"]
list_dihed = [i[13:] for i in df_delta_edat_bf_fit.columns[Torf:Torl]]
dict_dihed_atoms_types = {}
for i in df_delta_edat_bf_fit.columns[Torf:Torl]:
    dict_dihed_atoms_types[i[13:]] = i[4:12]


# In[ ]:


dict_atom_name_type = {'HO1': 'Ho', 'H2O': 'Ho', 'H3O': 'Ho', 'H4O': 'Ho', 'H6O': 'Ho',                       'H2': 'H1', 'H3': 'H1', 'H4': 'H1', 'H5': 'H1', 'H61': 'H1', 'H62': 'H1',                       'H1': 'H2', 'O1': 'Oh', 'O2': 'Oh', 'O3': 'Oh', 'O4': 'Oh', 'O6': 'Oh',                       'O5': 'Os', 'C1': 'Cg', 'C2': 'Cg', 'C3': 'Cg', 'C4': 'Cg', 'C5': 'Cg', 'C6': 'Cg'}


# In[ ]:


list_R_NB_eel  = [i for i in df_delta_edat_bf_fit.columns if i[:2] == "R_"]
list_R_i14_eel = [i for i in df_delta_edat_bf_fit.columns if i[:4] == "R14_"]


# In[ ]:


#　量子化学計算で最安定のエネルギーの構造のIndexを取得
#　min_index は　Index名を取得する。
#　min_index_num は　DataFrameの行数に対応し、この番号は0番からナンバリングされているのでmin_indexとはずれている。
#　min_index_num は　pd.DataFrame.iloc[] で値を取得する場合に使用する。

#　今まではカラム名をFor文で回して取得してためlocメソッドしか使用していなかったが、
#　このスクリプトからilocメソッドを使用する場合ができたのでmin_index_numを取得している。

sr               = df_delta_edat_bf_fit["Dedft"] - df_delta_edat_bf_fit["Demm"]
median_index     = sr.sort_values().index[15950]
median_index_num = df_delta_edat_bf_fit.index.get_loc(median_index)
min_index        = df_delta_edat_bf_fit[df_delta_edat_bf_fit["Dedft"] == df_delta_edat_bf_fit["Dedft"].min() ].index
min_index_num    = df_delta_edat_bf_fit.index.get_loc(min_index[0])
print(min_index)
print(min_index_num)
print(median_index)
print(median_index_num)


# 式　ΔEdft = ΔEmm　が等しくなることが理想であり、その為にパラメータフィッティングを行う<br>
# Edftで最安定の構造との差分を取ったDataFrameとして　**df_delta_edat_bf_fit** を用意<br>
# <br>$$ \Delta Edft = \Delta Emm $$<br>
# すなわち<br>
# <br>$${\Delta}{Edft} - \Delta Emm　=　0$$<br>
# が理想的なので<br>
# <br>$$Loss = (\Delta Edft - \Delta Emm)^2$$<br><br>
# を損失関数として最小化を行う<br>
# ここで　ΔEmm　の項は　ΔE_invar (パラメータを変更しない項)　と　ΔE_var(パラメータを変更する項)　に分解する。<br>
# 今回のフィッティングでは、　二面角パラメータと電荷がフィッティングの対象であるので、<br>
# 各構造から得られた二面角のねじれエネルギーと<br>
# 各構造から得られた静電相互作用エネルギーおよび1_4静電相互作用エネルギーの和が　ΔE_var　となる<br>
# ΔE_invar　は　これらのエネルギーを除いた項になるので<br>
# <br>$$\Delta E_{invar} = \Delta Emm - \Delta E_{var\_old}$$<br>
# (E_var_old　は最適化前のパラメータで計算した変更箇所のエネルギー）<br>
# よって　損失関数は　<br>$$\{\Delta Edft - (\Delta Emm - \Delta E_{var\_old}　+　\Delta E_{var})\}^2$$<br>　となる<br>
# $$Loss =　(　\Delta Edft　-　\Delta Emm　+　\Delta E_{var\_old}　-　\Delta E_{var}　)^2$$<br>
# <br>
# $$ y_{target} = \Delta Edft -\Delta Emm + \Delta E_{var\_old}$$
# と置くと変換すると<br>
# $$ Loss = ( y_{target} - \Delta E_{var} )^2$$<br>
# となり y_target とそれぞれの構造の二面角および参照構造の二面角、<br>
# それぞれの構造の非結合相互作用項と1_4相互作用項の距離データおよび参照構造の距離データを入力とすることで<br>
# 損失関数を計算し、パラメータフィッティングを行うことが可能となる<br>

# In[ ]:


print(df_delta_edat_bf_fit_train["y_target"].shape[0])
print(df_delta_edat_bf_fit_test["y_target"].shape[0])


# In[ ]:


if reference_index == "min":
    ref_index     = min_index
    ref_index_num = min_index_num
elif reference_index == "median":
    ref_index     = median_index
    ref_index_num = median_index_num
else:
    print("Input reference type")
    exit

#　Tensorflowの実行と最適化前後の二面角パラメータの比較
#　グラフは　二面角を　-π～π　まで変化させた際のエネルギーをプロットしている。
#　単位は　kcal/mol

train_sample_num = df_delta_edat_bf_fit_train["y_target"].shape[0]
test_sample_num  = df_delta_edat_bf_fit_test["y_target"].shape[0]
all_sample_num   = df_delta_edat_bf_fit["y_target"].shape[0]

##　入力データの設定
#　ターゲット関数の設定
y_data_train     = np.array(df_delta_edat_bf_fit_train["y_target"],
                            dtype=df_float_type).reshape(train_sample_num ,     )
y_data_test      = np.array(df_delta_edat_bf_fit_test["y_target"],
                            dtype=df_float_type).reshape(test_sample_num  ,     )
y_data_all       = np.array(df_delta_edat_bf_fit["y_target"],
                            dtype=df_float_type).reshape(all_sample_num   ,     )


# 二面角項に関係する二面角データの設定
Tor_f = dict_info_columns_num["Torsion"]["first"]
Tor_l = dict_info_columns_num["Torsion"]["last"]
Tor_train        = np.array(df_delta_edat_bf_fit_train.iloc[:,Tor_f:Tor_l],
                            dtype=df_float_type).reshape(train_sample_num ,  66 )
Tor_test         = np.array(df_delta_edat_bf_fit_test.iloc[:,Tor_f:Tor_l],
                            dtype=df_float_type).reshape(test_sample_num  ,  66 )
Tor_ref          = np.array(df_delta_edat_bf_fit.iloc[ref_index_num,Torf:Torl],
                            dtype=df_float_type).reshape(1                ,  66 )
Tor_all          = np.array(df_delta_edat_bf_fit.iloc[:,Tor_f:Tor_l],
                            dtype=df_float_type).reshape(all_sample_num   ,  66 )


# 静電相互作用項に関係する距離データの設定
R_f = dict_info_columns_num["R"]["first"]
R_l = dict_info_columns_num["R"]["last"]
R14_f = dict_info_columns_num["R14"]["first"]
R14_l = dict_info_columns_num["R14"]["last"]
Rnb_eel_train    = np.array(df_delta_edat_bf_fit_train.iloc[:,R_f:R_l],
                            dtype=df_float_type).reshape(train_sample_num ,  69 )
R14_eel_train    = np.array(df_delta_edat_bf_fit_train.iloc[:,R14_f:R14_l],
                            dtype=df_float_type).reshape(train_sample_num ,  28 )
Rnb_eel_test     = np.array(df_delta_edat_bf_fit_test.iloc[:,R_f:R_l],
                            dtype=df_float_type).reshape(test_sample_num  ,  69 )
R14_eel_test     = np.array(df_delta_edat_bf_fit_test.iloc[:,R14_f:R14_l],
                            dtype=df_float_type).reshape(test_sample_num  ,  28 )
Rnb_eel_ref      = np.array(df_delta_edat_bf_fit.iloc[ref_index_num,R_f:R_l],
                            dtype=df_float_type).reshape(1                ,  69 )
R14_eel_ref      = np.array(df_delta_edat_bf_fit.iloc[ref_index_num,R14_f:R14_l],
                            dtype=df_float_type).reshape(1                ,  28 )
Rnb_eel_all      = np.array(df_delta_edat_bf_fit.iloc[:,R_f:R_l],
                            dtype=df_float_type).reshape(all_sample_num   ,  69 )
R14_eel_all      = np.array(df_delta_edat_bf_fit.iloc[:,R14_f:R14_l],
                            dtype=df_float_type).reshape(all_sample_num   ,  28 )


# In[ ]:


fit = tf.Graph()
with fit.as_default():
    # Placeholder の設定
    tf_y_t        = tf.placeholder(shape=(None,),   dtype=tf_float_type,name="tf_y_t")
    tf_Tor     = tf.placeholder(shape=(None,66), dtype=tf_float_type,name="tf_Tor")
    tf_Rnb_eel = tf.placeholder(shape=(None,69), dtype=tf_float_type,name="tf_Rnb_eel")
    tf_R14_eel = tf.placeholder(shape=(None,28), dtype=tf_float_type,name="tf_R14_eel")
    

    ##　パラメータフィッティングを行う変数の設定
    #　フィッティングを行うパラメータとしてTorsionエネルギーのV1,V2,V3,gamma1,gamma2,gamma3を設定
    #　Gammaの単位はRadian
    for types in list_dihed_type:
        locals()["%s_v1" % types] = tf.Variable(df_params_torsion.loc[types,"V1"],    
                                                name="%s_v1" % types,dtype=tf_float_type)
        locals()["%s_v2" % types] = tf.Variable(df_params_torsion.loc[types,"V2"],
                                                name="%s_v2" % types,dtype=tf_float_type)
        locals()["%s_v3" % types] = tf.Variable(df_params_torsion.loc[types,"V3"],
                                                name="%s_v3" % types,dtype=tf_float_type)
        locals()["%s_v4" % types] = tf.Variable(df_params_torsion.loc[types,"V4"],
                                                name="%s_v4" % types,dtype=tf_float_type)
        locals()["%s_v5" % types] = tf.Variable(df_params_torsion.loc[types,"V5"],
                                                name="%s_v5" % types,dtype=tf_float_type)
        locals()["%s_v6" % types] = tf.Variable(df_params_torsion.loc[types,"V6"],
                                                name="%s_v6" % types,dtype=tf_float_type)
        locals()["%s_g1" % types] = tf.Variable(df_params_torsion.loc[types,"gamma1"],
                                                name="%s_g1" % types,dtype=tf_float_type)
        locals()["%s_g2" % types] = tf.Variable(df_params_torsion.loc[types,"gamma2"],
                                                name="%s_g2" % types,dtype=tf_float_type)
        locals()["%s_g3" % types] = tf.Variable(df_params_torsion.loc[types,"gamma3"],
                                                name="%s_g3" % types,dtype=tf_float_type)
        locals()["%s_g4" % types] = tf.Variable(df_params_torsion.loc[types,"gamma4"],
                                                name="%s_g4" % types,dtype=tf_float_type)
        locals()["%s_g5" % types] = tf.Variable(df_params_torsion.loc[types,"gamma5"],
                                                name="%s_g5" % types,dtype=tf_float_type)
        locals()["%s_g6" % types] = tf.Variable(df_params_torsion.loc[types,"gamma6"],
                                                name="%s_g6" % types,dtype=tf_float_type)

    #　フィッティングを行うパラメータとして、dict_charge に含まれる原子の原子電荷を設定
    #　なお、ここに含まれない原子としてアルキル水素がある。
    #　GLYCAM力場ではアルキル水素は結合している炭素に電荷を与え、自身は0とするユナイテッド原子として扱っている。
    for atom in dict_charge:
        locals()["%s_charge" % atom] = tf.Variable(dict_charge[atom],name="%s_charge" % atom , dtype=tf_float_type)
    #　さらに変数として14静電相互作用項のスケーリングファクターを設定
    SCEE = tf.Variable(1.0,name="SCEE14",dtype=tf_float_type)


    ## 各相互作用ごとに計算グラフを設定    
    #　Emm_var_tor　の計算の準備
    #　二面角項を計算するための二面角入力データに対応した二面角パラメータの配列の作成
    #　リストを作成し二面角入力データの順にパラメータをリストの中へ収納
    #　収納が終わった後、リストをテンソルに変換
    torsion_v1 , torsion_v2 , torsion_v3 , torsion_v4 , torsion_v5 , torsion_v6 = [] , [] , [] , [] , [] , []
    torsion_g1 , torsion_g2 , torsion_g3 , torsion_g4 , torsion_g5 , torsion_g6 = [] , [] , [] , [] , [] , []
    for i in df_delta_edat_bf_fit.columns[Torf:Torl]:
        types = i.split("_")[1]
        torsion_v1.append(locals()["%s_v1" % types])
        torsion_v2.append(locals()["%s_v2" % types])
        torsion_v3.append(locals()["%s_v3" % types])
        torsion_v4.append(locals()["%s_v4" % types])
        torsion_v5.append(locals()["%s_v5" % types])
        torsion_v6.append(locals()["%s_v6" % types])
        torsion_g1.append(locals()["%s_g1" % types])
        torsion_g2.append(locals()["%s_g2" % types])
        torsion_g3.append(locals()["%s_g3" % types])
        torsion_g4.append(locals()["%s_g4" % types])
        torsion_g5.append(locals()["%s_g5" % types])
        torsion_g6.append(locals()["%s_g6" % types])
    tf_torsion_v1 = tf.concat([torsion_v1],axis=1,name="Tor_V1")
    tf_torsion_v2 = tf.concat([torsion_v2],axis=1,name="Tor_V2")
    tf_torsion_v3 = tf.concat([torsion_v3],axis=1,name="Tor_V3")
    tf_torsion_v4 = tf.concat([torsion_v4],axis=1,name="Tor_V4")
    tf_torsion_v5 = tf.concat([torsion_v5],axis=1,name="Tor_V5")
    tf_torsion_v6 = tf.concat([torsion_v6],axis=1,name="Tor_V6")
    tf_torsion_g1 = tf.concat([torsion_g1],axis=1,name="Tor_G1")
    tf_torsion_g2 = tf.concat([torsion_g2],axis=1,name="Tor_G2")
    tf_torsion_g3 = tf.concat([torsion_g3],axis=1,name="Tor_G3")
    tf_torsion_g4 = tf.concat([torsion_g4],axis=1,name="Tor_G4")
    tf_torsion_g5 = tf.concat([torsion_g5],axis=1,name="Tor_G5")
    tf_torsion_g6 = tf.concat([torsion_g6],axis=1,name="Tor_G6")

    # 二面角エネルギーの計算
    # 各周期の二面角エネルギーの計算から二面角エネルギーの和の計算
    E_torsion_V1 = tf_torsion_v1 * ( 1. + tf.cos( 1. * tf_Tor - tf_torsion_g1 ))
    E_torsion_V2 = tf_torsion_v2 * ( 1. + tf.cos( 2. * tf_Tor - tf_torsion_g2 ))
    E_torsion_V3 = tf_torsion_v3 * ( 1. + tf.cos( 3. * tf_Tor - tf_torsion_g3 ))
    E_torsion_V4 = tf_torsion_v4 * ( 1. + tf.cos( 4. * tf_Tor - tf_torsion_g4 ))
    E_torsion_V5 = tf_torsion_v5 * ( 1. + tf.cos( 5. * tf_Tor - tf_torsion_g5 ))
    E_torsion_V6 = tf_torsion_v6 * ( 1. + tf.cos( 6. * tf_Tor - tf_torsion_g6 ))
    E_torsion    = E_torsion_V1 + E_torsion_V2 + E_torsion_V3 + E_torsion_V4 + E_torsion_V5 + E_torsion_V6

    # 参照構造の二面角エネルギーの計算
    E_torsion_V1_ref = tf_torsion_v1 * ( 1. + tf.cos( 1. * Tor_ref - tf_torsion_g1 ))
    E_torsion_V2_ref = tf_torsion_v2 * ( 1. + tf.cos( 2. * Tor_ref - tf_torsion_g2 ))
    E_torsion_V3_ref = tf_torsion_v3 * ( 1. + tf.cos( 3. * Tor_ref - tf_torsion_g3 ))
    E_torsion_V4_ref = tf_torsion_v4 * ( 1. + tf.cos( 4. * Tor_ref - tf_torsion_g4 ))
    E_torsion_V5_ref = tf_torsion_v5 * ( 1. + tf.cos( 5. * Tor_ref - tf_torsion_g5 ))
    E_torsion_V6_ref = tf_torsion_v6 * ( 1. + tf.cos( 6. * Tor_ref - tf_torsion_g6 ))
    E_torsion_ref    = ( E_torsion_V1_ref + E_torsion_V2_ref + E_torsion_V3_ref
                        + E_torsion_V4_ref + E_torsion_V5_ref + E_torsion_V6_ref)

    # DEmm_var_tor の計算
    DE_torsion_sum = tf.reduce_sum(E_torsion,axis=1) - tf.reduce_sum(E_torsion_ref,axis=1)


    # 静電相互作用の計算のための準備
    # 入力データに対応する原子i と原子j の電荷の配列の作成
    # 二面角と同様にリストを作成し、入力データの順に変数をリストに収納
    # 収納後にテンソルに変換してテンソルでエネルギーを計算している
    list_Chg_nb_atom_i , list_Chg_nb_atom_j = [],[]
    for i in list_R_NB_eel:
        atom_i , atom_j = i.split("_")[1],i.split("_")[2]
        list_Chg_nb_atom_i.append(locals()["%s_charge" % atom_i])
        list_Chg_nb_atom_j.append(locals()["%s_charge" % atom_j])
    tf_Chg_nb_atom_i = tf.concat([list_Chg_nb_atom_i],axis=1,name="Chg_nb_atom_i")
    tf_Chg_nb_atom_j = tf.concat([list_Chg_nb_atom_j],axis=1,name="Chg_nb_atom_j")

    list_Chg_i14_atom_i , list_Chg_i14_atom_j = [] , []
    for i in list_R_i14_eel:
        atom_i , atom_j = i.split("_")[1] , i.split("_")[2]
        list_Chg_i14_atom_i.append(locals()["%s_charge" % atom_i])
        list_Chg_i14_atom_j.append(locals()["%s_charge" % atom_j])
    tf_Chg_i14_atom_i = tf.concat([list_Chg_i14_atom_i],axis=1,name="Chg_i14_atom_i")
    tf_Chg_i14_atom_j = tf.concat([list_Chg_i14_atom_j],axis=1,name="Chg_i14_atom_j")

    # 静電相互作用の計算
    # 非結合ペアの静電相互作用の計算
    E_NB_eel_pair      = 332.05 * ( tf_Chg_nb_atom_i * tf_Chg_nb_atom_j ) / ( tf_Rnb_eel  * epsilon)
    E_NB_eel_pair_ref  = 332.05 * ( tf_Chg_nb_atom_i * tf_Chg_nb_atom_j ) / ( Rnb_eel_ref * epsilon)
    DE_NB_eel_pair_sum = tf.reduce_sum(E_NB_eel_pair,axis=1) - tf.reduce_sum(E_NB_eel_pair_ref,axis=1)

    # 1-4結合の静電相互作用の計算
    E_i14_eel_pair      = 332.05 * ( tf_Chg_i14_atom_i * tf_Chg_i14_atom_j ) / ( tf_R14_eel * epsilon  * SCEE)
    E_i14_eel_pair_ref  = 332.05 * ( tf_Chg_i14_atom_i * tf_Chg_i14_atom_j ) / ( R14_eel_ref * epsilon * SCEE)
    DE_i14_eel_pair_sum = tf.reduce_sum(E_i14_eel_pair,axis=1) - tf.reduce_sum(E_i14_eel_pair_ref,axis=1)

    #　DEmm_var の計算
    Devar = DE_torsion_sum + DE_NB_eel_pair_sum + DE_i14_eel_pair_sum

    #　損失関数の定義
    loss0 = tf.reduce_mean((tf_y_t - Devar ) ** 2 )

    #　正則化項の定義
    #　各原子電荷の分布からのずれに対する損失関数

    #　※ここから下についてはテンソル化はまだ行えていない
    charge_reg = 0.
    for i in dict_charge:
        charge_i_mean , charge_i_std = df_charge_distribution.loc[i,"Mean"] , df_charge_distribution.loc[i,"Std"]
        charge_err = (( locals()["%s_charge" % i] - charge_i_mean ) ** 2 )/ ((2 * charge_i_std) ** 2)
        charge_reg += charge_err


    #　ROH基を除いたグルコース部分の総電荷（0GBの総電荷）0.194からのずれに対する損失関数
    charge_sum = 0.
    for i in dict_charge:
        charge_sum    += locals()["%s_charge" % i]
    charge_sum_reg = ( charge_sum - 0.0 ) ** 2

    #　二面角の障壁エネルギーのGLYCAM力場を基準とする損失関数
    torsion_reg = 0.
    for types in list_dihed_type:
        torsion_reg += (  (locals()["%s_v1" % types] - float(df_params_torsion.loc[types,"V1"])) ** 2
                        + (locals()["%s_v2" % types] - float(df_params_torsion.loc[types,"V2"])) ** 2
                        + (locals()["%s_v3" % types] - float(df_params_torsion.loc[types,"V3"])) ** 2
                        + (locals()["%s_v4" % types] - float(df_params_torsion.loc[types,"V4"])) ** 2
                        + (locals()["%s_v5" % types] - float(df_params_torsion.loc[types,"V5"])) ** 2
                        + (locals()["%s_v6" % types] - float(df_params_torsion.loc[types,"V6"])) ** 2)
    
    #　損失関数の再定義
    
    loss = (  loss0 + c_charge_reg*charge_reg + c_charge_sum*charge_sum_reg
            + c_torsion * torsion_reg + c_SCEE * ( SCEE - 1. ) ** 2.)
        
    # 最適化関数の作成
    my_opt = tf.train.AdadeltaOptimizer(0.002)
    train_step = my_opt.minimize(loss)
    
    # 乱数の初期化メソッドの設定
    init = tf.global_variables_initializer()


# In[ ]:


# Train用、Test用のFeed_dict の作成

dict_train_feed = {"tf_y_t:0":y_data_train     , "tf_Tor:0":Tor_train ,
                   "tf_Rnb_eel:0":Rnb_eel_train, "tf_R14_eel:0":R14_eel_train}
dict_test_feed  = {"tf_y_t:0":y_data_test      , "tf_Tor:0":Tor_test  ,
                   "tf_Rnb_eel:0":Rnb_eel_test , "tf_R14_eel:0":R14_eel_test}
dict_all_feed   = {"tf_y_t:0":y_data_all       , "tf_Tor:0":Tor_all ,
                   "tf_Rnb_eel:0":Rnb_eel_all  , "tf_R14_eel:0":R14_eel_all }


# In[ ]:


## 変数保存用のリストの設定
#　電荷保存用のリスト
for atom in dict_charge:
    locals()["list_%s_charge" % atom]        = []
#　二面角パラメータ保存用のリスト
for types in list_dihed_type:
    locals()["list_%s_v1" % types ]          = []
    locals()["list_%s_v2" % types ]          = []
    locals()["list_%s_v3" % types ]          = []
    locals()["list_%s_v4" % types ]          = []
    locals()["list_%s_v5" % types ]          = []
    locals()["list_%s_v6" % types ]          = []
    locals()["list_%s_g1" % types ]          = []
    locals()["list_%s_g2" % types ]          = []
    locals()["list_%s_g3" % types ]          = []
    locals()["list_%s_g4" % types ]          = []
    locals()["list_%s_g5" % types ]          = []
    locals()["list_%s_g6" % types ]          = []
#　その他のパラメータ保存用リスト
list_SCEE                                    = []

#　損失関数の保存用リスト
Train_Loss                                   = []
Train_Loss0                                  = []
Train_LCharge_err                            = []
Train_LCharge_sum                            = []
Train_Ltorsion_reg                           = []
Test_Loss                                    = []
Test_Loss0                                   = []
Test_LCharge_err                             = []
Test_LCharge_sum                         = []
Test_Ltorsion_reg                            = []


# In[ ]:


## 変数保存用のリストの設定
#　電荷保存用のリスト
for atom in dict_charge:
    locals()["last_%s_charge" % atom]         = 0
    locals()["last1_%s_charge" % atom]        = 0
    
#　二面角パラメータ保存用のリスト
for types in list_dihed_type:
    locals()["last_%s_v1" % types ]           = 0
    locals()["last_%s_v2" % types ]           = 0
    locals()["last_%s_v3" % types ]           = 0
    locals()["last_%s_v4" % types ]           = 0
    locals()["last_%s_v5" % types ]           = 0
    locals()["last_%s_v6" % types ]           = 0
    locals()["last_%s_g1" % types ]           = 0
    locals()["last_%s_g2" % types ]           = 0
    locals()["last_%s_g3" % types ]           = 0
    locals()["last_%s_g4" % types ]           = 0
    locals()["last_%s_g5" % types ]           = 0
    locals()["last_%s_g6" % types ]           = 0
    locals()["last1_%s_v1" % types ]          = 0
    locals()["last1_%s_v2" % types ]          = 0
    locals()["last1_%s_v3" % types ]          = 0
    locals()["last1_%s_v4" % types ]          = 0
    locals()["last1_%s_v5" % types ]          = 0
    locals()["last1_%s_v6" % types ]          = 0
    locals()["last1_%s_g1" % types ]          = 0
    locals()["last1_%s_g2" % types ]          = 0
    locals()["last1_%s_g3" % types ]          = 0
    locals()["last1_%s_g4" % types ]          = 0
    locals()["last1_%s_g5" % types ]          = 0
    locals()["last1_%s_g6" % types ]          = 0
    

#　その他のパラメータ保存用リスト
last_SCEE                                     = 0
last1_SCEE                                    = 0

#　損失関数の保存用リスト
last_Train_Loss                                    = 0
last_Train_Loss0                                   = 0
last_Train_LCharge_err                             = 0
last_Train_LCharge_sum                             = 0
last_Train_Ltorsion_reg                            = 0
last_Test_Loss                                     = 0
last_Test_Loss0                                    = 0
last_Test_LCharge_err                              = 0
last_Test_LCharge_sum                              = 0
last_Test_Ltorsion_reg                             = 0

last1_Train_Loss                                   = 0
last1_Train_Loss0                                  = 0
last1_Train_LCharge_err                            = 0
last1_Train_LCharge_sum                            = 0
last1_Train_Ltorsion_reg                           = 0
last1_Test_Loss                                    = 0
last1_Test_Loss0                                   = 0
last1_Test_LCharge_err                             = 0
last1_Test_LCharge_sum                             = 0
last1_Test_Ltorsion_reg                            = 0


# In[ ]:


list_check_nan = ["train_loss" , "train_loss0" , "train_charge_reg" ,
                  "train_charge_sum" , "train_torsion_reg" ,
                  "test_loss"  , "test_loss0"  , "test_charge_reg" ,
                  "test_charge_sum" , "test_torsion_reg" ]


# In[ ]:


#　エラー終了した場合は出力ファイルに保存iteration番号を記録する
#　そのためのエラーフラグとエラー時の反復回数を記録
error_flag = False
error_iter = 0

# 学習の実行　学習回数は100000回

start = time.time()
sess = tf.Session(graph=fit)
sess.run(init)

out_iter       = int ( iteration / 10   )
loss_out_iter  = int ( iteration / 1000 )
param_out_iter = int ( iteration / 100  )
for i in range(iteration+1):

    #　最適化の実行
    _train , train_loss , train_loss0 , train_charge_reg , train_charge_sum , train_torsion_reg    = sess.run([train_step , loss , loss0 , charge_reg ,
                charge_sum_reg , torsion_reg] ,
               feed_dict=dict_train_feed)
    test_loss  , test_loss0  , test_charge_reg , test_charge_sum , test_torsion_reg     = sess.run([loss , loss0 , charge_reg , 
                charge_sum_reg , torsion_reg] ,
               feed_dict=dict_test_feed )
    
    #　Restart用に変数および各段階での計算結果を一時的に保存する
    #　各結果はエラーチェック前にLast～に保存
    #　エラーチェック後エラーがなければLast1～に保存する
    for atom in dict_charge:
        charge_temp = sess.run(locals()["%s_charge" % atom])
        locals()["last_%s_charge" % atom] = charge_temp
    for types in list_dihed_type:
        locals()["last_%s_v1" % types ] = sess.run(locals()["%s_v1" % types])
        locals()["last_%s_v2" % types ] = sess.run(locals()["%s_v2" % types])
        locals()["last_%s_v3" % types ] = sess.run(locals()["%s_v3" % types])
        locals()["last_%s_v4" % types ] = sess.run(locals()["%s_v4" % types])
        locals()["last_%s_v5" % types ] = sess.run(locals()["%s_v5" % types])
        locals()["last_%s_v6" % types ] = sess.run(locals()["%s_v6" % types])
        locals()["last_%s_g1" % types ] = sess.run(locals()["%s_g1" % types])
        locals()["last_%s_g2" % types ] = sess.run(locals()["%s_g2" % types])
        locals()["last_%s_g3" % types ] = sess.run(locals()["%s_g3" % types])
        locals()["last_%s_g4" % types ] = sess.run(locals()["%s_g4" % types])
        locals()["last_%s_g5" % types ] = sess.run(locals()["%s_g5" % types])
        locals()["last_%s_g6" % types ] = sess.run(locals()["%s_g6" % types])
    last_SCEE = sess.run(SCEE)
    last_Train_Loss = train_loss
    last_Train_Loss0 = train_loss0
    last_Train_LCharge_err = c_charge_reg * train_charge_reg
    last_Train_LCharge_sum = c_charge_sum * train_charge_sum
    last_Train_Ltorsion_reg = c_torsion * train_torsion_reg
    last_Test_Loss = test_loss
    last_Test_Loss0 = test_loss0
    last_Test_LCharge_err = c_charge_reg * test_charge_reg
    last_Test_LCharge_sum = c_charge_sum * test_charge_sum
    last_Test_Ltorsion_reg = c_torsion * test_torsion_reg
    
    for nan_num , value in enumerate([train_loss , train_loss0 , train_charge_reg ,
                                      train_charge_sum , train_torsion_reg ,
                                      test_loss  , test_loss0  , test_charge_reg ,
                                      test_charge_sum , test_torsion_reg ]):
        if np.isnan(value):
            print("%25s is NAN!!! Now Iteration is %6d" % (list_check_nan[nan_num],i))
            error_flag = True
            break
    else:
        time_stamp_filename = "time_stamp_" + file_date + ".log"
        
        for atom in dict_charge:
            charge_temp = sess.run(locals()["%s_charge" % atom])
            locals()["last1_%s_charge" % atom] = charge_temp
        for types in list_dihed_type:
            locals()["last1_%s_v1" % types ] = sess.run(locals()["%s_v1" % types])
            locals()["last1_%s_v2" % types ] = sess.run(locals()["%s_v2" % types])
            locals()["last1_%s_v3" % types ] = sess.run(locals()["%s_v3" % types])
            locals()["last1_%s_v4" % types ] = sess.run(locals()["%s_v4" % types])
            locals()["last1_%s_v5" % types ] = sess.run(locals()["%s_v5" % types])
            locals()["last1_%s_v6" % types ] = sess.run(locals()["%s_v6" % types])
            locals()["last1_%s_g1" % types ] = sess.run(locals()["%s_g1" % types])
            locals()["last1_%s_g2" % types ] = sess.run(locals()["%s_g2" % types])
            locals()["last1_%s_g3" % types ] = sess.run(locals()["%s_g3" % types])
            locals()["last1_%s_g4" % types ] = sess.run(locals()["%s_g4" % types])
            locals()["last1_%s_g5" % types ] = sess.run(locals()["%s_g5" % types])
            locals()["last1_%s_g6" % types ] = sess.run(locals()["%s_g6" % types])
        list1_SCEE = sess.run(SCEE)
        last1_Train_Loss = train_loss
        last1_Train_Loss0 = train_loss0
        last1_Train_LCharge_err = c_charge_reg * train_charge_reg
        last1_Train_LCharge_sum = c_charge_sum * train_charge_sum
        last1_Train_Ltorsion_reg = c_torsion * train_torsion_reg
        last1_Test_Loss = test_loss
        last1_Test_Loss0 = test_loss0
        last1_Test_LCharge_err = c_charge_reg * test_charge_reg
        last1_Test_LCharge_sum = c_charge_sum * test_charge_sum
        last1_Test_Ltorsion_reg = c_torsion * test_torsion_reg

        if i % param_out_iter == 0:
            error_iter = i
            for atom in dict_charge:
                charge_temp = sess.run(locals()["%s_charge" % atom])
                locals()["list_%s_charge" % atom].append(charge_temp)
            for types in list_dihed_type:
                locals()["list_%s_v1" % types ].append(sess.run(locals()["%s_v1" % types]))
                locals()["list_%s_v2" % types ].append(sess.run(locals()["%s_v2" % types]))
                locals()["list_%s_v3" % types ].append(sess.run(locals()["%s_v3" % types]))
                locals()["list_%s_v4" % types ].append(sess.run(locals()["%s_v4" % types]))
                locals()["list_%s_v5" % types ].append(sess.run(locals()["%s_v5" % types]))
                locals()["list_%s_v6" % types ].append(sess.run(locals()["%s_v6" % types]))
                locals()["list_%s_g1" % types ].append(sess.run(locals()["%s_g1" % types]))
                locals()["list_%s_g2" % types ].append(sess.run(locals()["%s_g2" % types]))
                locals()["list_%s_g3" % types ].append(sess.run(locals()["%s_g3" % types]))
                locals()["list_%s_g4" % types ].append(sess.run(locals()["%s_g4" % types]))
                locals()["list_%s_g5" % types ].append(sess.run(locals()["%s_g5" % types]))
                locals()["list_%s_g6" % types ].append(sess.run(locals()["%s_g6" % types]))
            list_SCEE.append(sess.run(SCEE))
        if i % loss_out_iter == 0:
            Train_Loss.append(train_loss)
            Train_Loss0.append(train_loss0)
            Train_LCharge_err.append(c_charge_reg * train_charge_reg)
            Train_LCharge_sum.append(c_charge_sum * train_charge_sum)
            Train_Ltorsion_reg.append(c_torsion * train_torsion_reg)
            Test_Loss.append(test_loss)
            Test_Loss0.append(test_loss0)
            Test_LCharge_err.append(c_charge_reg * test_charge_reg)
            Test_LCharge_sum.append(c_charge_sum * test_charge_sum)
            Test_Ltorsion_reg.append(c_torsion * test_torsion_reg)

        if i % out_iter == 0:
            print("iteration:: %8d , loss:: %8.3f , total_loss:: %8.3f" % (i,train_loss0,train_loss))
            with open(time_stamp_filename,"a") as timelog:
                print("iteration:: %8d , loss:: %8.3f , total_loss:: %8.3f" % (i,train_loss0,train_loss),file=timelog)
        continue
    break
else:  #　Iterationが全て正常に終了したら中間ファイルをすべて削除する
    for atom in dict_charge:
        del(locals()["last_%s_charge" % atom])
        del(locals()["last1_%s_charge" % atom])
    for types in list_dihed_type:
        del(locals()["last_%s_v1" % types ])
        del(locals()["last_%s_v2" % types ])
        del(locals()["last_%s_v3" % types ])
        del(locals()["last_%s_v4" % types ])
        del(locals()["last_%s_v5" % types ])
        del(locals()["last_%s_v6" % types ])
        del(locals()["last_%s_g1" % types ])
        del(locals()["last_%s_g2" % types ])
        del(locals()["last_%s_g3" % types ])
        del(locals()["last_%s_g4" % types ])
        del(locals()["last_%s_g5" % types ])
        del(locals()["last_%s_g6" % types ])
        del(locals()["last1_%s_v1" % types ])
        del(locals()["last1_%s_v2" % types ])
        del(locals()["last1_%s_v3" % types ])
        del(locals()["last1_%s_v4" % types ])
        del(locals()["last1_%s_v5" % types ])
        del(locals()["last1_%s_v6" % types ])
        del(locals()["last1_%s_g1" % types ])
        del(locals()["last1_%s_g2" % types ])
        del(locals()["last1_%s_g3" % types ])
        del(locals()["last1_%s_g4" % types ])
        del(locals()["last1_%s_g5" % types ])
        del(locals()["last1_%s_g6" % types ])
    del([last_SCEE,last1_SCEE,last_Train_Loss,last_Train_Loss0,
     last_Train_LCharge_err,last_Train_LCharge_sum,
     last_Train_Ltorsion_reg,
     last_Test_Loss,last_Test_Loss0,last_Test_LCharge_err,
     last_Test_LCharge_sum,last_Test_Ltorsion_reg,
     last1_Train_Loss,last1_Train_Loss0,last1_Train_LCharge_err,
     last1_Train_LCharge_sum,last1_Train_Ltorsion_reg,
     last1_Test_Loss,last1_Test_Loss0,
     last1_Test_LCharge_err,last1_Test_LCharge_sum,
     last1_Test_Ltorsion_reg])
        
end = time.time()
elapsed_hour,elapsed_min,elapsed_sec = 0. , 0. , 0.
elapsed_time = end - start
if elapsed_time >= 60:
    elapsed_min = elapsed_time // 60
    elapsed_sec = elapsed_time - elapsed_min * 60
if elapsed_min >= 60:
    elapsed_hour = elapsed_min // 60
    elapsed_min = elapsed_min - elapsed_hour * 60
if elapsed_hour >= 1:
    with open(time_stamp_filename,"a") as timelog:
        print("elapsed time //  %d hours  %d minutes  %d seconds" % (elapsed_hour,elapsed_min,elapsed_sec))
    print("elapsed time //  %d hours  %d minutes  %d seconds" % (elapsed_hour,elapsed_min,elapsed_sec))
elif elapsed_min >= 1:
    with open(time_stamp_filename,"a") as timelog:
        print("elapsed time // %d minutes  %d seconds" % (elapsed_min,elapsed_sec))
    print("elapsed time // %d minutes  %d seconds" % (elapsed_min,elapsed_sec))
else:
    with open(time_stamp_filename,"a") as timelog:
        print("elapsed time : %d seconds" % (elapsed_time))
    print("elapsed time : %d seconds" % (elapsed_time))
elapsed_time_per_iter = elapsed_time / iteration
print("elapsed time per iteration : %.3f"  % elapsed_time_per_iter )
with open(time_stamp_filename,"a") as timelog:
    print("elapsed time per iteration : %.3f"  % elapsed_time_per_iter )


# In[ ]:


#  各パラメータのフィッティング終了時の値の取得
#　異常終了した場合、パラメータ出力最終時点でのパラメータを .error.iter.という形式にして出力する
#　電荷
dict_charge_last = {}
for atom in dict_charge:
    dict_charge_last[atom] = locals()["list_%s_charge" % atom][-1]
df_charge = pd.DataFrame(index=dict_charge.keys())
df_charge["bf_fit"] = dict_charge.values()
df_charge["af_fit"] = dict_charge_last.values()
if error_flag:
    out_charge_csvname = "charge_after_fit." + file_date + ".error." + str(error_iter).zfill(6) + ".csv"
else:
    out_charge_csvname = "charge_after_fit." + file_date + ".csv"
df_charge.to_csv(out_charge_csvname)

#　二面角項
V1_opt , V2_opt , V3_opt , V4_opt , V5_opt , V6_opt = [] , [] , [] ,[] , [], []
gamma1_opt , gamma2_opt , gamma3_opt , gamma4_opt , gamma5_opt , gamma6_opt = [] , [] , [] ,[] , [], []
for types in df_params_torsion.index:
    V1_opt.append(     round( locals()["list_%s_v1" % types][-1],4) )
    V2_opt.append(     round( locals()["list_%s_v2" % types][-1],4) )
    V3_opt.append(     round( locals()["list_%s_v3" % types][-1],4) )
    V4_opt.append(     round( locals()["list_%s_v4" % types][-1],4) )
    V5_opt.append(     round( locals()["list_%s_v5" % types][-1],4) )
    V6_opt.append(     round( locals()["list_%s_v6" % types][-1],4) )
    gamma1_opt.append( round( locals()["list_%s_g1" % types][-1],4) )
    gamma2_opt.append( round( locals()["list_%s_g2" % types][-1],4) )
    gamma3_opt.append( round( locals()["list_%s_g3" % types][-1],4) )
    gamma4_opt.append( round( locals()["list_%s_g4" % types][-1],4) )
    gamma5_opt.append( round( locals()["list_%s_g5" % types][-1],4) )
    gamma6_opt.append( round( locals()["list_%s_g6" % types][-1],4) )
df_params_torsion["V1_opt"] = V1_opt
df_params_torsion["V2_opt"] = V2_opt
df_params_torsion["V3_opt"] = V3_opt
df_params_torsion["V4_opt"] = V4_opt
df_params_torsion["V5_opt"] = V5_opt
df_params_torsion["V6_opt"] = V6_opt
df_params_torsion["gamma1_opt"] = gamma1_opt
df_params_torsion["gamma2_opt"] = gamma2_opt
df_params_torsion["gamma3_opt"] = gamma3_opt
df_params_torsion["gamma4_opt"] = gamma4_opt
df_params_torsion["gamma5_opt"] = gamma5_opt
df_params_torsion["gamma6_opt"] = gamma6_opt
if error_flag:
    out_torsion_csvname = "torsion_after_fit." + file_date + ".error." + str(error_iter).zfill(6) + ".csv"
else:
    out_torsion_csvname = "torsion_after_fit." + file_date + ".csv"
df_params_torsion.to_csv(out_torsion_csvname)


#　1_4相互作用項のスケーリングファクター
if error_flag:
    SCEE_file = "scee." + file_date + ".error." + str(error_iter).zfill(6) + ".txt"
else:
    SCEE_file = "scee." + file_date + ".txt"
with open(SCEE_file, "w" ) as out:
    for index_num in range(len(list_SCEE)):
        scee = list_SCEE[index_num]
        print("SCEE : %9.7f" % (scee) , file=out)


# In[ ]:


if error_flag == True:
    print("Error Occured, Terminate this code!!!")
    exit


# In[ ]:


fig = plt.figure(figsize=(30,10))
ax = fig.add_subplot(1,3,1)
ax.plot(Train_Loss  , label="Train_Loss" ,color="blue")
ax.plot(Train_Loss0 , label="Train_Loss0",color="mediumblue")
ax.plot(Test_Loss   , label="Test_Loss"  ,color="red")
ax.plot(Test_Loss0  , label="Test_Loss0" ,color="orange")
ax.set_xlabel("Iteration")
ax.set_ylabel(r'Loss function [(kcal/mol)${2}$]')
ax.legend(fontsize=20)
ax.set_title("Loss function vs Iterations\nof Train and Test datasets" , fontsize=30)

ax = fig.add_subplot(1,3,2)
ax.plot(Train_Loss,            label="Loss")
ax.plot(Train_Loss0,           label="Loss0")
ax.plot(Train_LCharge_sum, label="charge_sum")
ax.plot(Train_LCharge_err,     label="charge_err")
ax.plot(Train_Ltorsion_reg,    label="torsion_term")
ax.set_xlabel("Iteration")
ax.set_ylabel(r'Components of the loss function [(kcal/mol)${2}$]')
ax.set_title("Components of the Loss function\nfor Train datasets" , fontsize=30)
ax.legend(fontsize=20)

ax = fig.add_subplot(1,3,3)
ax.plot(Test_Loss,            label="Loss")
ax.plot(Test_Loss0,           label="Loss0")
ax.plot(Test_LCharge_sum, label="charge_sum")
ax.plot(Test_LCharge_err,     label="charge_err")
ax.plot(Test_Ltorsion_reg,    label="torsion_term")
ax.set_xlabel("Iteration")
ax.set_ylabel(r'Components of the loss function [(kcal/mol)${2}$]')
ax.set_title("Components of the Loss function\nfor Test datasets" , fontsize=30)
ax.legend(fontsize=20)

loss_filename = "loss_component." + file_date + ".png"
plt.savefig(loss_filename,format="png")
plt.show()


# In[ ]:


#  20000構造の各原子のRESP電荷の分布から近似した正規分布と
#  GLYCAM力場および最適化パラメータの電荷の確認
fig = plt.figure(figsize=(20,20))
for num,atom_i in enumerate(dict_charge):
    ax = fig.add_subplot(4,5,num+1)
    x_lin = np.linspace(-1,1,1000)
    atom_i_mean , atom_i_std = df_charge_distribution.loc[atom_i,"Mean"] , df_charge_distribution.loc[atom_i,"Std"]
    pdf_x = np.exp( - (x_lin - atom_i_mean) ** 2 / ( 2 * atom_i_std ** 2 )) / np.sqrt( 2 * np.pi ) * atom_i_std
    ax.plot(x_lin,pdf_x)
    ax.vlines(df_charge.iloc[num,1],ymin=pdf_x.min(),ymax=pdf_x.max(),color="red",label="after_opt")
    ax.set_title(atom_i)
    ax.vlines(df_charge.iloc[num,0],ymin=pdf_x.min(),ymax=pdf_x.max(),color="green",label="before_opt")
    ax.set_xlim(atom_i_mean - 4. * atom_i_std , atom_i_mean + 4. * atom_i_std)
    ax.legend()
out_charge_distname = "charge_distribution_after_fit." + file_date + ".png"
plt.savefig(out_charge_distname,format="png")
plt.show()


# In[ ]:


#　入力の距離情報がベクトルにも対応した静電相互相互作用関数の作成
vfunc_eel   = np.vectorize(Electrostatic)
vfunc_eel14 = np.vectorize(Electrostatic14)
#　入力の距離情報がベクトルにも対応した二面角エネルギー関数の作成
vfunc_tor   = np.vectorize(Etorsion_var)


# In[ ]:


time_0 = time.time()
#　フィッティング後のパラメータを用いた構造の相対エネルギーデータの作成
df_delta_edat_af_fit = pd.DataFrame()
df_delta_edat_af_fit["Dedft"] = df_delta_edat_bf_fit["Dedft"]
df_delta_edat_af_fit["Demm_invar"] = df_delta_edat_bf_fit["Demm"] - df_delta_edat_bf_fit["Demm_var"]
df_delta_edat_af_fit["Demm_bf"] = df_delta_edat_bf_fit["Demm"]

#　距離データの引継ぎ
for nb_pair in list_R_NB_eel:
    df_delta_edat_af_fit[nb_pair] = df_delta_edat_bf_fit[nb_pair]
for i14_pair in list_R_i14_eel:
    df_delta_edat_af_fit[i14_pair] = df_delta_edat_bf_fit[i14_pair]
#　二面角データの引継ぎ
for i in range(dihed_num):
    torsion_name = "Tor_%s_%s" % (dict_dihed_atoms_types[list_dihed[i]],list_dihed[i])
    df_delta_edat_af_fit[torsion_name] = df_delta_edat_bf_fit[torsion_name]
time_1 = time.time()
print("time for getting distance_dihed_data %6.2f" % (time_1 - time_0 ))

#　フィッティング後の電荷パラメータを用いて静電相互作用エネルギーの計算
for i in list_R_NB_eel:
    atom_i = i.split("_")[1]
    if atom_i in dict_charge_last:
        charge_i = dict_charge_last[atom_i]
    else:
        charge_i = 0.
    atom_j = i.split("_")[2]
    if atom_j in dict_charge_last:
        charge_j = dict_charge_last[atom_j]
    else:
        charge_j = 0.
    column_name = "Denb_eel_" + i[2:]
    r_vec = df_delta_edat_af_fit[i]
    df_delta_edat_af_fit[column_name] = (vfunc_eel(charge_i,charge_j,r_vec)
                                      - float(Electrostatic(charge_i,charge_j,df_delta_edat_bf_fit.loc[ref_index,i]))
                                     )
time_2 = time.time()
print("time for calc nb eel %6.2f" % (time_2 - time_1))
    
#　フィッティング後の電荷パラメータを用いて14静電相互作用エネルギーの計算
for i in list_R_i14_eel:
    atom_i = i.split("_")[1]
    if atom_i in dict_charge_last:
        charge_i = dict_charge_last[atom_i]
    else:
        charge_i = 0.
    atom_j = i.split("_")[2]
    if atom_j in dict_charge_last:
        charge_j = dict_charge_last[atom_j]
    else:
        charge_j = 0.
    column_name = "De14_eel_" + i[4:]
    r_vec = df_delta_edat_af_fit[i]
    df_delta_edat_af_fit[column_name] = ( vfunc_eel14(charge_i,charge_j,r_vec,list_SCEE[-1])
                                      - float(Electrostatic14(charge_i,charge_j,
                                                              df_delta_edat_bf_fit.loc[ref_index,i],scee=list_SCEE[-1]))
                                     )
time_3 = time.time()
print("time for calc 14eel %6.2f" % (time_3 - time_2))

#　フィッティング後の二面角パラメータを用いてねじれ項のエネルギーを計算
for i in range(dihed_num):
    torsion_name = "Tor_%s_%s" % (dict_dihed_atoms_types[list_dihed[i]],list_dihed[i])
    Delta_E_torsion_name = "Detor_%s_%s" % (dict_dihed_atoms_types[list_dihed[i]],list_dihed[i])
    torsion_type = dict_dihed_atoms_types[list_dihed[i]]
    v1_     = df_params_torsion.loc[torsion_type,"V1_opt"]
    v2_     = df_params_torsion.loc[torsion_type,"V2_opt"]
    v3_     = df_params_torsion.loc[torsion_type,"V3_opt"]
    v4_     = df_params_torsion.loc[torsion_type,"V4_opt"]
    v5_     = df_params_torsion.loc[torsion_type,"V5_opt"]
    v6_     = df_params_torsion.loc[torsion_type,"V6_opt"]
    gamma1_ = df_params_torsion.loc[torsion_type,"gamma1_opt"]
    gamma2_ = df_params_torsion.loc[torsion_type,"gamma2_opt"]
    gamma3_ = df_params_torsion.loc[torsion_type,"gamma3_opt"]
    gamma4_ = df_params_torsion.loc[torsion_type,"gamma4_opt"]
    gamma5_ = df_params_torsion.loc[torsion_type,"gamma5_opt"]
    gamma6_ = df_params_torsion.loc[torsion_type,"gamma6_opt"]
    
    df_delta_edat_af_fit[Delta_E_torsion_name] = (  vfunc_tor(v1_,v2_,v3_,v4_,v5_,v6_,
                                                              gamma1_,gamma2_,gamma3_,gamma4_,gamma5_,gamma6_,
                                                              df_delta_edat_bf_fit[torsion_name])
                                               - Etorsion_var(v1_,v2_,v3_,v4_,v5_,v6_,
                                                              gamma1_,gamma2_,gamma3_,gamma4_,gamma5_,gamma6_,
                                                              float(df_delta_edat_bf_fit.loc[ref_index,torsion_name]))
                                              )
time_4 = time.time()
print("time for calc dihed %6.2f" % (time_4 - time_3))
    
# y_target はEdftとEmmからフィッティング変数に関係のない項の和なのでフィッティング前の値をそのまま使用
# Devar_af　はフィッティング後のパラメータで計算した、NB_eel , 1_4_eel , NB_vdw , 1_4_eel , E_torsion の和なので
# For文を回して計算
df_delta_edat_af_fit["y_target"] = df_delta_edat_bf_fit["y_target"]
df_delta_edat_af_fit["Demm_var_bf"] = df_delta_edat_bf_fit["Demm_var"]   
df_delta_edat_af_fit["Demm_eel_sum"] = 0.
df_delta_edat_af_fit["Demm_tor_sum"] = 0.
for nb_pair in list_R_NB_eel:
    column_name = "Denb_eel_" + nb_pair[2:]
    df_delta_edat_af_fit["Demm_eel_sum"] += df_delta_edat_af_fit[column_name]
for i14_pair in list_R_i14_eel:
    column_name = "De14_eel_" + i14_pair[4:]
    df_delta_edat_af_fit["Demm_eel_sum"] += df_delta_edat_af_fit[column_name]
for i in range(dihed_num):
    Delta_E_torsion_name = "Detor_%s_%s" % (dict_dihed_atoms_types[list_dihed[i]],list_dihed[i])
    df_delta_edat_af_fit["Demm_tor_sum"] += df_delta_edat_af_fit[Delta_E_torsion_name]
df_delta_edat_af_fit["Demm_var_af"] = (  df_delta_edat_af_fit["Demm_eel_sum"]
                                    + df_delta_edat_af_fit["Demm_tor_sum"])
df_delta_edat_af_fit["Demm_af"] = (df_delta_edat_af_fit["Dedft"] - df_delta_edat_af_fit["y_target"]
                                   + df_delta_edat_af_fit["Demm_var_af"])
df_delta_edat_af_fit["DDE_af"]       = df_delta_edat_af_fit["Dedft"]       - df_delta_edat_af_fit["Demm_af"]
df_delta_edat_af_fit["DDE_bf"]       = df_delta_edat_af_fit["Dedft"]       - df_delta_edat_af_fit["Demm_bf"]
time_5 = time.time()
print("time for calc result %6.2f" % (time_5 - time_4))


# In[ ]:


out_energy_profile_name = "delta_energy_profile_fit_edft_af." + file_date + ".csv"
df_delta_edat_af_fit.to_csv(out_energy_profile_name)
df_delta_edat_af_fit_train = df_delta_edat_af_fit.loc[train_index , :]
df_delta_edat_af_fit_train = df_delta_edat_af_fit_train.sort_index()
df_delta_edat_af_fit_test = df_delta_edat_af_fit.loc[test_index , :]
df_delta_edat_af_fit_test = df_delta_edat_af_fit_test.sort_index()


# In[ ]:


# 損失関数とΔEdft-ΔEmm　のパラメータフィッティング前後のヒストグラムの表示
# 上が最適化の際に用いた損失関数、下がΔΔE
# 左側がフィッティング前、右側がフィッティング後
# 損失関数とΔΔEが等しいことが確認でき、
# またフィッティング前後でわずかながら0に近づいてることが確認できる。
# 損失関数のグラフを見ると、二乗和の平均がフィッティング前は平均約8,5、
# フィッティング後は平均約5ほどでヒストグラムとも対応している。
# 損失関数として二乗和を使用しているが、これは誤差の分布が正規分布に近いことに由来

fig = plt.figure(figsize=(25,35))
ax = fig.add_subplot(3,2,1)
ax.hist(df_delta_edat_af_fit["Dedft"] - df_delta_edat_af_fit["Demm_bf"],bins=30)
avg = (df_delta_edat_af_fit["Dedft"] - df_delta_edat_af_fit["Demm_bf"]).mean()
std = (df_delta_edat_af_fit["Dedft"] - df_delta_edat_af_fit["Demm_bf"]).std()
title = 'Before fit\ndelta_delta_E_mean    %.3f\ndelta_delta_E_std    %.3f' % (avg,std)
ax.set_title(title,fontsize=30)
ax.set_xlabel(r'$\Delta \Delta$E [kcal/mol]',fontsize=25)
ax.set_ylabel(r'Frequency of Samples',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax = fig.add_subplot(3,2,2)
ax.hist(df_delta_edat_af_fit["Dedft"] - df_delta_edat_af_fit["Demm_af"],bins=30)
avg = (df_delta_edat_af_fit["Dedft"] - df_delta_edat_af_fit["Demm_af"]).mean()
std = (df_delta_edat_af_fit["Dedft"] - df_delta_edat_af_fit["Demm_af"]).std()
title = "After fit\ndelta_delta_E_mean    %.3f\ndelta_delta_E_std    %.3f" % (avg,std)
ax.set_title(title,fontsize=30)
ax.set_xlabel(r'$\Delta \Delta$E [kcal/mol]',fontsize=25)
ax.set_ylabel(r'Frequency of Samples',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax = fig.add_subplot(3,2,3)
ax.hist(df_delta_edat_af_fit_train["Dedft"] - df_delta_edat_af_fit_train["Demm_bf"],bins=30)
avg = (df_delta_edat_af_fit_train["Dedft"] - df_delta_edat_af_fit_train["Demm_bf"]).mean()
std = (df_delta_edat_af_fit_train["Dedft"] - df_delta_edat_af_fit_train["Demm_bf"]).std()
title = "Before fit of Training Data\ndelta_delta_E_mean    %.3f\ndelta_delta_E_std    %.3f" % (avg,std)
ax.set_title(title,fontsize=30)
ax.set_xlabel(r'$\Delta \Delta$E [kcal/mol]',fontsize=25)
ax.set_ylabel(r'Frequency of Samples',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax = fig.add_subplot(3,2,4)
ax.hist(df_delta_edat_af_fit_train["Dedft"] - df_delta_edat_af_fit_train["Demm_af"],bins=30)
avg = (df_delta_edat_af_fit_train["Dedft"] - df_delta_edat_af_fit_train["Demm_af"]).mean()
std = (df_delta_edat_af_fit_train["Dedft"] - df_delta_edat_af_fit_train["Demm_af"]).std()
title = "After fit of Training Data\ndelta_delta_E_mean    %.3f\ndelta_delta_E_std    %.3f" % (avg,std)
ax.set_title(title,fontsize=30)
ax.set_xlabel(r'$\Delta \Delta$E [kcal/mol]',fontsize=25)
ax.set_ylabel(r'Frequency of Samples',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax = fig.add_subplot(3,2,5)
ax.hist(df_delta_edat_af_fit_test["Dedft"] - df_delta_edat_af_fit_test["Demm_bf"],bins=30)
avg = (df_delta_edat_af_fit_test["Dedft"] - df_delta_edat_af_fit_test["Demm_bf"]).mean()
std = (df_delta_edat_af_fit_test["Dedft"] - df_delta_edat_af_fit_test["Demm_bf"]).std()
title = "Before fit of Test Data\ndelta_delta_E_mean    %.3f\ndelta_delta_E_std    %.3f" % (avg,std)
ax.set_title(title,fontsize=30)
ax.set_xlabel(r'$\Delta \Delta$E [kcal/mol]',fontsize=25)
ax.set_ylabel(r'Frequency of Samples',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax = fig.add_subplot(3,2,6)
ax.hist(df_delta_edat_af_fit_test["Dedft"] - df_delta_edat_af_fit_test["Demm_af"],bins=30)
avg = (df_delta_edat_af_fit_test["Dedft"] - df_delta_edat_af_fit_test["Demm_af"]).mean()
std = (df_delta_edat_af_fit_test["Dedft"] - df_delta_edat_af_fit_test["Demm_af"]).std()
title = "After fit of Test Data\ndelta_delta_E_mean    %.3f\ndelta_delta_E_std    %.3f" % (avg,std)
ax.set_title(title,fontsize=30)
ax.set_xlabel(r'$\Delta \Delta$E [kcal/mol]',fontsize=25)
ax.set_ylabel(r'Frequency of Samples',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.subplots_adjust(hspace=0.4,wspace=0.2)
hist_filename = "delta_delta_E_hist." + file_date + ".png"
plt.savefig(hist_filename,format="png",bbox_inches="tight")
plt.show()


# In[ ]:


#　続いて損失関数を最小にする二面角パラメータを用いてMMエネルギーを計算した結果を示す。
#　散布図は横軸にΔEdft、縦軸にΔEmmを取り、参照構造はEdftが最小であった164番目の構造である。
#　（補足：Pythonで表示されるIndexは0からなので構造が1つずれている）
#　エネルギーの単位はともに　kcal/mol
#　散布図の左側がフィッティング前、すなわちGLYCAM06jの二面角パラメータを用いたもので
#　散布図の右側がフィッティング後の二面角パラメータを用いたものである。
#　グラフの上に線形近似曲線と相関係数をそれぞれ示している。
#　

D_dft_min = df_delta_edat_bf_fit["Dedft"].min()
D_dft_max = df_delta_edat_bf_fit["Dedft"].max()
fig = plt.figure(figsize=(25,35))
x_approximate = np.linspace(D_dft_min,D_dft_max,1000)
ax = fig.add_subplot(3,2,1)
x    = df_delta_edat_af_fit["Dedft"]
y_bf = df_delta_edat_af_fit["Demm_bf"]
corrcoef_bf = np.corrcoef(x,y_bf)
a_bf, b_bf = np.polyfit(x, y_bf, 1)
# フィッティング直線
y_approximate_bf = a_bf * x_approximate + b_bf
if b_bf >= 0:
    approx_sec_flag = "+"
else:
    approx_sec_flag = "-"
approx_txt = "y = %5.2fx %s %.2f" % (a_bf,approx_sec_flag,abs(b_bf))
ax.plot(x_approximate,y_approximate_bf,c="red",label=approx_txt,lw=3)
ax.plot(x_approximate,x_approximate,c="green",label=r'$\Delta E_{QM} = \Delta E_{MM}$',lw=3)
title_txt = "Before fit of All Data\ncorrelation coefficients::%5.2f" % corrcoef_bf[0,1]
ax.set_title(title_txt,fontsize=30)
ax.set_xlabel(r'$\Delta E_{QM}$',fontsize=25)
ax.set_ylabel(r'$\Delta E_{MM}$',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.legend(fontsize=25)
ax.scatter(x,y_bf,s=8)
ax.set_xlim(D_dft_min,D_dft_max)
#ax.set_ylim(-5,32)

ax = fig.add_subplot(3,2,2)
x    = df_delta_edat_af_fit["Dedft"]
y_af = df_delta_edat_af_fit["Demm_af"]
corrcoef_af = np.corrcoef(x,y_af)
a_af, b_af = np.polyfit(x, y_af, 1)
# フィッティング直線
y_approximate_af = a_af * x_approximate + b_af
if b_af >= 0:
    approx_sec_flag = "+"
else:
    approx_sec_flag = "-"
approx_txt = "y = %5.2fx %s %.2f" % (a_af,approx_sec_flag,abs(b_af))
ax.plot(x_approximate,y_approximate_af,c="red",label=approx_txt,lw=3)
ax.plot(x_approximate,x_approximate,c="green",label=r'$\Delta E_{QM} = \Delta E_{MM}$',lw=3)
title_txt = "After fit of All Data\ncorrelation coefficients::%5.2f" % corrcoef_af[0,1]
ax.set_title(title_txt,fontsize=30)
ax.set_xlabel(r'$\Delta E_{QM}$',fontsize=25)
ax.set_ylabel(r'$\Delta E_{MM}$',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.legend(fontsize=25)
ax.scatter(x,y_af,s=8)
ax.set_xlim(D_dft_min,D_dft_max)
#ax.set_ylim(-5,32)

ax = fig.add_subplot(3,2,3)
x    = df_delta_edat_af_fit_train["Dedft"]
y_bf = df_delta_edat_af_fit_train["Demm_bf"]
corrcoef_bf = np.corrcoef(x,y_bf)
a_bf, b_bf = np.polyfit(x, y_bf, 1)
# フィッティング直線
y_approximate_bf = a_bf * x_approximate + b_bf
if b_bf >= 0:
    approx_sec_flag = "+"
else:
    approx_sec_flag = "-"
approx_txt = "y = %5.2fx %s %.2f" % (a_bf,approx_sec_flag,abs(b_bf))
ax.plot(x_approximate,y_approximate_bf,c="red",label=approx_txt,lw=3)
ax.plot(x_approximate,x_approximate,c="green",label=r'$\Delta E_{QM} = \Delta E_{MM}$',lw=3)
title_txt = "Before fit of Train Data\ncorrelation coefficients::%5.2f" % corrcoef_bf[0,1]
ax.set_title(title_txt,fontsize=30)
ax.set_xlabel(r'$\Delta E_{QM}$',fontsize=25)
ax.set_ylabel(r'$\Delta E_{MM}$',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.legend(fontsize=25)
ax.scatter(x,y_bf,s=8)
ax.set_xlim(D_dft_min,D_dft_max)

ax = fig.add_subplot(3,2,4)
x    = df_delta_edat_af_fit_train["Dedft"]
y_af = df_delta_edat_af_fit_train["Demm_af"]
corrcoef_af = np.corrcoef(x,y_af)
a_af, b_af = np.polyfit(x, y_af, 1)
# フィッティング直線
y_approximate_af = a_af * x_approximate + b_af
if b_af >= 0:
    approx_sec_flag = "+"
else:
    approx_sec_flag = "-"
approx_txt = "y = %5.2fx %s %.2f" % (a_af,approx_sec_flag,abs(b_af))
ax.plot(x_approximate,y_approximate_af,c="red",label=approx_txt,lw=3)
ax.plot(x_approximate,x_approximate,c="green",label=r'$\Delta E_{QM} = \Delta E_{MM}$',lw=3)
title_txt = "After fit of Train Data\ncorrelation coefficients::%5.2f" % corrcoef_af[0,1]
ax.set_title(title_txt,fontsize=30)
ax.set_xlabel(r'$\Delta E_{QM}$',fontsize=25)
ax.set_ylabel(r'$\Delta E_{MM}$',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.legend(fontsize=25)
ax.scatter(x,y_af,s=8)
ax.set_xlim(D_dft_min,D_dft_max)

ax = fig.add_subplot(3,2,5)
x    = df_delta_edat_af_fit_test["Dedft"]
y_bf = df_delta_edat_af_fit_test["Demm_bf"]
corrcoef_bf = np.corrcoef(x,y_bf)
a_bf, b_bf = np.polyfit(x, y_bf, 1)
# フィッティング直線
y_approximate_bf = a_bf * x_approximate + b_bf
if b_bf >= 0:
    approx_sec_flag = "+"
else:
    approx_sec_flag = "-"
approx_txt = "y = %5.2fx %s %.2f" % (a_bf,approx_sec_flag,abs(b_bf))
ax.plot(x_approximate,y_approximate_bf,c="red",label=approx_txt,lw=3)
ax.plot(x_approximate,x_approximate,c="green",label=r'$\Delta E_{QM} = \Delta E_{MM}$',lw=3)
title_txt = "Before fit of Test Data\ncorrelation coefficients::%5.2f" % corrcoef_bf[0,1]
ax.set_title(title_txt,fontsize=30)
ax.set_xlabel(r'$\Delta E_{QM}$',fontsize=25)
ax.set_ylabel(r'$\Delta E_{MM}$',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.legend(fontsize=25)
ax.scatter(x,y_bf,s=8)
ax.set_xlim(D_dft_min,D_dft_max)

ax = fig.add_subplot(3,2,6)
x    = df_delta_edat_af_fit_test["Dedft"]
y_af = df_delta_edat_af_fit_test["Demm_af"]
corrcoef_af = np.corrcoef(x,y_af)
a_af, b_af = np.polyfit(x, y_af, 1)
# フィッティング直線
y_approximate_af = a_af * x_approximate + b_af
if b_af >= 0:
    approx_sec_flag = "+"
else:
    approx_sec_flag = "-"
approx_txt = "y = %5.2fx %s %.2f" % (a_af,approx_sec_flag,abs(b_af))
ax.plot(x_approximate,y_approximate_af,c="red",label=approx_txt,lw=3)
ax.plot(x_approximate,x_approximate,c="green",label=r'$\Delta E_{QM} = \Delta E_{MM}$',lw=3)
title_txt = "After fit of Test Data\ncorrelation coefficients::%5.2f" % corrcoef_af[0,1]
ax.set_title(title_txt,fontsize=30)
ax.set_xlabel(r'$\Delta E_{QM}$',fontsize=25)
ax.set_ylabel(r'$\Delta E_{MM}$',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.legend(fontsize=25)
ax.scatter(x,y_af,s=8)
ax.set_xlim(D_dft_min,D_dft_max)

plt.subplots_adjust(wspace=0.2,hspace=0.3)
deltaE_plot_filename = "deltaE_qm_vs_deltaE_mm_after_fit." + file_date + ".png"
plt.savefig(deltaE_plot_filename,format="png",bbox_inches="tight")
plt.show()


# In[ ]:


fig = plt.figure(figsize=(20,30))
x_lin = np.pi * np.linspace(-1,1,200)
vfunc = np.vectorize(Etorsion_var)
for num,types in enumerate(list_dihed_type):
    ax = fig.add_subplot(4,5,num+1)
    v1_f , v1_l = locals()["list_%s_v1" % types][0] , locals()["list_%s_v1" % types][-1]
    v2_f , v2_l = locals()["list_%s_v2" % types][0] , locals()["list_%s_v2" % types][-1]
    v3_f , v3_l = locals()["list_%s_v3" % types][0] , locals()["list_%s_v3" % types][-1]
    v4_f , v4_l = locals()["list_%s_v4" % types][0] , locals()["list_%s_v4" % types][-1]
    v5_f , v5_l = locals()["list_%s_v5" % types][0] , locals()["list_%s_v5" % types][-1]
    v6_f , v6_l = locals()["list_%s_v6" % types][0] , locals()["list_%s_v6" % types][-1]
    g1_f , g1_l = locals()["list_%s_g1" % types][0] , locals()["list_%s_g1" % types][-1]
    g2_f , g2_l = locals()["list_%s_g2" % types][0] , locals()["list_%s_g2" % types][-1]
    g3_f , g3_l = locals()["list_%s_g3" % types][0] , locals()["list_%s_g3" % types][-1]
    g4_f , g4_l = locals()["list_%s_g4" % types][0] , locals()["list_%s_g4" % types][-1]
    g5_f , g5_l = locals()["list_%s_g5" % types][0] , locals()["list_%s_g5" % types][-1]
    g6_f , g6_l = locals()["list_%s_g6" % types][0] , locals()["list_%s_g6" % types][-1]
    y_before = vfunc(v1_f,v2_f,v3_f,v4_f,v5_f,v6_f,g1_f,g2_f,g3_f,g4_f,g5_f,g6_f,x_lin)
    y_after  = vfunc(v1_l,v2_l,v3_l,v4_l,v5_l,v6_l,g1_l,g2_l,g3_l,g4_l,g5_l,g6_l,x_lin)
    ax.plot(x_lin*180/np.pi,y_before,label="before",c="red")
    ax.plot(x_lin*180/np.pi,y_after,label="after",c="blue")
    ax.set_title(types)
    ax.legend()
filename = "torsion_energy_after_fit." + file_date + ".png" 
plt.savefig(filename,format="png",bbox_inches="tight")
plt.show()


# In[ ]:


fig = plt.figure(figsize=(40,90))
x_lin = np.pi * np.linspace(-1,1,200)
list_index = list(sorted(set(dict_dihed_atoms_types.values())))
columns_num_dict = {}
for i in list_index:
    columns_num_dict[i] = 0
for i in range(dihed_num):
    torsion_name = list_dihed[i]
    torsion_type = dict_dihed_atoms_types[list_dihed[i]]
    target_tor = "Tor_%s_%s" % (dict_dihed_atoms_types[list_dihed[i]],list_dihed[i])
    index_num = list_index.index(torsion_type)
    columns_num_dict[torsion_type] += 1
    plot_num = index_num * 8 + columns_num_dict[torsion_type]
    target_Etor = "Detor_%s_%s" % (dict_dihed_atoms_types[list_dihed[i]],list_dihed[i])
    locals()["dihed_%s_data" % list_dihed[i]] = np.array(df_delta_edat_bf_fit[target_tor])
    locals()["dihed_%s_ref" % list_dihed[i]] = float(df_delta_edat_bf_fit.loc[ref_index,target_tor])
    ax = fig.add_subplot(20,8,plot_num)
    n, bins, pathces = ax.hist(locals()["dihed_%s_data" % list_dihed[i]]*180/np.pi,bins=30,density=False)
    y_max = n.max()
    y_min = n.min()
    #print(locals()["dihed_%s_ref" % list_dihed[i]]*180/np.pi)
    ax.vlines(locals()["dihed_%s_ref" % list_dihed[i]]*180/np.pi,ymin=y_min,ymax=y_max,color="orange")
    v1_f , v1_l = locals()["list_%s_v1" % torsion_type][0] , locals()["list_%s_v1" % torsion_type][-1]
    v2_f , v2_l = locals()["list_%s_v2" % torsion_type][0] , locals()["list_%s_v2" % torsion_type][-1]
    v3_f , v3_l = locals()["list_%s_v3" % torsion_type][0] , locals()["list_%s_v3" % torsion_type][-1]
    v4_f , v4_l = locals()["list_%s_v4" % torsion_type][0] , locals()["list_%s_v4" % torsion_type][-1]
    v5_f , v5_l = locals()["list_%s_v5" % torsion_type][0] , locals()["list_%s_v5" % torsion_type][-1]
    v6_f , v6_l = locals()["list_%s_v6" % torsion_type][0] , locals()["list_%s_v6" % torsion_type][-1]
    g1_f , g1_l = locals()["list_%s_g1" % torsion_type][0] , locals()["list_%s_g1" % torsion_type][-1]
    g2_f , g2_l = locals()["list_%s_g2" % torsion_type][0] , locals()["list_%s_g2" % torsion_type][-1]
    g3_f , g3_l = locals()["list_%s_g3" % torsion_type][0] , locals()["list_%s_g3" % torsion_type][-1]
    g4_f , g4_l = locals()["list_%s_g4" % torsion_type][0] , locals()["list_%s_g4" % torsion_type][-1]
    g5_f , g5_l = locals()["list_%s_g5" % torsion_type][0] , locals()["list_%s_g5" % torsion_type][-1]
    g6_f , g6_l = locals()["list_%s_g6" % torsion_type][0] , locals()["list_%s_g6" % torsion_type][-1]
    y_before = vfunc(v1_f,v2_f,v3_f,v4_f,v5_f,v6_f,g1_f,g2_f,g3_f,g4_f,g5_f,g6_f,x_lin)
    y_after  = vfunc(v1_l,v2_l,v3_l,v4_l,v5_l,v6_l,g1_l,g2_l,g3_l,g4_l,g5_l,g6_l,x_lin)
    ax2 = ax.twinx()
    ax2.plot(x_lin*180/np.pi,y_before,label="before",c="red")
    ax2.plot(x_lin*180/np.pi,y_after,label="after",c="blue")
    ax2.legend()
    ax.set_title("%s // %s" % (torsion_name,torsion_type))
fig.subplots_adjust(wspace=0.35,hspace=0.3)
hist_filename = "Each_torsion_hist_and_torsion_function" + file_date + ".png"
plt.savefig(hist_filename,bbox_inches="tight")
plt.show()


# In[ ]:


file_bf_fit = "../../input_data_opt_reduce_datasets/0GB_energy_profile_dft_opt_reduce_datasets.csv"
df_energy_profile_bf = pd.read_csv(file_bf_fit,index_col=0)


# In[ ]:


#　フィッティング後の電荷パラメータを用いた構造のエネルギーデータの作成
df_energy_profile_af_fit_calc           = pd.DataFrame()
df_energy_profile_af_fit_calc["Edft"]   = df_energy_profile_bf["Edft"]
df_energy_profile_af_fit_calc["Emm_bf"] = df_energy_profile_bf["Emm"]

#　距離データの引継ぎ
for nb_pair in list_R_NB_eel:
    df_energy_profile_af_fit_calc[nb_pair]      = df_energy_profile_bf[nb_pair]
for i14_pair in list_R_i14_eel:
    df_energy_profile_af_fit_calc[i14_pair]     = df_energy_profile_bf[i14_pair]
#　二面角データの引継ぎ
for i in range(dihed_num):
    torsion_name = "Tor_%s_%s" % (dict_dihed_atoms_types[list_dihed[i]],list_dihed[i])
    df_energy_profile_af_fit_calc[torsion_name] = df_energy_profile_bf[torsion_name]

#　フィッティング後の電荷パラメータを用いて非結合静電相互作用エネルギーの計算
for nb_eel_pair in list_R_NB_eel:
    atom_i      = nb_eel_pair.split("_")[1]
    charge_i    = dict_charge_last[atom_i]
    atom_j      = nb_eel_pair.split("_")[2]
    charge_j    = dict_charge_last[atom_j]
    column_name = "Enb_eel_" + nb_eel_pair[2:]
    np_r_vec       = df_energy_profile_af_fit_calc[nb_eel_pair]
    df_energy_profile_af_fit_calc[column_name] = vfunc_eel(charge_i,charge_j,np_r_vec)
    
#　フィッティング後の電荷パラメータを用いて14静電相互作用エネルギーの計算
for i14_eel_pair in list_R_i14_eel:
    atom_i      = i14_eel_pair.split("_")[1]
    charge_i    = dict_charge_last[atom_i]
    atom_j      = i14_eel_pair.split("_")[2]
    charge_j    = dict_charge_last[atom_j]
    column_name = "E14_eel_" + i14_eel_pair[4:]
    np_r_vec       = df_energy_profile_af_fit_calc[i14_eel_pair]
    df_energy_profile_af_fit_calc[column_name] = vfunc_eel14(charge_i,charge_j,np_r_vec,scee)
    
#　フィッティング後の二面角パラメータを用いてねじれ項のエネルギーを計算
for dihed_pair in range(dihed_num):
    torsion_name   = "Tor_%s_%s"  % (dict_dihed_atoms_types[list_dihed[dihed_pair]],list_dihed[dihed_pair])
    E_torsion_name = "Etor_%s_%s" % (dict_dihed_atoms_types[list_dihed[dihed_pair]],list_dihed[dihed_pair])
    torsion_type = dict_dihed_atoms_types[list_dihed[dihed_pair]]
    v1_     = df_params_torsion.loc[torsion_type,"V1_opt"]
    v2_     = df_params_torsion.loc[torsion_type,"V2_opt"]
    v3_     = df_params_torsion.loc[torsion_type,"V3_opt"]
    v4_     = df_params_torsion.loc[torsion_type,"V4_opt"]
    v5_     = df_params_torsion.loc[torsion_type,"V5_opt"]
    v6_     = df_params_torsion.loc[torsion_type,"V6_opt"]
    gamma1_ = df_params_torsion.loc[torsion_type,"gamma1_opt"]
    gamma2_ = df_params_torsion.loc[torsion_type,"gamma2_opt"]
    gamma3_ = df_params_torsion.loc[torsion_type,"gamma3_opt"]
    gamma4_ = df_params_torsion.loc[torsion_type,"gamma4_opt"]
    gamma5_ = df_params_torsion.loc[torsion_type,"gamma5_opt"]
    gamma6_ = df_params_torsion.loc[torsion_type,"gamma6_opt"]
    np_tonp_r_vec = df_energy_profile_af_fit_calc[torsion_name]
    df_energy_profile_af_fit_calc[E_torsion_name] = vfunc_tor(v1_,v2_,v3_,v4_,v5_,v6_,
                                                              gamma1_,gamma2_,gamma3_,gamma4_,gamma5_,gamma6_,
                                                              np_tonp_r_vec)
    
#　上で計算した、静電相互作用、VDW相互作用、二面角エネルギーを合計し
#　パラメータフィッティングの際に変化する項として Emm_var_af　とする
#　パラメータフィッティングの際に変化しない項 Emm_invar を
#　フィッティング前のエネルギーから Emm_invar = Emm_bf - Emm_var_bf により計算し
#　最後に　Emm_af = Emm_invar + Emm_var_af を計算してフィッティング後のエネルギーとする

np_emm_eel_sum = np.zeros((len(df_energy_profile_af_fit_calc),))
np_emm_tor_sum = np.zeros((len(df_energy_profile_af_fit_calc),))
for nb_pair in list_R_NB_eel:
    column_name     = "Enb_eel_" + nb_pair[2:]
    np_emm_eel_sum += df_energy_profile_af_fit_calc[column_name]
for i14_pair in list_R_i14_eel:
    column_name     = "E14_eel_" + i14_pair[4:]
    np_emm_eel_sum += df_energy_profile_af_fit_calc[column_name]
for dihed_pair in range(dihed_num):
    column_name = "Etor_%s_%s" % (dict_dihed_atoms_types[list_dihed[dihed_pair]],list_dihed[dihed_pair])
    np_emm_tor_sum += df_energy_profile_af_fit_calc[column_name]
df_energy_profile_af_fit_calc["Emm_var_bf"] = df_energy_profile_bf["Emm_var"]
df_energy_profile_af_fit_calc["Emm_var_af"] = (  np_emm_eel_sum + np_emm_tor_sum)
df_energy_profile_af_fit_calc["Emm_invar"] = (  df_energy_profile_af_fit_calc["Emm_bf"]
                                              - df_energy_profile_af_fit_calc["Emm_var_bf"])
df_energy_profile_af_fit_calc["Emm_af"] = (  df_energy_profile_af_fit_calc["Emm_var_af"] +
                                           df_energy_profile_af_fit_calc["Emm_invar"] )


# In[ ]:


energy_profile_af_fit_name = "0GB_energy_profile_reduce_datasets_af_fit_" + file_date + ".csv"
df_energy_profile_af_fit_calc.to_csv(energy_profile_af_fit_name)


# In[ ]:


outcsvname = "torsion_after_fit." + file_date + ".csv"
df_params_torsion_af_fit = pd.read_csv(outcsvname,index_col=0)
df_params_torsion_af_fit


# In[ ]:


fig = plt.figure(figsize=(20,15))
x_lin = np.pi * np.linspace(-1,1,200)
vfunc = np.vectorize(Etorsion_var)
for num,types in enumerate(list_dihed_type):
    ax = fig.add_subplot(4,5,num+1)
    v1_f , v1_l = df_params_torsion_af_fit.loc[types,"V1"] , df_params_torsion_af_fit.loc[types,"V1_opt"]
    v2_f , v2_l = df_params_torsion_af_fit.loc[types,"V2"] , df_params_torsion_af_fit.loc[types,"V2_opt"]
    v3_f , v3_l = df_params_torsion_af_fit.loc[types,"V3"] , df_params_torsion_af_fit.loc[types,"V3_opt"]
    v4_f , v4_l = df_params_torsion_af_fit.loc[types,"V4"] , df_params_torsion_af_fit.loc[types,"V4_opt"]
    v5_f , v5_l = df_params_torsion_af_fit.loc[types,"V5"] , df_params_torsion_af_fit.loc[types,"V5_opt"]
    v6_f , v6_l = df_params_torsion_af_fit.loc[types,"V6"] , df_params_torsion_af_fit.loc[types,"V6_opt"]
    g1_f , g1_l = df_params_torsion_af_fit.loc[types,"gamma1"] , df_params_torsion_af_fit.loc[types,"gamma1_opt"]
    g2_f , g2_l = df_params_torsion_af_fit.loc[types,"gamma2"] , df_params_torsion_af_fit.loc[types,"gamma2_opt"]
    g3_f , g3_l = df_params_torsion_af_fit.loc[types,"gamma3"] , df_params_torsion_af_fit.loc[types,"gamma3_opt"]
    g4_f , g4_l = df_params_torsion_af_fit.loc[types,"gamma4"] , df_params_torsion_af_fit.loc[types,"gamma4_opt"]
    g5_f , g5_l = df_params_torsion_af_fit.loc[types,"gamma5"] , df_params_torsion_af_fit.loc[types,"gamma5_opt"]
    g6_f , g6_l = df_params_torsion_af_fit.loc[types,"gamma6"] , df_params_torsion_af_fit.loc[types,"gamma6_opt"]
    y_before = vfunc(v1_f,v2_f,v3_f,v4_f,v5_f,v6_f,g1_f,g2_f,g3_f,g4_f,g5_f,g6_f,x_lin)
    y_after  = vfunc(v1_l,v2_l,v3_l,v4_l,v5_l,v6_l,g1_l,g2_l,g3_l,g4_l,g5_l,g6_l,x_lin)
    ax.plot(x_lin*180/np.pi,y_before,label="before",c="red")
    ax.plot(x_lin*180/np.pi,y_after,label="after",c="blue")
    ax.set_title(types)
    ax.legend()
filename = "torsion_energy_after_fit_." + file_date + ".png" 
plt.savefig(filename,format="png",bbox_inches="tight")
plt.show()


# In[ ]:


out_charge_csvname = "charge_after_fit." + file_date + ".csv"
charge_df = pd.read_csv(out_charge_csvname,index_col=0)
fig = plt.figure(figsize=(20,15))
for num,atom_i in enumerate(dict_charge):
    ax = fig.add_subplot(3,6,num+1)
    x_lin = np.linspace(-1,1,1000)
    atom_i_mean , atom_i_std = df_charge_distribution.loc[atom_i,"Mean"] , df_charge_distribution.loc[atom_i,"Std"]
    pdf_x = np.exp( - (x_lin - atom_i_mean) ** 2 / ( 2 * atom_i_std ** 2 )) / np.sqrt( 2 * np.pi ) * atom_i_std
    ax.plot(x_lin,pdf_x)
    ax.vlines(charge_df.iloc[num,1],ymin=pdf_x.min(),ymax=pdf_x.max(),color="red",label="after_opt")
    ax.set_title(atom_i)
    ax.vlines(charge_df.iloc[num,0],ymin=pdf_x.min(),ymax=pdf_x.max(),color="green",label="before_opt")
    ax.set_xlim(atom_i_mean - 4. * atom_i_std , atom_i_mean + 4. * atom_i_std)
    ax.legend()
out_charge_distname = "charge_distribution_after_fit_." + file_date + ".png"
plt.savefig(out_charge_distname,format="png",bbox_inches="tight")
plt.show()


# In[ ]:


print('DDE   = %6.3f' % (df_delta_edat_af_fit["Dedft"] - df_delta_edat_af_fit["Demm_af"]).mean())
print('DDE^2 = %6.3f' % ((df_delta_edat_af_fit["Dedft"] - df_delta_edat_af_fit["Demm_af"])**2).mean())


# In[ ]:


df_delta_edat_af_fit.columns[331:]


# In[ ]:


dict_info_columns_num


# df_delta_edat_af_fitのデータフレームは<br>
# 
#  |列番号|列名|説明| 
#  |:-------|:-----|:----| 
#  | 0列目 | Dedft | QM計算での相対エネルギー | 
#  | 1列目 | Demm_invar | MM計算のうちフィッティング変数に依存しない、Bond,Angle,VDW項の和 | 
#  | 2列目 | Demm_bf | フィッティング前のパラメータによるMM計算での相対エネルギー | 
#  | 3-71列目 | R_ | 非結合相互作用原子ペアの距離 | 
#  | 72-100列目 | R14_ | 1-4相互作用原子ペアの距離 | 
#  | 101-166列目 | Tor_ | 二面角 | 
#  | 167-234列目 | Denb_eel_ |非結合相互作用原子ペアの静電相互作用エネルギーの相対エネルギー | 
#  | 235-262列目 | De14_eel_ |1-4相互作用原子ペアの静電相互作用エネルギーの相対エネルギー | 
#  | 263-329列目 | Detor_ |二面角エネルギーの相対エネルギー | 
#  | 330列目 | y_targe | $De_{dft}-(De_{mm_invar}-De_{mm_bf})$|
#  |331列目|Devar_bf|MM計算のうちフィッティング変数に依存する項のGLYCAM力場による和の相対エネルギー|
#  |332列目|Demm_eel_sum|相対エネルギーの静電相互作用項の和|
#  |333列目|Demm_tor_sum|相対エネルギーの二面角項の和|
#  |334列目|Devar_af|MM計算のうちフィッティング変数に依存する項のフィッティング変数による和の相対エネルギー|
#  |335列目|Demm_af|フィッティングパラメータによるMM計算の相対エネルギー|
#  |336列目|DDE_bf|GLYCAM力場による$\Delta\Delta$E|
#  |337列目|DDE_af|フィッティングパラメータによる$\Delta\Delta$E|

# In[ ]:


# 新たにTf.Sessionを立てる。これはフィッティングが終了しているセッションとは別に
# フィッティング前のパラメータを用いて計算結果の確認を行うためである。


# In[ ]:


# ここでは最適化は実行しない

start = time.time()
sess2 = tf.Session(graph=fit)
sess2.run(init)


_loss , _demm_tor , _demm_nb_eel , _demm_i14_eel,  _devar , _y_t , _loss0= sess2.run([loss , DE_torsion_sum , DE_NB_eel_pair_sum , DE_i14_eel_pair_sum ,             Devar , tf_y_t , loss0] , feed_dict=dict_train_feed)
print("loss0 at no optimization::  %.3f" % _loss0 )


# ここからコードによる計算結果とデータ処理の結果に違いがあるかを確認する<br>
# コードによる計算では初期変数で計算する<br>
# データ処理結果は　df_delta_edat_bf_fit を参考にする<br>
# <br>
# $$
#  \begin{align}
#  {\Delta}{\Delta}E &= {\Delta}E_{DFT} - {\Delta}E_{MM}\\
#  &= {\Delta}E_{DFT} - ( {\Delta}E_{MM\_invar} + {\Delta}E_{MM\_var} )\\
#  &= {\Delta}E_{DFT} - {\Delta}E_{MM\_invar} &- {\Delta}E_{MM\_var} \\
#  &= {\Delta}E_{DFT} - {\Delta}E_{MM\_invar} &- ( {\Delta}E_{MM\_tor} + {\Delta}E_{MM\_i14eel} + {\Delta}E_{MM\_i14vdw} + {\Delta}E_{MM\_nbeel} + {\Delta}E_{MM\_nbvdw} )\\
#  &= y\_target &- ( {\Delta}E_{MM\_tor} + {\Delta}E_{MM\_i14eel} + {\Delta}E_{MM\_i14vdw} + {\Delta}E_{MM\_nbeel} + {\Delta}E_{MM\_nbvdw} )\\
#  \end{align}
#  $$
#  
#  ${\Delta}E_{MM\_tor} , {\Delta}E_{MM\_i14eel} , {\Delta}E_{MM\_i14vdw} , {\Delta}E_{MM\_nbeel} , {\Delta}E_{MM\_nbvdw},{\Delta\Delta}E$ について<br>
#  それぞれ確認を進めていく

# In[ ]:


np_code_calc_demm_tor_bf_fit    = sess2.run(DE_torsion_sum      , feed_dict=dict_all_feed)
np_code_calc_demm_i14eel_bf_fit = sess2.run(DE_i14_eel_pair_sum , feed_dict=dict_all_feed)
np_code_calc_demm_nbeel_bf_fit  = sess2.run(DE_NB_eel_pair_sum  , feed_dict=dict_all_feed)
np_code_calc_dde_bf_fit         = sess2.run(tf_y_t - Devar      , feed_dict=dict_all_feed)
np_code_calc_dde2_bf_fit        = sess2.run(loss0               , feed_dict=dict_all_feed)
np_code_calc_y_target_bf_fit    = sess2.run(tf_y_t              , feed_dict=dict_all_feed)
np_code_calc_evar_bf_fit        = (np_code_calc_demm_tor_bf_fit + np_code_calc_demm_i14eel_bf_fit 
                                   + np_code_calc_demm_nbeel_bf_fit )


# In[ ]:


# データから計算されたそれぞれの値を計算する前に
# データ中の該当するカラムインデックスを取得する
list_detor_columns_num_bf_fit = []
for num,i in enumerate(df_delta_edat_bf_fit.columns):
    if "Detor" in i:
        list_detor_columns_num_bf_fit.append(num)
list_denbeel_columns_num_bf_fit = []
for num,i in enumerate(df_delta_edat_bf_fit.columns):
    if "Deel" in i:
        list_denbeel_columns_num_bf_fit.append(num)
list_dei14eel_columns_num_bf_fit = []
for num,i in enumerate(df_delta_edat_bf_fit.columns):
    if "De14eel" in i:
        list_dei14eel_columns_num_bf_fit.append(num)


# In[ ]:


# インデックスリストを使用してデータから計算された値を取り出す。
np_data_calc_demm_tor_bf_fit    = np.array(df_delta_edat_bf_fit.iloc[:,list_detor_columns_num_bf_fit].sum(axis=1))
np_data_calc_demm_i14eel_bf_fit = np.array(df_delta_edat_bf_fit.iloc[:,list_dei14eel_columns_num_bf_fit].sum(axis=1))
np_data_calc_demm_nbeel_bf_fit  = np.array(df_delta_edat_bf_fit.iloc[:,list_denbeel_columns_num_bf_fit].sum(axis=1))
np_data_calc_dde_bf_fit         = np.array(df_delta_edat_bf_fit["Dedft"] - df_delta_edat_bf_fit["Demm"])
np_data_calc_y_target_bf_fit    = np.array(df_delta_edat_bf_fit["y_target"])
np_data_calc_evar_bf_fit        = (np_data_calc_demm_tor_bf_fit + np_data_calc_demm_i14eel_bf_fit 
                                   + np_data_calc_demm_nbeel_bf_fit)
np_data_calc_dde2_bf_fit        = np.mean(np_data_calc_dde_bf_fit ** 2)


# In[ ]:


# コードによる計算結果とデータによる計算結果の差分を計算する
tor_avg_bf_fit      = ( np_code_calc_demm_tor_bf_fit    - np_data_calc_demm_tor_bf_fit    ).mean()
tor_std_bf_fit      = ( np_code_calc_demm_tor_bf_fit    - np_data_calc_demm_tor_bf_fit    ).std()
i14eel_avg_bf_fit   = ( np_code_calc_demm_i14eel_bf_fit - np_data_calc_demm_i14eel_bf_fit ).mean()
i14eel_std_bf_fit   = ( np_code_calc_demm_i14eel_bf_fit - np_data_calc_demm_i14eel_bf_fit ).std()
nbeel_avg_bf_fit    = ( np_code_calc_demm_nbeel_bf_fit  - np_data_calc_demm_nbeel_bf_fit  ).mean()
nbeel_std_bf_fit    = ( np_code_calc_demm_nbeel_bf_fit  - np_data_calc_demm_nbeel_bf_fit  ).std()
dde_avg_bf_fit      = ( np_code_calc_dde_bf_fit         - np_data_calc_dde_bf_fit         ).mean()
dde_std_bf_fit      = ( np_code_calc_dde_bf_fit         - np_data_calc_dde_bf_fit         ).std()
y_target_avg_bf_fit = ( np_code_calc_y_target_bf_fit    - np_data_calc_y_target_bf_fit    ).mean()
y_target_std_bf_fit = ( np_code_calc_y_target_bf_fit    - np_data_calc_y_target_bf_fit    ).std()
evar_avg_bf_fit     = ( np_code_calc_evar_bf_fit        - np_data_calc_evar_bf_fit        ).mean()
evar_std_bf_fit     = ( np_code_calc_evar_bf_fit        - np_data_calc_evar_bf_fit        ).std()

print("Tor      Avg : %8.4f // Std : %7.4f" % ( tor_avg_bf_fit       , tor_std_bf_fit      ))
print("i14 EEL  Avg : %8.4f // Std : %7.4f" % ( i14eel_avg_bf_fit    , i14eel_std_bf_fit   ))
print("NB  EEL  Avg : %8.4f // Std : %7.4f" % ( nbeel_avg_bf_fit     , nbeel_std_bf_fit    ))
print("DDE      Avg : %8.4f // Std : %7.4f" % ( dde_avg_bf_fit       , dde_std_bf_fit      ))
print("y_target Avg : %8.4f // Std : %7.4f" % ( y_target_avg_bf_fit  , y_target_std_bf_fit ))
print("evar     Avg : %8.4f // Std : %7.4f" % ( evar_avg_bf_fit      , evar_std_bf_fit     ))
print()
print("DDE^2 from code      : %8.4f" % np_code_calc_dde2_bf_fit)
print("DDE^2 from data      : %8.4f" % np_data_calc_dde2_bf_fit)
print("Diference of DDE^2   : %8.4f" % (np_code_calc_dde2_bf_fit - np_data_calc_dde2_bf_fit))

DDE2c_DDE2d_bf_fit = (-2 * np_code_calc_y_target_bf_fit * (np_code_calc_evar_bf_fit - np_data_calc_evar_bf_fit ) 
               + (np_code_calc_evar_bf_fit**2 - np_data_calc_evar_bf_fit**2))
print("DDE^2 with code - DDE^2 with data  Avg : %8.4f // Std : %8.4f" 
      % (DDE2c_DDE2d_bf_fit.mean(),DDE2c_DDE2d_bf_fit.std()))


# In[ ]:


# ついでフィッティング後のパラメータを用いたコードによる計算結果と
# 最適化後のパラメータから計算し、保存したデータによる結果との比較を行う


# In[ ]:


np_code_calc_demm_tor_af_fit    = sess.run(DE_torsion_sum      , feed_dict=dict_all_feed)
np_code_calc_demm_i14eel_af_fit = sess.run(DE_i14_eel_pair_sum , feed_dict=dict_all_feed)
np_code_calc_demm_nbeel_af_fit  = sess.run(DE_NB_eel_pair_sum  , feed_dict=dict_all_feed)
np_code_calc_dde_af_fit         = sess.run(tf_y_t - Devar         , feed_dict=dict_all_feed)
np_code_calc_dde2_af_fit        = sess.run(loss0               , feed_dict=dict_all_feed)
np_code_calc_y_target_af_fit    = sess.run(tf_y_t                 , feed_dict=dict_all_feed)
np_code_calc_evar_af_fit        = (np_code_calc_demm_tor_af_fit + np_code_calc_demm_i14eel_af_fit 
                                   + np_code_calc_demm_nbeel_af_fit)


# In[ ]:


# データから計算されたそれぞれの値を計算する前に
# データ中の該当するカラムインデックスを取得する
list_detor_columns_num_af_fit = []
for num,i in enumerate(df_delta_edat_af_fit.columns):
    if "Detor" in i:
        list_detor_columns_num_af_fit.append(num)
list_denbeel_columns_num_af_fit = []
for num,i in enumerate(df_delta_edat_af_fit.columns):
    if "Denb_eel" in i:
        list_denbeel_columns_num_af_fit.append(num)
list_dei14eel_columns_num_af_fit = []
for num,i in enumerate(df_delta_edat_af_fit.columns):
    if "De14_eel" in i:
        list_dei14eel_columns_num_af_fit.append(num)


# In[ ]:


# インデックスリストを使用してデータから計算された値を取り出す。
np_data_calc_demm_tor_af_fit    = np.array(df_delta_edat_af_fit.iloc[:,list_detor_columns_num_af_fit].sum(axis=1))
np_data_calc_demm_i14eel_af_fit = np.array(df_delta_edat_af_fit.iloc[:,list_dei14eel_columns_num_af_fit].sum(axis=1))
np_data_calc_demm_nbeel_af_fit  = np.array(df_delta_edat_af_fit.iloc[:,list_denbeel_columns_num_af_fit].sum(axis=1))
np_data_calc_dde_af_fit         = np.array(df_delta_edat_af_fit["Dedft"] - df_delta_edat_af_fit["Demm_af"])
np_data_calc_y_target_af_fit    = np.array(df_delta_edat_af_fit["y_target"])
np_data_calc_evar_af_fit        = (np_data_calc_demm_tor_af_fit + np_data_calc_demm_i14eel_af_fit 
                                   + np_data_calc_demm_nbeel_af_fit)
np_data_calc_dde2_af_fit        = np.mean(np_data_calc_dde_af_fit ** 2)


# In[ ]:


# コードによる計算結果とデータによる計算結果の差分を計算する
tor_avg_af_fit      = ( np_code_calc_demm_tor_af_fit    - np_data_calc_demm_tor_af_fit    ).mean()
tor_std_af_fit      = ( np_code_calc_demm_tor_af_fit    - np_data_calc_demm_tor_af_fit    ).std()
i14eel_avg_af_fit   = ( np_code_calc_demm_i14eel_af_fit - np_data_calc_demm_i14eel_af_fit ).mean()
i14eel_std_af_fit   = ( np_code_calc_demm_i14eel_af_fit - np_data_calc_demm_i14eel_af_fit ).std()
nbeel_avg_af_fit    = ( np_code_calc_demm_nbeel_af_fit  - np_data_calc_demm_nbeel_af_fit  ).mean()
nbeel_std_af_fit    = ( np_code_calc_demm_nbeel_af_fit  - np_data_calc_demm_nbeel_af_fit  ).std()
dde_avg_af_fit      = ( np_code_calc_dde_af_fit         - np_data_calc_dde_af_fit         ).mean()
dde_std_af_fit      = ( np_code_calc_dde_af_fit         - np_data_calc_dde_af_fit         ).std()
y_target_avg_af_fit = ( np_code_calc_y_target_af_fit    - np_data_calc_y_target_af_fit    ).mean()
y_target_std_af_fit = ( np_code_calc_y_target_af_fit    - np_data_calc_y_target_af_fit    ).std()
evar_avg_af_fit     = ( np_code_calc_evar_af_fit        - np_data_calc_evar_af_fit        ).mean()
evar_std_af_fit     = ( np_code_calc_evar_af_fit        - np_data_calc_evar_af_fit        ).std()

print("Diference of Tor      Avg : %8.4f // Std : %7.4f" % ( tor_avg_af_fit       , tor_std_af_fit      ))
print("Diference of i14 EEL  Avg : %8.4f // Std : %7.4f" % ( i14eel_avg_af_fit    , i14eel_std_af_fit   ))
print("Diference of NB  EEL  Avg : %8.4f // Std : %7.4f" % ( nbeel_avg_af_fit     , nbeel_std_af_fit    ))
print("Diference of DDE      Avg : %8.4f // Std : %7.4f" % ( dde_avg_af_fit       , dde_std_af_fit      ))
print("Diference of y_target Avg : %8.4f // Std : %7.4f" % ( y_target_avg_af_fit  , y_target_std_af_fit ))
print("Diference of evar     Avg : %8.4f // Std : %7.4f" % ( evar_avg_af_fit      , evar_std_af_fit     ))
print()
print("DDE^2 from code      : %8.4f" % np_code_calc_dde2_af_fit)
print("DDE^2 from data      : %8.4f" % np_data_calc_dde2_af_fit)
print("Diference of DDE^2   : %8.4f" % (np_code_calc_dde2_af_fit - np_data_calc_dde2_af_fit))

DDE2c_DDE2d_af_fit = (-2 * np_code_calc_y_target_af_fit * (np_code_calc_evar_af_fit - np_data_calc_evar_af_fit ) 
               + (np_code_calc_evar_af_fit**2 - np_data_calc_evar_af_fit**2))
print("DDE^2 with code - DDE^2 with data  Avg : %8.4f // Std : %8.4f" 
      % (DDE2c_DDE2d_af_fit.mean(),DDE2c_DDE2d_af_fit.std()))


# In[ ]:


print("Fitting Results")
print("DDE after fitting:: Avg %8.4f / Std %8.4f" % (np_code_calc_dde_af_fit.mean(),np_code_calc_dde_af_fit.std()))

