# DeepCTR

[![Python Versions](https://img.shields.io/pypi/pyversions/deepctr.svg)](https://pypi.org/project/deepctr)
[![TensorFlow Versions](https://img.shields.io/badge/TensorFlow-1.4+/2.0+-blue.svg)](https://pypi.org/project/deepctr)
[![Downloads](https://pepy.tech/badge/deepctr)](https://pepy.tech/project/deepctr)
[![PyPI Version](https://img.shields.io/pypi/v/deepctr.svg)](https://pypi.org/project/deepctr)
[![GitHub Issues](https://img.shields.io/github/issues/shenweichen/deepctr.svg
)](https://github.com/shenweichen/deepctr/issues)
<!-- [![Activity](https://img.shields.io/github/last-commit/shenweichen/deepctr.svg)](https://github.com/shenweichen/DeepCTR/commits/master) -->


[![Documentation Status](https://readthedocs.org/projects/deepctr-doc/badge/?version=latest)](https://deepctr-doc.readthedocs.io/)
[![Build Status](https://travis-ci.org/shenweichen/DeepCTR.svg?branch=master)](https://travis-ci.org/shenweichen/DeepCTR)
[![Coverage Status](https://coveralls.io/repos/github/shenweichen/DeepCTR/badge.svg?branch=master)](https://coveralls.io/github/shenweichen/DeepCTR?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d4099734dc0e4bab91d332ead8c0bdd0)](https://www.codacy.com/app/wcshen1994/DeepCTR?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=shenweichen/DeepCTR&amp;utm_campaign=Badge_Grade)
[![Disscussion](https://img.shields.io/badge/chat-wechat-brightgreen?style=flat)](./README.md#disscussiongroup)
[![License](https://img.shields.io/github/license/shenweichen/deepctr.svg)](https://github.com/shenweichen/deepctr/blob/master/LICENSE)
<!-- [![Gitter](https://badges.gitter.im/DeepCTR/community.svg)](https://gitter.im/DeepCTR/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) -->


DeepCTR is a **Easy-to-use**,**Modular** and **Extendible** package of deep-learning based CTR models along with lots of core components layers which can be used to easily build custom models.It is compatible with **tensorflow 1.4+ and 2.0+**.You can use any complex model with `model.fit()`and `model.predict()` .

## Models List

|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  Convolutional Click Prediction Model  | [CIKM 2015][A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)             |
| Factorization-supported Neural Network | [ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)                    |
|      Product-based Neural Network      | [ICDM 2016][Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                   |
|              Wide & Deep               | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                 |
|                 DeepFM                 | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                           |
|        Piece-wise Linear Model         | [arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)                                 |
|          Deep & Cross Network          | [ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                   |
|   Attentional Factorization Machine    | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |
|      Neural Factorization Machine      | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                               |
|                xDeepFM                 | [KDD 2018][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                         |
|                AutoInt                 | [arxiv 2018][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                              |
|         Deep Interest Network          | [KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)                                                       |
|    Deep Interest Evolution Network     | [AAAI 2019][Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)                                            |
|                  ONN                  | [arxiv 2019][Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                                |
|                 FGCNN                  | [WWW 2019][Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction ](https://arxiv.org/pdf/1904.04447)                             |
|     Deep Session Interest Network      | [IJCAI 2019][Deep Session Interest Network for Click-Through Rate Prediction ](https://arxiv.org/abs/1905.06482)                                                |
|                FiBiNET                 | [RecSys 2019][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)   |
|                FLEN                    | [arxiv 2019][FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/pdf/1911.04690.pdf)   |



## Using Dense Feature with higher dimensions + Var Len Features
```python3
train_table.tp = train_table.tp.fillna("0|0|0|0|0|0|0|0|0|0")
test_table.tp = test_table.tp.fillna("0|0|0|0|0|0|0|0|0|0")
test_table.tw = [[0,0,0,0,0,0,0,0,0,0] if type(x) != list else x for x in test_table.tw]
train_table.tw = [[0,0,0,0,0,0,0,0,0,0] if type(x) != list else x for x in train_table.tw]

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr.models import DeepFM, DIEN
from deepctr.inputs import SparseFeat, VarLenSparseFeat,get_feature_names


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index_pos:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index_pos[key] = len(key2index_pos) + 1
    return list(map(lambda x: key2index_pos[x], key_ans))

def split_t(x):
    key_ans = x.split(' ')
    for key in key_ans:
        if key not in key2index_tit:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index_tit[key] = len(key2index_tit) + 1
    return list(map(lambda x: key2index_tit[x], key_ans))

def split_tp(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index_tp:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index_tp[key] = len(key2index_tp) + 1
    return list(map(lambda x: key2index_tp[x], key_ans))

def split2(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index_neg:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index_neg[key] = len(key2index_neg) + 1
    return list(map(lambda x: key2index_neg[x], key_ans))

totdata = pd.concat([test_table, train_table], ignore_index=True, sort=False)
totdata['topic'] = totdata['topicMax']
tit_max = totdata.title_text.apply(lambda x: x.count(" ")).max()
vocab_size_tit = ad_title_table.wordId.nunique()
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM, FiBiNET
from deepctr.inputs import  SparseFeat, DenseFeat,get_feature_names

sparse_features = ["sparse_features"]
dense_features = ["dense_feat"]

target = ['clicked']
# 1.Label Encoding for sparse features,and do simple Transformation for dense features
encoders = dict()
for feat in sparse_features:
    lbe = LabelEncoder()
    lbe = lbe.fit(totdata[feat])
    encoders[feat] = lbe
mms = MinMaxScaler(feature_range=(0, 1))
mms.fit(totdata[dense_features])

sp_feat = [SparseFeat(feat, vocabulary_size=totdata[feat].nunique(), embedding_dim=4)
                        for i,feat in enumerate(sparse_features)]
dn_feat = [DenseFeat(feat, 1,)
                      for feat in dense_features]      

dn_feat += [DenseFeat("image", 512,)]       
fixlen_feature_columns =   sp_feat + dn_feat

varlen_feature_columns = [VarLenSparseFeat(SparseFeat('title_text',vocabulary_size= vocab_size_tit + 1,
                                                      embedding_dim=4),
                                           length_name="title_len", maxlen=tit_max,
                                           combiner='mean',weight_name=None),
                          VarLenSparseFeat(SparseFeat('hist_topic',vocabulary_size= 51,
                                                      embedding_dim=4, embedding_name="topic"),
                                           length_name="tp_len", maxlen=10,
                                           combiner='mean',weight_name="tw")]
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
linear_feature_columns = fixlen_feature_columns + varlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
data = train_table[3000000:].copy()
data["topic"] = data["topicMax"]
for feat in sparse_features:
    lbe = encoders[feat]
    data[feat] = lbe.transform(data[feat])
data[dense_features] = mms.transform(data[dense_features])

key2index_tit = {}
tit_list = list(map(split_t, data['title_text'].values))
tit_length = np.array(list(map(len, tit_list)))
tit_list = pad_sequences(tit_list, maxlen=tit_max, padding='post', )

key2index_tp = {}
tp_list = list(map(split_tp, data['tp'].values))
tp_length = np.array(list(map(len, tp_list)))
tp_list = pad_sequences(tp_list, maxlen=10, padding='post', )
data["title_len"] = 0
data["tp_len"] = 0
data["hist_topic"] = data["tp"]
# data["neg_len"] = 0
# train, test = train_test_split(data, test_size=0.2)
train_model_input = {name:data[name] for name in feature_names}
train_model_input["title_text"] = tit_list
train_model_input["title_len"] = tit_length
train_model_input["hist_topic"] = tp_list
train_model_input["tp_len"] = tp_length
data["tw"] = data["tw"].apply(lambda x: np.asarray(x))
twa = np.asarray(list(data["tw"]))
weights = twa.reshape(len(twa),-1,1)
train_model_input["tw"] = weights

# train_model_input["pos_len"] = pos_length
# train_model_input["neg_len"] = neg_length
# test_model_input = {name:test[name] for name in feature_names}
def split_t2(x):
    key_ans = x.split(' ')
    for key in key_ans:
        if key not in key2index_tit_test:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index_tit_test[key] = len(key2index_tit_test) + 1
    return list(map(lambda x: key2index_tit_test[x], key_ans))
def split_tp2(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index_tp_test:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index_tp_test[key] = len(key2index_tp_test) + 1
    return list(map(lambda x: key2index_tp_test[x], key_ans))

tdata = test_table.copy()
tdata["topic"] = tdata["topicMax"]
# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    e = encoders[feat]
    tdata[feat] = e.transform(tdata[feat])
tdata[dense_features] = mms.transform(tdata[dense_features])

# 3.generate input data for model
key2index_tit_test = {}
tit_list_test = list(map(split_t2, tdata['title_text'].values))
tit_length_test = np.array(list(map(len, tit_list_test)))
tit_list_test = pad_sequences(tit_list_test, maxlen=tit_max, padding='post', )

key2index_tp_test = {}
tp_list_test = list(map(split_tp2, tdata['tp'].values))
tp_length_test = np.array(list(map(len, tp_list_test)))
tp_list_test = pad_sequences(tp_list_test, maxlen=10, padding='post', )


tdata["title_len"] = 0
tdata["tp_len"] = 0
tdata["hist_topic"] = tdata["tp"]
model_input = {name:tdata[name] for name in feature_names}

model_input["title_text"] = tit_list_test
model_input["title_len"] = tit_length_test
model_input["hist_topic"] = tp_list_test
model_input["tp_len"] = tp_length_test


tdata["tw"] = tdata["tw"].apply(lambda x: np.asarray(x))
twa2 = np.asarray(list(tdata["tw"]))
weights2 = twa2.reshape(len(twa2),-1,1)
model_input["tw"] = weights2

model_input["topic"] = model_input["topicMax"]
data["topic"] = data["topicMax"]
train_model_input["topic"] = train_model_input["topicMax"]

model_input["seq_length"] = model_input["tp_len"]
train_model_input["seq_length"] = train_model_input["tp_len"]

import keras
model = FiBiNET(linear_feature_columns, dnn_feature_columns, task='binary')
# model = NFM(linear_feature_columns, dnn_feature_columns, task='binary')
# model = WDL(linear_feature_columns, dnn_feature_columns, task='binary')

# model = ONN(linear_feature_columns, dnn_feature_columns, task='binary')
# model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary')

# model = FGCNN(linear_feature_columns, dnn_feature_columns, pooling_width=(1, 1, 1, 1), task='binary')

# model = DIN(dnn_feature_columns, ["topic"], task="binary")
# print(model)
model.compile("adam", "binary_crossentropy",
              metrics=[auc])

image = np.asarray(list(train_model_input["image"]))
image2 = np.asarray(list(model_input["image"]))
model_input["image"] = image2
train_model_input["image"] = image

model.fit(train_model_input, data[target].values,validation_split=0.0, epochs=18, batch_size=32000)
```
## Using VarLen Features And Neg Pos Hist
```python3
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr.models import DeepFM, DIEN
from deepctr.inputs import SparseFeat, VarLenSparseFeat,get_feature_names


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index_pos:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index_pos[key] = len(key2index_pos) + 1
    return list(map(lambda x: key2index_pos[x], key_ans))

def split_t(x):
    key_ans = x.split(' ')
    for key in key_ans:
        if key not in key2index_tit:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index_tit[key] = len(key2index_tit) + 1
    return list(map(lambda x: key2index_tit[x], key_ans))

def split2(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index_neg:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index_neg[key] = len(key2index_neg) + 1
    return list(map(lambda x: key2index_neg[x], key_ans))
    
totdata = pd.concat([test_table, train_table], ignore_index=True, sort=False)
neg_hist_max = totdata.neg_hist.apply(lambda x: x.count("|")).max()
pos_hist_max = totdata.pos_hist.apply(lambda x: x.count("|")).max()
tit_max = totdata.title_text.apply(lambda x: x.count(" ")).max()
vocab_size = totdata.adId.nunique()
vocab_size_tit = ad_title_table.wordId.nunique()
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM, FiBiNET
from deepctr.inputs import  SparseFeat, DenseFeat,get_feature_names

sparse_features = ["sparse_columns"]
dense_features = ["dense_columns_name"]

target = ['clicked']
# 1.Label Encoding for sparse features,and do simple Transformation for dense features
encoders = dict()
for feat in sparse_features:
    lbe = LabelEncoder()
    lbe = lbe.fit(totdata[feat])
    encoders[feat] = lbe
mms = MinMaxScaler(feature_range=(0, 1))
mms.fit(totdata[dense_features])

sp_feat = [SparseFeat(feat, vocabulary_size=totdata[feat].nunique(), embedding_dim=4)
                        for i,feat in enumerate(sparse_features)]
dn_feat = [DenseFeat(feat, 1,)
                      for feat in dense_features]             
fixlen_feature_columns =   sp_feat + dn_feat

# pos hist + neg hist + var len features
varlen_feature_columns = [VarLenSparseFeat(SparseFeat('pos_hist',vocabulary_size= vocab_size + 1,
                                                      embedding_dim=4),
                                           length_name="pos_len", maxlen=pos_hist_max,
                                           combiner='mean',weight_name=None),
        VarLenSparseFeat(SparseFeat('neg_hist',vocabulary_size= vocab_size + 1,
                                    embedding_dim=4),length_name="neg_len",
                          maxlen=neg_hist_max, combiner='mean',weight_name=None)]
varlen_feature_columns = [VarLenSparseFeat(SparseFeat('title_text',vocabulary_size= vocab_size_tit + 1,
                                                      embedding_dim=4),
                                           length_name="title_len", maxlen=tit_max,
                                           combiner='mean',weight_name=None)]
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
linear_feature_columns = fixlen_feature_columns + varlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
for feat in sparse_features:
    lbe = encoders[feat]
    data[feat] = lbe.transform(data[feat])
data[dense_features] = mms.transform(data[dense_features])

key2index_tit = {}
tit_list = list(map(split_t, data['title_text'].values))
tit_length = np.array(list(map(len, tit_list)))
tit_list = pad_sequences(tit_list, maxlen=tit_max, padding='post', )

key2index_pos = {}
pos_list = list(map(split, data['pos_hist'].values))
pos_length = np.array(list(map(len, pos_list)))
max_len_pos = max(pos_length)
pos_list = pad_sequences(pos_list, maxlen=pos_hist_max, padding='post', )

key2index_neg = {}
neg_list = list(map(split2, data['neg_hist'].values))
neg_length = np.array(list(map(len, neg_list)))
max_len_neg = max(neg_length)
neg_list = pad_sequences(neg_list, maxlen=neg_hist_max, padding='post', )

import tensorflow as tf

data["title_len"] = 0
# data["neg_len"] = 0
# train, test = train_test_split(data, test_size=0.2)
train_model_input = {name:data[name] for name in feature_names}
train_model_input["title_text"] = tit_list
train_model_input["title_len"] = tit_length
train_model_input["pos_len"] = pos_length
train_model_input["neg_len"] = neg_length

# 4.Define Model,train,predict and evaluate
# model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
# model = FGCNN(linear_feature_columns, dnn_feature_columns, pooling_width=(1, 1, 1, 1), task='binary')
# model = FGCNN(linear_feature_columns, dnn_feature_columns, task='binary')
# model = AFM(fixlen_feature_columns, sp_feat, task='binary')
# model = NFM(linear_feature_columns, dnn_feature_columns, task='binary')
# model = WDL(linear_feature_columns, dnn_feature_columns, task='binary')
# model = FNN(linear_feature_columns, dnn_feature_columns, task='binary')
# model = AutoInt(linear_feature_columns, dnn_feature_columns, task='binary')
# model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
# model = DIEN(dnn_feature_columns,["pos_hist","neg_hist"],gru_type="AUGRU", use_negsampling=True, task='binary')
# model = DIN(dnn_feature_columns,["pos_hist","neg_hist"], task='binary')
import keras
model = FiBiNET(linear_feature_columns, dnn_feature_columns,dnn_hidden_units=(1024,1024), dnn_dropout=0.08, task='binary')

model.compile("adam", "binary_crossentropy",
              metrics=[tf.keras.metrics.AUC()], )
```
