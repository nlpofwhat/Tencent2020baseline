import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from tqdm import tqdm
names = ["class", "title", "content"]
import sklearn as sk

print(sk.__version__)

#第一种方式直接使用id本身作为嵌入
# 把placeholder的数据直接赋值为embedding
#第二种方式使用embedding
#把每一个id作为一个单词
def to_one_hot(y, n_class):
    return np.eye(n_class)[y.astype(int)-1]
    # 产生一个ndarray行数和y的行数相同，列数为n_class

#减少内存
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def load_train_data(ad_file,click_log_file,user_file, n_class=2, one_hot=True):
    '''load data from .csv file'''
    
    ad = pd.read_csv(ad_file, sep=',')
    click_log = pd.read_csv(click_log_file, sep=',')
    ad_click = pd.merge(ad,click_log,on='creative_id')
    ad_click = ad_click.sort_values(by=['user_id','time'], ascending=[True,True])
    # 335
    # 44313
    # 进行\N的处理
    ad_click['industry'] = ad_click['industry'].astype(str).replace(r'\N','336').astype(int)
    #test_ad_click['industry'] = test_ad_click['industry'].apply(lambda x: 0 if str(x)==r'\N' else int(str(x)))
    ad_click['product_id'] =ad_click['product_id'].astype(str).replace(r'\N','44314').astype(int)
    
    #data.loc[data['Y'] == 'T'] = 1
    #进行unk处理
    #print(ad_click.columns)
    """
    Index(['creative_id', 'ad_id', 'product_id', 'product_category',
       'advertiser_id', 'industry', 'time', 'user_id', 'click_times'],
      dtype='object')
    """
    #进行unk处理
    # 进行unk处理
    ad_click.loc[ad_click['ad_id'] > 3812200,'ad_id'] = 3812201
    #print("6",len(ad_click.groupby('user_id')))
    ad_click.loc[ad_click['advertiser_id'] > 62965,'advertiser_id'] = 62966 # 多了1
    #print("7",len(ad_click.groupby('user_id')))
    # ad_click.loc[ad_click['click_times']>3812200] = 3812201
    ad_click.loc[ad_click['creative_id'] > 4445718,'creative_id'] = 4445719
    #print("8",len(ad_click.groupby('user_id')))
    ad_click.loc[ad_click['industry'] > 336,'industry'] = 337
    #print("9",len(ad_click.groupby('user_id')))
    ad_click.loc[ad_click['product_category'] > 18,'product_category'] = 19
    #print("10",len(ad_click.groupby('user_id')))
    ad_click.loc[ad_click['product_id'] > 44314,'product_id'] = 44315  #这里
    #print("11",len(ad_click.groupby('user_id')))
    # ad_click.loc[ad_click['time']>3812200] = 3812201

    ad_id = []
    advertiser_id = []
    click_times = []
    creative_id = []
    industry = []
    product_category = []
    product_id = []
    time = []

    print("start train data......")
    # 这里相当耗时
    ad_click  = reduce_mem_usage(ad_click)
    for user_id,df in  tqdm(ad_click.groupby('user_id')):
        ad_id.append(list(df['ad_id'].values))
        advertiser_id.append(list(df['advertiser_id'].values))
        click_times.append(list(df['click_times'].values))
        creative_id.append(list(df['creative_id'].values))
        industry.append(list(df['industry'].values))
        product_category.append(list(df['product_category'].values))
        product_id.append(list(df['product_id'].values))
        time.append(list(df['time'].values))

    age = pd.read_csv(user_file, sep=',')['age'].values
    gender  = pd.read_csv(user_file, sep=',')['gender'].values

    #shuffle_csv = csv_file.sample(frac=sample_ratio)
    #x_ad_id = pd.Series(shuffle_csv["content"])

    x = []

    #
    x.append(ad_id)
    x.append(advertiser_id)
    x.append(click_times)
    x.append(creative_id)
    x.append(industry)
    x.append(product_category)
    x.append(product_id)
    x.append(time)


    """
    ad_id = pd.Series(shuffle_csv["ad_id"])
    advertiser_id = pd.Series(shuffle_csv["advertiser_id"])
    click_times = pd.Series(shuffle_csv["click_times"])
    creative_id = pd.Series(shuffle_csv["creative_id"])
    industry= pd.Series(shuffle_csv["industry"])
    product_category = pd.Series(shuffle_csv["product_category"])
    product_id = pd.Series(shuffle_csv["product_id"])
    time = pd.Series(shuffle_csv["time"])
    
    y1 = pd.Series(shuffle_csv["age"])
    y2 = pd.Series(shuffle_csv["gender"])
    """
    #y = pd.Series(shuffle_csv["class"])
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #y = age   n_class=10
   
    y = gender #n_class=2
    if one_hot:
        y = to_one_hot(y, n_class)
    return x, y


def load_test_data(ad_file,click_log_file):
    '''load data from .csv file'''

    ad = pd.read_csv(ad_file, sep=',')
    click_log = pd.read_csv(click_log_file, sep=',')
    ad_click = pd.merge(ad,click_log,on='creative_id')
    ad_click = ad_click.sort_values(by=['user_id','time'], ascending=[True,True])
    # 335
    # 44313
    ad_click['industry'] = ad_click['industry'].astype(str).replace(r'\N','336').astype(int)
    #test_ad_click['industry'] = test_ad_click['industry'].apply(lambda x: 0 if str(x)==r'\N' else int(str(x)))
    ad_click['product_id'] =ad_click['product_id'].astype(str).replace(r'\N','443140').astype(int)

    # 进行unk处理
    # 进行unk处理
    ad_click.loc[ad_click['ad_id'] > 3812200,'ad_id'] = 3812201
    #print("6",len(ad_click.groupby('user_id')))
    ad_click.loc[ad_click['advertiser_id'] > 62965,'advertiser_id'] = 62966 # 多了1
    #print("7",len(ad_click.groupby('user_id')))
    # ad_click.loc[ad_click['click_times']>3812200] = 3812201
    ad_click.loc[ad_click['creative_id'] > 4445718,'creative_id'] = 4445719
    #print("8",len(ad_click.groupby('user_id')))
    ad_click.loc[ad_click['industry'] > 336,'industry'] = 337
    #print("9",len(ad_click.groupby('user_id')))
    ad_click.loc[ad_click['product_category'] > 18,'product_category'] = 19
    #print("10",len(ad_click.groupby('user_id')))
    ad_click.loc[ad_click['product_id'] > 44314,'product_id'] = 44315  #这里
    #print("11",len(ad_click.groupby('user_id')))
    # ad_click.loc[ad_click['time']>3812200] = 3812201

    x = []
    ad_id = []
    advertiser_id = []
    click_times = []
    creative_id = []
    industry = []
    product_category = []
    product_id = []
    time = []
    print("load test data......")
    # 这里相当耗时
    ad_click  = reduce_mem_usage(ad_click)
    for user_id,df in  tqdm(ad_click.groupby('user_id')):
        ad_id.append(list(df['ad_id'].values))
        advertiser_id.append(list(df['advertiser_id'].values))
        click_times.append(list(df['click_times'].values))
        creative_id.append(list(df['creative_id'].values))
        industry.append(list(df['industry'].values))
        product_category.append(list(df['product_category'].values))
        product_id.append(list(df['product_id'].values))
        time.append(list(df['time'].values))
    #

    x.append(ad_id)
    x.append(advertiser_id)
    x.append(click_times)
    x.append(creative_id)
    x.append(industry)
    x.append(product_category)
    x.append(product_id)
    x.append(time)

    #x = (ad_id,advertiser_id,click_times,creative_id,industry,product_category,product_id,time)
    #x = tuple([])
    user_id = click_log[['user_id']]
    # user_id是DataFrame数据用来生成提交文件
    return x,user_id

# max_len = 300/400
def data_preprocessing(train, test, max_len):
    """transform to one-hot idx vector by VocabularyProcessor"""
    """VocabularyProcessor is deprecated, use v2 instead"""
    #vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
    x_transform_train = vocab_processor.fit_transform(train)
    x_transform_test = vocab_processor.transform(test)
    vocab = vocab_processor.vocabulary_
    vocab_size = len(vocab)
    x_train_list = list(x_transform_train)
    x_test_list = list(x_transform_test)
    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)

    return x_train, x_test, vocab, vocab_size
    #不需要词典但是需要词典大小

#max_len = 300
# 测试数据或者训练数据padding
def data_preprocessing_v2(train, max_len):
    #进行pad的处理
    #print(train)
    # sequences：浮点数或整数构成的两层嵌套列表

    #print(train[0])
    ad_id = pad_sequences(train[0], maxlen=max_len, padding='post', truncating='post')
    advertiser_id = pad_sequences(train[1], maxlen=max_len, padding='post', truncating='post')
    click_times = pad_sequences(train[2], maxlen=max_len, padding='post', truncating='post')
    creative_id = pad_sequences(train[3], maxlen=max_len, padding='post', truncating='post')
    industry = pad_sequences(train[4], maxlen=max_len, padding='post', truncating='post')
    product_category = pad_sequences(train[5], maxlen=max_len, padding='post', truncating='post')

    product_id = pad_sequences(train[6], maxlen=max_len, padding='post', truncating='post')
    #print(train[7])
    time = pad_sequences(train[7], maxlen=max_len, padding='post', truncating='post')
    #train_idx = tokenizer.texts_to_sequences(train)
    #test_idx = tokenizer.texts_to_sequences(test)
    train_padded = [ad_id,advertiser_id,click_times,creative_id,industry,product_category,product_id,time]
    #train_padded = [ad_id,advertiser_id,click_times,creative_id,industry,product_category,product_id,time]
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded

"""
def data_preprocessing_v2(train, test, max_len, max_words=50000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, max_words + 2
"""

def data_preprocessing_with_dict(train, test, max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, tokenizer.word_docs, tokenizer.word_index, len(tokenizer.word_docs) + 2

# 随机划分数据集
def split_dataset(x_test, y_test, dev_ratio):
    """split test dataset to test and dev set with ratio """
    """split train dataset to train and dev set with ratio """
    test_size = len(x_test[0])
    #print(test_size)
    dev_size = (int)(test_size * dev_ratio)
    #print(dev_size)
    x_0 = x_test[0][:dev_size]
    x_1 = x_test[1][:dev_size]
    x_2 = x_test[2][:dev_size]
    x_3 = x_test[3][:dev_size]
    x_4 = x_test[4][:dev_size]
    x_5 = x_test[5][:dev_size]
    x_6 = x_test[6][:dev_size]
    x_7 = x_test[7][:dev_size]

    x_test_0 = x_test[0][dev_size:]
    x_test_1 = x_test[1][dev_size:]
    x_test_2 = x_test[2][dev_size:]
    x_test_3 = x_test[3][dev_size:]
    x_test_4 = x_test[4][dev_size:]
    x_test_5 = x_test[5][dev_size:]
    x_test_6 = x_test[6][dev_size:]
    x_test_7 = x_test[7][dev_size:]

    x_dev =  [x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7]
    x_test =  [x_test_0, x_test_1, x_test_2, x_test_3, x_test_4, x_test_5, x_test_6, x_test_7]
    y_dev = y_test[:dev_size]
    y_test = y_test[dev_size:]
    return x_test, x_dev, y_test, y_dev,dev_size

# 打乱数据
def shuffle_me(x_train,y_train):
    x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, shuffled_y = shuffle(x_train[0], x_train[1], x_train[2], x_train[3],
                                                                 x_train[4], \
                                                                 x_train[5], x_train[6], x_train[7], y_train)
    shuffled_x = [x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7]
    return shuffled_x,shuffled_y

# 我们的数据为3维列表data_x
def fill_feed_dict(data_X, data_Y, batch_size):
    #Generator to yield batches
    # Shuffle data first.
    shuffled_X, shuffled_Y = shuffle_me(data_X, data_Y)
    # print("before shuffle: ", data_Y[:10])
    # print(data_X.shape[0])
    # perm = np.random.permutation(data_X.shape[0])
    # data_X = data_X[perm]
    # shuffled_Y = data_Y[perm]
    # print("after shuffle: ", shuffled_Y[:10])
    for idx in range(data_X[0].shape[0] // batch_size+1):
        if idx!=data_X[0].shape[0] // batch_size:
            x_0 = shuffled_X[0][batch_size * idx: batch_size * (idx + 1)]
            x_1 = shuffled_X[1][batch_size * idx: batch_size * (idx + 1)]
            x_2 = shuffled_X[2][batch_size * idx: batch_size * (idx + 1)]
            x_3 = shuffled_X[3][batch_size * idx: batch_size * (idx + 1)]
            x_4 = shuffled_X[4][batch_size * idx: batch_size * (idx + 1)]
            x_5 = shuffled_X[5][batch_size * idx: batch_size * (idx + 1)]
            x_6 = shuffled_X[6][batch_size * idx: batch_size * (idx + 1)]
            x_7 = shuffled_X[7][batch_size * idx: batch_size * (idx + 1)]
        #处理最后一个batch碎片
        else:
            x_0 = shuffled_X[0][batch_size * idx: ]
            x_1 = shuffled_X[1][batch_size * idx: ]
            x_2 = shuffled_X[2][batch_size * idx: ]
            x_3 = shuffled_X[3][batch_size * idx: ]
            x_4 = shuffled_X[4][batch_size * idx: ]
            x_5 = shuffled_X[5][batch_size * idx: ]
            x_6 = shuffled_X[6][batch_size * idx: ]
            x_7 = shuffled_X[7][batch_size * idx: ]
        x_batch = [x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7]

        y_batch = shuffled_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch, y_batch
def fill_feed_dev_dict(data_X, data_Y, batch_size):
    #Generator to yield batches
    # Shuffle data first.
    #shuffled_X, shuffled_Y = shuffle_me(data_X, data_Y)
    # print("before shuffle: ", data_Y[:10])
    # print(data_X.shape[0])
    # perm = np.random.permutation(data_X.shape[0])
    # data_X = data_X[perm]
    # shuffled_Y = data_Y[perm]
    # print("after shuffle: ", shuffled_Y[:10])
    for idx in range(data_X[0].shape[0] // batch_size+1):
        if idx!=data_X[0].shape[0] // batch_size:
            x_0 = data_X[0][batch_size * idx: batch_size * (idx + 1)]
            x_1 = data_X[1][batch_size * idx: batch_size * (idx + 1)]
            x_2 = data_X[2][batch_size * idx: batch_size * (idx + 1)]
            x_3 = data_X[3][batch_size * idx: batch_size * (idx + 1)]
            x_4 = data_X[4][batch_size * idx: batch_size * (idx + 1)]
            x_5 = data_X[5][batch_size * idx: batch_size * (idx + 1)]
            x_6 = data_X[6][batch_size * idx: batch_size * (idx + 1)]
            x_7 = data_X[7][batch_size * idx: batch_size * (idx + 1)]
            y_batch = data_Y[batch_size * idx: batch_size * (idx + 1)]
        #处理最后一个batch碎片
        else:
            x_0 = data_X[0][batch_size * idx: ]
            x_1 = data_X[1][batch_size * idx: ]
            x_2 = data_X[2][batch_size * idx: ]
            x_3 = data_X[3][batch_size * idx: ]
            x_4 = data_X[4][batch_size * idx: ]
            x_5 = data_X[5][batch_size * idx: ]
            x_6 = data_X[6][batch_size * idx: ]
            x_7 = data_X[7][batch_size * idx: ]
            y_batch = data_Y[batch_size * idx: ]
        x_batch = [x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7]

        
        yield x_batch, y_batch
#测试的时候不打乱
def fill_feed_test_dict(shuffled_X, batch_size):
    #num_batch = data_X[0].shape[0] // batch_size
    for idx in range(shuffled_X[0].shape[0] // batch_size):
        if idx!=shuffled_X[0].shape[0] // batch_size:
            x_0 = shuffled_X[0][batch_size * idx: batch_size * (idx + 1)]
            x_1 = shuffled_X[1][batch_size * idx: batch_size * (idx + 1)]
            x_2 = shuffled_X[2][batch_size * idx: batch_size * (idx + 1)]
            x_3 = shuffled_X[3][batch_size * idx: batch_size * (idx + 1)]
            x_4 = shuffled_X[4][batch_size * idx: batch_size * (idx + 1)]
            x_5 = shuffled_X[5][batch_size * idx: batch_size * (idx + 1)]
            x_6 = shuffled_X[6][batch_size * idx: batch_size * (idx + 1)]
            x_7 = shuffled_X[7][batch_size * idx: batch_size * (idx + 1)]
        #处理最后一个batch碎片
        else:
            x_0 = shuffled_X[0][batch_size * idx: ]
            x_1 = shuffled_X[1][batch_size * idx: ]
            x_2 = shuffled_X[2][batch_size * idx: ]
            x_3 = shuffled_X[3][batch_size * idx: ]
            x_4 = shuffled_X[4][batch_size * idx: ]
            x_5 = shuffled_X[5][batch_size * idx: ]
            x_6 = shuffled_X[6][batch_size * idx: ]
            x_7 = shuffled_X[7][batch_size * idx: ]
        x_batch = [x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7]

        yield  x_batch
        
        
"""
def fill_feed_dict(data_X, data_Y, batch_size):
    #Generator to yield batches
    # Shuffle data first.
    shuffled_X, shuffled_Y = shuffle(data_X, data_Y)
    # print("before shuffle: ", data_Y[:10])
    # print(data_X.shape[0])
    # perm = np.random.permutation(data_X.shape[0])
    # data_X = data_X[perm]
    # shuffled_Y = data_Y[perm]
    # print("after shuffle: ", shuffled_Y[:10])
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = shuffled_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = shuffled_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch, y_batch
"""


# pad  unk 