import jieba      #分词包
import gensim     #NLP包
import copy
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
tf.compat.v1.disable_eager_execution()      #消除默认情况下启用的立即执行
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import multiprocessing
cpu_count = multiprocessing.cpu_count()   #获取cpu核的数目
import sys
sys.setrecursionlimit(1000000)    #解除python默认对递归深度的限制
import yaml       #一种文件格式，调用该模块进行该文件（model）读写
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt


def loadfile():
    """
    加载neg.csv，pos.csv，neutral.csv数据集，输出combined为一系列21088个评论，y为对应结果
    """
    neg = pd.read_csv('neg.csv', header=None, index_col=None)
    pos = pd.read_csv('pos.csv', header=None, index_col=None, error_bad_lines=False)   #error_bad_lines=False处理某行出错直接跳过
    neu = pd.read_csv('neutral.csv', header=None, index_col=None)
    combined = np.concatenate((pos[0], neu[0], neg[0]))     #pos[0]即pos表格的首列，按列合并,结果为单列数组
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neu), dtype=int), -1*np.ones(len(neg), dtype=int)))  #训练集的实际结果y
    return combined, y


def tokenizer(text):
    """
    对输入评论集text去掉换行后进行分词,输出二维词组(21088行，列对应各评论长度)
    """
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


def create_dictionaries(model, combined):
    gensim_dict = gensim.corpora.dictionary.Dictionary(combined)
    gensim_dict.filter_extremes(no_below=10)
    gensim_dict.save('dictionary.dict')    #输出gensim_dict为语料库combined中7753个词（去重后且词频不低于10)和编号(从0开始)的键值对，{ID:word}
    print('success save dict')             #通过gensim_dict.id2token查看具体键值对(word:ID)
    # corpus = [gensim_dict.doc2bow(text) for text in combined]   #稀疏矩阵。对语料库中的每个句子，一系列元组（词语ID，本句出现次数）构成的列表
    # gensim_dict.doc2bow(model.vocab.keys(), allow_update=True)
    w2indx = {word: id+1 for id, word in gensim_dict.items()}   #所有频数超过10的词语的索引,(k->v)=>(v->k)，dict.items()返回dict中所有键值对的list数据
    w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量, (word:model(word)) ,dict.keys()返回dict中所有键的list数据
    """
    w2indx：其中index=0空闲，留作记录生僻词；其他index相对gensim_dict中的ID+1
    w2vec：word: model[word]键值对的字典
    """
    def parse_dataset(combined): # 返回语料库combind中各单词在w2indx的索引，生僻词用索引0表示，数据类型二维数组
        data = []
        for sentence in combined:
            new_txt = []
            for word in sentence:
                try:
                    new_txt.append(w2indx[word])
                except:
                    new_txt.append(0) # freqxiao10->0
            data.append(new_txt)
        return data
    combined = parse_dataset(combined)      #将二维词表combind中的词转化为对应id（根据w2indx）
    combined = tf.keras.preprocessing.sequence.pad_sequences(combined, maxlen=100)  #截取句子，使得每个句子所含词语对应的索引，为固定长度100
    return w2indx, w2vec, combined



def word2vec_train(combined):
    model = gensim.models.word2vec.Word2Vec(size=100, min_count=10, window=7, workers=cpu_count, iter=1)
    """
    size：特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百
    min_count：词频少于min_count次数的单词会被丢弃掉, 默认值为5
    window：当前词与预测词在一个句子中的最大距离，某一个中心词可能与前后多个词相关
    workers：参数控制训练的并行数
    iter：迭代次数，默认为5
    输出model为词向量矩阵
    """
    model.build_vocab(combined)   # input: list
    model.train(combined, epochs=model.epochs, total_examples=model.corpus_count)
    model.save('Word2vec_model.pkl')           #依据combined训练并保存词向量模型,通过model[word]获取词word的词向量
    print('success save word2vec model')

    index_dict, word_vectors, combined = create_dictionaries(model, combined)
    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):
    '''
    n_symbols：所有单词索引数目，加上频数小于10的词语索引为0
    embedding_weights：词向量数组，行号为单词对应索引号
    '''
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, 100))  # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items():          # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]   #为n_symbols * 100的矩阵，保存index_dict中每个word的词向量，word的index即行号

    # combined_y = np.concatenate((combined, y.reshape(-1, 1)), axis=1)     #将combined，y按列合并(21088, 101)
    # train_combined_y = copy.deepcopy(combined_y)
    # test_combined_y = copy.deepcopy(combined_y)
    # random_list_less = random.sample(range(len(y)), int(len(y)*0.2))
    # random_list_more = list(set(range(len(y))).difference(set(random_list_less)))
    # train_combined_y = np.delete(train_combined_y, random_list_less, axis=0)              #(16871, 101)
    # test_combined_y = np.delete(test_combined_y, random_list_more, axis=0)                #(4217, 101) 将combined_y分割为train_combined_y和test_combined_y
    # train_y = train_combined_y[:, -1].reshape(-1, 1)                  #(16871, 1)
    # train_x = train_combined_y[:, 0:-1]                        #(16871, 100)
    # test_y = test_combined_y[:, -1].reshape(-1, 1)                    #(4217, 1)
    # test_x = test_combined_y[:, 0:-1]                          #(4217, 100) 对train_combined_y和test_combined_y的combined_y切分

    x_train, x_valid, y_train, y_valid = train_test_split(combined, y.reshape(-1, 1), test_size=0.2)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_valid = tf.keras.utils.to_categorical(y_valid, num_classes=3)    #将y结果由一维扩展为三维one_hot（1:[0 1 0],0:[1 0 0],-1:[0 0 1]）
    print(x_train.shape)
    print(y_train.shape)
    return n_symbols, embedding_weights, x_train, y_train, x_valid, y_valid


batch_size = 32
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_valid, y_valid):
    print('Defining a Simple LSTM Model...')
    model = keras.models.Sequential()  # or Graph or whatever
    model.add(keras.layers.Embedding(output_dim=100,              #词向量维度
                                     input_dim=n_symbols,         #样本数
                                     mask_zero=True,              #是否将输入中的‘0’看作是应该被忽略的‘填充’
                                     weights=[embedding_weights], #直接给定词向量初始值
                                     input_length=100))           # 每个评论的词数
    """输入为(batch_size, input_length)，输出为(batch_size, input_length, output_dim)"""
    model.add(keras.layers.LSTM(units=128, activation='tanh'))    #units为LSTM节点个数
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(3, activation='softmax'))   # Dense=>全连接层,输出维度=3
    model.add(keras.layers.Activation('softmax'))

    print('trainning LSTM Model...')
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=10, validation_data=(x_valid, y_valid))

    """保存模型结构和模型参数"""
    yaml_string = model.to_yaml()
    with open('lstm_model_structure.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    print('success save lstm_model_structure')
    model.save_weights('lstm_model_weights.h5')
    print('success save lstm_model_weights')

    def plot_learning_curves(history, label, epochs, min, max):
        data = {}
        data[label] = history.history[label]
        data['val_' + label] = history.history['val_' + label]
        pd.DataFrame(data).plot(figsize=(8, 5))
        plt.grid(True)
        plt.axis([0, epochs, min, max])
        plt.show()

    """作图"""
    plot_learning_curves(history, 'accuracy', 30, 0, 1)
    plot_learning_curves(history, 'loss', 30, 0, 1)


print('loading data')
combined, y = loadfile()
print('the length of trainning test is %s' % len(combined))

print('cutting sentence')
combined = tokenizer(combined)

index_dict, word_vectors, combined = word2vec_train(combined)
n_symbols, embedding_weights, x_train, y_train, x_valid, y_valid = get_data(index_dict, word_vectors, combined, y)
train_lstm(n_symbols, embedding_weights, x_train, y_train, x_valid, y_valid)



