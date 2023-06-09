import spacy, os, numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from gensim.corpora import Dictionary as gensim_dico
def semeval_dataset(cor,max_words):
    dico = gensim_dico()
    datasets = {}
    tokens_list = []
    label_string = []
    word = []
    X= []
    Y = []
    #首先将句子和标签分离开来
    for i in cor:
        label_string.append(i[0])
        word.append(i[1])

    #first pass list to bulid dictionary(首先根据word里面的句子来构建字典)
    for tok in word:
            ts = tok.split(sep=' ')
            _= dico.doc2bow(ts,allow_update= True)

    #processing_options
    dico.filter_extremes(no_below=2, keep_n= max_words)
    dico.compactify()


    #标签和词袋特征
    for i in label_string:
        if i == '"positive"':
            lv = 1
            Y.append(lv)
        elif i == '"negative"':
            lv = 0
            Y.append(lv)
        else:
            raise Exception("Invalid Label Value")
    docid = -1
    for tok in word:
            ts = tok.split(sep=' ')
            count_list = dico.doc2bow(ts,allow_update= False)
            docid += 1
            X.append((docid,count_list))
    datasets = (X,np.array(Y))

    return datasets,dico

if __name__ == '__main__':
    bow_size = 5000
    x = np.load('semeval_2013_train.npy', allow_pickle=True)
    semeval_train = x.tolist()
    _,dico= semeval_dataset(semeval_train, bow_size)
    y = np.load('semeval_2013_test.npy', allow_pickle=True)
    X1 = np.load('semeval_2016_train.npy', allow_pickle=True)
    y1 = np.load('semeval_2016_test.npy', allow_pickle=True)








