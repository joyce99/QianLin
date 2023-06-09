import spacy, os, numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from gensim.corpora import Dictionary as gensim_dico

nlp = spacy.load("en_core_web_sm")

def semeval_train_dataset(cor,max_words):
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
    dico.filter_extremes(no_below=1, keep_n= max_words)
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
    datasets = (X, np.array(Y))

    return datasets,dico
def semeval_test_dataset(cor,max_words):
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
    dico.filter_extremes(no_below=1, keep_n= max_words)
    dico.compactify()


    #标签和词袋特征
    for i in label_string:
        if i == 'positive':
            lv = 1
            Y.append(lv)
        elif i == 'negative':
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
    datasets = (X, np.array(Y))

    return datasets,dico
def semeval_dataset(cor, max_words):
    dico = gensim_dico()
    datasets = {}
    tokens_list = []
    label_string = []
    word = []
    X = []
    Y = []
    # 首先将句子和标签分离开来
    for i in cor:
        label_string.append(i[0])
        word.append(i[1])

    # first pass list to bulid dictionary(首先根据word里面的句子来构建字典)
    for tok in word:
        ts = tok.split(sep=' ')
        _ = dico.doc2bow(ts, allow_update=True)

    # processing_options
    dico.filter_extremes(no_below=1, keep_n=max_words)
    dico.compactify()

    # 标签和词袋特征

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
        count_list = dico.doc2bow(ts, allow_update=False)
        docid += 1
        X.append((docid, count_list))
    datasets = (X, np.array(Y))

    return datasets, dico


def parse_processed_amazon_dataset(FNames, max_words):
    datasets = {}
    dico = gensim_dico()

    # First pass on document to build dictionary
    for fname in FNames:
        f = open(fname, encoding='utf-8')
        for l in f:
            tokens = l.split(sep=' ')
            label_string = tokens[-1]
            tokens_list = []
            for tok in tokens[:-1]:
                ts, tfreq = tok.split(':')
                freq = int(tfreq)
                tokens_list += [ts] * freq

            _ = dico.doc2bow(tokens_list, allow_update=True)

        f.close()

    # Preprocessing_options
    dico.filter_extremes(no_below=2, keep_n=max_words)
    dico.compactify()

    for fname in FNames:
        X = []
        Y = []
        docid = -1
        f = open(fname, encoding='utf-8')
        for l in f:
            tokens = l.split(sep=' ')
            label_string = tokens[-1]
            tokens_list = []
            for tok in tokens[:-1]:
                ts, tfreq = tok.split(':')
                freq = int(tfreq)
                tokens_list += [ts] * freq

            count_list = dico.doc2bow(tokens_list, allow_update=False)

            docid += 1

            X.append((docid, count_list))

            # Preprocess Label
            ls, lvalue = label_string.split(':')
            if ls == "#label#":
                if lvalue.rstrip() == 'positive':
                    lv = 1
                    Y.append(lv)
                elif lvalue.rstrip() == 'negative':
                    lv = 0
                    Y.append(lv)
                else:
                    raise Exception("Invalid Label Value")
            else:
                raise Exception('Invalid Format')

        datasets[fname] = (X, np.array(Y))
        f.close()
        del f

    return datasets, dico


def count_list_to_sparse_matrix(X_list, dico):
    ndocs = len(X_list)
    voc_size = len(dico.keys())

    X_spmatrix = sp.lil_matrix((ndocs, voc_size))
    for did, counts in X_list:
        for wid, freq in counts:
            X_spmatrix[did, wid] = freq

    return X_spmatrix.tocsr()


def get_dataset_path(domain_name, exp_type):
    prefix = './dataset/'
    if exp_type == 'small':
        fname = 'labelled.review'
    elif exp_type == 'all':
        fname = 'all.review'
    elif exp_type == 'test':
        fname = 'unlabeled.review'

    return os.path.join(prefix, domain_name, fname)

def sem_get_domain_dataset(sem, max_words=5000):
    """
    Returns training (small) / test split of a single domain
    """
    datasets, dico = semeval_dataset(sem,max_words)
    L_s, _ = datasets
    X_s = count_list_to_sparse_matrix(L_s, dico)
    X_s = np.array(X_s.todense())
    return X_s, dico
def get_domain_dataset(domain_name, max_words=5000, exp_type='small'):
    """
    Returns training (small) / test split of a single domain
    """
    domain_path = get_dataset_path(domain_name, exp_type)
    datasets, dico = parse_processed_amazon_dataset([domain_path], max_words)

    L_s, _ = datasets[domain_path]
    X_s = count_list_to_sparse_matrix(L_s, dico)
    X_s = np.array(X_s.todense())

    return X_s, dico

def sem_get_dataset(semeval_train,max_words=5000):
    """
    Returns training (small) / test split of a single domain
    """
    datasets1, dico1 = semeval_train_dataset(semeval_train,max_words)
    #datasets2, dico2 = semeval_test_dataset(semeval_test, max_words)
    L_s, Y_s = datasets1
    X_s = count_list_to_sparse_matrix(L_s,dico1)
    #L_t, Y_t = datasets2
    #X_t = count_list_to_sparse_matrix(L_t, dico2)
    return X_s, Y_s
def get_dataset(target_name, max_words=5000):
    """
    Returns source domain, target domain paired dataset
    """
    #source_path = get_dataset_path(source_name, 'small')
    target_path1 = get_dataset_path(target_name, 'small')
    target_path2 = get_dataset_path(target_name, 'test')

    dataset_list = [target_path1,target_path2]
    datasets, dico = parse_processed_amazon_dataset(dataset_list, max_words)

    #L_s, Y_s = datasets[source_path]
    L_t1, Y_t1 = datasets[target_path1]
    L_t2, Y_t2 = datasets[target_path2]

    #X_s = count_list_to_sparse_matrix(L_s, dico)
    X_t1 = count_list_to_sparse_matrix(L_t1, dico)
    X_t2 = count_list_to_sparse_matrix(L_t2, dico)

    return X_t1,Y_t1,X_t2,Y_t2,dico


def spacy_seed_concepts(dico):
    """
    Returns concepts which belongs to proper noun, noun, adjective, or adverb parts-of-speech-tag category
    """
    seeds = []
    concepts = list(dico.values())
    tags = ['PROPN', 'NOUN', 'ADJ', 'ADV']

    for item in tqdm(concepts):
        if '_' not in item:
            doc = nlp(item)
            switch = 0
            for token in doc:
                if token.pos_ not in tags:
                    switch = 1
                    break
                else:
                    continue

            if switch == 0:
                seeds.append(item)

    return set(seeds)


def spacy_seed_concepts_list(concepts):
    """
    Returns concepts which belongs to proper noun, noun, adjective, or adverb parts-of-speech-tag category
    """
    seeds = []
    tags = ['PROPN', 'NOUN', 'ADJ', 'ADV']

    for item in concepts:
        if '_' not in item:
            doc = nlp(item)
            switch = 0
            for token in doc:
                if token.pos_ not in tags:
                    switch = 1
                    break
                else:
                    continue

            if switch == 0:
                seeds.append(item)

    return set(seeds)


def obtain_all_seed_concepts(max_words):
    """
    Returns seed concepts drwan from all the domains
    """
    _, dico1 = get_domain_dataset('dvd', max_words)
    _, dico2 = get_domain_dataset('electronics', max_words)
    _, dico3 = get_domain_dataset('kitchen', max_words)
    _, dico4 = get_domain_dataset('books', max_words)
    x = np.load('semeval_2013_train.npy', allow_pickle=True)
    semeval_train = x.tolist()
    _, dico5 = semeval_dataset(semeval_train, 5000)
    concepts = list(set(dico1.values()) \
                    .union(set(dico2.values())) \
                    .union(set(dico3.values())) \
                    .union(set(dico4.values())))

    seeds1 = spacy_seed_concepts(dico1)
    seeds2 = spacy_seed_concepts(dico2)
    seeds3 = spacy_seed_concepts(dico3)
    seeds4 = spacy_seed_concepts(dico4)
    seeds5 = spacy_seed_concepts(dico5)
    all_seeds = list(seeds5)

    return all_seeds