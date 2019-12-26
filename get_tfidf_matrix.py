import datetime
import numpy as np
import pandas as pd
import random
from sklearn import preprocessing, datasets
from sklearn.datasets import fetch_20newsgroups, load_wine, load_diabetes, fetch_rcv1, load_svmlight_file
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import jieba
import jieba.posseg as pseg
from gensim import corpora, models
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
np.random.seed(10000)
data_folder = r'dataset\\'

def get_20newsgroups_data(categories,n_features):
    #通过sklearn自带数据集提取数据
    dataset = fetch_20newsgroups(subset='all', categories=categories,
                                 shuffle=True, random_state=42)
    dataset=dataset[:len(categories)*100]
    labels=dataset.target
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
    X=vectorizer.fit_transform(dataset.data)
    print("n_samples: %d, n_features: %d" % X.shape)
    return labels,X

def reduce_dimension(data):
    print("Performing dimensionality reduction using LSA")
    svd = TruncatedSVD()
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(data)
    return X

def tokenization(text):
    corpus = []
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    stopwords = open("stopwords.txt").read().split()
    for content in text:
        # 分词  and not word.isdigit()
        words = []
        # print(content)
        content.replace("reuter", "")
        word_list = pseg.cut(content)
        for word, flag in word_list:
            if flag not in stop_flag and word not in stopwords:
                words.append(word)
        corpus.append(words)
    return corpus

def jieba_tokenize(text):
    return jieba.lcut(text)

def tfidf(corpus):
    dictionary = corpora.Dictionary(corpus)
    doc_vectors = [dictionary.doc2bow(text) for text in corpus]
    tfidf = models.TfidfModel(doc_vectors, normalize=True)
    tfidf_vector = tfidf[doc_vectors]
    #words = [doc for doc in tfidf_vector]
    #print(words)
    return tfidf_vector

def to_csr_matrix(tfidf_vector):
    data = []
    rows = []
    cols = []
    line_count = 0
    for line in tfidf_vector:
        for elem in line:
            # print(elem)
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1
    tfidf_matrix = csr_matrix((data, (rows, cols))).toarray()
    return tfidf_matrix

'''
分词，去停用词并按主题收集文档
max_test_doc_num:每个主题（类别）下测试的最大文档数
low_words_num：分词后每个文档包含的最小词项数
up_words_num：分词后每个文档包含的最大词项数
'''
def text_dic(topics, file_name,max_test_doc_num,low_words_num,up_words_num):
    labels = []
    corpus = []
    topic_docs = dict.fromkeys(topics,0)
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    stopwords = open("stopwords.txt").read().split()
    stopwords.append("reuter")
    itt = 0
    jtt = 0
    lmax = 0
    dataset_info={}
    with open(data_folder + file_name, 'r') as fp:
        text = fp.readlines()
        for content in text:
            itt = itt+1
            words = []
            topic=content.strip().split("\t")[0] #the topic uses tab to seperate itself and its contents
            doc=content.strip().split("\t")[1:] #the rest is the content
            #print(topic)
            word_list = pseg.cut(str(doc))
            for word,flag in word_list:
                if flag not in stop_flag and word not in stopwords:
                    words.append(word)

            if topic in topics:
                if topic not in topic_docs:
                    topic_docs[topic] = 0  #Num of documents in each topics
                if topic_docs[topic] <max_test_doc_num:
                    lmax=max(lmax,len(words))
                    if len(words)>low_words_num:   #the documents should have enough words
                        topic_docs[topic] += 1
                        corpus.append(words)    #all documents No.
                        labels.append(topics.index(topic)) #if no word should not be appended
                    else:
                        jtt = jtt+1 #count the lines with not enough words

    tfidf_vector = tfidf(corpus)
    tfidf_matrix = to_csr_matrix(tfidf_vector)
    labels=np.array(labels)
    ########################################
    path_info = data_folder + file_name + "_dic.txt"
    mylog=open(path_info, mode = 'w', encoding = 'utf-8')
    dataset_info['file_name']=file_name
    dataset_info['max_test_doc_num']=max_test_doc_num
    dataset_info['low_words_num']=low_words_num
    dataset_info['total_docs']=itt
    dataset_info['words no enough count']=jtt
    dataset_info['words enough count']=itt-jtt
    dataset_info['max len for one document']=lmax
    dataset_info['NO. of documents']=len(corpus)
    dataset_info["n_cluster"]=len(topics)
    dataset_info["shape pf tfidf_matrix"]=tfidf_matrix.shape
    dataset_info['NO. of documents group by topic']=''
    for topic,docs_num in topic_docs.items():
        dataset_info[topic]=docs_num

    for key, val in dataset_info.items():
        # print(key + ": " + str(val))
        mylog.write(key + ": " + str(val) + '\n')

    mylog.close()
    return dataset_info,labels, tfidf_matrix, topic_docs

def get_uci_data(file_dir,data_id,file_list):
    dataset_info={}
    print(file_list[data_id])
    file_path = file_dir + file_list[data_id] + ".dat"
    df = pd.read_csv(file_path, header=None,sep='\t').values
    labels=df[:,-1]
    data=df[:,:-1]
    n_cluster=len(np.unique(labels))

    dataset_info["dataset_name"]=file_list[data_id]
    dataset_info["n_cluster"]=n_cluster
    dataset_info["shape pf tfidf_matrix"]=data.shape
    # dataset_info['labels']=labels
    return data,labels,n_cluster

def get_data_matrix(test_data_id,max_test_doc_num,low_words_num,up_words_num,is_shuff):
    topics_webkb4=["project", "course", "faculty",  "student"]
    topics_r5 = ["coffee", "crude", "interest", "sugar", "trade"]
    news20_topics=["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc","comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
                   "comp.windows.x","misc.forsale","rec.autos","rec.motorcycles", "rec.sport.baseball","rec.sport.hockey","sci.crypt",
                   "sci.electronics", "sci.med", "sci.space","soc.religion.christian","talk.politics.guns","talk.politics.mideast",
                   "talk.politics.misc","talk.religion.misc"]
    topics_r8=["coffee", "crude", "interest", "sugar", "trade","money-fx", "money-supply", "ship"]
    topics_r10=["coffee", "crude", "interest", "sugar", "trade","gold", "gnp", "ship","cocoa","acq"]
    topics_cade12=["01_servicos", "02_sociedade", "03_lazer", "04_informatica", "05_saude","06_educacao", "07_internet",
                   "08_cultura","09_esportes","10_noticias","11_ciencias","12_compras-online"]
    topics_r15 = ["coffee", "crude", "interest", "sugar", "trade", "money-fx", "money-supply", "ship","acq","coffee",
                  "earn","gold","cocoa","cpi","gnp"]
    topics_cade12=["01_servicos", "02_sociedade", "03_lazer", "04_informatica", "05_saude","06_educacao", "07_internet",
                   "08_cultura","09_esportes","10_noticias","11_ciencias","12_compras-online"]
    topics_r15 = ["coffee", "crude", "interest", "sugar", "trade", "money-fx", "money-supply", "ship","acq","coffee",
                  "earn","gold","cocoa","cpi","gnp"]
    ng20_t6=["alt.atheism","comp.sys.ibm.pc.hardware","rec.sport.baseball","sci.crypt","sci.space","talk.religion.misc"]
    ng20_t15= ["alt.atheism", "comp.graphics", "comp.sys.ibm.pc.hardware", "comp.windows.x","misc.forsale","rec.autos","rec.motorcycles",
               "rec.sport.baseball","sci.crypt","sci.electronics", "sci.space","soc.religion.christian","talk.politics.guns",
               "talk.politics.mideast","talk.religion.misc"]
    if test_data_id==0:
        topics=topics_webkb4
        dataFilename = "webkb4.txt"
       # labels, tfidf_matrix , corpus, itt, jtt,lmax,max_test_doc_num,low_words_num,topic,topic_docs= text_dic(topics, 'webkb4.txt')
    elif test_data_id==1:
        topics = topics_r5
        dataFilename = "r52-train.txt"
        #labels, tfidf_matrix = text_dic(topics, "r52-train.txt")
    elif test_data_id==2:
        topics = topics_r8
        dataFilename = "r52-train.txt"
        #labels, tfidf_matrix = text_dic(topics, "r52-train.txt")
    elif test_data_id==3:
        topics = topics_r10
        dataFilename = "r52-train.txt"
        #labels, tfidf_matrix= text_dic(topics, "r52-train.txt")
    elif test_data_id == 4:
        topics = news20_topics
        dataFilename = "mini20-train.txt"
        #labels, tfidf_matrix = text_dic(topics, 'mini20-train.txt')
    elif test_data_id==5:
        topics = ng20_t6
        dataFilename = "mini20-train.txt"
        #labels, tfidf_matrix = text_dic(topics, 'mini20-train.txt')
    else:
        topics = ng20_t15
        dataFilename = "mini20-train.txt"
        #labels, tfidf_matrix = text_dic(topics, 'mini20-train.txt')
        
    dataset_info,labels, tfidf_matrix, topic_docs= text_dic(topics, dataFilename,max_test_doc_num,low_words_num,up_words_num)

    n_cluster = len(topics)
    # data=np.column_stack((tfidf_matrix,labels))
    # pd.DataFrame(np.around(np.array(data), decimals=4)).to_csv("1total_result.txt", sep="\t", header=None, index=None)
    #打乱数据
    if is_shuff==True:
        index=[i for i in range(tfidf_matrix.shape[0])]
        random.shuffle(index)
        tfidf_matrix = tfidf_matrix[index]
        labels = labels[index]
    return dataset_info,tfidf_matrix, labels, n_cluster
if __name__ == '__main__':
    file_dir = "dataset/"
    test_data_id=1
    max_test_doc_num = 100
    low_words_num = 30
    up_words_num = 1e9
    is_shuff=False
    dataset_info,tfidf_matrix, labels, n_cluster = get_data_matrix(test_data_id,max_test_doc_num,low_words_num,up_words_num,is_shuff)
    # for i in range(0,29):
    #     get_uci_data(file_dir,i,datasets1)
