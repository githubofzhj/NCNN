import random
import  os
import  re
# from bson import json_util
import json
import spacy
from collections import Counter
import numpy as np

en_nlp = spacy.load("en_core_web_sm")

def read_event_data(fname,opt):
    """{
            "id": "CNN_ENG_20030428_130651.4-EV3-1",
            "text": "but just to put the threat of sars in perspective, in the united states, 284 people died last year from the west nile virus while 36,000 people die every year from the flu",
            "trigger": "died",
            "trigger_sub_type": "Die",
            "argument": "['people', 'united states', 'last year']",
            "argument_type": "['Victim', 'Place', 'Time-Within']",
            "trigger_start": "84",
            "trigger_end": "87",
            "argument_start": "[77, 58, 89]",
            "argument_end": "[82, 70, 97]"
        }"""
    if os.path.isfile(fname) == False:
        raise ("[!] Data %s not found" % fname)
    file = open(fname, 'r', encoding='utf-8')
    events=json.load(file)
    data_x=[]
    data_y=[]
    source_words, target_words, max_sent_len = [], [], 0
    trigger_words_locations=[]
    source_len=[]
    trigger_not_in_text=0
    muti_word_trigger=0
    none_trigger_sentence=0
    for event in events:
        text=event["text"]
        trigger=event["trigger"]
        eventtype=event["trigger_sub_type"]
        sptoks = en_nlp(text)  # 取出了句子的文本内容，并分词
        # print(sptoks)
        words=[sp.text.lower() for sp in sptoks]
        if trigger=='':
            none_trigger_sentence+=1
        trigger_words=[sp.text.lower() for sp in en_nlp(trigger)]
        ##trigger location##
        t_w_location=[]
        if(len(trigger_words)>1):
            muti_word_trigger+=1
        error_sentence_flag=False
        for t_word in trigger_words:
            if t_word not in words:#处理分词问题
                error_sentence_flag=True
                print('trigger not in words:')
                print(words)
                print(trigger_words)
                trigger_not_in_text+=1
                break
            t_w_location.append(words.index(t_word))
        if len(sptoks) > max_sent_len:
            max_sent_len = len(sptoks)  # 不断更新最大句子长度

        if(not error_sentence_flag):#正确的句子才作为网络输入
            source_words.append(words)  # 得到了单词的集合，所有句子拼接在一起然后分词
            target_words.append(eventtype.lower())
            trigger_words_locations.append(t_w_location)
            source_len.append(len(words))

    print("type:"+str(opt))
    print("trigger_not_in_text:"+str(trigger_not_in_text))
    print("muti_word_trigger:"+str(muti_word_trigger))
    print("none_trigger_sentence:"+str(none_trigger_sentence))
    return source_words, target_words, trigger_words_locations,max_sent_len,source_len

def getall_word2id(source_words,target_words):
    source_count=[]
    if len(source_count) == 0:
        source_count.append(['<pad>', 0])
    #处理成1维的
    sourcewordlist=[]
    for words in source_words:
        sourcewordlist.extend(words)
    source_count.extend(Counter(sourcewordlist).most_common())

    source_word2idx,target_word2idx= {},{}
    source_idx2word,targat_idx2word={},{}

    for word, _ in source_count:
        if word not in source_word2idx:
            #source_idx2word[len(source_word2idx)] = word
            source_word2idx[word] = len(source_word2idx)
    target_word2idx['']=0#没有trigger位置，类型为none,和后面补齐的0对应
    targat_idx2word[0]=''
    for word in target_words:
        if word not in target_word2idx:
            #targat_idx2word[len(target_word2idx)] = word
            target_word2idx[word] = len(target_word2idx)
    return source_word2idx,target_word2idx,source_idx2word,targat_idx2word
def data2id(source_words, target_words, source_word2idx, target_word2idx,trigger_words_locations):
    source_id,target_id=[],[]
    trigger_id_sparse=[]
    for i,sentence in enumerate(source_words):
        source_id.append([source_word2idx[word] for word in sentence])#append2层list
        #把trigger位置变成sparse的，并且换成id
        trigger_sparse_list=[]
        for j in range(len(sentence)):
            trigger_sparse_list.append(0)#0为none类型的id
        for location in trigger_words_locations[i]:
            trigger_sparse_list[location]=target_word2idx[target_words[i]]
        trigger_id_sparse.append(trigger_sparse_list)#2层list
    for target in target_words:
        target_id.append(target_word2idx[target])#1层list
    return source_id,target_id,trigger_id_sparse
def get_word_embeddingMatrix(embeddingfile,word2idx,dim,init_std):
    wt = np.random.uniform(-1.0*init_std, init_std, [len(word2idx), dim])
    i=0
    file = open(embeddingfile, 'r', encoding='utf-8')
    word2vec=json.load(file)
    for word in word2idx:
        if word in word2vec:
            i+=1
            wt[word2idx[word]]=word2vec[word]
    print("%d word in Glove %d word in all" % (i, len(word2idx)))
    return wt.tolist()

