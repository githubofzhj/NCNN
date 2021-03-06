from gensim.models import word2vec
from dataprocess.xml_parse import xml_parse_base
import gensim
from xml.dom import minidom
import  re
import json
import os
import spacy

en_nlp = spacy.load("en_core_web_sm")
#sgm文件读取
i=0
def splitToWords(text):
    nlps=en_nlp(text)
    sentences=[]#保存文章的分词结果
    return  [sp.text.lower() for sp in nlps]
    # 将文章分成句子，再分词单词
    # for ix, sent in enumerate(nlps.sents, 1):#获取每一个句子token，用token.text访问句子
    #     #print("Sentence number {}:{}".format(ix, sent))
    #     for wordtoken in en_nlp(sent.text):
    #         sentences.append(wordtoken.text.lower())
    # print(sentences)
    # return  sentences
def getsentence(sgm_file):
    #print(sgm_file)
    sentences = []  # 保存sgm文本的句子
    doc=None
    try:
        doc = minidom.parse(sgm_file+".sgm")  # 从xml文件得到doc对象
    except:
        print("this file cannot minidom"+sgm_file)
        doc = minidom.parse(sgm_file + ".sgm")
        global i
        i+=1
        return sentences
    root = doc.documentElement  #
    post_nodes = xml_parse_base.get_xmlnode(None, root, 'POST')
    for post_node in post_nodes:
        node=post_node.childNodes[-1]
        textWordList=splitToWords(node.nodeValue.replace("\n",' ').strip())
        sentences.extend(textWordList)
    text_nodes = xml_parse_base.get_xmlnode(None, root, 'TEXT')
    for text_node in text_nodes:
        textWordList=splitToWords(text_node.firstChild.data.replace("\n",' ').strip())
        sentences.extend(textWordList)#得到sgm文件中所有文本，去除\n后，按句子分句

    turn_nodes = xml_parse_base.get_xmlnode(None, root, 'TURN')
    for text_node in turn_nodes:
        node = text_node.childNodes[-1]
        textWordList=splitToWords(node.nodeValue.replace("\n",' ').strip())
        sentences.extend(textWordList)
    return sentences

def getACECorpus(ACEEnglistDir="./data/EnglishLDC2006/"\
                            ):
    "输入ACE_English语料,返回train,test,dev文件名和保存成txt"
    ACEFileList=os.listdir(ACEEnglistDir)#6大类文件夹
    print(ACEFileList)
    count=0
    sentences=[]
    for dirname in ACEFileList:
        fileList=os.listdir(ACEEnglistDir +dirname+"/adj/")
        for fileItem in fileList:
            # print("## Processing ", fileItem)
            if not fileItem.endswith(".sgm"):continue
            count+=1
            filename = fileItem.strip(".sgm").strip(".apf.xml").strip(".ag.xml").strip(".tab")
            filename=ACEEnglistDir +dirname+"/adj/"+filename
            sentences.append(getsentence(filename))#每个文本分词变成一个list,sentences[[],[]]
    return sentences
if __name__ == "__main__":
    # 模型训练
    # sgm_file = "./data/English/nw/timex2norm/AFP_ENG_20030311.0491"
    sentences=getACECorpus()
    for i in range(30,50):
        print(sentences[i])
    print(i)
    model = gensim.models.Word2Vec(sentences,sg=1,size=300, min_count=1)
    print(len(model['el']))
    print(model['will'])
    vocab = model.wv.vocab
    word2vecjson={}
    count=0
    for item in vocab:
        count+=1
        word2vecjson[item]=[str(i) for i in model[item]]
    print("have"+str(count)+"word!")
    with open("./data/Englishsplit/word2vec.json","w") as f:
        json.dump(word2vecjson, f, indent=4)
