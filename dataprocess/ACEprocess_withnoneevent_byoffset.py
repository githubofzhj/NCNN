from dataprocess.xml_parse import xml_parse_base
from xml.dom import minidom
import  decimal
import random
import  os
import  re
# from bson import json_util
import json
import spacy
from nltk.tokenize import sent_tokenize
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords as sw

en_nlp = spacy.load("en_core_web_sm")


class ACE_info:
    # go gain the event mention from ACE dataset
    eventsubType=set() #所有事件的eventtype集合
    def __init__(self):
        self.id = None                       # 获取事件编号
        self.text = None                     # 获取整个事件的内容
        self.text_unremoved = None           # ldcscope中的没除去回车的text
        self.text_start=None                 # 获取候选事件的起始位置,设置为0了，即以每个句子为开始
        self.text_end=None                   # 获取候选事件的终止位置
        self.trigger = ''                  # 获取事件触发词
        self.trigger_start = None            # 获取事件触发词的起始位置
        self.trigger_end = None              # 获取事件触发词的终止位置
        self.trigger_sub_type = ''         # 获取事件触发词的子类型
        self.argument = []                   # 获取事件元素
        self.argument_start = []             # 获取事件元素的起始位置
        self.argument_end = []               # 获取事件元素的终止位置
        self.argument_type = []              # 获取事件元素的类型
        self.argument_entity_type = []
        self.entity = []                     # 获取实体
        self.entity_start = []               # 获取实体的起始位置
        self.entity_end = []                 # 获取实体的终止位置
        self.entity_type = []                # 获取实体的类型

    def toString(self):
        return 'id:' + str(self.id) + '\t text:' + str(self.text) + '\t trigger:' + str(
            self.trigger) + '\t trigger_sub_type:' + str(self.trigger_sub_type)+ \
               '\t argument:' + str(self.argument) + '\t argument_type:' + str(self.argument_type)\
               + '\t trigger_start:' + str(self.trigger_start)+ '\t trigger_end:' + str(self.trigger_end) \
               + '\t argument_start:' + str(self.argument_start)+ '\t argument_end:' + str(self.argument_end)
    def toDict(self):
        return {'id': str(self.id) , 'text': str(self.text) ,'trigger' :str(
            self.trigger) ,'trigger_sub_type':str(self.trigger_sub_type),\
               'argument' :str(self.argument) ,'argument_type' :str(self.argument_type)\
               , 'trigger_start' :str(self.trigger_start), 'trigger_end': str(self.trigger_end) \
               ,'argument_start': str(self.argument_start), 'argument_end' : str(self.argument_end)}


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# def ace_sgmfile(sgm_file,R=[]):
#     #sgm文件读取
#     doc = minidom.parse(sgm_file+".sgm")  # 从xml文件得到doc对象
#     root = doc.documentElement  #
#     sentences=[]#保存sgm文本的句子
#     post_nodes = xml_parse_base.get_xmlnode(None, root, 'POST')
#     for post_node in post_nodes:
#         node=post_node.childNodes[-1]
#         sentences.extend(re.split("[.?!]",node.nodeValue.replace("\n",' ')))
#     text_nodes = xml_parse_base.get_xmlnode(None, root, 'TEXT')
#     for text_node in text_nodes:
#         # print("text_node:\n"+text_node.firstChild.data)
#         # print("___________")
#         # print(text_node.firstChild.data.replace("\n",''))
#         sentences.extend(re.split("[.?!]",text_node.firstChild.data.replace("\n",' ')))#得到sgm文件中所有文本，去除\n后，按句子分句
#     # apf_file=sgm_file.strip("sgm")+"apf.xml"
#     # extract_ace_info(apf_file,R)
#     haveeventset=set()
#     for event in R:
#         flag=0#交际event的text是否被找到了
#         for i in range(len(sentences)):
#             if event['text'] in sentences[i]:
#                 flag=1
#                 haveeventset.add(i)
#         if flag==0:
#             print("in file:"+sgm_file)
#             print("event text not match:"+event['text'])
#             for sentence in sentences:
#                 print(sentence)
#             exit(-2)
#     noEventSentences=[]
#     for i in range(len(sentences)):
#         if i not in haveeventset:
#             noEventSentences.append(sentences[i])
#     for i in range(len(noEventSentences)):
#         R_element=ACE_info()
#         R_element.text=noEventSentences[i]
#         R.append(R_element.toDict())#更新加入没有event的句子
def ace_sgm_ET(sgm_file):
    # text = open(sgm_file+".sgm").read()
    # text = re.sub(r'&', '&amp;', text)
    # print(sgm_file)
    # root = ET.fromstring(text)
    # # root = ET.parse(sgm_file+".sgm").getroot()
    return

def ace_sgm_re(sgm_file):
    '''输入sgm文件，输出sgm文件中去除xml标记的数据'''
    str = ''
    with open(sgm_file+".sgm", "r")as f:
        lines = f.readlines()
    for line in lines:
        str += line
    # print(str)
    '''正则表达式去除xml标签'''
    pre = re.compile('>([^</]+)</')
    s1 = ''.join(pre.findall(str))
    return s1
def ace_sgm_deleteXMLtag_TEXT(sgm_file):
    '''输入sgm文件，输出sgm文件中去除<TEXT> xml标记的数据,以及<TEXT>在文件中的偏移'''
    str = ''
    with open(sgm_file+".sgm", "r")as f:
        lines = f.readlines()
    for line in lines:
        str += line
    ##取所有非tag的句子
    s_all = ''
    __flag_num = 0  # 记录未匹配的<数目
    for char in str:
        if char == '<':
            __flag_num += 1
            continue
        if char == '>':
            __flag_num -= 1
            continue
        if __flag_num == 0:  # 不在<>之间，则保存
            s_all += char
        if __flag_num < 0:
            print("error __flag<0 || file:" + sgm_file)

    # print(str)
    TEXT_content=''
    __flag_num=0#表示是否在<>之间
    xml_tag_string=''
    __text_flag=0

    TEXT_offset=0

    #get <TEXT>... </TEXT>
    for i,char in enumerate(str):

        if __flag_num == 1:  # 在<>之间，则保存
            xml_tag_string += char
        #不在<>且在<TEXT>... </TEXT>之间
        if(__flag_num ==0 and __text_flag==1):
            TEXT_content+=char

        if char =='<' :
            __flag_num=1
            TEXT_content=TEXT_content[:-1]#去掉最后一个
            xml_tag_string+=char
            continue
        if char == '>':
            __flag_num =0
            #结束时判断是否为<TEXT>
            if (xml_tag_string == '<TEXT>'):
                __text_flag = 1
                xml_tag_string = ''
                continue
            #判断是否为</TEXT>
            if (xml_tag_string == '</TEXT>'):
                break
            xml_tag_string = ''


    #
    # ##取<TEXT>... </TEXT>之间的句子
    # s1 = ''
    # __flag_num = 0  # 记录未匹配的<数目
    # for char in TEXT_content:
    #     if char == '<':
    #         __flag_num += 1
    #         continue
    #     if char == '>':
    #         __flag_num -= 1
    #         continue
    #     if __flag_num == 0:  # 不在<>之间，则保存
    #         s1 += char
    #     if __flag_num < 0:
    #         print("error __flag<0 || file:" + sgm_file)

    TEXT_offset=s_all.find(TEXT_content)
    if(TEXT_offset<=0):
        print("not find <TEXT>")
    return TEXT_content,TEXT_offset


def ace_sgm_deleteXMLtag(sgm_file):
    '''输入sgm文件，输出sgm文件中去除xml标记的数据'''
    str = ''
    with open(sgm_file+".sgm", "r")as f:
        lines = f.readlines()
    for line in lines:
        str += line
    # print(str)
    s1=''
    __flag_num=0#记录未匹配的<数目
    for char in str:
        if char =='<':
            __flag_num+=1
            continue
        if char == '>':
            __flag_num-=1
            continue
        if __flag_num==0:#不在<>之间，则保存
            s1+=char
        if __flag_num<0: print("error __flag<0 || file:"+sgm_file)

    return s1

refailed=0
def get_none_events(apf_events=[],R=[],sgm_str='',apf_file=''):
    text_set=set()
    global  refailed
    sentence_start=0
    sgm_tmp=sgm_str

    #sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_tokenize(sgm_tmp)

    for sentence in list(sentences):
        sentence=str(sentence)

        sentence_start=sgm_str.find(sentence)
        if(sentence_start<0):
            print("error: sentence cannot find in sgm_text!")
            exit(-1)
        #创建一个最后需要输入网络的句子
        R_e=ACE_info()
        R_e.text=sentence.strip().replace("\n", ' ')

        haveevent_flag=False#标记这句话是否有事件

        s2 = sgm_str[sentence_start:sentence_start + len(sentence)]
        for R_element in apf_events:
            start = R_element.trigger_start
            end = R_element.trigger_end
            trigger = R_element.trigger
            if (start>=sentence_start and end <=sentence_start+len(sentence)):#in this sentence
                haveevent_flag=True

                s=start-sentence_start
                e=end-sentence_start
                sentence_trigger=sentence[s:e+1]
                if(trigger != sentence_trigger):
                    print("error!")
                    print("sentence_trigger:"+sentence_trigger+"     trigger:"+trigger+"  sgm_str:"+sgm_str[start:end+1])
                    continue
                R_e.trigger=trigger.strip().replace("\n",' ')
                R_e.trigger_sub_type=R_element.trigger_sub_type
                R.append(R_e.toDict())
                #R_element.text = sentence.strip().replace("\n", ' ')
        #如果本句话没有event，则保存为没有事件
        if( not haveevent_flag):
            R.append(R_e.toDict())
def get_none_events_TEXT(apf_events=[],R=[],sgm_str='',apf_file='',TEXT_offset=0):
    text_set = set()
    global refailed
    sentence_start = 0
    sgm_tmp = sgm_str

    finded_event = 0
    # sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_tokenize(sgm_tmp)

    for sentence in list(sentences):
        sentence = str(sentence)

        sentence_start = sgm_str.find(sentence)+TEXT_offset
        if (sentence_start < 0):
            print("error: sentence cannot find in sgm_text!")
            exit(-1)
        # 创建一个最后需要输入网络的句子
        R_e = ACE_info()
        R_e.text = sentence.strip().replace("\n", ' ')

        haveevent_flag = False  # 标记这句话是否有事件

        for R_element in apf_events:
            start = R_element.trigger_start
            end = R_element.trigger_end
            trigger = R_element.trigger
            if (start >= sentence_start and end <= sentence_start + len(sentence)):  # in this sentence
                haveevent_flag = True
                finded_event+=1
                s = start - sentence_start
                e = end - sentence_start
                sentence_trigger = sentence[s:e + 1]
                if (trigger != sentence_trigger):
                    print("error!")
                    print("sentence_trigger:" + sentence_trigger + "     trigger:" + trigger)
                    continue
                R_e.trigger = trigger.strip().replace("\n", ' ')
                R_e.trigger_sub_type = R_element.trigger_sub_type
                R.append(R_e.toDict())
                # R_element.text = sentence.strip().replace("\n", ' ')
        # 如果本句话没有event，则保存为没有事件
        if (not haveevent_flag):
            R.append(R_e.toDict())
    if (finded_event)!=len(apf_events) :
        print("%d , %d"%(finded_event,len(apf_events)))
        print("some event was not foundd!")


def extract_ace_info(apf_file, \
                     R=[],sgm_str='',TEXT_offset=0):
    # 存储事件实体的list
    doc = minidom.parse(apf_file+".apf.xml")#从xml文件得到doc对象
    root = doc.documentElement#获得根对象source_file
    entity = {}
    # 获取实体提及
    entity_nodes = xml_parse_base.get_xmlnode(None, root, 'entity')
    for entity_node in entity_nodes:
        entity_type = xml_parse_base.get_attrvalue(None, entity_node, 'SUBTYPE')
        entity_mention_nodes = xml_parse_base.get_xmlnode(None, entity_node, 'entity_mention')
        for entity_mention_node in entity_mention_nodes:
            entity_mention_id = xml_parse_base.get_attrvalue(None, entity_mention_node, 'ID')
            entity_mention_head = xml_parse_base.get_xmlnode(None, entity_mention_node, 'head')
            entity_mention_head_charseq = xml_parse_base.get_xmlnode(None, entity_mention_head[0], 'charseq')
            for charse in entity_mention_head_charseq:
                entity_mention_start = xml_parse_base.get_attrvalue(None,charse, 'START')
                entity_mention_end = xml_parse_base.get_attrvalue(None, charse, 'END')
                entity[entity_mention_id] = [entity_mention_start, entity_mention_end, entity_type]
    #获得value提及
    value_nodes = xml_parse_base.get_xmlnode(None, root, 'value')
    for value_node in value_nodes:
        value_type = xml_parse_base.get_attrvalue(None, value_node, 'TYPE')
        value_mention_nodes = xml_parse_base.get_xmlnode(None, value_node, 'value_mention')
        for value_mention_node in value_mention_nodes:
            value_mention_id = xml_parse_base.get_attrvalue(None, value_mention_node, 'ID')
            value_mention_extent = xml_parse_base.get_xmlnode(None, value_mention_node, 'extent')
            value_mention_extent_charseq = xml_parse_base.get_xmlnode(None, value_mention_extent[0], 'charseq')
            for charse in value_mention_extent_charseq:
                value_mention_start = xml_parse_base.get_attrvalue(None,charse, 'START')
                value_mention_end = xml_parse_base.get_attrvalue(None, charse, 'END')
                entity[value_mention_id] = [value_mention_start,value_mention_end,value_type]
    #获得time提及
    timex2_nodes = xml_parse_base.get_xmlnode(None, root, 'timex2')
    for timex2_node in timex2_nodes:
        timex2_mention_nodes = xml_parse_base.get_xmlnode(None, timex2_node, 'timex2_mention')
        for timex2_mention_node in timex2_mention_nodes:
            timex2_mention_id = xml_parse_base.get_attrvalue(None, timex2_mention_node, 'ID')
            timex2_mention_extent = xml_parse_base.get_xmlnode(None, timex2_mention_node, 'extent')
            timex2_mention_extent_charseq = xml_parse_base.get_xmlnode(None, timex2_mention_extent[0], 'charseq')
            for charse in timex2_mention_extent_charseq:
                timex2_mention_start = xml_parse_base.get_attrvalue(None,charse, 'START')
                timex2_mention_end = xml_parse_base.get_attrvalue(None, charse, 'END')
                entity[timex2_mention_id] = [timex2_mention_start,timex2_mention_end,'timex2']


    event_nodes = xml_parse_base.get_xmlnode(None, root, 'event')#获得对象名为event的节点列表

    apf_events=[]##保存本apf_中的所有event
    for node in event_nodes:
        # 获取事件mention
        mention_nodes = xml_parse_base.get_xmlnode(None, node, 'event_mention')#获得对象名为event_mention的节点列表
        for mention_node in mention_nodes:
            R_element = ACE_info()#这里只是提取apf文件里面的event，利用这些信息来标记sgm文件的每一个句子，sgm文件的每一个句子才是最后使用的
            # 获取事件id
            R_element.id = xml_parse_base.get_attrvalue(None, mention_node, 'ID')#获得event_mention的ID属性值
            # 获取事件子类型
            R_element.trigger_sub_type = xml_parse_base.get_attrvalue(None, node, 'SUBTYPE')#获得event的SUBTYPE属性值
            ACE_info.eventsubType.add(R_element.trigger_sub_type)#加入到全局eventType中去，统计所有type
            # 获取事件所在语句
            mention_ldc_scope = xml_parse_base.get_xmlnode(None, mention_node, 'ldc_scope')#获得ldc_scope列表
            mention_ldc_scope_charseq = xml_parse_base.get_xmlnode(None, mention_ldc_scope[0], 'charseq')
            #获得事件语句并将/n换位空格
            #text_str = xml_parse_base.get_nodevalue(None, mention_ldc_scope_charseq[0], 0).replace("\n", " ")#xml文件中有换行的符号
            #不替换/n
            text_str = xml_parse_base.get_nodevalue(None, mention_ldc_scope_charseq[0], 0)
            R_element.text_unremoved=text_str
            R_element.text = text_str.replace("\n", " ")
            s = None
            m = 0
            #ldcscope_text开始均为0
            for charse in mention_ldc_scope_charseq:
                start = xml_parse_base.get_attrvalue(None,charse, 'START')#charseq的START属性值
                end = xml_parse_base.get_attrvalue(None, charse, 'END')
                s = int(start)
                R_element.text_start = 0
                R_element.text_end = int(end)-int(start)
                for j, x in enumerate(entity):
                    #print entity[x][0]
                    if int(entity[x][0])>=int(start) and int(entity[x][1])<=int(end):
                        R_element.entity_start.append(int(entity[x][0]) - int(s))#保存event这句话中entity开始位置
                        R_element.entity_end.append(int(entity[x][1]) - int(s))#保存event这句话中entity结束位置
                        R_element.entity_type.append(entity[x][2])#保存实体类型
                        R_element.entity.append(R_element.text[R_element.entity_start[m]:R_element.entity_end[m]+1])#保存实体字符串
                        m+=1


            # 获取事件触发词
            mention_anchor = xml_parse_base.get_xmlnode(None, mention_node, 'anchor')#获得anchor列表
            mention_anchor_charseq = xml_parse_base.get_xmlnode(None, mention_anchor[0], 'charseq')
            for anch in mention_anchor_charseq:
                start = xml_parse_base.get_attrvalue(None, anch, 'START')
                end = xml_parse_base.get_attrvalue(None, anch, 'END')
                trigger=xml_parse_base.get_nodevalue(None, anch, 0)
                R_element.trigger_start = int(start)#绝对位置
                R_element.trigger_end = int(end)
                R_element.trigger=trigger
                #ts= R_element.text[R_element.trigger_start-s:R_element.trigger_end-s+1]#begin 这样的



            # 获取事件元素
            mention_arguments = xml_parse_base.get_xmlnode(None, mention_node, 'event_mention_argument')#event_mention_argument列表
            i = 0
            arg = []
            for mention_argument in mention_arguments:
                mention_argument_refid = xml_parse_base.get_attrvalue(None,mention_argument, 'REFID')
                #mention_argument_extent = get_xmlnode(None, mention_argument, 'extent')
                #mention_argument_charseq = get_xmlnode(None, mention_argument_extent[0], 'charseq')
                try:#先try用id索引读信息，如果不在，再尝试用extent来读信息。
                    argument_position = entity[mention_argument_refid]
                    start = argument_position[0]
                    end = argument_position[1]
                    entity_type = argument_position[2]
                except KeyError:
                    print ('error')
                    mention_argument_extent = xml_parse_base.get_xmlnode(None, mention_argument, 'extent')
                    mention_argument_charseq = xml_parse_base.get_xmlnode(None, mention_argument_extent[0], 'charseq')
                    for argument_charse in mention_argument_charseq:
                        start = xml_parse_base.get_attrvalue(None,argument_charse, 'START')
                        end = xml_parse_base.get_attrvalue(None, argument_charse, 'END')
                        entity_type = None

                R_element.argument_start.append(int(start) - int(s))#多个事件元素
                R_element.argument_end.append( int(end) - int(s))
                R_element.argument.append(R_element.text[R_element.argument_start[i]:R_element.argument_end[i]+1])
                #arg.append(get_nodevalue(None, mention_argument_charseq[0], 0).replace("\n", " "))#事件元素列表
                R_element.argument_entity_type.append(entity_type)
                R_element.argument_type.append(xml_parse_base.get_attrvalue(None, mention_argument, 'ROLE'))#事件元素类型
                i+=1
            apf_events.append(R_element)
    get_none_events_TEXT(apf_events, R,sgm_str,apf_file,TEXT_offset)#R中加入不存在事件的句子
            # R.append(R_element.toDict())
    return R
def getACETrainTestFilename(ACEEnglistDir="./data/English/"\
                            ,outDir='./data/Englishsplit/'\
                            ):
    "输入ACE_English语料,返回train,test,dev文件名和保存成txt"
    ACEFileList=os.listdir(ACEEnglistDir)#6大类文件夹
    print(ACEFileList)
    nwFilelist=set()
    otherFileList=set()
    for dirname in ACEFileList:
        fileList=os.listdir(ACEEnglistDir +dirname+"/timex2norm/")
        for fileItem in fileList:
            if not fileItem.endswith(".apf.xml"): continue #只要一个文件即可
            filename = fileItem[:len(fileItem)-len(".apf.xml")]
            if dirname == "nw":
                nwFilelist.add(filename)
            else:
                otherFileList.add(filename)
    nwFilelist=list(nwFilelist)
    otherFileList=list(otherFileList)

    random.shuffle(nwFilelist)
    otherFileList=otherFileList+nwFilelist[40:]
    testFileList=nwFilelist[:40]
    random.shuffle(otherFileList)
    devFileList=otherFileList[0:30]
    trainFileList=otherFileList[30:]

    with open(outDir+"trainFileName.txt", 'w') as f1:
        json.dump(list(trainFileList), f1,indent=4)
    with open(outDir+"testFileName.txt", 'w') as f2:
        json.dump(list(testFileList), f2,indent=4)
    with open(outDir+"devFileName.txt", 'w') as f3:
        json.dump(list(devFileList), f3,indent=4)


    ###生成对应train,test,dev的event的json文件###
    trainEventList=[]
    devEventList = []
    testEventList = []

    train, test, dev, allfile = 0, 0, 0, 0
    ACEFileList = os.listdir(ACEEnglistDir)
    print(ACEFileList)
    for dirname in ACEFileList:
        fileList = os.listdir(ACEEnglistDir + dirname + "/timex2norm/")
        for fileItem in fileList:
            if not fileItem.endswith(".apf.xml"): continue#每篇文章选中一个即可
            filename_no_end=fileItem[:len(fileItem)-len(".apf.xml")]
            filename=ACEEnglistDir + dirname + "/timex2norm/"+filename_no_end

            if filename_no_end in trainFileList:
                train+=1
                sgm_str,TEXT_offset=ace_sgm_deleteXMLtag_TEXT(filename)#返回sgm文件中去除xml标签的部分
                extract_ace_info(filename,trainEventList,sgm_str,TEXT_offset)#用apf文件读事件
            if filename_no_end in devFileList:
                dev+=1
                sgm_str,TEXT_offset =ace_sgm_deleteXMLtag_TEXT(filename)
                extract_ace_info(filename,devEventList,sgm_str,TEXT_offset)
            if filename_no_end in testFileList:
                test+=1
                sgm_str,TEXT_offset =ace_sgm_deleteXMLtag_TEXT(filename)
                extract_ace_info(filename,testEventList,sgm_str,TEXT_offset)
    print(train, test, dev)
    print("refailed:"+str(refailed))
    with open(outDir+"trainevent.json", 'w') as f:
        json.dump(trainEventList, f,indent=4)
    with open(outDir+"devevent.json", 'w') as f:
        json.dump(devEventList, f,indent=4)
    with open(outDir+"testevent.json", 'w') as f:
        json.dump(testEventList, f,indent=4)
    with open(outDir+"eventsubtype.json", 'w') as f:
        json.dump(list(ACE_info.eventsubType), f,indent=4)
    print("trainevents:"+str(len(trainEventList)))
    print("devevents:"+str(len(devEventList)))
    print("testevents:"+str(len(testEventList)))
    print("eventsubtypes:" + str(len(list(ACE_info.eventsubType))))
    return

if __name__ == "__main__":
    # testfile="data/AGGRESSIVEVOICEDAILY_20041101.1144.apf.xml"
    # extract_ace_info(testfile)
    # sgm_file1="./data/English/nw/timex2norm/AFP_ENG_20030311.0491.sgm"
    # ace_sgmfile(sgm_file1)
    # sgm_file1="./data/English/wl/timex2norm/AGGRESSIVEVOICEDAILY_20041101.1806.sgm"
    # ace_sgmfile(sgm_file1)
    getACETrainTestFilename()
