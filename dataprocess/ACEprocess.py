from dataprocess.xml_parse import xml_parse_base
from xml.dom import minidom
import  decimal
import random
import  os
# from bson import json_util
import json


class ACE_info:
    # go gain the event mention from ACE dataset
    eventsubType=set() #所有事件的eventtype集合
    def __init__(self):
        self.id = None                       # 获取事件编号
        self.text = None                     # 获取整个事件的内容
        self.text_start=None                 # 获取候选事件的起始位置
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

import json


# class DecimalEncoder(json.JSONEncoder):
#
#
#     def default(self, o):
#
#         if isinstance(o, decimal.Decimal):
#
#             return float(o)
#
#         super(DecimalEncoder, self).default(o)

def extract_ace_info(apf_file, \
                     R=[]):
    # 存储事件实体的list
    doc = minidom.parse(apf_file)#从xml文件得到doc对象
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
    for node in event_nodes:
        # 获取事件mention
        mention_nodes = xml_parse_base.get_xmlnode(None, node, 'event_mention')#获得对象名为event_mention的节点列表
        for mention_node in mention_nodes:
            R_element = ACE_info()
            # 获取事件id
            R_element.id = xml_parse_base.get_attrvalue(None, mention_node, 'ID')#获得event_mention的ID属性值
            # 获取事件子类型
            R_element.trigger_sub_type = xml_parse_base.get_attrvalue(None, node, 'SUBTYPE')#获得event的SUBTYPE属性值
            ACE_info.eventsubType.add(R_element.trigger_sub_type)#加入到全局eventType中去，统计所有type
            # 获取事件所在语句
            mention_ldc_scope = xml_parse_base.get_xmlnode(None, mention_node, 'ldc_scope')#获得ldc_scope列表
            mention_ldc_scope_charseq = xml_parse_base.get_xmlnode(None, mention_ldc_scope[0], 'charseq')
            #获得事件语句并将/n换位空格
            text_str = xml_parse_base.get_nodevalue(None, mention_ldc_scope_charseq[0], 0).replace("\n", " ")#xml文件中有换行的符号
            R_element.text = text_str
            s = None
            m = 0
            #text开始均为0
            for charse in mention_ldc_scope_charseq:
                start = xml_parse_base.get_attrvalue(None,charse, 'START')#charseq的START属性值
                end = xml_parse_base.get_attrvalue(None, charse, 'END')
                s = start
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
                R_element.trigger_start = int(start)-int(s)#相对语句位置
                R_element.trigger_end = int(end)-int(s)
                # R_element.trigger = R_element.text[R_element.trigger_start:R_element.trigger_end+1]#从text中获取，text可能把\n替换了
                R_element.trigger = xml_parse_base.get_nodevalue(None, mention_anchor_charseq[0], 0).replace("\n"," ")  # 直接获取文中标注的字段



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

            R.append(R_element.toDict())
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
            #filename = fileItem[:len(fileItem)-len(".apf.xml")]
            filename = fileItem.strip(".apf.xml").strip(".sgm").strip(".ag.xml").strip(".tab")
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

    ACEFileList = os.listdir(ACEEnglistDir)
    print(ACEFileList)
    train,test,dev,allfile=0,0,0,0
    for dirname in ACEFileList:
        fileList = os.listdir(ACEEnglistDir + dirname + "/timex2norm/")
        for fileItem in fileList:
            trainf, testf, devf=0,0,0
            if not fileItem.endswith(".apf.xml"): continue
            allfile+=1
            filename=ACEEnglistDir + dirname + "/timex2norm/"+fileItem
            # if fileItem[:len(fileItem)-len(".apf.xml")] in trainFileList:
            if fileItem.strip(".apf.xml") in trainFileList:
                train+=1
                trainf+=1
                extract_ace_info(filename,trainEventList)
            # if fileItem[:len(fileItem)-len(".apf.xml")] in devFileList:
            if fileItem.strip(".apf.xml") in devFileList:
                dev+=1
                devf+=1
                extract_ace_info(filename,devEventList)
            #if fileItem[:len(fileItem)-len(".apf.xml")] in testFileList:
            if fileItem.strip(".apf.xml") in testFileList:
                test+=1
                testf+=1
                extract_ace_info(filename,testEventList)
            if(trainf+testf+devf!=1):
                print("num:"+str(trainf+testf+devf)+" filename:"+filename+" fileItem:"+fileItem)
    print(allfile)
    print(train,test,dev)
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
    getACETrainTestFilename()
