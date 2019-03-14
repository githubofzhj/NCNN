#coding: utf-8
import xml.dom.minidom

file_name = 'axml.apf.xml'
dom = xml.dom.minidom.parse(file_name)  #打开xml文档

root = dom.documentElement              #得到xml文档对象
print ("nodeName:", root.nodeName)        #每一个结点都有它的nodeName，nodeValue，nodeType属性
print ("nodeValue:", root.nodeValue)      #nodeValue是结点的值，只对文本结点有效
print ("nodeType:", root.nodeType)
print ("ELEMENT_NODE:", root.ELEMENT_NODE)