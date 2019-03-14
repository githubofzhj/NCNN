from xml.dom import minidom

def getE(Elem,sgm_str):
    # print( \
    #     "ElemNodeName:" + str(Elem.nodeName) + " ElemNodeType:" + str(Elem.nodeType) + " ElemValue:" + str(
    #         Elem.nodeValue))
    if(Elem.nodeValue!=None):
        sgm_str+=Elem.nodeValue
    for child in Elem.childNodes:
        sgm_str=getE(child,sgm_str)
    return  sgm_str

def ace_sgmfile(sgm_file,sgm_str):
    #sgm文件读取
    doc = minidom.parse(sgm_file)  # 从xml文件得到doc对象
    root = doc.documentElement  #
    sgm_str=getE(root,sgm_str)
    # for child in root.childNodes:
    #     print(child.nodeValue)
    return sgm_str
sgm_str=''
file_name = 'axml.apf.xml'
sgm_str=ace_sgmfile(file_name,sgm_str)
print(sgm_str[2558:2582])