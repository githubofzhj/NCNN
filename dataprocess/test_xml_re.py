import re
str=''
with open("axml.apf.xml","r")as f:
    lines=f.readlines()
for line in lines:
    str+=line
# print(str)
pre = re.compile('>([^</]+)</')
s1 = ''.join(pre.findall(str))
print(s1)
#print(s1[2558:2582])