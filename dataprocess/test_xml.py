import re

htmlString = '''<ul id="TopNav"><li><a href="/EditPosts.aspx" id="TabPosts">随笔</a></li>
        <li><a href="/EditArticles.aspx" id="TabArticles">文章</a></li>
        <li><a href="/EditDiary.aspx" id="TabDiary">日记</a></li>
        <li><a href="/Feedback.aspx" id="TabFeedback">评论</a></li>
        <li><a href="/EditLinks.aspx" id="TabLinks">链接</a></li>
        <li id="GalleryTab"><a href="/EditGalleries.aspx" id="TabGalleries">相册</a></li>
        <li id="FilesTab"><a href="Files.aspx" id="TabFiles">文件</a></li>
        <li><a href="/Configure.aspx" id="TabConfigure">设置</a></li>
        <li><a href="/Preferences.aspx" id="TabPreferences">选项</a></li></ul>'''

# 方法 1
pre = re.compile('>(.*?)<')
s1 = ''.join(pre.findall(htmlString))
print(s1)  # '随笔文章日记评论链接相册文件设置选项'

# 方法 2
s2 = re.sub(r'<.*?>', '', htmlString)
print(s2)  # '\n\n随笔\n文章\n日记\n评论\n链接\n相册\n文件\n设置\n选项\n\n'

# 再用str.replace()函数去掉'\n'
s2 = s2.replace('\n', '')
print(s2)  # '随笔文章日记评论链接相册文件设置选项'