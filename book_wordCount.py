import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Okt

fp = codecs.open("2BEXXX09.txt", "r", encoding = "utf-16")
soup = BeautifulSoup(fp, "html.parser")
text = soup.findAll('p')

#print(text[2].text)

okt = Okt()
word_dic = {}
for line in text:
    malist = okt.pos(line.text)
    for word in malist:
        if word[1] == "Noun" or word[1] == "Verb": # 리스트 내부에 (a,b)튜플이 들어가있으므로 word[1]은 단어의 품사, [0]는 단어가 들어있다
            if not (word[0] in word_dic):
                word_dic[word[0]] = 0
            word_dic[word[0]] += 1 #같은 단어가 나올때마다 카운팅    

sortWord = sorted(word_dic.items(), reverse = True, key = lambda x:x[1])   # 키값이없으면 그냥 순서없이 정렬됨 (딕셔너리는 두개로 이루어져있기때문)          
for word, count in sortWord[:100]:
    print(f"{word}({count}) / ", end = "")
print()    
# malist = okt.pos("아버지 가방에 들어가신다")
# print(malist[0][0])