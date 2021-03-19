import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from gensim.models import word2vec

def texting(line):
    return line.text

fp = codecs.open("2BEXXX09.txt", "r", encoding = "utf-16")
soup = BeautifulSoup(fp, "html.parser")
body = soup.findAll("p")

lines = list(map(texting, body))

results = []

okt = Okt()
for line in lines:
    malist = okt.pos(line, norm=True, stem = True)
    r = []
    for word in malist:
        if not word[1] in ["Josa", "Eomi", "Punctuation"]:
            r.append(word[0])
    rl = (" ".join(r)).strip()
    results.append(rl)
    print(rl)

wakati_file = 'toji.wakati'
with open(wakati_file, 'w', encoding ='utf-8') as fp: 
    fp.write("\n".join(results))

data = word2vec.LineSentence(wakati_file)
model = word2vec.Word2Vec(data, size=200, window=10, hs=1, min_count=2, sg=1)
model.save('toji.model')
print("done")
