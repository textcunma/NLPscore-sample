from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import meteor
from nltk import word_tokenize

#単一文章の場合
candidate = "an apple on this tree"
candidate = word_tokenize(candidate)

# reference=["this is an apple", "that is an apple"]
reference=["this is an apple"]
for i in range(len(reference)):
    tmp=word_tokenize(reference[i])
    reference[i]=tmp

sm=SmoothingFunction().method1
# sm=None   # sm=Noneでも可
print("単一文章の場合")
print('BLEU-1: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=sm))
print('BLEU-2: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0), smoothing_function=sm))
print('BLEU-3: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0), smoothing_function=sm))
print('BLEU-4: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1), smoothing_function=sm))
print('METEOR: %f' % round(meteor(reference,candidate),4))      #小数点切り捨てでMETEOR

# 複数文章の場合
from statistics import mean

BLEU1=[]
BLEU2=[]
BLEU3=[]
BLEU4=[]
METEOR=[]

with open("hyps.txt") as f:
    cands = [line.strip() for line in f]

with open("refs.txt") as f:
    refs = [line.strip() for line in f]

length=len(cands)

for i in range(length):
    candidate=cands[i]
    candidate = word_tokenize(candidate)
    reference=refs[i]
    reference=[word_tokenize(reference)]

    B1=sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=sm)
    BLEU1.append(B1)

    B2=sentence_bleu(reference, candidate, weights=(0, 1, 0, 0), smoothing_function=sm)
    BLEU2.append(B2)

    B3=sentence_bleu(reference, candidate, weights=(0, 0, 1, 0), smoothing_function=sm)
    BLEU3.append(B3)

    B4=sentence_bleu(reference, candidate, weights=(0, 0, 0, 1), smoothing_function=sm)
    BLEU4.append(B4)

    M=meteor(reference,candidate)
    METEOR.append(M)
print("複数文章の場合")
print('BLEU-1: %f' % mean(BLEU1))
print('BLEU-2: %f' % mean(BLEU2))
print('BLEU-3: %f' % mean(BLEU3))
print('BLEU-4: %f' % mean(BLEU4))
print('METEOR: %f' % round(mean(METEOR),4))      #小数点切り捨てでMETEOR

