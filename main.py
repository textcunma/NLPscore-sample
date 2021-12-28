from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import meteor
from nltk import word_tokenize

candidate = "an apple on this tree"
candidate = word_tokenize(candidate)

# reference=["this is an apple", "that is an apple"]
reference=["this is an apple"]
for i in range(len(reference)):
    tmp=word_tokenize(reference[i])
    reference[i]=tmp

sm=SmoothingFunction().method1
# sm=None   # sm=Noneでも可
print('BLEU-1: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=sm))
print('BLEU-2: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0), smoothing_function=sm))
print('BLEU-3: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0), smoothing_function=sm))
print('BLEU-4: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1), smoothing_function=sm))
print('METEOR: %f' % round(meteor(reference,candidate),4))      #小数点切り捨てでMETEOR
