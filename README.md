# NLPscore-sample
calc BLEU1,2,3,4  METEOR  BERTscore

自然言語処理で使用されるBLEU1,2,3,4 METEOR BERTscoreを計算するサンプルコード

## セットアップ
```bash
conda create -n NLPsample  python=3.8
conda activate NLPsample
pip install bert-score
pip install -U nltk --user
python setup.py
```

### 参考
https://qiita.com/48saaan/items/24d38174e07358169ce1
https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
https://github.com/itscassie/NLP_tools
https://naomichi-dev.hatenablog.com/entry/2017/10/18/032244
https://github.com/Tiiiger/bert_score
