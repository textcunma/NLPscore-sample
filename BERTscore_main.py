# @inproceedings{bert-score,
#   title={BERTScore: Evaluating Text Generation with BERT},
#   author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
#   booktitle={International Conference on Learning Representations},
#   year={2020},
#   url={https://openreview.net/forum?id=SkeHuCVFDr}
# }

from bert_score import score
import bert_score
print(bert_score.__version__)
def calc_bert_score(cands, refs):
    Precision, Recall, F1 = score(cands, refs, lang="en", verbose=True,batch_size=10, device="cuda:0")
    return F1.numpy().tolist() #F1のみ返す

if __name__ == "__main__":
    """ サンプル実行 """
    with open("hyps.txt") as f:
        cands = [line.strip() for line in f]

    with open("refs.txt") as f:
        refs = [line.strip() for line in f]

    (P, R, F), hashname = score(cands, refs, lang="en", return_hash=True)
    for p,r, f1 in zip(P, R, F):
        print("P:%f, R:%f, F1:%f" %(p, r, f1))
    print(f"{hashname}:\n Precision={P.mean().item():.6f}\n Recall={R.mean().item():.6f}\n F1スコア={F.mean().item():.6f}")