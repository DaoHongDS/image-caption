from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from neighbors import *
from itertools import combinations

def get_each_BLEU_score(list_s_references,s_candidate):
    candidate = s_candidate.split()
    references = []
    for i in range(0,len(list_s_references)):
        l = list_s_references[i].split()
        references.append(l)
    bleu_score = sentence_bleu(references, candidate,smoothing_function=SmoothingFunction().method1)
    return bleu_score

def get_zone_BLEU_score(captions): #captions is list of dict
    n_len = len(captions)
    zone_bleu_score = []        #captions with bleu score
    for i in range(0,n_len):
        ref= [captions[i]["caption"] for i in range(0, n_len)]      #ref is list of string of caption
        ref.remove(ref[i])
        n_score = get_each_BLEU_score(ref,captions[i]["caption"])

        tmp_bleu_score = dict(captions[i])
        tmp_bleu_score["bleu_score"] = n_score
        zone_bleu_score.append(tmp_bleu_score)
    return zone_bleu_score

def get_max_score_zone(mother_set, size_of_subset):
    subsets = list(combinations(mother_set, size_of_subset))
    max_score = 0   #max score of zones
    max_i=0           #zone which get max score
    for i in range(0,len(subsets)):
        subset_scores = get_zone_BLEU_score(subsets[i])
        tot_subset_scores = 0
        for j in range(0,len(subset_scores)):
            tot_subset_scores += subset_scores[j]["bleu_score"]
        if max_score < tot_subset_scores:
            max_score = tot_subset_scores
            max_i = i
    return subsets[max_i]

imgFile = "./image/test.jpg"
annFile = "./coco/annotations/captions_val2017.json"
train_img_folder = "./coco/val2017/"
k =3

kn = k_neighbors(imgFile, annFile, train_img_folder, k)
captions = kn.get_k_neighbors()
#print("k neighbors")
#print(captions)
max_score_zone = get_max_score_zone(captions,10)
print(max_score_zone[0])
#sorted_zone = sorted(max_score_zone,key=lambda x: x["bleu_score"], reverse=True)
#print("the best match caption: ", sorted_zone[0]["caption"])



