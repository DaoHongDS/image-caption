import cv2
from PythonAPI.pycocotools.coco import COCO
import numpy as np
from GIST.img_gist_feature.utils_gist import *
from GIST.img_gist_feature.util__cal import *
import time

class k_neighbors:
    def __init__(self, imgFile=None, annFile=None, train_img_folder=None, k=0):
        self.imgFile = imgFile                      #file need to predict caption
        self.annFile = annFile                      #annotation file of coco dataset
        self.train_img_folder = train_img_folder    #image folder of coco dataset
        self.k = k                                  #k image neighbors
        self.coco = COCO(annFile)
        self.np_gist = None                         #gist vector of input image
        self.train_vectors = []                     #list of feature vectors of coco img set

    def get_img_gist(self,file_name):                 #get gist vector of input image
        gist_helper = GistUtils();
        np_img = cv2.imread(file_name, -1)
        np_gist = gist_helper.get_gist_vec(np_img, mode="rgb")
        return np_gist

    def load_train_img_vectors(self):
        img_ids = self.coco.getImgIds()
        imgs = self.coco.loadImgs(img_ids)
        tic = time.time()
        for i in range(0, len(imgs)):
            tmp_vector = {}
            tmp_vector["id"] = imgs[i]["id"]
            img_file_name = self.train_img_folder + imgs[i]["file_name"]
            tmp_vector["feature"] = self.get_img_gist(img_file_name)
            print("calculate feature vector ", i)
            self.train_vectors.append(tmp_vector)
        print("time to calculate {} feature vectors: {}s".format(len(imgs), time.time() - tic))

    def get_cos_sim(self):      #get cosin similarity of feature vector of input image and feature vectors of coco
        cos_sims = []
        for i in range(0, len(self.train_vectors)):
            tmp_cos_sim = {}
            tmp_cos_sim["id"] = self.train_vectors[i]["id"]

            tmp_cos_sim["cos_sim"] = np.inner(np_l2norm(self.train_vectors[i]["feature"]), np_l2norm(self.np_gist))
            cos_sims.append(tmp_cos_sim)
        return cos_sims

    def get_k_neighbors(self):        #get captions of k neighbors in coco, which have highest cosin similarity
        self.np_gist = self.get_img_gist(self.imgFile)
        self.load_train_img_vectors()

        cos_sims = self.get_cos_sim()
        sorted_cos_sims = sorted(cos_sims, key=lambda x: x["cos_sim"], reverse=True)

        k_neighbor_ids = []
        for i in range(0, self.k):
            k_neighbor_ids.append(sorted_cos_sims[i]["id"])
        ann_ids = self.coco.getAnnIds(k_neighbor_ids)
        anns = self.coco.loadAnns(ann_ids)
        return anns

#------------test--------------
'''
imgFile = "./image/test.jpg"
annFile = "./coco/annotations/captions_val2017.json"
train_img_folder = "./coco/val2017/"
k =3

kn = k_neighbors(imgFile, annFile, train_img_folder, k)
captions = kn.get_k_neighbors()
print(captions)
print(len(captions))
'''