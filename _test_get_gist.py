import cv2
import sys

sys.path.append("./GIST/")

from img_gist_feature.utils_gist import *

s_img_url = "GIST/test/A.jpg"
gist_helper = GistUtils()
np_img = cv2.imread(s_img_url, -1)
print("default: rgb")
np_gist = gist_helper.get_gist_vec(np_img, mode="rgb")
print("shape ", np_gist.shape)
print("noly show 10dim", np_gist[0,:10], "...")
print()
print("convert rgb image")
np_gist = gist_helper.get_gist_vec(np_img, mode="rgb")
print("shape ", np_gist.shape)
print("noly show 10dim", np_gist[0,:10], "...")
print()
print("convert gray image")
np_gist = gist_helper.get_gist_vec(np_img, mode="gray")
print("shape ", np_gist.shape)
print("noly show 10dim", np_gist[0,:10], "...")
print()