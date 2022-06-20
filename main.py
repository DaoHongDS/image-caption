# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#import os
import cv2
import sys
#import json
import matplotlib.pyplot as plt

sys.path.append("./GIST/")

from img_gist_feature.utils_gist import *
from util__base import *
from util__cal import *

def proc_main(O_IN):
    s_img_url_a = O_IN["s_img_url_a"]
    np_img_gist_a = get_img_gist_feat(s_img_url_a)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    O_IN = {}
    O_IN['s_img_url_a'] = "GIST/test/A.jpg"
    O_IN['s_img_url_b'] = "GIST/test/B.jpg"
    proc_main(O_IN)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
