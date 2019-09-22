import os
import cv2
import csv
import numpy as np
from scipy import misc

L2_sum = 0
with open('securityAI_round1_dev.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        origin_filepath = os.path.join("./securityAI_round1_images/", row['ImageName'])
        adv_filepath = os.path.join("./images/", row['ImageName'])
        raw_image = misc.imread(origin_filepath).astype(np.float)
        adv_image = misc.imread(adv_filepath).astype(np.float)
        diff = adv_image.reshape(-1, 3) - raw_image.reshape(-1, 3)
        L2 = np.mean(np.sqrt(np.sum((diff ** 2), axis=1)))
        L2_sum += L2
        print(row['ImageName'] + ", L2: " + str(L2))
L2_aver = L2_sum / 712.0
print("aver L2: " + str(L2_aver))