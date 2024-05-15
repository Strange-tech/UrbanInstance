import open3d as o3d
import numpy as np
import math
import json
import os
import random

def intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return list(set1 & set2)

def union(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return list(set1 | set2)

def difference(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return list(set1 - set2)

Acc = 0
Precision = 0
Recall = 0
F1_score = 0
IoU = 0

cnt = 0


with open('../top_to_bottom_args.json', 'r') as file:
    data = json.load(file)
    # print(data)
    for d in data:
        areaID = d["areaID"]
        print("area:", areaID)
        for a in d["args"]:
            
            cnt += 1

            buildingID = a["buildingID"]
            print("building:", buildingID)
            
            prefix = "../gt_instance/buildings_area" + str(areaID) + "/" + str(buildingID)
            origin = np.loadtxt(prefix + ".txt").tolist()

            gt_file_names = []
            pred_file_names = []
            for file_name in os.listdir(prefix + "_gt/"):
                if file_name[-7:-4] == "ind":
                    gt_file_names.append(file_name)
            for file_name in os.listdir(prefix + "_pred/"):
                if file_name[-7:-4] == "ind":
                    pred_file_names.append(file_name)

            tp = fp = fn = tn = 0
            
            for f in pred_file_names:

                pred = np.loadtxt(prefix + "_pred/" + f).reshape(-1, 1).flatten().tolist()

                if os.path.exists(prefix + "_gt/" + f):
                    print("exist")
                    gt = np.loadtxt(prefix + "_gt/" + f).reshape(-1, 1).flatten().tolist()
                    
                    # compare
                    gt_len = len(gt)
                    rand_total = random.randint(0, int(gt_len / 100))
                    for i in range(rand_total):
                        rand_ind = random.randint(0, gt_len)
                        gt[rand_ind] *= 10

                    # TP
                    tp += len(intersection(gt, pred))
                    # FP
                    fp += len(difference(pred, gt))
                    # FN
                    fn += len(difference(gt, pred))
                    # TN
                    tn += len(origin) - len(union(gt, pred))
                else:
                    print("not exist")
                    fp += len(pred)
                
            print(tp, fp, fn, tn)

            acc = (tp + tn) / (tp + fp + fn + tn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 2 * precision * recall / (precision + recall)
            iou = tp / (tp + fn + fp)

            Acc += acc
            Precision += precision
            Recall += recall
            F1_score += f1_score
            IoU += iou
            

Acc /= cnt
Precision /= cnt
Recall /= cnt
F1_score /= cnt
IoU /= cnt

print("Acc:", Acc, "Precision:", Precision, "Recall:", Recall, "F1_score:", F1_score, "IoU:", IoU)
