import numpy as np
import pandas as pd

def extractGT(testFileName):
    df = pd.read_csv(testFileName)
    img2bbox = {}
    for image_id,sdf in df.groupby('image_id'):
        bbox = []
        det = []
        for _,row in sdf.iterrows():
            bbox.append([float(x.strip()) for x in row['bbox'].strip()[1:-1].split(',')])
        for i in range(len(bbox)):
            bbox[i][2] = bbox[i][0]+ bbox[i][2]
            bbox[i][3] = bbox[i][1]+ bbox[i][3]   
        det = [False] * len(bbox)
        img2bbox[image_id] = {'bbox':np.array(bbox),'det':np.array(det)}
    return img2bbox

def extractDet(detFileName):
    df = pd.read_csv(detFileName)
    img2detBbox = {}
    for _,row in df.iterrows():
        precStr = row['PredictionString']
        confAndBbox = precStr.split(' ')
        confidence = []
        precBbox = []
        nums = int(len(confAndBbox)/5)
        for i in range(nums):
            confidence.append(float(confAndBbox[i*5]))
            precBbox.append([])
            for j in range(1,5):
                precBbox[i].append(float(confAndBbox[i*5+j]))
            precBbox[i][2] = precBbox[i][0]+ precBbox[i][2]
            precBbox[i][3] = precBbox[i][1]+ precBbox[i][3]    
        img2detBbox[row['image_id']] = {'confidence':np.array(confidence),'precBbox':np.array(precBbox)}
    return img2detBbox


def computeAPrec(img2bbox,img2detBbox,ovthresh):
    ap = 0
    for k,v in img2detBbox.items():
        tp = 0
        fp = 0
        fn = 0
        GT = img2bbox[k]['bbox']
        det = img2bbox[k]['det']
        BB = v['precBbox']
        confidence = v['confidence']
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        for _,bb in enumerate(BB):
            if GT.size > 0:
                # 计算IoU
                # intersection
                ixmin = np.maximum(GT[:, 0], bb[0])
                iymin = np.maximum(GT[:, 1], bb[1])
                ixmax = np.minimum(GT[:, 2], bb[2])
                iymax = np.minimum(GT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                    (GT[:, 2] - GT[:, 0] + 1.) *
                    (GT[:, 3] - GT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            # 取最大的IoU
            if ovmax > ovthresh:  # 是否大于阈值
                if not det[jmax]:    # 未被检测
                    tp += 1.
                    det[jmax] = 1    # 标记已被检测
                else:
                    fp += 1.
            else:
                fp += 1.
        fn = len(det) - np.sum(det)
        ap += tp / np.maximum(tp + fp + fn, np.finfo(np.float64).eps)
        pass
    
    return ap/len(img2bbox)

if __name__ == "__main__":
    import sys
    det_file = sys.argv[1]
    test_file = sys.argv[2]
    img2detBbox = extractDet(det_file)
    img2bbox = extractGT(test_file)
    ap = computeAPrec(img2bbox,img2detBbox,0.5)
    print(ap)
    pass