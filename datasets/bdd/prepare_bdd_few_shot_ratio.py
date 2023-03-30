import argparse
import copy
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
# from fsdet.utils.file_io import PathManager

# from ucbdrive few-shot-objection-detection gitHub

BDD_CLASSES = ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle', 'traffic light', 'traffic sign']  # fmt: skip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 20], help="Range of seeds"
    )
    parser.add_argument(
        "--path", type=str, nargs="?", help="data_path"
    )
    parser.add_argument(
        "--ratio", type=float, nargs="?", help="support shot ratio"
    )
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data = []
    data_per_cat = {c: [] for c in BDD_CLASSES}
    dirname = "BDD"
    f = open(os.path.join(args.path, dirname, "ImageSets", "Main", "trainval" + ".txt"))
    fileids = np.loadtxt(f, dtype=str).tolist()

    data.extend(fileids)
    for fileid in data:
        # dirname = os.path.join("datasets", "BDD")
        anno_file = os.path.join(args.path, dirname, "Annotations", fileid + ".xml")
        tree = ET.parse(anno_file)
        clses = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            clses.append(cls)
        for cls in set(clses):
            data_per_cat[cls].append(anno_file)

    result = {cls: {} for cls in data_per_cat.keys()}
    total_instance_per_cat = {c: len(data_per_cat[c]) for c in BDD_CLASSES}
    shots = {c: int(total_instance_per_cat[c]*args.ratio) for c in BDD_CLASSES}
    print(total_instance_per_cat)
    print(shots)
    #shots = [1, 5]
    for i in range(args.seeds[0] , args.seeds[1]):
        random.seed(i)
        for c in data_per_cat.keys():
            c_data = []
            #for j, shot in enumerate(shots[c]):
            #diff_shot = shots[c][j] - shots[c][j - 1] if j != 0 else 1
            shots_c = random.sample(data_per_cat[c], shots[c])
            num_objs = 0
            for s in shots_c:
                if s not in c_data:
                    tree = ET.parse(s)
                    file = tree.find("filename").text
                    name = "JPEGImages/{}".format(file)
                    c_data.append(name)
                    for obj in tree.findall("object"):
                        if obj.find("name").text == c:
                            num_objs += 1
                    if num_objs >= shots[c]:
                        break
            result[c][shots[c]] = copy.deepcopy(c_data)
        save_path = "bddsplit/seed{}".format(i)
        os.makedirs(save_path, exist_ok=True)
        for c in result.keys():
            #for shot in result[c].keys():
            filename = "box_{}ratio_{}_train.txt".format(args.ratio, c)
            with open(os.path.join(save_path, filename), "w") as fp:
                fp.write("\n".join(result[c][shots[c]]) + "\n")


if __name__ == "__main__":
    args = parse_args()
    generate_seeds(args)