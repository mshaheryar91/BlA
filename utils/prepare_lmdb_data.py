"""
    Refer to https://github.com/rosinality/stylegan2-pytorch/blob/master/prepare_data.py
"""

import argparse
from io import BytesIO
import multiprocessing
from functools import partial
import os, glob, sys

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, (size, size), resample)
    # img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(
    img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100
):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, file, img_id = img_file
    # print("check resize_worker:", i, file, img_id)
    img = Image.open(file)
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, out, img_id


def file_to_list(filename):
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()
    files = [f.rstrip() for f in files]
    return files



def prepare(
    env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS
):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)
    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file, file.split('/')[-1].split('.')[0]) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs, img_id in tqdm(pool.imap_unordered(resize_fn, files)):
            key_label = f"{str(i).zfill(5)}".encode("utf-8")
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")
                with env.begin(write=True) as txn:
                    txn.put(key, img)
                    txn.put(key_label, str(img_id).encode("utf-8"))

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))

def prepare_identity(
    env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, label_file='identity_labels.txt'
):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)
    files = sorted(dataset.imgs, key=lambda x: x[0])
    labels = file_to_list(label_file)  # New: Read identity labels from a file
    label_dict = {line.split()[0]: int(line.split()[1]) for line in labels}

    files_identity = []
    for i, (file, split) in enumerate(files):
        img_name = file.split('/')[-1]
        identity_label = label_dict.get(img_name)
        if identity_label is not None:
            files_identity.append((i, file, identity_label))

    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs, label in tqdm(pool.imap_unordered(resize_fn, files_identity)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")
                key_label = f"{'label'}-{str(i).zfill(5)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)
                    txn.put(key_label, str(label).encode("utf-8"))

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))




def prepare_attr(
    env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, label_attr='identity', identity_file=None
):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)
    files = sorted(dataset.imgs, key=lambda x: x[0])
    if label_attr == 'identity' and identity_file is not None:
        # Read identity labels
        identity_labels = {}  # mapping from image filename to identity label
        identity_counts = {}  # mapping from identity label to count

        with open(identity_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        img_filename, identity = parts
                        identity = int(identity)
                        identity_labels[img_filename] = identity
                        if identity in identity_counts:
                            identity_counts[identity] += 1
                        else:
                            identity_counts[identity] = 1

        # Find the identity with the most images
        max_identity = max(identity_counts, key=identity_counts.get)
        print(f"Identity with the most images: {max_identity} ({identity_counts[max_identity]} images)")

        files_attr = []
        for i, (file, _) in enumerate(files):
            img_filename = os.path.basename(file)
            identity = identity_labels.get(img_filename)
            if identity is None:
                # Image not found in identity labels
                continue
            label = 1 if identity == max_identity else 0
            files_attr.append((i, file, label))

        files = files_attr
        
        # Write labels to text file
        with open('label.txt', 'w') as label_file:
            for i, file, label in files:
                img_filename = os.path.basename(file)
                label_file.write(f"{img_filename} {label}\n")

    else:
        # Handle other attributes if needed
        pass

    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for item in tqdm(pool.imap_unordered(resize_fn, files), total=len(files)):
            if item is None:
                continue
            i, imgs, label = item
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")
                key_label = f"label-{str(i).zfill(5)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)
                    txn.put(key_label, str(label).encode("utf-8"))

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str)
    parser.add_argument("--size", type=str, default="128,256,512,1024")
    parser.add_argument("--n_worker", type=int, default=5)
    parser.add_argument("--resample", type=str, default="bilinear")
    parser.add_argument("--attr", type=str, default="identity")
    parser.add_argument("--identity_file", type=str, required=True, help="Path to the identity label file")
    #parser.add_argument("--label_output_file", type=str)
    parser.add_argument("path", type=str)
   

    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]

    sizes = [int(s.strip()) for s in args.size.split(",")]
    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare_attr(
            env, imgset, args.n_worker, sizes=sizes, resample=resample,
            label_attr=args.attr, identity_file=args.identity_file
        )
