import argparse
import glob
import os
import torchvision
from multiprocessing import Pool
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="D:/dissertation/version 2/AvatarGAN-main/data/cartoonset100k/")
    parser.add_argument("--dest", type=str, default="D:/dissertation/version 2/AvatarGAN-main/data/cartoon")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def work(i):
    args = get_args()
    root = args.root
    dest = args.dest
    os.makedirs(dest, exist_ok=True)
    transform = transforms.Compose([transforms.CenterCrop(350), transforms.Resize((128, 128))])
    img_paths = []
    for folder in os.listdir(root):
        img_paths += glob.glob(os.path.join(root, folder, "*.png"))
    img_path = img_paths[i]
    img_name = os.path.split(img_path)[-1]
    dest_path = os.path.join(dest, img_name)
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    img.save(dest_path)
           


if __name__ == "__main__":
    args = get_args()
    root = args.root
    dest = args.dest
    os.makedirs(dest, exist_ok=True)

   

    img_paths = []
    print(type(img_paths))
    for folder in os.listdir(root):
        img_paths += glob.glob(os.path.join(root, folder, "*.png"))
        print(len(img_paths)) 

    pool = Pool(args.num_workers)
    for i in tqdm(pool.imap(work, range(len(img_paths))), total=len(img_paths)):
        pass
    pool.close()
    pool.join()
