import os
import gc
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from configs import get_args
from dataset import Dataset
from models import Decoder, Encoder

IMG_EXTENSIONS = (".jpg", ".png", ".jpeg")

class Dataset:
    def __init__(self, root, transform=None):
        self.root = root
        torch.cuda.empty_cache()
        gc.collect()
        self.img_paths = [
            os.path.join(root, img_name)
            for img_name in os.listdir(root)
            if img_name.lower().endswith(IMG_EXTENSIONS)
        ]
 
        self.transform = transform

    def __getitem__(self, i):
        img = Image.open(self.img_paths[i])

        if self.transform is not None:
            img = self.transform(img)

        return img, os.path.split(self.img_paths[i])[-1]

    def __len__(self):
        return len(self.img_paths)

def main():
    args = get_args()
    
    device = torch.device("cpu")

    # Dataset
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    src_dataset = Dataset(args.src_root, transform=transform)

    src_loader = DataLoader(
        src_dataset, batch_size=1, num_workers=args.num_workers, shuffle=True, pin_memory=True,
    )

    encoder = Encoder(args.num_channels, args.num_features).to(device)
    decoder = Decoder(args.num_channels, args.num_features).to(device)
    exp_dir_path = os.path.join(args.exp_dir, args.exp_name)
    dst_dir_path = os.path.join(args.exp_dir, args.exp_name, "results")
    os.makedirs(dst_dir_path, exist_ok=True)
    os.makedirs(os.path.join(dst_dir_path, "gt"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir_path, "pred"), exist_ok=True)

    ckpt = torch.load(os.path.join(exp_dir_path, "save", "27000.ckpt"))
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
        pbar = tqdm(src_loader, total=len(src_loader))
        for src_data, src_name in pbar:
            gc.collect()
            torch.cuda.empty_cache()
            print(src_name)
            src_data = src_data.to(device)

            latent = encoder(src_data, mode="AB")
            pred_img = decoder(latent)

            pred_img = (pred_img + 1.0) * 0.5

            for pred, s_name in zip(pred_img, src_name):
                save_image(
                    pred, os.path.join(dst_dir_path, "pred", s_name), padding=0,
                )


if __name__ == "__main__":
    main()
