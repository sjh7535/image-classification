import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from model import BaseModel
from tqdm import tqdm
from PIL import Image

class ImageDataset(Dataset):

    def __init__(self, root_dir, transform=None, fmt=':04d', extension='.jpg'):
        self.root_dir = root_dir
        self.fmtstr = '{' + fmt + '}' + extension
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.fmtstr.format(idx)
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        data = self.transform(img)
        return data

def inference(args, data_loader, model):
    """ model inference """

    model.eval()
    preds = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for i, x in enumerate(pbar):
            
            image = x.to(args.device)
            
            y_hat = model(image)
            
            y_hat.argmax()

            _, predicted = torch.max(y_hat, 1)
            preds.extend(map(lambda t: t.item(), predicted))

    return preds



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2023 DL Term Project')
    parser.add_argument('--load-model', default='checkpoints/model.pth', help="Model's state_dict")
    parser.add_argument('--batch-size', default=16, help='test loader batch size')
    parser.add_argument('--dataset', default='test_images/', help='image dataset directory')

    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # instantiate model
    # model = BaseModel()
    # model.load_state_dict(torch.load(args.load_model))
    # model.to(device)

    # torchvision model
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(args.load_model))
    model.to(device)

    # load dataset in test image folder
    # you may need to edit transform
    test_data = ImageDataset(args.dataset, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    # write model inference
    preds = inference(args, test_loader, model)
        
    with open('result.txt', 'w') as f:
        f.writelines('\n'.join(map(str, preds)))