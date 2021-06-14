import argparse
import os
from math import log10
from tqdm import tqdm

import numpy as np
import pandas as pd
from model import RDBHSSANet
from data_utils import TestDatasetFromFolder, display_transform
import pytorch_ssim

import torch
import torchvision.utils as utils
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from AverageMeter import *

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')

# parameters
parser.add_argument('--test_dir', type=str, default='./data/rat2019/test')
parser.add_argument('--data_name', type=str, default='rat2019')
parser.add_argument('--gpu_id', type=str, default='1', help='specify the gpu ids')
parser.add_argument('--checkpoint', type=str, default='./output/checkpoint/model_epoch010.pth')
args = parser.parse_args()

TEST_DIR = args.test_dir
DATA_NAME = args.data_name
GPU_ID = args.gpu_id
CHECKPOINT = args.checkpoint

# results = {'rat2019001': {'psnr': [], 'ssim': []}, 'testserial': {'psnr': [], 'ssim': []}}
results = {}
results[DATA_NAME] = {'psnr':[],'ssim':[]}

def main():
    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu'))
    kernel_size = checkpoint['kernel_size']
    model = RDBHSSANet()
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    epoch = checkpoint['epoch']

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    cudnn.benchmark =True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available:
        model = model.to(device)

    test_set = TestDatasetFromFolder(TEST_DIR)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')
    out_path = 'benchmark_results/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    interp_error = AverageMeter()
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    model.eval()
    for image_name, frame0, frame1, frame2 in test_bar:
        image_name = image_name[0]
        # print('combine size:',val_input.size())
        # print('target size:',val_target.size())
        if torch.cuda.is_available():
            frame0 = frame0.to(device)
            frame1 = frame1.to(device)
            frame2 = frame2.to(device)
        with torch.no_grad():
            frame_out = model(frame0, frame2)

        # IE
        rec_rgb = frame_out.cpu().numpy() * 255.0
        gt_rgb = frame1.cpu().numpy() * 255.0
        # print(rec_rgb)
        diff_rgb_abs = np.abs(rec_rgb - gt_rgb)
        avg_interp_error_abs = np.mean(diff_rgb_abs).item()

        interp_error.update(avg_interp_error_abs, 1)

        psnr = -10 * log10(torch.mean((frame1 - frame_out) * (frame1 - frame_out)).item())
        ssim = pytorch_ssim.ssim(frame_out, frame1).item()

        test_images = torch.stack(
            [display_transform()(frame1.data.cpu().squeeze(0)),
             display_transform()(frame_out.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=2, padding=5)
        utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                         image_name.split('.')[-1], padding=5)

        # save psnr\ssim
        PSNR.update(psnr, 1)
        SSIM.update(ssim, 1)

    out_path = 'statistics/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    saved_results = {'psnr': [], 'ssim': [], 'ie':[]}

    saved_results['psnr'].append(round(PSNR.avg, 4))
    saved_results['ssim'].append(round(SSIM.avg, 4))
    saved_results['ie'].append(round(interp_error.avg, 4))
    data_frame = pd.DataFrame(saved_results, results.keys())
    data_frame.to_csv(out_path + 'test_results.csv', index_label='DataSet')


if __name__ == "__main__":
    main()
