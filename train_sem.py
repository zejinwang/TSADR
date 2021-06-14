import argparse
import os
from tqdm import tqdm
from math import log10
import pandas as pd
import numpy

from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from modules.model import PCDNet
from loss import CNNLoss

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.backends.cudnn as cudnn
import torchvision.utils as utils
from torch.utils.data import DataLoader
import torch

from AverageMeter import *
from collections import OrderedDict
import util

parser = argparse.ArgumentParser(description='Train Frame Interpolation Models')

# parameters
parser.add_argument('--train_dir', type=str, default='./data/rat2019/train', help='train dataset path')
parser.add_argument('--val_dir', default='data/rat2019/val', type=str, help='val dataset path')
parser.add_argument('--gpu_id', default='1', type=str, help='specified gpu ids you want to use')
parser.add_argument('--out_dir', type=str, default='./output')
parser.add_argument('--num_epochs', type=int, default=10, help='train epoch number')
parser.add_argument('--batch_size', type=int, default=16, help='iter train samples')
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint model path to load')
parser.add_argument('--seed', default=121, type=int, help='random seed')
parser.add_argument('--lr', default=4e-4, type=float, help='initial learning rate ')

args = parser.parse_args()
TRAIN_DIR = args.train_dir
VAL_DIR = args.val_dir
OUT_DIR = args.out_dir
GPU_ID = args.gpu_id
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
CHECKPOINT = args.checkpoint
SEED = args.seed
LR = args.lr


def get_current_learning_rate(optimizer):
    # return self.schedulers[0].get_lr()[0]
    return optimizer.param_groups[0]["lr"]


def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    result_dir = OUT_DIR + '/result'
    ckpt_dir = OUT_DIR + '/checkpoint'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    total_epoch = NUM_EPOCHS
    batch_size = BATCH_SIZE

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #### set random seed
    seed = SEED
    if seed is None:
        seed = random.randint(1, 10000)
    util.set_random_seed(seed)

    train_set = TrainDatasetFromFolder(TRAIN_DIR, RCrop=(256, 256))
    # train_set = TrainDBreader(TRAIN_DIR, resize=(128, 128))
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    val_set = ValDatasetFromFolder(VAL_DIR)
    # val_set = ValDBreader(VAL_DIR, resize=None)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    if CHECKPOINT is not None:
        checkpoint = torch.load(CHECKPOINT)
        netG = PCDNet()
        state_dict = checkpoint['state_dict']
        netG.load_state_dict(state_dict)
        epoch = checkpoint['epoch']
    else:
        netG = PCDNet()
        epoch = 0

    start =epoch + 1

    if torch.cuda.is_available():
        netG = netG.to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=LR)

    # adjust learning rate
    #scheduler1 = StepLR(optimizerG, step_size=10, gamma=0.1)
    scheduler1 = ReduceLROnPlateau(optimizerG, 'min', factor=0.1, patience=5, verbose=True)

    criterion = CNNLoss().to(device)


    results = {'train_loss': [],'val_loss': [], 'psnr': [], 'ssim': [], 'ie': []}

    saved_total_loss = 10e10
    saved_total_ie = 10e10

    while True:
        if epoch == total_epoch:
            break
        train_bar = tqdm(train_loader)

        running_results = OrderedDict()
        running_results["loss"] = []

        netG.train()

        for frame0, frame1, frame2 in train_bar:
            if torch.cuda.is_available():
                frame0 = frame0.to(device)
                frame1 = frame1.to(device)
                frame2 = frame2.to(device)

            netG.zero_grad()
            fake_img = netG(frame0, frame2)
            loss = criterion(fake_img, frame1)
            loss.backward()
            optimizerG.step()

            running_results["loss"].append(loss.item())

            train_bar.set_description(desc='[%d/%d] lr: %.4e Loss: %.4f ' % (
            epoch, NUM_EPOCHS, get_current_learning_rate(optimizerG), numpy.mean(running_results['loss'])))

            # scheduler1.step(running_results['loss'] / running_results['batch_sizes'])

        #scheduler1.step()
        #scheduler1.step(numpy.mean(running_results['loss']))
        if os.path.exists(ckpt_dir+'/netG_epoch_%d.pth'%(epoch-1)):
            os.remove(ckpt_dir +'/netG_epoch_%d.pth'%(epoch-1))
        torch.save({'epoch': epoch, 'state_dict': netG.state_dict()}, ckpt_dir + '/netG_epoch_%d.pth' % epoch)
        epoch += 1

        netG.eval()

        interp_error = AverageMeter()

        val_images = []
        index = 0

        val_results = OrderedDict()
        val_results["psnr"] = []
        val_results["ssim"] = []
        val_results["loss"] = []

        val_bar = tqdm(val_loader)
        for frame0, frame1, frame2 in val_bar:
            if torch.cuda.is_available():
                frame0 = frame0.to(device)
                frame1 = frame1.to(device)
                frame2 = frame2.to(device)
            with torch.no_grad():
                frame_out = netG(frame0, frame2)
                loss = criterion(frame_out, frame1)

            # metrics: PSNR, SSIM, IEs, Loss
            # IE
            rec_rgb = frame_out.cpu().numpy() * 255.0
            gt_rgb = frame1.cpu().numpy() * 255.0

            diff_rgb_abs = numpy.abs(rec_rgb - gt_rgb)
            avg_interp_error_abs = numpy.mean(diff_rgb_abs).item()

            interp_error.update(avg_interp_error_abs, 1)

            gt_img = util.tensor2img(frame1)
            fk_img = util.tensor2img(frame_out)
            gt_img = gt_img / 255.0
            fk_img = fk_img / 255.0

            psnr = util.calculate_psnr(fk_img * 255, gt_img * 255)
            ssim = util.calculate_ssim(fk_img * 255, gt_img * 255)

            val_results["psnr"].append(psnr)
            val_results["ssim"].append(ssim)
            val_results["loss"].append(loss.item())

            val_bar.set_description(
                desc='[generating predict images] PSNR: %.4f dB SSIM: %.4f IE: %.4f' % (
                   numpy.mean(val_results['psnr']), numpy.mean(val_results['ssim']), interp_error.avg))

            if index % 1 == 0:
                val_images.extend(
                    [display_transform()(frame0.data.cpu().squeeze(0)),
                     display_transform()(frame1.data.cpu().squeeze(0)),
                     display_transform()(frame_out.data.cpu().squeeze(0)),
                     display_transform()(frame2.data.cpu().squeeze(0))])
            # 验证阶段结束后，输出的 val_images 尺寸为(K x W x H x 2)
            # 每个epoch将所有验证集图片的 tensor 进行一次stack
            index += 1

        scheduler1.step(numpy.mean(val_results['loss']))

        val_images = torch.stack(val_images)
        val_images = torch.chunk(val_images, val_images.size(0) // 4)
        val_save_bar = tqdm(val_images, desc='[saving training results]')
        index = 1

        for image in val_save_bar:
            image = utils.make_grid(image, nrow=4, padding=5)
            utils.save_image(image, result_dir + '/epoch_%d_index_%d.png' % (epoch, index), padding=5)
            index += 1

        # save model parameters
        #if epoch % 10 == 0:
        #    torch.save({'epoch': epoch, 'state_dict': netG.state_dict(), 'kernel_size': kernel_size}, ckpt_dir + '/netG_epoch' + str(epoch).zfill(3) + '.pth')
        if saved_total_loss >= numpy.mean(val_results['loss']) and saved_total_ie >= interp_error.avg:
            saved_total_loss = numpy.mean(val_results['loss'])
            saved_total_ie = interp_error.avg
            torch.save({'epoch': epoch, 'state_dict': netG.state_dict()}, ckpt_dir + '/best.pth')

        # save loss\scores\psnr\ssim
        results['train_loss'].append(numpy.mean(running_results['loss']))
        results['val_loss'].append(numpy.mean(val_results['loss']))
        results['psnr'].append(numpy.mean(val_results['psnr']))
        results['ssim'].append(numpy.mean(val_results['ssim']))
        results['ie'].append(interp_error.avg)

        #if epoch % 10 == 0 and epoch != 0:
        if epoch != 0:
            data_frame = pd.DataFrame(
                data={'Train Loss': results['train_loss'],
                      'Val Loss': results['val_loss'],
                      'PSNR': results['psnr'], 'SSIM': results['ssim'],'IE':results['ie']},
                index=range(start, epoch + 1))
            data_frame.to_csv(OUT_DIR + '/train_results.csv', index_label='Epoch')


if __name__ == "__main__":
    main()
