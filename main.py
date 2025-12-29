import argparse
import os
import sys
import time
import torch
import cv2
import math
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchnet import meter
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from data import HSTrainingData
from data import HSTestData
from MSDformer import MSDformer
from common import *
from metrics import compare_mpsnr
# loss
from loss import HLoss
from metrics import quality_assessment
from datetime import datetime

# 获取当前时间戳并格式化为无空格的字符串
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # 格式化时间戳





# global settings
resume = True
log_interval = 50
model_name = ''
test_data_dir = ''

folder_path = './results'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 创建文件名
filename = os.path.join(folder_path, 'loss_log.txt')

def visualize_and_save(image, filename):
    if image.shape[2] > 100:  # 确保有足够的波段
        # 提取指定波段
        band_70 = np.clip(image[:, :, 70], 0, 1)  # 第70个波段
        band_100 = np.clip(image[:, :, 100], 0, 1)  # 第100个波段
        band_36 = np.clip(image[:, :, 36], 0, 1)  # 第36个波段

        # 合成RGB图像
        img_rgb = np.zeros((image.shape[0], image.shape[1], 3))  # 创建一个空的RGB图像
        img_rgb[:, :, 0] = band_70  # R通道
        img_rgb[:, :, 1] = band_100  # G通道
        img_rgb[:, :, 2] = band_36  # B通道

        plt.imshow(img_rgb)
        plt.axis('off')  # 不显示坐标轴
        plt.savefig(filename, bbox_inches='tight')  # 保存图像
        plt.show()  # 显示图像
    else:
        print("Image does not have enough bands.")

def main():
    # parsers
    main_parser = argparse.ArgumentParser(description="parser for SR network")
    subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_parser.add_argument("--cuda", type=int, required=False,default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    train_parser.add_argument("--batch_size", type=int, default=32, help="batch size, default set to 32")
    train_parser.add_argument("--epochs", type=int, default=300, help="epochs, default set to 300")
    train_parser.add_argument("--n_feats", type=int, default=240, help="n_feats, default set to 240")

    train_parser.add_argument("--n_depth", type=int, default=4, help="n_blocks, default set to 4")
    train_parser.add_argument("--n_subs", type=int, default=8, help="n_subs, default set to 8")
    train_parser.add_argument("--n_ovls", type=int, default=0, help="n_ovls, default set to 0")
    train_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 4")
    train_parser.add_argument("--dataset_name", type=str, default="Chikusei", help="dataset_name, default set to dataset_name")
    train_parser.add_argument("--model_title", type=str, default="MSDformer", help="model_title, default set to model_title")
    train_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4,
                              help="learning rate, default set to 1e-4")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    train_parser.add_argument("--gpus", type=str, default="1", help="gpu ids (default: 7)")

    test_parser = subparsers.add_parser("test", help="parser for testing arguments")
    test_parser.add_argument("--cuda", type=int, required=False,default=1,
                             help="set it to 1 for running on GPU, 0 for CPU")
    test_parser.add_argument("--gpus", type=str, default="0,1", help="gpu ids (default: 7)")
    test_parser.add_argument("--dataset_name", type=str, default="Chikusei",help="dataset_name, default set to dataset_name")
    test_parser.add_argument("--model_title", type=str, default="MSDformer",help="model_title, default set to model_title")
    test_parser.add_argument("--n_feats", type=int, default=240, help="n_feats, default set to 240")

    test_parser.add_argument("--n_depth", type=int, default=4, help="n_blocks, default set to 4")
    test_parser.add_argument("--n_subs", type=int, default=8, help="n_subs, default set to 8")
    test_parser.add_argument("--n_ovls", type=int, default=0, help="n_ovls, default set to 1")
    test_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 2")

    args = main_parser.parse_args()
    print('===>GPU:',args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.subcommand is None:
        print("ERROR: specify either train or test")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    if args.subcommand == "train":
        train(args)
    else:
        test(args)
    pass


def train(args):
    # 使用合法的字符格式化日期，避免空格和冒号
    traintime = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')  # 格式化为合法的字符串
    device = torch.device("cuda" if args.cuda else "cpu")
    # args.seed = random.randint(1, 10000)
    print("Start seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    print('===> Loading datasets')
    train_path = './datasets32/'+args.dataset_name+'_x'+str(args.n_scale)+'/trains/'
    eval_path = './datasets32/' + args.dataset_name + '_x' + str(args.n_scale) + '/evals/'
    test_data_dir = './datasets32/' + args.dataset_name + '_x' + str(args.n_scale) + '/' + args.dataset_name + '_test.mat'

    train_set = HSTrainingData(image_dir=train_path, augment=True)
    eval_set =  HSTrainingData(image_dir=eval_path, augment=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_set = HSTestData(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    if args.dataset_name=='Cave':
        colors = 31
    elif args.dataset_name=='Pavia':
        colors = 102
    elif args.dataset_name=='Houston':
        colors = 48
    else:
        colors = 128

    print('===> Building model:{}'.format(args.model_title))
    net = MSDformer(
        n_subs=args.n_subs,
        n_ovls=args.n_ovls,
        n_colors=colors,
        n_depth=args.n_depth,
        n_feats=args.n_feats,
        n_scale=args.n_scale,
        datasetname=args.dataset_name

    )
    # print(net)
    model_title =  args.model_title +'_Depth='+str(args.n_depth)+'_Subs'+str(args.n_subs)+'_Ovls'+str(args.n_ovls)+'_Feats='+str(args.n_feats)
    model_name = './checkpoints/' + args.dataset_name+'_'+ args.model_title +'_Depth='+str(args.n_depth)+'_Subs'+str(args.n_subs)+'_Ovls'+str(args.n_ovls)+'_Feats='+str(args.n_feats)+'_'+"ckpt_epoch_"+"300" + ".pth"
    args.model_title = model_title


    start_epoch = 0
    if resume:
        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name,weights_only=True)
            start_epoch = checkpoint["epoch"]
            net.load_state_dict(checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(model_name))

    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)

    net.to(device).train()
    print_network(net)
    h_loss = HLoss(0.5, 0.1)
    L1_loss = torch.nn.L1Loss()

    print("===> Setting optimizer and logger")
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    epoch_meter = meter.AverageValueMeter()
    writer = SummaryWriter(f'runs/{model_title}_{traintime}')

    best_epoch = 28
    best_loss = 0.012334

    print('===> Start training')
    for e in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()

        adjust_learning_rate(args.learning_rate, optimizer, e + 1)
        epoch_meter.reset()
        net.train()
        print("Start epoch {}, learning rate = {}".format(e + 1, optimizer.param_groups[0]["lr"]))
        for iteration, (x, lms, gt) in enumerate(tqdm(train_loader, leave=False)):
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            optimizer.zero_grad()
            y = net(x, lms)
            loss = h_loss(y, gt)
            epoch_meter.add(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm(net.parameters(), clip_para)
            optimizer.step()
            # tensorboard visualization
            if (iteration + log_interval) % log_interval == 0:
                print("===> {} \tEpoch[{}]({}/{}): Loss: {:.6f}".format(time.ctime(), e + 1, iteration + 1,
                                                                        len(train_loader)-1, loss.item()))
                n_iter = e * len(train_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss', loss, n_iter)

        # run validation set every epoch
        eval_loss = validate(args, eval_loader, net, L1_loss)
        if e == 0:
            best_loss = eval_loss
        else:
            if eval_loss <= best_loss:
                best_loss = eval_loss
                best_epoch = e+1

        print("===> {}\tEpoch evaluation Complete: Avg. Loss: {:.6f}, best_epoch: {}, best_loss: {:.6f}".format
              (time.ctime(), eval_loss, best_epoch, best_loss))

        # 写入文件
        with open(filename, 'a') as f:
            f.write("Epoch: {}\tEval Loss: {:.6f}\tBest Loss: {:.6f}\n".format(e + 1, eval_loss, best_loss))


        # tensorboard visualization
        writer.add_scalar('scalar/avg_epoch_loss', epoch_meter.value()[0], e + 1)
        writer.add_scalar('scalar/avg_validation_loss', eval_loss, e + 1)
        # save model weights at checkpoints every 5 epochs
        if (e + 1) % 1 == 0:
            save_checkpoint(args, net, e+1, traintime)


    print("===> Start testing")
    result_path = './results/' + args.dataset_name + '_x' + str(args.n_scale) + '/'
    model_name = './checkpoints/' + traintime + '/' + args.dataset_name + '_' + model_title + "_ckpt_epoch_" + str(best_epoch) + ".pth"



    with torch.no_grad():
        test_number = 0
        epoch_meter = meter.AverageValueMeter()
        epoch_meter.reset()
        # loading model
        net = MSDformer(
            n_subs=args.n_subs,
            n_ovls=args.n_ovls,
            n_colors=colors,
            n_depth=args.n_depth,
            n_feats=args.n_feats,
            n_scale=args.n_scale,
            datasetname=args.dataset_name

        )

        state_dict = torch.load(model_name,weights_only=True)
        net.load_state_dict(state_dict['model'])
        net.to(device).eval()

        output = []
        for i, (ms, lms, gt) in enumerate(test_loader):
            # compute output
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            y = net(ms, lms)
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            ms = ms.squeeze().cpu().numpy().transpose(1, 2, 0)
            lms = lms.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0],:gt.shape[1],:]

            ms_image = ms  # 去掉批量维度，形状变为 [128, 128, 128]
            lms_image = lms  # 去掉批量维度，形状变为 [128, 128, 128]
            gt_image = gt  # 去掉批量维度，形状变为 [128, 128, 128]
            # visualize_and_save(ms_image, f'ms_image_{i}.png')
            # visualize_and_save(lms_image, f'lms_image_{i}.png')
            # visualize_and_save(gt_image, f'gt_image_{i}.png')
            # visualize_and_save(y, f'y_image_{i}.png')
            if i==0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number









    save_dir = result_path + model_title + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)
    QIstr = model_title + '_' + str(timestamp) + ".txt"
    json.dump(indices, open(QIstr, 'w'))



def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp


def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""
    lr = start_lr * (0.1 ** (epoch // 22 ))  # Chikusei x4

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(args, loader, model, criterion):
    device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    with torch.no_grad():
        for i, (ms, lms, gt) in enumerate(loader):
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            y = model(ms, lms)
            loss = criterion(y, gt)
            epoch_meter.add(loss.item())

    # back to training mode
    model.train()
    return epoch_meter.value()[0]


def test(args):
    if args.dataset_name=='Cave':
        colors = 31
    elif args.dataset_name=='Pavia':
        colors = 102
    elif args.dataset_name=='Houston':
        colors = 48
    else:
        colors = 128
    test_data_dir = './datasets32/' + args.dataset_name + '_x' + str(args.n_scale) + '/' + args.dataset_name + '_test.mat'
    result_path = './results/' + args.dataset_name + '_x' + str(args.n_scale) + '/'
    model_title = args.model_title +'_Depth=' + str(args.n_depth) + '_Subs' + str(args.n_subs) + '_Ovls' + str(args.n_ovls) +'_Feats=' + str(args.n_feats)
    model_name = f'./checkpoints/{args.dataset_name}_{model_title}_ckpt_epoch_{str(300)}.pth'

    # model_name = './checkpoints/' + {traintime}+ args.dataset_name +'_'+ model_title + "_ckpt_epoch_" + str(300) + ".pth"
    device = torch.device("cuda" if args.cuda else "cpu")
    print('===> Loading testset')

    test_set = HSTestData(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')

    with torch.no_grad():
        test_number = 0
        epoch_meter = meter.AverageValueMeter()
        epoch_meter.reset()
        # loading model
        net = MSDformer(
            n_subs=args.n_subs,
            n_ovls=args.n_ovls,
            n_colors=colors,
            n_depth=args.n_depth,
            n_feats=args.n_feats,
            n_scale=args.n_scale,
            datasetname=args.dataset_name

        )
        net.to(device).eval()
        state_dict = torch.load(model_name,weights_only=True)
        net.load_state_dict(state_dict["model"])

        output = []
        for i, (ms, lms, gt) in enumerate(test_loader):
            # compute output
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            y = net(ms, lms)
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            ms = ms.squeeze().cpu().numpy().transpose(1, 2, 0)
            lms = lms.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0], :gt.shape[1], :]

            ms_image = ms  # 去掉批量维度，形状变为 [128, 128, 128]
            lms_image = lms  # 去掉批量维度，形状变为 [128, 128, 128]
            gt_image = gt  # 去掉批量维度，形状变为 [128, 128, 128]
            # visualize_and_save(ms_image, f'ms_image_{i}.png')
            # visualize_and_save(lms_image, f'lms_image_{i}.png')
            # visualize_and_save(gt_image, f'gt_image_{i}.png')
            # visualize_and_save(y, f'y_image_{i}.png')
            if i == 0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    # save the results
    save_dir = result_path + model_title + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)

    QIstr = f"{result_path}{model_title}_{timestamp}.txt"
    json.dump(indices, open(QIstr, 'w'))


def save_checkpoint(args, model, epoch, traintime):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval().cpu()
    checkpoint_model_dir = './checkpoints/'+traintime+'/'
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)
    ckpt_model_filename = args.dataset_name + "_" + args.model_title + "_ckpt_epoch_" + str(epoch) + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)

    if torch.cuda.device_count() > 1:
        state = {"epoch": epoch, "model": model.module.state_dict()}
    else:
        state = {"epoch": epoch, "model": model.state_dict()}
    torch.save(state, ckpt_model_path)
    model.to(device).train()
    print("Checkpoint saved to {}".format(ckpt_model_path))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


if __name__ == "__main__":
    main()
