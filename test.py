import argparse
import os

import torch

from models import resnet
from utils import util
from utils.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser()
    # Seed option
    parser.add_argument('--seed', default=0, type=int)
    # GPU option
    parser.add_argument('--gpu_id', type=int, default=0)
    # Dataset option
    parser.add_argument('--dataset', type=str, default='cifar10to5')
    # Genrator option
    parser.add_argument('--g_path', type=str, required=True)
    # Output options
    parser.add_argument('--out', type=str, default='samples')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    args = parser.parse_args()

    # Set up seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Set up GPU
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device('cuda:%d' % args.gpu_id)
    else:
        device = torch.device('cpu')

    # Set up dataset
    if args.dataset == 'cifar10':
        vis_label_list = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    elif args.dataset == 'cifar10to5':
        vis_label_list = [[0], [0, 1], [1], [1, 2], [2], [2, 3], [3], [3, 4],
                          [4], [4, 0]]
    elif args.dataset == 'cifar7to3':
        vis_label_list = [[0], [0, 1], [1], [1, 2], [2], [2, 0], [0, 1, 2]]

    # Set up generator
    g_root = os.path.dirname(args.g_path)
    g_params = util.load_params(os.path.join(g_root, 'netG_params.pkl'))
    g_iteration = int(
        os.path.splitext(os.path.basename(args.g_path))[0].split('_')[-1])
    netG = resnet.Generator(**g_params)
    netG.to(device)
    netG.load_state_dict(
        torch.load(args.g_path, map_location=lambda storage, loc: storage))
    netG.eval()

    # Set up output
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # Set up visualizer
    visualizer = Visualizer(netG, vis_label_list, device, args.out,
                            args.num_samples, args.eval_batch_size)

    # Visualize
    visualizer.visualize(g_iteration)


if __name__ == '__main__':
    main()
