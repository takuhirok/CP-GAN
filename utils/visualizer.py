import os

import torch
import torchvision.utils

from utils import util


class Visualizer():
    def __init__(self,
                 netG,
                 label_list,
                 device,
                 out,
                 num_samples=10,
                 batch_size=100,
                 data_range=(-1, 1)):
        self.netG = netG
        self.label_list = label_list
        self.device = device
        self.out = out
        self.num_samples = num_samples
        self.num_columns = len(label_list)
        self.batch_size = batch_size
        self.data_range = data_range

        z_base = netG.sample_z(num_samples).to(device)
        z = z_base.clone().unsqueeze(1).repeat(1, self.num_columns, 1)
        self.fixed_z = z.view(-1, netG.latent_dim)

        s_base = []
        for i in range(len(label_list)):
            s = 0
            for j in range(len(label_list[i])):
                y = torch.tensor(label_list[i][j]).unsqueeze(0).to(device)
                s += util.one_hot(y, netG.num_classes) / len(label_list[i])
            s_base.append(s)
        self.fixed_s = torch.cat(s_base, dim=0).repeat(self.num_samples, 1)

    def visualize(self, iteration):
        netG = self.netG
        netG.eval()

        with torch.no_grad():
            if self.fixed_s.size(0) < self.batch_size:
                x = netG(self.fixed_z, class_weight=self.fixed_s)
            else:
                xs = []
                for i in range(0, self.fixed_s.size(0), self.batch_size):
                    x = netG(self.fixed_z[i:i + self.batch_size],
                             class_weight=self.fixed_s[i:i + self.batch_size])
                    xs.append(x)
                x = torch.cat(xs, dim=0)
            torchvision.utils.save_image(
                x.detach(),
                os.path.join(self.out, 'samples_iter_%d.png' % iteration),
                self.num_columns,
                0,
                normalize=True,
                range=self.data_range)
