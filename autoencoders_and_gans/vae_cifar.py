from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
import random

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='./checkpoints/')
parser.add_argument('--reconstruct_path', type=str, default='./reconstructions')
parser.add_argument('--train', action='store_true', default=True)
parser.add_argument('--continue_epoch', type=int, default=10)
parser.add_argument('--d_z', type=int, default=20, help='dimensionality of latent space')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
if not os.path.exists(args.reconstruct_path):
    os.makedirs(args.reconstruct_path)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    # Architecture:
    #            fc (400)
    #               |
    #            fc (400)
    #            |     |
    #       fc(400) fc(400)
    #            |     |
    #
    #
    #       fc_mu (20)  fc_logvar(20)
    #            |       |
    #             -------
    #                |
    #              fc(400)
    #                |
    #              fc(784)

    def __init__(self):
        super(VAE, self).__init__()
        self.fc_0 = torch.nn.Linear(3*32*32, 400)
        self.fc_1 = torch.nn.Linear(400, 400)

        self.fc_1_1 = torch.nn.Linear(400, 400)
        self.fc_1_2 = torch.nn.Linear(400, 400)

        self.fc_2_1 = torch.nn.Linear(400, 20)
        self.fc_2_2 = torch.nn.Linear(400, 20)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.dec_fc0 = torch.nn.Linear(20, 400)
        self.dec_fc1 = torch.nn.Linear(400, 3*32*32)

    def encode(self, x):
        res_0 = self.relu(self.fc_0(x))
        res_1 = self.relu(self.fc_1(res_0))
        return self.fc_2_1(self.relu(self.fc_1_1(res_1))), self.fc_2_2(self.relu(self.fc_1_2(res_1)))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        """Task2. Define decoder"""
        res_0 = self.relu(self.dec_fc0(z))
        res_1 = self.sigmoid(self.dec_fc1(res_0))
        return res_1

    def forward(self, x):
        """Task2 define forward path. which should consist of
        encode reparametrize and decode function calls"""
        x = x.view(-1, 3 * 32 * 32)
        mu, logvar = self.encode(x)
        reped = self.reparametrize(mu, logvar)
        decoded = self.decode(reped)

        # Todo train GAN

        return decoded, mu, logvar

    def save_model(self, epoch):
        model_file = os.path.join(args.save_path, 'vae_{}.th'.format(epoch))
        torch.save(self.state_dict(), model_file)
        print('model saved after epoch: {:03d}'.format(epoch))

    def load_model(self, epoch):
        model_file = os.path.join(args.save_path, 'vae_{}.th'.format(epoch))
        self.load_state_dict(torch.load(model_file))
        print('model restored from checkpoint on epoch: {:03d}'.format(epoch))



reconstruction_function = nn.MSELoss()
reconstruction_function.size_average = False

model = VAE()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    recosntruction_loss = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return recosntruction_loss + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch, ):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        recon_file = os.path.join(args.reconstruct_path, 'im_{}_{}_mse.jpg'.format(epoch, i))
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            utils.save_image(recon_batch.data.resize_((args.batch_size, 1, 32, 32, 3)), recon_file)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def manifold_visulisation(model):
    """Train VAE with len(z) = 2
    sample z values from closed interval of [-3, 3].
    visualize generated examples on the grid"""
    xs = None
    #for x in range(10):
    #    z = torch.FloatTensor(1, 2)
    #    z[0][0] = random.uniform(-3, 3)
    #    z[0][1] = random.uniform(-3, 3)
    #    latent_sp_file = os.path.join(args.reconstruct_path, 'latent_{}_{}_{}_{}.jpg'.format(epoch, 'test', z[0][0], z[0][1]))
    #    xs = model.decode(Variable(z))
    #    utils.save_image(xs.data.resize_(1, 28, 28), latent_sp_file, normalize=True)

    k = 30

    z = torch.FloatTensor(k * k, 2)
    idx = 0

    for i in range(0, k):
        for j in range(0, k):
            z[idx][0] = -3 + (6.0 / k) * i
            z[idx][1] = -3 + (6.0 / k) * j
            idx += 1
   
    latent_sp_file = os.path.join(args.reconstruct_path, 'latent_{}_{}_all.jpg'.format(epoch, 'test'))
    xs = model.decode(Variable(z))
    utils.save_image(xs.data.resize_((k * k, 1, 32, 32)), latent_sp_file, nrow=k)

if __name__ == '__main__':
    if True: # args.train:
        print('training model')
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            model.save_model(epoch)
            test(epoch)
    else:
        print('testing model')
        if args.continue_epoch != -1:
            epoch = args.continue_epoch
            model.load_model(epoch)
            manifold_visulisation(model)
