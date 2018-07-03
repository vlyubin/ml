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
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')

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
    # Define NN layers for VAE task
    # Follow this architecture
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
        self.fc_0 = torch.nn.Linear(3*32*32, 800)
        self.fc_1 = torch.nn.Linear(800, 400)

        self.fc_1_1 = torch.nn.Linear(400, 400)
        self.fc_1_2 = torch.nn.Linear(400, 400)

        self.fc_2_1 = torch.nn.Linear(400, 100)
        self.fc_2_2 = torch.nn.Linear(400, 100)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.dec_fc0 = torch.nn.Linear(100, 400)
        self.dec_fc1 = torch.nn.Linear(400, 800)
        self.dec_fc2 = torch.nn.Linear(800, 3*32*32)

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
        res_0 = self.relu(self.dec_fc0(z))
        res_1 = self.relu(self.dec_fc1(res_0))
        res_2 = self.sigmoid(self.dec_fc2(res_1))
        #print('Dec Debug')
        #print(self.dec_fc0.weight)
        #print(self.dec_fc1.weight)
        return res_2

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        mu, logvar = self.encode(x)
        reped = self.reparametrize(mu, logvar)
        decoded = self.decode(reped)
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

# Black magic and useless crap
nz = 100
ndf = 64
nc = 3

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 16, 4, 2, 1, bias=False),
        )
        self.fc = nn.Linear(256, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        pre_fc = output.view(args.batch_size, -1)
        fc_out = self.fc(pre_fc)
        return self.sigmoid(fc_out), pre_fc

netD = _netD(0)
netD.apply(weights_init)
criterion = nn.BCELoss()
print(netD)

optimizer_encoder = optim.Adam(model.parameters(), lr=2*1e-3)
optimizer_decoder = optim.Adam(model.parameters(), lr=2*1e-3)
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

def loss_function(mu, logvar, scores_real, scores_fake):
    diff_loss = (scores_real - scores_fake).pow(2).sum()

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return KLD, diff_loss


def train(epoch):
    model.train()
    train_loss = 0
    label = torch.FloatTensor(args.batch_size)
    real_label = 1
    fake_label = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        if data.size()[0] != args.batch_size:
            continue

        var_data = Variable(data)
        if args.cuda:
            var_data = var_data.cuda()
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        recon_batch, mu, logvar = model(var_data)

        # Now feed data and recon_bratch into the D
        # train with real
        netD.zero_grad()
        real_cpu = data

        batch_size = real_cpu.size(0)
        if False: # cuda
            real_cpu = real_cpu.cuda()
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(real_cpu)
        labelv = Variable(label)

        output, scores_real = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward(retain_variables=True)
        D_x = output.data.mean()

        # train with fake (what we generated)
        labelv = Variable(label.fill_(fake_label))
        output, scores_fake = netD(recon_batch.view(128, 3, 32, 32))
        errD_fake = criterion(output, labelv)
        errD_fake.backward(retain_variables=True)
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        loss_encoder, loss_decoder = loss_function(mu, logvar, scores_real, scores_fake)
        loss = loss_encoder.data[0] + loss_decoder.data[0]
        train_loss += loss

        netD.zero_grad()
        #optimizer_decoder.zero_grad()
        #optimizer_encoder.zero_grad()
        loss_decoder.backward(retain_variables=True)

        #optimizer_decoder.zero_grad()
        #optimizer_encoder.zero_grad()

        loss_encoder.backward(retain_variables=True)

        optimizer_encoder.step()
        optimizer_decoder.step()

        if batch_idx % args.log_interval == 0:
            print('Discriminator error: {}', errD)
            print('loss_encoder: {}', loss_encoder.data[0])
            print('loss_decoder: {}', loss_decoder.data[0])
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch, ):
    model.eval()
    for i, (data, _) in enumerate(test_loader):
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        recon_file = os.path.join(args.reconstruct_path, 'im_{}_{}_cifar.jpg'.format(epoch, i))
        if i == 0:
            utils.save_image(recon_batch.data.resize_((args.batch_size, 1, 32, 32, 3)), recon_file)

    print('Done testing')

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
