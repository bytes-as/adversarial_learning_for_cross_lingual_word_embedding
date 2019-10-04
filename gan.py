import argparse
import os
import numpy as np
import math
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--mapping_size", type=int, default=28, help="size of each mapping dimension(in our case only one mapping)")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen mapping samples")
args = parser.parse_args()
print(args)

cuda = True if torch.cuda.is_available() else False
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, int(np.prod(args.mapping_size))),
            nn.Tanh()
        )

    def forward(self, z):
        W_generated = self.model(z)
        return W_generated
        


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(args.mapping_size)), 512),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(mapped_embeddings)
        return validity

adversarial_loss = torch.nn.BCELoss()

generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for epoch in range(args.n_epochs):
    for i, embeddings in enumerate(monolingual_embeddings_of_seed_x):
        # Adversarial ground truthsq
        valid = Variable(Tensor(embeddings.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(embeddings.size(0), 1).fill_(0.0), requires_grad=False)
        # print(valid.size)
        # Configure input
        real_embeddings = Variable(embeddings.type(Tensor))

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (embeddings.shape[0], args.latent_dim))))

        # Generate a batch of images
        wx = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(wx), valid)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_embeddings), valid)
        fake_loss = adversarial_loss(discriminator(wx.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, args.n_epochs, i, len(dataloader), d_loss, g_loss)
        )

        batches_done = epoch * monolingual_embeddings_of_seed_x.size(0) + i
        if batches_done % args.sample_interval == 0:
            # save the genereated mapping 'W'