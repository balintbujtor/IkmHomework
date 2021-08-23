import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import torch.optim as optim
from bayes_opt import BayesianOptimization
from IPython.display import HTML, display
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

# first part / creating the network


# layer
class Layer(nn.Module):

    def __init__(self, inplanes, planes, kernelsize, stride=1, nlintype=0, bnorm=True, residual=True):
        super(Layer, self).__init__()

        if nlintype is 1:
            nlintype = nn.LeakyReLU()
        if nlintype is 2:
            nlintype = nn.Sigmoid()
        else:
            nlintype = nn.ReLU(inplace=True)

        # if we do not use bnorm, then bnorm is an identical layer but we turn on bias instead
        if bnorm is True:
            bnorm = nn.BatchNorm2d
            bias = False
        else:
            bnorm = nn.Identity
            bias = True

        if residual is True:
            self.res = True
        else:
            self.res = False

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernelsize, stride=stride, bias=bias,
                              padding=kernelsize // 2)
        self.bnorm = bnorm(planes)
        self.activation = nlintype

    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.bnorm(out)

        if self.res is True:
            out += identity

        out = self.activation(out)

        return out


# class to create a level
class Level(nn.Module):
    def __init__(self, inpl, outpl, kernel, layersperlvl=1, nlin=0, bn=True, resi=True):
        super(Level, self).__init__()

        self.layers = []

        # first layer, if we want to use different inplane number for example
        self.layers.append(
            Layer(inplanes=inpl, planes=outpl, kernelsize=kernel, stride=1, nlintype=nlin, bnorm=bn,
                  residual=False))

        # we do not count the strided layers in  layersperlevel
        for _ in range(1, layersperlvl):
            self.layers.append(
                Layer(inplanes=outpl, planes=outpl, kernelsize=kernel, stride=1, nlintype=nlin,
                      bnorm=bn,
                      residual=resi))

        # strided conv layer double stride double conv not residual
        self.layers.append(
            Layer(inplanes=outpl, planes=2 * outpl, kernelsize=kernel,
                  stride=2, nlintype=nlin, bnorm=bn, residual=False))

        self.lvl = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.lvl(x)
        return out


# network
class Network(nn.Module):
    def __init__(self, inplanes, nFeat, nC, nLevels, dropOut, layersperlevel, kernel, nlin, bnorm, residual):
        super(Network, self).__init__()

        # first level for different input channel size
        self.firstlevel = Level(inplanes, nFeat, layersperlevel, kernel, nlin, bnorm, residual)

        self.lvls = []

        # appending the network with nlevels, doubling the plane number with each iteration
        for idx in range(1, nLevels + 1):

            self.lvls.append(Level((2**idx)*nFeat, (2**idx)*nFeat, layersperlevel, kernel, nlin, bnorm, residual))

        self.middleLayers = nn.Sequential(*self.lvls)

        self.dropout = nn.Dropout2d(dropOut)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # fully connected layer
        self.fc = nn.Linear((2**(nLevels+1))*nFeat, nC)

    def forward(self, x):
        out = self.firstlevel(x)
        out = self.middleLayers(out)
        out = self.dropout(out)
        out = self.pooling(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


# second part / training


# initializing the pseudo random generators to use the same values
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def progress(value, maximum=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=maximum))


# creating a dataset to store the data
class ListDataset(Dataset):
    def __init__(self, path, img_size=(32, 32)):

        self.path = path
        # store all subdirectories, by name
        self.directories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        self.img_files = []
        self.labels = []

        # fill the img list with the images sorted by categories
        for it in range(0, len(self.directories)):
            for img in os.listdir(os.path.join(self.path, self.directories[it])):
                self.img_files.append(img)

        # fill the labels with the index of each directory
        for label in range(0, len(self.directories)):
            for _ in os.listdir(os.path.join(self.path, self.directories[label])):
                self.labels.append(label)

        self.imgsize = img_size
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):

        # get currentdir index and opening the image
        currentdir = self.labels[index % len(self.img_files)]
        img_path = os.path.join(self.path, os.path.join(self.directories[currentdir],
                                                        self.img_files[index % len(self.img_files)]))
        img = Image.open(img_path).convert('RGB')

        input_img = self.transform(img)

        label = self.labels[index % len(self.img_files)]

        return img_path, input_img, label

    def __len__(self):
        return len(self.img_files)


trainroot = "C:/Users/Balint/Downloads/trafficSigns/trainFULL"
valroot = "C:/Users/Balint/Downloads/trafficSigns/testFULL"


# train function with parameters below
# inplanes, nFeat, nC, nLevels, dropOut, layersperlevel, stride, kernel, nlin, bnorm, residual
def train(inplanes, nfeat, nc, nlevels, dropout, layersperlevel, kernel, nlin, bnorm, residual,
          bsize, lr, lr_ratio, numepoch, decay, visu, sample):
    if sample is False:
        # creating a dataloader for both validation and train data
        traindataloader = torch.utils.data.DataLoader(ListDataset(trainroot), batch_size=bsize, shuffle=True)
        valdataloader = torch.utils.data.DataLoader(ListDataset(valroot), batch_size=bsize, shuffle=False)

    else:

        leng = ListDataset(trainroot).__len__()
        rndindices = torch.randperm(leng)
        size = ((leng // 5) // bsize) * bsize
        divrndind = rndindices[:int(size)]
        traindataloader = torch.utils.data.DataLoader(ListDataset(trainroot), batch_size=bsize, shuffle=False,
                                                      sampler=torch.utils.data.SubsetRandomSampler(divrndind))

        leng = ListDataset(valroot).__len__()
        rndindices = torch.randperm(leng)
        size = ((leng // 5) // bsize) * bsize
        divrndind = rndindices[:int(size)]
        valdataloader = torch.utils.data.DataLoader(ListDataset(valroot), batch_size=bsize, shuffle=False,
                                                    sampler=torch.utils.data.SubsetRandomSampler(divrndind))

    # creating the network
    model = Network(inplanes, nfeat, nc, nlevels, dropout, layersperlevel, kernel, nlin, bnorm, residual).cuda()

    # Cross Entropy loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    # Cosine annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numepoch, eta_min=lr * lr_ratio)

    # values to be shown during training

    # train and validation running loss
    trainlosses = []
    vallosses = []

    # train and validation accuracies
    trainaccuracies = []
    valaccuracies = []

    # best loss value and best accuracy during validation
    bestloss = 1000
    bestacc = 0

    # training loop
    for epoch in range(numepoch):
        runningloss = 0.0
        total = 0
        correct = 0

        # print("train:")
        # bar = display(progress(0, len(traindataloader)), display_id=True)
        # training
        model.train()

        # we do not use the image path during training
        for idxx, (_, imgs, labels) in enumerate(traindataloader):
            imgs = imgs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # calculating loss and accuracy
            runningloss += loss.item()
            total += labels.size(0)

            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

            # bar.update(progress(idxx+1, len(traindataloader)))

        # appending the lists with current values
        runningloss /= len(traindataloader)
        trainlosses.append(runningloss)
        accuracy = correct / total
        trainaccuracies.append(accuracy)

        # validation
        model.eval()

        correct = 0
        total = 0
        runningloss = 0

        # print("validation:")
        # bar = display(progress(0, len(valdataloader)), display_id=True)

        with torch.no_grad():
            for k, (_, imgs, labels) in enumerate(valdataloader):
                imgs = imgs.cuda()
                labels = labels.cuda()
                outputs = model(imgs)

                loss = criterion(outputs, labels)

                runningloss += loss.item()
                total += labels.size(0)

                pred = outputs.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability.
                correct += pred.eq(labels.view_as(pred)).sum().item()

                # bar.update(progress(k+1, len(valdataloader)))

        runningloss /= len(valdataloader)
        vallosses.append(runningloss)
        accuracy = correct / total
        valaccuracies.append(accuracy)

        # if the current loss is better than the previous best
        if runningloss < bestloss:
            bestloss = runningloss
            torch.save(model.state_dict(), "model.pth")

        if accuracy > bestacc:
            bestacc = accuracy

        # updating the lr scheduler
        # print("epoch: %d" % (epoch + 1))
        scheduler.step(epoch)

    # visualizing data
    if visu is True:
        plt.plot(trainlosses)
        plt.plot(vallosses)
        plt.show()

        plt.plot(trainaccuracies)
        plt.plot(valaccuracies)
        plt.show()

        print(bestloss)
        print(bestacc)

    # returning the best accuracy / value to be maximized during bayesian optimization
    return bestacc


# third part / bayesian optimization

# the optimization can not deal with discrete values thus we create a function to convert continuous values
# to discrete which we give to the original training function
def cont2disctrain(nf, nlvls, dout, layersplvl, krnl, resid, bsize, lr, lr_ratio, decay):
    # converting the cont values to discr

    dnf = int(2 ** (np.round(nf)))
    dnlvls = int(np.round(nlvls))
    dlayersplvl = int(np.round(layersplvl))
    dkrnl = int(2 * np.round(krnl) + 1)
    # dnonlin = int(nonlin)
    dresid = int(np.round(resid))
    dbsize = int(2 ** (np.round(bsize)))
    # dnumepoch = int(numepoch)

    # calling the train function with discrete values
    return train(inplanes=3, nfeat=dnf, nc=55, nlevels=dnlvls, dropout=dout, layersperlevel=dlayersplvl,
                 kernel=dkrnl, nlin=1, bnorm=True, residual=dresid,
                 bsize=dbsize, lr=lr, lr_ratio=lr_ratio, numepoch=20, decay=decay, visu=False, sample=True)


# bayesian optimization / parameters to be optimized with the possible bounds
pbounds = {'nf': (2, 5), 'nlvls': (2, 5), 'dout': (0, 0.5), 'layersplvl': (1, 3), 'krnl': (0, 2), 'resid': (0, 1),
           'bsize': (2, 8), 'lr': (1e-5, 1e-1), 'lr_ratio': (0.01, 1), 'decay': (1e-3, 1e-7)}

# calling the optimization
bayoptimizer = BayesianOptimization(f=cont2disctrain, pbounds=pbounds, random_state=1, verbose=2)

for i in range(10):
    if i != 0:
        load_logs(bayoptimizer, logs=["./logs.json"])
        print("New optimizer is now aware of {} points.".format(len(bayoptimizer.space)))

    logger = JSONLogger(path="./logs.json")
    bayoptimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # maximizing the function with 6 random initializing steps and 60 total steps
    bayoptimizer.maximize(init_points=5, n_iter=5)

    # printing the best parameters
    print(bayoptimizer.max)

    for j, res in enumerate(bayoptimizer.res):
        print("Iteration {}: \n\t{}".format(j, res))
