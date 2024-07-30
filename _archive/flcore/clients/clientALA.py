import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.ALA import ALA
from torch.optim import lr_scheduler
from loguru import logger


class clientALA(object):
    def __init__(self, args, id, train_samples, test_samples):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        if self.dataset == 'vgg':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9,
                                             nesterov=True)
            scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[40], gamma=0.1)
        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx

        train_data = read_client_data(self.dataset, self.id, is_train=True)
        self.ALA = ALA(self.id, self.loss, train_data, self.batch_size,
                       self.rand_percent, self.layer_idx, self.eta, self.device)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        for step in range(self.local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

    def local_initialization(self, received_global_model):
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)

    def test_metrics(self, model=None):

        testloader = self.load_test_data()
        if model == None:
            model = self.model
        model.eval()

        # if self.dataset == 'vgg':
        #     loss, accuracy = 0, 0
        #
        #     with torch.no_grad():
        #         for batch_x, batch_y in testloader:
        #             batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        #
        #             logits = model(batch_x)
        #             error = self.loss(logits, batch_y)
        #             loss += error.item()
        #
        #             probs, pred_y = logits.data.max(dim=1)
        #             accuracy += (pred_y == batch_y.data).float().sum() / batch_y.size(0)
        #
        #     loss /= len(testloader)
        #     accuracy = accuracy * 100.0 / len(testloader)
        #     logger.info('Test on {}, loss {:.4f}, accuracy {:.2f}%'.format(self.dataset, loss, accuracy))
        #     return accuracy, len(testloader), 0

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self, model=None):
        trainloader = self.load_train_data()
        if model == None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
