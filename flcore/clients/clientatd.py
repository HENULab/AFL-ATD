import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics

import flcore.servers.serveratd
from utils.data_utils import read_client_data
from flcore.clients.clientbase import Client
from loguru import logger
from utils.ALA import ALA
# from flcore.conditional_selection import ConditionalSelection

class clientCP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        # kwargs['ConditionalSelection'] = copy.deepcopy(flcore.servers.servercp.ConditionalSelection)
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

        self.lamda = args.lamda
        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx
        in_dim = list(args.model.head.parameters())[0].shape[1]
        self.context = torch.rand(1, in_dim).to(self.device)

        self.model = Ensemble(
            model=self.model,
            cs=copy.deepcopy(kwargs['ConditionalSelection']),
            head_g=copy.deepcopy(self.model.head),
            feature_extractor=copy.deepcopy(self.model.feature_extractor)
        )
        self.opt= torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.pm_train = []
        self.pm_test = []

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
            
    def set_parameters(self, feature_extractor):
        for new_param, old_param in zip(feature_extractor.parameters(), self.model.model.feature_extractor.parameters()):
            old_param.data = new_param.data.clone()

    def set_head_g(self, head):
        headw_ps = []
        for name, mat in self.model.model.head.named_parameters():
            if 'weight' in name:
                headw_ps.append(mat.data)
        headw_p = headw_ps[-1]
        for mat in headw_ps[-2::-1]:
            headw_p = torch.matmul(headw_p, mat)
        headw_p.detach_()
        self.context = torch.sum(headw_p, dim=0, keepdim=True)
        
        for new_param, old_param in zip(head.parameters(), self.model.head_g.parameters()):
            old_param.data = new_param.data.clone()

    def set_cs(self, cs):
        for new_param, old_param in zip(cs.parameters(), self.model.gate.cs.parameters()):
            old_param.data = new_param.data.clone()

    def save_con_items(self, items, tag='', item_path=None):
        self.save_item(self.pm_train, 'pm_train' + '_' + tag, item_path)


    def generate_upload_head(self):
        for (np, pp), (ng, pg) in zip(self.model.model.head.named_parameters(), self.model.head_g.named_parameters()):
            pg.data = pp * 0.5 + pg * 0.5

    def train_cs_model(self):
        trainloader = self.load_train_data()
        self.model.train()
        
        for _ in range(self.local_steps):
            self.model.gate.pm = []
            self.model.gate.gm = []
            self.pm_train = []
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output, rep, rep_base = self.model(x, is_rep=True, context=self.context)
                loss = self.loss(output, y)
                loss += MMD(rep, rep_base, 'rbf', self.device) * self.lamda
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        self.pm_train.extend(self.model.gate.pm)
        scores = [torch.mean(pm).item() for pm in self.pm_train]
        #print(np.mean(scores), np.std(scores))
        logger.info(f'The number {self.id} client is trained.')

def MMD(x, y, kernel, device='cpu'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
    return torch.mean(XX + YY - 2. * XY)


class Ensemble(nn.Module):
    def __init__(self, model, cs, head_g, feature_extractor) -> None:
        super().__init__()

        self.model = model
        self.head_g = head_g
        self.feature_extractor = feature_extractor

        for param in self.head_g.parameters():
            param.requires_grad = False
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.flag = 0
        self.tau = 1
        self.hard = False
        self.context = None

        self.gate = Gate(cs)

    def forward(self, x, is_rep=False, context=None):
        rep = self.model.feature_extractor(x)

        gate_in = rep

        if context != None:
            context = F.normalize(context, p=2, dim=1)
            if type(x) == type([]):
                self.context = torch.tile(context, (x[0].shape[0], 1))
            else:
                self.context = torch.tile(context, (x.shape[0], 1))

        if self.context != None:
            gate_in = rep * self.context

        if self.flag == 0:
            rep_p, rep_g = self.gate(rep, self.tau, self.hard, gate_in, self.flag)
            output = self.model.head(rep_p) + self.head_g(rep_g)
        elif self.flag == 1:
            rep_p = self.gate(rep, self.tau, self.hard, gate_in, self.flag)
            output = self.model.head(rep_p)
        else:
            rep_g = self.gate(rep, self.tau, self.hard, gate_in, self.flag)
            output = self.head_g(rep_g)

        if is_rep:
            return output, rep, self.feature_extractor(x)
        else:
            return output


class Gate(nn.Module):
    def __init__(self, cs) -> None:
        super().__init__()

        self.cs = cs
        self.pm = []
        self.gm = []
        self.pm_ = []
        self.gm_ = []

    def forward(self, rep, tau=1, hard=False, context=None, flag=0):
        pm, gm = self.cs(context, tau=tau, hard=hard)
        if self.training:
            self.pm.extend(pm)
            self.gm.extend(gm)
        else:
            self.pm_.extend(pm)
            self.gm_.extend(gm)

        if flag == 0:
            rep_p = rep * pm
            rep_g = rep * gm
            return rep_p, rep_g
        elif flag == 1:
            return rep * pm
        else:
            return rep * gm