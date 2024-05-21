import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = [
    'linear', 'mlp', 'LeNet'
]

class linear(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(linear, self).__init__()

        self.L1 = nn.Linear(n_inputs, n_outputs)
        init.xavier_uniform_(self.L1.weight)
        self.n_inputs = n_inputs

    def forward(self, x):
        x = x.view(-1, self.n_inputs)
        x = self.L1(x)
        return x


class mlp_(nn.Module):
    def __init__(self, n_inputs, n_outputs, parameter_momentum=0.1):
        super(mlp, self).__init__()

        self.L1 = nn.Linear(n_inputs, 300, bias=False)
        init.xavier_uniform_(self.L1.weight)
        self.bn1 = nn.BatchNorm1d(300, momentum=parameter_momentum)
        init.ones_(self.bn1.weight)

        self.L2 = nn.Linear(300, 301, bias=False)
        init.xavier_uniform_(self.L2.weight)
        self.bn2 = nn.BatchNorm1d(301, momentum=parameter_momentum)
        init.ones_(self.bn2.weight)

        self.L3 = nn.Linear(301, 302, bias=False)
        init.xavier_uniform_(self.L3.weight)
        self.bn3 = nn.BatchNorm1d(302, momentum=parameter_momentum)
        init.ones_(self.bn3.weight)

        self.L4 = nn.Linear(302, 303, bias=False)
        init.xavier_uniform_(self.L4.weight)
        self.bn4 = nn.BatchNorm1d(303, momentum=parameter_momentum)
        init.ones_(self.bn4.weight)

        self.L5 = nn.Linear(303, n_outputs, bias=True)
        init.xavier_uniform_(self.L5.weight)
        init.zeros_(self.L5.bias)
        
        self.n_inputs = n_inputs

    def forward(self, x):

        hiddens = []
        x = x.view(-1, self.n_inputs)

        x = self.L1(x)
        x = self.bn1(x)
        x = F.relu(x)
        hiddens.append(x)


        x = self.L2(x)
        x = self.bn2(x)
        x = F.relu(x)
        hiddens.append(x)

        x = self.L3(x)
        x = self.bn3(x)
        x = F.relu(x)
        hiddens.append(x)

        x = self.L4(x)
        x = self.bn4(x)
        x = F.relu(x)
        hiddens.append(x)

        x = self.L5(x)
        hiddens.append(torch.nn.functional.softmax(x, dim=1))

        return x, hiddens


class mlp(nn.Module):
    def __init__(self, n_inputs, n_outputs, parameter_momentum=0.1):
        super(mlp, self).__init__()

        self.L1 = nn.Linear(n_inputs, 1024, bias=False)
        init.xavier_uniform_(self.L1.weight)
        self.bn1 = nn.BatchNorm1d(1024, momentum=parameter_momentum)
        init.ones_(self.bn1.weight)

        self.L2 = nn.Linear(1024, 20, bias=False)
        init.xavier_uniform_(self.L2.weight)
        self.bn2 = nn.BatchNorm1d(20, momentum=parameter_momentum)
        init.ones_(self.bn2.weight)

        self.L3 = nn.Linear(20, 20, bias=False)
        init.xavier_uniform_(self.L3.weight)
        self.bn3 = nn.BatchNorm1d(20, momentum=parameter_momentum)
        init.ones_(self.bn3.weight)

        self.L4 = nn.Linear(20, 20, bias=False)
        init.xavier_uniform_(self.L4.weight)
        self.bn4 = nn.BatchNorm1d(20, momentum=parameter_momentum)
        init.ones_(self.bn4.weight)

        self.L5 = nn.Linear(20, n_outputs, bias=True)
        init.xavier_uniform_(self.L5.weight)
        init.zeros_(self.L5.bias)
        
        self.n_inputs = n_inputs

    def forward(self, x):

        hiddens = []
        x = x.view(-1, self.n_inputs)

        x = self.L1(x)
        x = self.bn1(x)
        x = F.relu(x)
        hiddens.append(x)


        x = self.L2(x)
        x = self.bn2(x)
        x = F.relu(x)
        hiddens.append(x)

        x = self.L3(x)
        x = self.bn3(x)
        x = F.relu(x)
        hiddens.append(x)

        x = self.L4(x)
        x = self.bn4(x)
        x = F.relu(x)
        hiddens.append(x)

        x = self.L5(x)
        hiddens.append(torch.nn.functional.softmax(x, dim=1))

        return x, hiddens

class LeNet(nn.Module):

    def __init__(self, input_dim, out_dim, in_channel=1, img_sz=28):
        super(LeNet, self).__init__()
        feat_map_sz = img_sz//4
        self.n_feat = 50 * feat_map_sz * feat_map_sz

        # !!! [Architecture design tip] !!!
        # The KCL has much better convergence of optimization when the BN layers are added.
        # MCL is robust even without BN layer.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 20, 5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(self.n_feat, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(500, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, self.n_feat))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x