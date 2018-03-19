import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable

cuda = torch.cuda.is_available()


class ElasticWeightConsolidation(object):
    """
    Elastic Weight Consolidation (EWC) class template
    based on Kirkpatrick et al. (2016).
    """
    def __init__(self, model):
        self.model = model

        thetas = [p for p in self.model.parameters() if p.requires_grad]

        # Parameters learned on task A
        self.theta_A = copy.deepcopy(thetas)
        self.fisher_matrix = None

    def calculate_fisher(self, data_loader):
        """
        Caculate the Fisher information matrix for
        the parameters of the domain spanned by
        data_loader.
        """
        self.fisher_matrix = copy.deepcopy(self.theta_A)

        self.model.eval()
        for x, _ in data_loader:
            self.model.zero_grad()

            x = Variable(x)
            if cuda:
                x = x.cuda()

            logits = self.model(x).view(1, -1)
            _, prediction = logits.max(1)

            loss = F.cross_entropy(logits, prediction.view(-1))
            loss.backward()

            thetas = [p for p in self.model.parameters() if p.requires_grad]

            for i, p in enumerate(thetas):
                self.fisher_matrix[i].data += p.grad.data**2 / len(data_loader.dataset)

        return self

    def penalty(self):
        """
        Calculate the penalty induced by straying too far
        from the original weights.
        """
        theta_B = [p for p in self.model.parameters() if p.requires_grad]

        sum = 0
        for i in range(len(theta_B)):
            fisher_distance = self.fisher_matrix[i] * (theta_B[i] - self.theta_A[i])**2
            sum += fisher_distance.sum()

        return sum