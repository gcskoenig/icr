import torch
from torch.distributions import Distribution, Uniform

class TransformedUniform(torch.distributions.Distribution):

    def __init__(self, sigma, p_y_1, **kwargs):
        self.phi = torch.tensor(sigma)
        self.p_y_1 = torch.tensor(p_y_1)

        self.p_smaller = self.p_y_1 / self.phi
        self.p_larger = (1 - self.p_y_1) / (1 - self.phi)

        self.factor_smaller = torch.tensor(1.0) / self.p_smaller
        self.factor_larger = torch.tensor(1.0) / self.p_larger

        super().__init__(**kwargs)


    def rsample(self, sample_shape=torch.Size()):
        v = Uniform(0, 1).rsample(sample_shape)
        smpl = torch.min(v, self.p_y_1) * self.factor_smaller + torch.max(torch.tensor(0), v - self.p_y_1) * self.factor_larger
        return smpl

    def log_prob(self, value):
        res = (value <= self.phi) * self.p_smaller + (value > self.phi) * self.p_larger
        res = torch.log(res)
        return res