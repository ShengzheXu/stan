import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, OneHotCategorical
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SingleTaskNet(nn.Module):
    def __init__(self, dim_in, dim_out, dim_window=1, mask_mode=None, encoder_arch=None, decoder_arch=None, model_tag=None):
        super().__init__()
        # data dimension
        self.model_tag = model_tag
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.window_cnn_network = None
        self.gmm_network = None
        self.dec_network = None
        self.memory = []

        # mask
        self.mask_mode = mask_mode
        self.mask = None

        if mask_mode is None:
            self.mask = torch.ones(((dim_window+1), dim_in))
            self.mask[dim_window, :] = 0
        else:
            self.mask = torch.ones(((dim_window+1), dim_in))
            self.mask[dim_window, mask_mode:] = 0

        curr_in = dim_in * dim_window if mask_mode is None else dim_in * (dim_window+1)
        # curr_in = dim_in * (dim_window+1)
        if encoder_arch is not None:
            self.window_cnn_network = self.make_layers(encoder_arch)
            curr_in = 512
        assert decoder_arch[0] in ['gmm', 'softmax'], "Unknown Decoder Type"
        self.decoder_type = decoder_arch[0]
        if self.decoder_type == 'gmm':
            self.gmm_network = MixtureDensityNetwork(curr_in, dim_out, n_components=decoder_arch[1])
        else: #softmax
            self.dec_network = self.make_decs(curr_in, dim_out, hidden_dim=decoder_arch[1], is_onehot=True)

    def forward(self, x):
        # print('forward x', x.shape)
        # print('example x', x[0])
        out = self.make_mask(x)

        if self.window_cnn_network is not None:
            out = torch.unsqueeze(out, 1)
            # print(out.shape)
            out = self.window_cnn_network(out)
        out = out.view(out.size()[0], -1)
        if self.gmm_network is not None:
            pi, normal = self.gmm_network(out)
            return pi, normal
        else:
            out = self.dec_network(out)
            # print('dec_out', out)
            return out
    
    def loss(self, x, y, y_reshape_func=None, bin_type=False):
        if self.decoder_type == 'gmm':
            pi, normal = self.forward(x)
            if bin_type:
                batch_loss = self.gmm_network.bin_loss(pi, normal, y)
            else:
                batch_loss = self.gmm_network.loss(pi, normal, y)
        else:
            out = self.forward(x)
            # print('model_tag',self.model_tag, y.shape, 'y_type', type(y), y)
            # input()
            if y.size()[1] > 1:
                y = torch.max(y, 1)[1].long()
            else:
                if y_reshape_func is not None:
                    y = y_reshape_func(y)
                y = y.long().squeeze(1)
            # print('out', out)
            # print('fixed', y)
            try:
                batch_loss = torch.nn.CrossEntropyLoss()(out, y)
            except:
                print('y', y)
                print('out', out)
        self.memory.append(batch_loss.cpu().detach().numpy())
        # print('batch_loss', batch_loss.detach().numpy())
        return batch_loss
    
    def batch_reset(self):
        self.memory = []
    
    def get_batch_loss(self):
        # print(self.memory)
        # print(np.mean(self.memory[0]))
        # input()
        return np.mean(self.memory[0])
        
    def sample(self, x):
        x = torch.unsqueeze(x, 0) 
        if self.decoder_type == 'gmm':
            pi, normal = self.forward(x)
            return self.gmm_network.sample(pi, normal)
        else:
            out = self.forward(x)
            return self.softmax_sample(out)

    def make_mask(self, x):
        if self.mask_mode is None:
            # print('masking', x, self.dim_in)
            x = x[:, :-self.dim_in]
            # x = x * self.mask
        else:
            # print(x.shape)
            # print('mask', self.mask.shape)
            x = x * self.mask.to(device)
        return x

    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 1
        print(cfg)
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def make_decs(self, input_dim, output_dim, hidden_dim=100, is_onehot=True):
        if is_onehot:
            dec_layer = nn.Sequential(
                #nn.Linear(input_num, hidden_num),
                #nn.ReLU(inplace=True),
                #nn.Linear(hidden_num, output_num),
                nn.Linear(input_dim, output_dim),
                # nn.Softmax()
            )
        else:
            dec_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(inplace=True)
            )
        return dec_layer
    
    def softmax_sample(self, out):
        probs = F.softmax(out, dim=1)
        dist = torch.distributions.Categorical(probs)
        sample = dist.sample().data.tolist()[0]
        # print(out, probs, dist, sample)
        # input()
        return sample, dist


class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components):
        super().__init__()
        self.pi_network = CategoricalNetwork(dim_in, n_components)
        self.normal_network = MixtureDiagNormalNetwork(dim_in, dim_out,
                                                       n_components)

    def forward(self, x):
        return self.pi_network(x), self.normal_network(x)

    def manual_logsumexp(self, x, dim=1):
        return torch.log(torch.sum(torch.exp(x), dim=dim)+1e-10)

    def cdf_func(self, sigma, mu, value):
        return 0.5 * (1 + torch.erf((value - mu) * sigma.reciprocal() / math.sqrt(2)))

    def bin_loss(self, pi, normal, y):
        binwidth = 1.0/200 /2.0
        y = y.unsqueeze(1).expand_as(normal.loc)
        # loglik = self.cdf_func(normal.scale, normal.loc, y+binwidth) - self.cdf_func(normal.scale, normal.loc, y-binwidth)
        loglik = normal.cdf(y + binwidth) - normal.cdf(y - binwidth)
        loglik = torch.prod(loglik, dim=2)
        loss = -torch.log(torch.sum(pi.probs * loglik, dim=1)+1e-10)

        return torch.mean(loss)

    def loss(self, pi, normal, y):
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        # loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        loss = -self.manual_logsumexp(torch.log(pi.probs) + loglik, dim=1)
        return torch.mean(loss)

    def sample(self, pi, normal):
        # pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples, normal


class MixtureDiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, n_components, hidden_dim=None):
        super().__init__()
        self.n_components = n_components
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * out_dim * n_components),
        )

    def forward(self, x):
        params = self.network(x)
        # print(params)
        # input()
        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
        return Normal(mean.transpose(0, 1), torch.exp(sd).transpose(0, 1))

class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)
