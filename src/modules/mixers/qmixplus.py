import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixerPlus(nn.Module):
    def __init__(self, args):
        super(QMixerPlus, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        self.state_network_hidden_dim = args.state_network_hidden_dim
        self.state_network_dim = args.state_network_dim
        self.state_network = nn.Sequential(
            nn.Linear(self.state_dim, self.state_network_hidden_dim),
            nn.BatchNorm1d(self.state_network_hidden_dim),
            nn.ELU(),
            nn.Linear(self.state_network_hidden_dim, self.state_network_dim)
        )

        self.qs_w = th.nn.Parameter(
            th.nn.init.kaiming_normal_(th.randn(self.state_network_dim, self.n_agents, requires_grad=True, device=args.device))
        )
        self.b_w = th.nn.Parameter(th.tensor(np.random.uniform(-1e-5, 1e-5, self.state_network_dim), requires_grad=True, dtype=th.float, device=args.device))

        self.w_1 = th.nn.Parameter(
            th.nn.init.kaiming_normal_(th.randn(self.embed_dim, self.state_network_dim, requires_grad=True, device=args.device))
        )
        self.w_final = th.nn.Parameter(
            th.nn.init.kaiming_normal_(th.randn(1, self.embed_dim, requires_grad=True, device=args.device))
        )

        self.b_1 = th.nn.Parameter(th.tensor(np.random.uniform(-1e-5, 1e-5, self.embed_dim), requires_grad=True, dtype=th.float, device=args.device))
        self.b_2 = th.nn.Parameter(th.tensor(np.random.uniform(-1e-5, 1e-5, 1), requires_grad=True, dtype=th.float, device=args.device))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        states_outputs = self.state_network(states) # (Batch size x timesteps, state_dim)
        states_outputs = states_outputs.view(-1, 1, self.state_network_dim)

        qs = F.elu(F.linear(agent_qs, th.abs(self.qs_w), self.b_w) ) # (..., n_agents, state dim)
        qs = qs.permute(0, 2, 1)
        attention = th.softmax(th.bmm(qs, states_outputs) / np.sqrt(self.state_network_dim), -1)
        qs_outputs = th.matmul(attention, qs).permute(0, 2, 1) # (..., 1, state_network_dim)

        # matmul weights must be 3D, stack this bs times
        hidden = F.linear(qs_outputs, th.abs(self.w_1), self.b_1)
        # (Batch size x timesteps, 1, embed_dim)
        bnorm = nn.BatchNorm1d(1)
        if self.args.device == 'cuda':
            bnorm = bnorm.cuda()

        hidden = F.elu(bnorm(hidden))
        # Second layer
        # Compute final output
        y = F.linear(hidden, th.abs(self.w_final), self.b_2)
        # Reshape and return
        return y.view(bs, -1, 1)