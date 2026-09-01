import torch
import torch.nn as nn


class VariateGrouping(nn.Module):
    """
    Compresses N variate tokens into G compact group tokens.

    Input:
        x: [B, N, D]

    Output:
        group_tokens: [B, G, D]
        assignment:   [B, N, G]

    The routing matrix is produced independently for each variate and
    therefore does not construct an N x N interaction matrix.
    """

    def __init__(self, d_model, n_groups, dropout=0.1):
        super(VariateGrouping, self).__init__()
        self.n_groups = n_groups

        self.router = nn.Linear(d_model, n_groups)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, D]
        assignment = torch.softmax(self.router(x), dim=-1)
        # assignment = self.dropout(assignment)

        # Normalize the routing weights so that each group represents
        # a weighted average rather than a sum whose scale depends on N.
        group_mass = assignment.sum(dim=1, keepdim=False).unsqueeze(-1)
        group_mass = group_mass.clamp_min(1e-6)

        group_tokens = torch.einsum('bng,bnd->bgd', assignment, x)
        group_tokens = group_tokens / group_mass

        return group_tokens, assignment


class VariateUngrouping(nn.Module):
    """
    Broadcasts compact group representations back to the original
    variate-token space.

    Input:
        group_tokens: [B, G, D]
        assignment:   [B, N, G]

    Output:
        context: [B, N, D]
    """

    def __init__(self):
        super(VariateUngrouping, self).__init__()

    def forward(self, group_tokens, assignment):
        return torch.einsum('bng,bgd->bnd', assignment, group_tokens)
