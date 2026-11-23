"""Causal CPC (El Bouchattaoui et al., NeurIPS 2024) adapted to 'when to defend'.

Differences from the paper (all deliberate, see notes/01, 02):
  * the history holds no outcome: encoder rows are [X, W, t, init] (Y is an oracle quantity here);
  * the decoder runs to the end of the run (not tau steps), with or without feeding back its own predictions
    (`feedback`; the start value is a learned constant, never a true Y);
  * twins (same seed) are never negatives of each other in the contrastive losses;
  * W is binary (Krum on / off), so the treatment head is one logit.

Decoder step j (target row cut+1+j) reads [W of the previous row, t of the target row (, previous prediction)]:
its hidden state therefore never contains the planned W of its own step, which only the outcome head sees. The
balancing representation br_j is thus a function of the history and the plan BEFORE step j, as in the paper, and
the treatment head predicting W_j from br_j is a valid CLUB critic.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm


def contrastive_mask(seed: torch.Tensor) -> torch.Tensor:
    """(B, B) True where column b must not serve as a negative for row a: another run with the same seed (twin)."""
    same = seed[:, None] == seed[None, :]
    return same & ~torch.eye(len(seed), dtype=torch.bool, device=seed.device)


def info_nce(targets: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """InfoNCE: for each row a, the true prediction preds[a] against the other runs' predictions."""
    logits = (targets @ preds.T).masked_fill(mask, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(targets), device=targets.device))


class CausalCPC(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, tau: int = 20, feedback: bool = False, n_out: int = 2):
        super().__init__()
        self.tau, self.feedback, self.n_out = tau, feedback, n_out
        z = hidden
        self.local = nn.Sequential(weight_norm(nn.Linear(in_dim, 3 * z // 2)), nn.SELU(),
                                   weight_norm(nn.Linear(3 * z // 2, z)))  # Z_t = phi1([X, W, t, init])
        self.ar = nn.GRU(z, hidden, batch_first=True)  # C_t
        self.wk = nn.ModuleList([nn.Linear(hidden, z, bias=False) for _ in range(tau)])  # CPC, horizons 1..tau
        self.infomax_map = nn.Linear(hidden, hidden, bias=False)  # C^h -> prediction of C^f
        self.dec = nn.GRU(2 + (n_out if feedback else 0), hidden, batch_first=True)
        self.br = nn.Sequential(weight_norm(nn.Linear(hidden, hidden)), nn.SELU())  # balancing representation
        self.out = nn.Sequential(weight_norm(nn.Linear(hidden + 1, hidden)), nn.SELU(),
                                 weight_norm(nn.Linear(hidden, n_out)))
        self.treat = nn.Sequential(spectral_norm(nn.Linear(hidden, hidden)), nn.SELU(),
                                   spectral_norm(nn.Linear(hidden, 1)))  # CLUB critic q(W_j | br_j)
        self.y0 = nn.Parameter(torch.zeros(n_out))  # learned start value of the fed-back outcome

    # ----------------------------------------------------------------- groups of parameters
    def encoder_params(self):
        return [*self.local.parameters(), *self.ar.parameters()]

    def pretrain_params(self):
        return [*self.encoder_params(), *self.wk.parameters(), *self.infomax_map.parameters()]

    def decoder_params(self):
        return [*self.dec.parameters(), *self.br.parameters(), *self.out.parameters(), self.y0]

    def treat_params(self):
        return list(self.treat.parameters())

    # ----------------------------------------------------------------- encoder
    def encode(self, enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.local(enc)
        c, _ = self.ar(z)
        return z, c  # (B, n, H) each; c[:, i] summarises rows 0..i

    def cpc_loss(self, z: torch.Tensor, c: torch.Tensor, mask: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Mean InfoNCE over the sampled times and horizons j = 1..tau (only where t + j is a row of the run)."""
        n, losses = z.shape[1], []
        for t in times.tolist():
            for j in range(1, self.tau + 1):
                if t + j > n - 1:
                    break
                losses.append(info_nce(z[:, t + j], self.wk[j - 1](c[:, t]), mask))
        return torch.stack(losses).mean()

    def infomax_loss(self, z: torch.Tensor, mask: torch.Tensor, t: int, t0: int) -> torch.Tensor:
        """InfoNCE between the past view rows [0, t0) and the future view rows [t0, t] (each encoded from scratch)."""
        ch = self.ar(z[:, :t0])[0][:, -1]
        cf = self.ar(z[:, t0:t + 1])[0][:, -1]
        return info_nce(cf, self.infomax_map(ch), mask)

    # ----------------------------------------------------------------- decoder
    def decode(self, h0: torch.Tensor, w_prev: torch.Tensor, w_now: torch.Tensor,
               t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h0 (N, H); w_prev, w_now, t (N, S). Returns outcome predictions (N, S, n_out) in [0, 1], br (N, S, H)."""
        hidden = h0.unsqueeze(0).contiguous()
        if not self.feedback:
            o, _ = self.dec(torch.stack([w_prev, t], -1), hidden)
            br = self.br(o)
            return torch.sigmoid(self.out(torch.cat([br, w_now[..., None]], -1))), br
        y = torch.sigmoid(self.y0).expand(len(h0), -1)
        ys, brs = [], []
        for s in range(w_now.shape[1]):  # autoregressive, no teacher forcing (the paper's decoder)
            x = torch.cat([w_prev[:, s, None], t[:, s, None], y], -1)[:, None]
            o, hidden = self.dec(x, hidden)
            b = self.br(o[:, 0])
            y = torch.sigmoid(self.out(torch.cat([b, w_now[:, s, None]], -1)))
            ys.append(y)
            brs.append(b)
        return torch.stack(ys, 1), torch.stack(brs, 1)

    def cut_predictions(self, enc: torch.Tensor, W: torch.Tensor, t_rows: torch.Tensor, cuts: torch.Tensor,
                        c: torch.Tensor | None = None):
        """Predictions from history rows 0..i for each cut i, for rows i+1 .. n-1, under the schedule W (B, n).

        Returns preds (B, C, S, n_out), br (B, C, S, H), valid (C, S), idx (C, S) with target row = idx + 1.
        Same convention as gru_experiment.cut_predictions."""
        B, n = W.shape
        n_dec = n - 1
        if c is None:
            c = self.encode(enc)[1]
        h = c[:, cuts]  # (B, C, H)
        S = n_dec - int(cuts.min())
        raw = cuts[:, None] + torch.arange(S, device=W.device)[None]
        valid, idx = raw < n_dec, raw.clamp(max=n_dec - 1)
        w_now, w_prev, tt = W[:, 1:][:, idx], W[:, :-1][:, idx], t_rows[1:][idx].expand(B, -1, -1)
        C = len(cuts)
        y, br = self.decode(h.reshape(B * C, -1), w_prev.reshape(B * C, S), w_now.reshape(B * C, S),
                            tt.reshape(B * C, S))
        return y.reshape(B, C, S, -1), br.reshape(B, C, S, -1), valid, idx
