"""
Module: multi_task_flexible
---------------------------

Tensor shape legend (used across all docstrings):

* **B** – batch size
* **L** – sequence length (e.g. number of temperature points)
* **D** – latent / embedding feature dimension
* **C** – channel dimension for 1‑D convolutions

Throughout the code shapes are written as `(L, B, D)` or `(B, L, D)` to indicate
exact tensor layouts while keeping the same letter meanings.
"""

import math
from typing import Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .layers import LinearBlock  # same helper used by the old model


# --------------------------------------------------------------------------- #
#                    helpers for formula‑structure fusion                     #
# --------------------------------------------------------------------------- #
class StructureEncoder(nn.Module):
    """
    Simple MLP encoder for structure descriptors (vector form).

    Parameters
    ----------
    d_in : int
        Input dimension of structure descriptor vector.
    hidden_dims : list[int]
        Widths of hidden layers (last element should equal formula latent dim).
    norm, residual : bool
        Same switches as LinearBlock.
    """

    def __init__(
        self,
        d_in: int,
        hidden_dims: List[int],
        norm: bool = True,
        residual: bool = False,
    ):
        super().__init__()
        self.net = LinearBlock(
            [d_in] + hidden_dims,
            normalization=norm,
            residual=residual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LoRAAdapter(nn.Module):
    """
    LoRA:  ΔW = α / r  ·  U @ V
    * Wraps an existing nn.Linear; base weight kept frozen when freeze=True.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: float = 1.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base = base_linear
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)
        in_dim, out_dim = base_linear.in_features, base_linear.out_features
        self.U = nn.Parameter(torch.randn(in_dim, r) * 0.01)
        self.V = nn.Parameter(torch.randn(r, out_dim) * 0.01)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.U @ self.V)


class GatedFusion(nn.Module):
    """
    h = h_formula + sigmoid(W [h_f, h_s]) * h_structure
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, 1)

    def forward(
        self, h_f: torch.Tensor, h_s: torch.Tensor, has_s: torch.Tensor
    ) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([h_f, h_s], dim=-1))) * has_s
        return h_f + g * h_s


# --------------------------------------------------------------------------- #
#                               sequence heads                                #
# --------------------------------------------------------------------------- #
class SequenceHeadRNN(nn.Module):
    """
    GRU/LSTM sequence head with FiLM conditioning.

    Parameters
    ----------
    d_in : int
        Dimension of the latent vector from the encoder.
    hidden : int, optional
        Hidden size of the recurrent layer (default: 128).
    cell : {"gru","lstm"}, optional
        Select GRU or LSTM cell (default: "gru").
    """

    def __init__(self, d_in: int, hidden: int = 128, cell: str = "gru"):
        super().__init__()
        rnn_cls = nn.GRU if cell.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=1, hidden_size=hidden, num_layers=2, batch_first=True
        )
        self.film = nn.Linear(d_in, 2 * hidden)  # γ & β
        self.out = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor
            Latent tensor of shape (B, D).
        temps : torch.Tensor
            Temperature points, shape (B, L, 1).

        Returns
        -------
        torch.Tensor
            Predicted sequence, shape (B, L).
        """
        # h: (B,D) / temps: (B,L,1)  -> y:(B,L)
        gamma_beta = self.film(h).unsqueeze(1)  # (B,1,2H)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        rnn_out, _ = self.rnn(temps)  # (B,L,H)
        fused = gamma * rnn_out + beta
        return self.out(fused).squeeze(-1)


class SequenceHeadFixedVec(nn.Module):
    """
    Simple multi‑output MLP that predicts a *fixed‑length* vector.

    Parameters
    ----------
    d_in : int
        Latent dimension.
    seq_len : int
        Length of the output sequence (each element is a scalar).
    """

    def __init__(self, d_in: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, seq_len),
        )

    def forward(
        self, h: torch.Tensor, temps: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor
            Latent tensor (B, D).
        temps : torch.Tensor | None
            Ignored; kept for interface compatibility.

        Returns
        -------
        torch.Tensor
            Fixed‑length output (B, seq_len).
        """
        return self.net(h)  # (B,L)


class _PosEnc(nn.Module):
    """
    Classic sinusoidal positional encoding (length, dim) → additive tensor.
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens (L, B, D).

        Returns
        -------
        torch.Tensor
            Same shape as x with positional information added.
        """
        return x + self.pe[: x.size(0)]


# --------------------------------------------------------------------------- #
#                      minimal Flash‑Attention MHA block                      #
# --------------------------------------------------------------------------- #
class _FlashMHABlock(nn.Module):
    """
    Transformer block that uses PyTorch‑native Flash Attention (SDPA).

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    nhead : int
        Number of attention heads.
    ff_mult : int, optional
        Expansion factor for the feed‑forward layer (default: 4).
    causal : bool, optional
        If True, apply causal mask for autoregressive decoding.
    """

    def __init__(
        self, d_model: int, nhead: int, ff_mult: int = 4, causal: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.causal = causal

    def _mha(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L,B,D)
        L, B, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)  # each (L,B,D)
        # reshape to (B, nH, L, Dh)
        q, k, v = [
            t.reshape(L, B, self.nhead, D // self.nhead).permute(1, 2, 0, 3)
            for t in qkv
        ]
        # Flash kernel on GPU, math path on CPU – automatic
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal, dropout_p=0.0
        )  # (B,nH,L,Dh)
        out = out.permute(2, 0, 1, 3).contiguous().reshape(L, B, D)  # (L,B,D)
        return self.proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer sub‑block forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (L, B, D).

        Returns
        -------
        torch.Tensor
            Output tensor of same shape.
        """
        # Multi‑head flash attention
        x = x + self._mha(self.ln1(x))
        # Feed‑forward
        x = x + self.ff(self.ln2(x))
        return x


class SequenceHeadTransformer(nn.Module):
    """
    Transformer‑based sequence head with Flash‑Attention.

    Parameters
    ----------
    d_in : int
        Dimension of the latent vector coming from the foundation encoder.
    d_model : int, optional
        Model width used inside the Transformer (default: 256).
    nhead : int, optional
        Number of attention heads (default: 4).
    num_layers : int, optional
        How many `_FlashMHABlock` layers to stack (default: 4).
    autoregressive : bool, optional
        If ``True`` uses *causal* self‑attention (auto‑regressive decoding);
        if ``False`` the attention is bidirectional (BERT‑style).  The same
        forward method is kept – just the mask differs.

    Notes
    -----
    * Uses :pyfunc:`torch.nn.functional.scaled_dot_product_attention` with
      ``is_causal=autoregressive`` – GPU devices that support Flash‑Attention
      kernels will accelerate automatically, CPU gracefully falls back.
    * Output shape is ``(B, L)`` – identical to all other sequence heads so the
      loss pipeline stays unchanged.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        autoregressive: bool = False,
    ):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = _PosEnc(d_model)
        self.layers = nn.ModuleList(
            [
                _FlashMHABlock(d_model, nhead, causal=autoregressive)
                for _ in range(num_layers)
            ]
        )
        self.out = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:
        B, L, _ = temps.shape
        cls = self.proj(h).unsqueeze(0)  # (1,B,D)
        t_emb = self.proj(temps.squeeze(-1)).permute(1, 0, 2)  # (L,B,D)
        tok = torch.cat([cls, t_emb], dim=0)  # (1+L,B,D)
        tok = self.pos(tok)
        for blk in self.layers:
            tok = blk(tok)
        seq_feat = tok[1:]  # strip CLS
        y = self.out(seq_feat).permute(1, 0, 2).squeeze(-1)  # (B,L)
        return y


# --------------------------------------------------------------------------- #
#                 advanced heads: Dilated TCN + FiLM / Hybrid                #
# --------------------------------------------------------------------------- #
class _DilatedTCN(nn.Module):
    """Simple 1‑D Dilated TCN block stack with residual connections."""

    def __init__(self, channels: int, n_layers: int = 4, kernel_size: int = 3):
        super().__init__()
        dilations = [2**i for i in range(n_layers)]
        pads = [((kernel_size - 1) * d) // 2 for d in dilations]
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d,
                    padding=p,
                )
                for d, p in zip(dilations, pads)
            ]
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,L)
        out = x
        for conv in self.layers:
            res = out
            out = self.act(conv(out))
            out = out + res
        return out


class SequenceHeadTCNFiLM(nn.Module):
    """
    Dilated‑TCN + FiLM modulation.

    * temps → linear → (B,hidden,L)  → Dilated‑TCN
    * FiLM: γ,β from latent h  (B,1,hidden)
    * output linear → (B,L)
    """

    def __init__(self, d_in: int, hidden: int = 128, n_layers: int = 4):
        super().__init__()
        self.temp_proj = nn.Linear(1, hidden)
        self.tcn = _DilatedTCN(hidden, n_layers=n_layers)
        self.film = nn.Linear(d_in, 2 * hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:
        # temps:(B,L,1)
        B, L, _ = temps.shape
        x = self.temp_proj(temps).permute(0, 2, 1)  # (B,hidden,L)
        x = self.tcn(x).permute(0, 2, 1)  # (B,L,hidden)
        gamma, beta = self.film(h).unsqueeze(1).chunk(2, dim=-1)  # (B,1,H)
        fused = gamma * x + beta
        y = self.out(fused).squeeze(-1)  # (B,L)
        return y


class SequenceHeadHybrid(nn.Module):
    """
    Hybrid: Dilated‑TCN (local) → Transformer (global).

    hidden channels == d_model for simplicity.
    """

    def __init__(
        self,
        d_in: int,
        hidden: int = 128,
        n_tcn_layers: int = 4,
        num_layers: int = 2,
        nhead: int = 4,
    ):
        super().__init__()
        self.tcn_film = SequenceHeadTCNFiLM(d_in, hidden=hidden, n_layers=n_tcn_layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:
        # Use the TCN‑FiLM to get local features first
        feats = self.tcn_film.temp_proj(temps)  # reuse linear
        B, L, C = feats.shape
        # run Dilated TCN (private)
        feats = self.tcn_film.tcn(feats.permute(0, 2, 1)).permute(0, 2, 1)
        gamma, beta = self.tcn_film.film(h).unsqueeze(1).chunk(2, dim=-1)
        feats = gamma * feats + beta  # FiLM
        feats = self.encoder(feats)  # (B,L,C)
        y = self.out(feats).squeeze(-1)
        return y


def build_sequence_head(
    mode: str,
    d_latent: int,
    seq_len: int | None = None,
    hidden: int = 128,
    cell: str = "gru",
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    n_tcn_layers: int = 4,
    autoregressive: bool = False,
) -> nn.Module:
    """
    Factory helper that instantiates the proper sequence head.

    Parameters
    ----------
    mode : {"none","rnn","vec","transformer","tcn","hybrid"}
        Which architecture to build.
    d_latent : int
        Dimension of the latent feature vector.
    seq_len : int, optional
        Required when ``mode="vec"`` – sets fixed output length.
    hidden, cell
        Width / cell type for ``mode="rnn"``.
    d_model, nhead, num_layers, autoregressive
        Hyper‑parameters for ``mode="transformer"``.
    n_tcn_layers
        #layers for TCN variants.
    ...
    """
    mode = mode.lower()
    if mode == "rnn":
        return SequenceHeadRNN(d_latent, hidden=hidden, cell=cell)
    if mode == "vec":
        if seq_len is None:
            raise ValueError("`seq_len` must be provided when mode='vec'.")
        return SequenceHeadFixedVec(d_latent, seq_len)
    if mode == "transformer":
        return SequenceHeadTransformer(
            d_latent,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            autoregressive=autoregressive,
        )
    if mode == "tcn":
        return SequenceHeadTCNFiLM(d_latent, hidden=hidden, n_layers=n_tcn_layers)
    if mode == "hybrid":
        return SequenceHeadHybrid(
            d_latent,
            hidden=d_model,
            n_tcn_layers=n_tcn_layers,
            num_layers=num_layers,
            nhead=nhead,
        )
    raise ValueError(f"Unknown sequence head mode: {mode}")


# --------------------------------------------------------------------------- #
#                             flexible main module                            #
# --------------------------------------------------------------------------- #
class FlexibleMultiTaskModel(L.LightningModule):
    """
    Drop‑in replacement for the original **MultiTaskAttributePredictor** with
    extra sequence‑prediction options.

    Parameters
    ----------
    shared_block_dims : list[int]
        Widths of shared MLP layers (foundation encoder output → deposit).
    task_block_dims : list[int]
        Widths of per‑attribute MLP head before the final scalar output.
    n_attr_tasks : int
        How many scalar regression attributes.
    norm_shared, residual_shared
        Toggles for LayerNorm / residual in the shared block.
    norm_tasks, residual_tasks
        Same toggles for every attribute head.
    shared_block_lr, task_block_lr, seq_head_lr
        Learning rates for each parameter group when *manual optimisation* is
        active.
    sequence_mode : {"none","rnn","vec","transformer","tcn","hybrid"}
        Choose which sequence head architecture – all disabled when "none".
    seq_len, rnn_hidden, cell, d_model, nhead, num_layers
        Assorted hyper‑parameters forwarded to the chosen sequence head.
    """

    def __init__(
        self,
        shared_block_dims: List[int],
        task_block_dims: List[int],
        n_attr_tasks: int,
        *,
        # residual / norm switches
        norm_shared: bool = True,
        residual_shared: bool = False,
        norm_tasks: bool = True,
        residual_tasks: bool = False,
        # optimiser hparams
        shared_block_lr: float = 5e-3,
        task_block_lr: float = 5e-3,
        seq_head_lr: float = 5e-3,
        # sequence head options
        sequence_mode: str = "none",  # 'none'|'rnn'|'vec'|'transformer'
        seq_len: Optional[int] = None,
        rnn_hidden: int = 128,
        cell: str = "gru",
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        # structure fusion options
        with_structure: bool = False,
        struct_block_dims: Optional[List[int]] = None,
        modality_dropout_p: float = 0.3,
        # --- pre‑training options ---
        pretrain: bool = False,
        loss_weights: Optional[Dict[str, float]] = None,
        mask_ratio: float = 0.15,
        temperature: float = 0.07,
        # --- LoRA options ---
        freeze_encoder: bool = False,
        lora_rank: int = 0,  # 0 = off
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        # ---------- pre‑train flags ----------
        self.pretrain = pretrain
        self.tau = temperature
        self.mask_ratio = mask_ratio

        # LoRA flags
        self.freeze_encoder = freeze_encoder
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # loss‑weight dict（keys: con, cross, mask, attr, seq）
        self.w = {"con": 1, "cross": 1, "mask": 1, "attr": 1, "seq": 1}
        if loss_weights:
            self.w.update(loss_weights)

        # ---------------- shared + attribute heads ---------------- #
        self.shared = LinearBlock(
            shared_block_dims, normalization=norm_shared, residual=residual_shared
        )
        self.deposit = nn.Sequential(
            nn.Linear(shared_block_dims[-1], task_block_dims[0]),
            nn.Tanh(),
        )
        # attribute regression heads: reuse LinearBlock with built‑in output layer
        self.attr_heads = nn.ModuleList()
        for _ in range(n_attr_tasks):
            blk = LinearBlock(
                task_block_dims[:-1],
                normalization=norm_tasks,
                residual=residual_tasks,
                dim_output_layer=task_block_dims[-1],  # adds final LinearLayer
            )
            # last LinearLayer is blk[-1] in Sequential
            if self.lora_rank > 0:
                blk[-1] = LoRAAdapter(
                    blk[-1], r=self.lora_rank, alpha=self.lora_alpha, freeze_base=True
                )
            self.attr_heads.append(blk)

        # -------------------- optional sequence head --------------- #
        d_latent = shared_block_dims[-1]  # before deposit
        self.sequence_mode = sequence_mode.lower()
        if self.sequence_mode != "none":
            self.seq_head = build_sequence_head(
                self.sequence_mode,
                d_latent=d_latent,
                seq_len=seq_len,
                hidden=rnn_hidden,
                cell=cell,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
            )
        else:
            self.seq_head = None

        # -------------- optional structure encoder & fusion -------------- #
        self.with_structure = with_structure
        self.mod_dropout_p = modality_dropout_p
        if self.with_structure:
            if not struct_block_dims or struct_block_dims[-1] != shared_block_dims[-1]:
                raise ValueError(
                    "struct_block_dims must be provided and last dim equal to formula latent dim"
                )
            self.struct_enc = StructureEncoder(
                struct_block_dims[0],
                struct_block_dims[1:],
                norm=norm_shared,
                residual=residual_shared,
            )
            self.fusion = GatedFusion(shared_block_dims[-1])

        # decoders for cross‑reconstruction / MFM
        if self.pretrain:
            latent_dim = shared_block_dims[-1]
            self.dec_formula = nn.Linear(latent_dim, shared_block_dims[0], bias=False)
            if self.with_structure:
                self.dec_struct = nn.Linear(
                    latent_dim, struct_block_dims[0], bias=False
                )

        # manual optimisation
        self.automatic_optimization = False
        self._init_weights()

        # lrs
        self._lr_shared = shared_block_lr
        self._lr_attrs = task_block_lr
        self._lr_seq = seq_head_lr

    # ------------------------------------------------------------------ #
    #                         internal utilities                         #
    # ------------------------------------------------------------------ #
    def _init_weights(self):
        # -------- freeze encoder if asked --------
        if self.freeze_encoder:
            for p in self.shared.parameters():
                p.requires_grad_(False)
            if self.with_structure:
                for p in self.struct_enc.parameters():
                    p.requires_grad_(False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _masked_mse(
        pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        losses = F.mse_loss(pred, tgt, reduction="none") * mask
        per_attr = torch.nan_to_num(
            losses.sum(0) / mask.sum(0), nan=0.0, posinf=0.0, neginf=0.0
        )
        return losses.sum() / mask.sum(), per_attr

    # ---------------- pre‑train helpers ---------------- #
    def _mask_feat(self, x: torch.Tensor):
        mask = torch.rand_like(x) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask

    def _contrastive(self, h_f: torch.Tensor, h_s: torch.Tensor):
        h_f = F.normalize(h_f, dim=-1)
        h_s = F.normalize(h_s, dim=-1)
        logits = (h_f @ h_s.T) / self.tau
        tgt = torch.arange(h_f.size(0), device=h_f.device)
        return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))

    # ------------------------------------------------------------------ #
    #                            forward pass                            #
    # ------------------------------------------------------------------ #
    def _encode(
        self, x_formula: torch.Tensor, x_struct: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h_latent : torch.Tensor  (B, D_latent)
        h_task   : torch.Tensor  (B, first_task_dim)
        """
        h_f = self.shared(x_formula)
        if self.with_structure:
            if x_struct is None:
                h_s = torch.zeros_like(h_f)
                has_s = torch.zeros(h_f.size(0), 1, device=h_f.device)
            else:
                h_s = self.struct_enc(x_struct)
                has_s = torch.ones(h_f.size(0), 1, device=h_f.device)
            # gated fusion
            h_latent = self.fusion(h_f, h_s, has_s)
        else:
            h_latent = h_f
        h_task = self.deposit(h_latent)
        return h_latent, h_task

    def forward(
        self,
        x: torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]],
        temps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.with_structure and isinstance(x, (list, tuple)):
            x_formula, x_struct = x
        else:
            x_formula = x
            x_struct = None
        h_latent, h_task = self._encode(x_formula, x_struct)
        attr_preds = torch.cat(
            [head(h_task) for head in self.attr_heads], dim=1
        )  # (B, n_attr)
        out = {"attr": attr_preds}
        if self.seq_head is not None and temps is not None:
            out["seq"] = self.seq_head(h_latent, temps)  # (B,L)
        return out

    # ------------------------------------------------------------------ #
    #                         lightning workflow                         #
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        x, y_attr, mask_attr, temps, y_seq, mask_seq = (
            batch  # seq parts may be None if no seq head
        )
        # unpack formula / structure
        if self.with_structure and isinstance(x, (list, tuple)):
            x_formula, x_struct = x
        else:
            x_formula, x_struct = x, None

        # raw embeddings (before fusion) — needed for contrastive
        h_f = self.shared(x_formula)
        h_s = (
            self.struct_enc(x_struct)
            if (self.with_structure and x_struct is not None)
            else None
        )

        # modality dropout during pre‑train
        if (
            self.pretrain
            and h_s is not None
            and torch.rand(1).item() < self.mod_dropout_p
        ):
            h_s, x_struct = None, None

        # fused path + downstream heads
        h_latent, h_task = self._encode(x_formula, x_struct)
        attr_pred = torch.cat([head(h_task) for head in self.attr_heads], dim=1)
        preds = {"attr": attr_pred}
        if self.seq_head is not None and temps is not None:
            preds["seq"] = self.seq_head(h_latent, temps)

        preds = self((x_formula, x_struct) if self.with_structure else x_formula, temps)
        attr_pred = preds["attr"]
        attr_loss, per_attr = self._masked_mse(attr_pred, y_attr, mask_attr)

        total_loss = self.w["attr"] * attr_loss
        logs = {"train_attr_loss": attr_loss}
        # ---------- extra pre‑train losses ----------
        if self.pretrain:
            # 1) Contrastive
            if h_s is not None and self.w["con"] > 0:
                l_con = self._contrastive(h_f, h_s)
                total_loss += self.w["con"] * l_con
                logs["train_con"] = l_con

            # 2) Cross‑reconstruction
            if h_s is not None and self.w["cross"] > 0:
                l_cross = 0.5 * (
                    F.mse_loss(self.dec_struct(h_f), x_struct)
                    + F.mse_loss(self.dec_formula(h_s), x_formula)
                )
                total_loss += self.w["cross"] * l_cross
                logs["train_cross"] = l_cross

            # 3) Masked‑feature modelling
            if self.w["mask"] > 0:
                xf_mask, mf = self._mask_feat(x_formula)
                l_mask = F.mse_loss(
                    self.dec_formula(self.shared(xf_mask))[mf], x_formula[mf]
                )
                if x_struct is not None:
                    xs_mask, ms = self._mask_feat(x_struct)
                    l_mask_s = F.mse_loss(
                        self.dec_struct(self.struct_enc(xs_mask))[ms], x_struct[ms]
                    )
                    l_mask = 0.5 * (l_mask + l_mask_s)
                total_loss += self.w["mask"] * l_mask
                logs["train_mask"] = l_mask

        # sequence loss
        if "seq" in preds:
            seq_loss = F.mse_loss(preds["seq"] * mask_seq, y_seq * mask_seq)
            seq_loss = seq_loss / mask_seq.sum().clamp_min(1.0)
            logs["train_seq_loss"] = seq_loss
            total_loss = total_loss + seq_loss

        # ---- manual optimisation ---- #
        opt_shared, opt_attr, *rest = self.optimizers()
        opt_seq = rest[0] if len(rest) > 0 else None
        opt_shared.zero_grad()
        opt_attr.zero_grad()
        if opt_seq is not None:
            opt_seq.zero_grad()
        self.manual_backward(total_loss)
        opt_shared.step()
        opt_attr.step()
        if opt_seq is not None:
            opt_seq.step()

        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=False)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y_attr, mask_attr, temps, y_seq, mask_seq = batch
        # unpack formula / structure
        if self.with_structure and isinstance(x, (list, tuple)):
            x_formula, x_struct = x
        else:
            x_formula, x_struct = x, None

        # raw embeddings (before fusion) — needed for contrastive
        h_f = self.shared(x_formula)
        h_s = (
            self.struct_enc(x_struct)
            if (self.with_structure and x_struct is not None)
            else None
        )

        # modality dropout during pre‑train
        if (
            self.pretrain
            and h_s is not None
            and torch.rand(1).item() < self.mod_dropout_p
        ):
            h_s, x_struct = None, None

        # fused path + downstream heads
        h_latent, h_task = self._encode(x_formula, x_struct)
        attr_pred = torch.cat([head(h_task) for head in self.attr_heads], dim=1)
        preds = {"attr": attr_pred}
        if self.seq_head is not None and temps is not None:
            preds["seq"] = self.seq_head(h_latent, temps)
        preds = self((x_formula, x_struct) if self.with_structure else x_formula, temps)
        attr_loss, _ = self._masked_mse(preds["attr"], y_attr, mask_attr)
        logs = {"val_attr_loss": attr_loss}
        total_loss = attr_loss
        if "seq" in preds:
            seq_loss = F.mse_loss(preds["seq"] * mask_seq, y_seq * mask_seq)
            seq_loss = seq_loss / mask_seq.sum().clamp_min(1.0)
            logs["val_seq_loss"] = seq_loss
            total_loss += seq_loss
        self.log_dict(logs, prog_bar=True, on_epoch=True)
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _, _, temps, _, _ = batch
        # unpack formula / structure
        if self.with_structure and isinstance(x, (list, tuple)):
            x_formula, x_struct = x
        else:
            x_formula, x_struct = x, None
        return self((x_formula, x_struct) if self.with_structure else x_formula, temps)

    # ------------------------------------------------------------------ #
    #                          optimisers                                #
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        def _trainable(p):
            return p.requires_grad

        opt_shared = optim.Adam(
            filter(_trainable, self.shared.parameters()), lr=self._lr_shared
        )
        opt_attr = optim.Adam(
            filter(_trainable, self.attr_heads.parameters()), lr=self._lr_attrs
        )
        opt_seq = None
        if self.seq_head is not None:
            opt_seq = optim.Adam(
                filter(_trainable, self.seq_head.parameters()), lr=self._lr_seq
            )
        if self.with_structure:
            opt_struct = optim.Adam(
                filter(_trainable, self.struct_enc.parameters()), lr=self._lr_shared
            )
            opts = [opt_shared, opt_attr, opt_struct]
            if opt_seq is not None:
                opts.append(opt_seq)
            return opts
        else:
            if opt_seq is not None:
                return [opt_shared, opt_attr, opt_seq]
            return [opt_shared, opt_attr]
