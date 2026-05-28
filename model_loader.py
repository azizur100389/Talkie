"""
Download and load Talkie custom models.

Architecture (from reference src/talkie/model.py):
  - Pre-norm GPT with *parameter-free* F.rms_norm (no learnable γ/β)
  - Per-layer scalar gains (attn_gain, mlp_gain, embed_skip)
  - QK-norm + per-head gain on queries
  - RoPE (base=1e6) positional encoding
  - SwiGLU FFN
  - LM head with weight gain
  - Embedding skip-connection (added after MLP, not with attention)

Checkpoint keys follow `_orig_mod.blocks.{i}.*` naming (torch.compile wrapper).
"""

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download

import config


# ═══════════════════════════════════════════════════════════════════════
#  Architecture  (matches checkpoint key layout exactly)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TalkieConfig:
    n_layer: int = 40
    n_head: int = 40
    n_embd: int = 5120
    head_dim: int = 128
    vocab_size: int = 65536
    max_seq_len: int = 2048
    intermediate_size: int = 13696


class HeadGain(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.head_g = nn.Parameter(torch.ones(n_head))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.head_g.type_as(x).view(1, 1, -1, 1)


class WeightGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_g = nn.Parameter(torch.ones(1))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w * self.w_g.type_as(w)


class ActGain(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.a_g = nn.Parameter(torch.ones(1) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.a_g.type_as(x)


# ── RoPE (base=1e6, no learnable params) ─────────────────────────────

def _precompute_rotary(seq_len: int, head_dim: int,
                       base: int = 1_000_000) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()[None, :, None, :]
    sin = freqs.sin()[None, :, None, :]
    return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


# ── Attention (with QK-norm and per-head gain on Q) ──────────────────

class Attention(nn.Module):
    def __init__(self, cfg: TalkieConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        n_state = cfg.n_embd

        self.attn_query = nn.Linear(n_state, n_state, bias=False)
        self.attn_key   = nn.Linear(n_state, n_state, bias=False)
        self.attn_value = nn.Linear(n_state, n_state, bias=False)
        self.attn_resid = nn.Linear(n_state, n_state, bias=False)
        self.head_gain  = HeadGain(cfg.n_head)

    def forward(self, x: torch.Tensor, cos_sin: tuple) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.attn_query(x).view(B, T, self.n_head, self.head_dim)
        k = self.attn_key(x).view(B, T, self.n_head, self.head_dim)
        v = self.attn_value(x).view(B, T, self.n_head, self.head_dim)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = self.head_gain(q)

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.attn_resid(y)


# ── SwiGLU FFN ────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, cfg: TalkieConfig):
        super().__init__()
        self.mlp_gate   = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.mlp_linear = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.mlp_resid  = nn.Linear(cfg.intermediate_size, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_resid(F.silu(self.mlp_gate(x)) * self.mlp_linear(x))


# ── Transformer block (pre-norm with parameter-free RMS norm) ────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg: TalkieConfig):
        super().__init__()
        self.attn       = Attention(cfg)
        self.mlp        = MLP(cfg)
        self.attn_gain  = ActGain((2 * cfg.n_layer) ** -0.5)
        self.mlp_gain   = ActGain((2 * cfg.n_layer) ** -0.5)
        self.embed_skip = ActGain(0.0)

    def forward(self, e_x: torch.Tensor, x: torch.Tensor,
                cos_sin: tuple) -> torch.Tensor:
        x = x + self.attn_gain(self.attn(F.rms_norm(x, (x.shape[-1],)), cos_sin))
        x = x + self.mlp_gain(self.mlp(F.rms_norm(x, (x.shape[-1],))))
        x = x + self.embed_skip(e_x)
        return x


# ── Full model ────────────────────────────────────────────────────────

class TalkieModel(nn.Module):
    def __init__(self, cfg: TalkieConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.lm_head = nn.Parameter(torch.zeros(cfg.vocab_size, cfg.n_embd))
        self.lm_head_gain = WeightGain()

        cos, sin = _precompute_rotary(cfg.max_seq_len, cfg.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False,
                last_positions: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        cos_sin = self.cos[:, :T].to(input_ids.device), self.sin[:, :T].to(input_ids.device)

        x = self.embed(input_ids)
        x = F.rms_norm(x, (x.shape[-1],))
        e_x = x

        hidden_states = [x] if output_hidden_states else None

        for block in self.blocks:
            x = block(e_x, x, cos_sin)
            if output_hidden_states:
                hidden_states.append(x)

        x = F.rms_norm(x, (x.shape[-1],))

        if last_positions is not None:
            x_sel = x[torch.arange(B, device=x.device), last_positions]
            logits = F.linear(x_sel, self.lm_head_gain(self.lm_head)).float()
        else:
            logits = F.linear(x, self.lm_head_gain(self.lm_head)).float()

        if output_hidden_states:
            hidden_states.append(x)
            return logits, tuple(hidden_states)
        return (logits,)


# ═══════════════════════════════════════════════════════════════════════
#  Checkpoint loading
# ═══════════════════════════════════════════════════════════════════════

def _strip_prefix(state_dict: dict, prefix: str = "_orig_mod.") -> dict:
    """Remove torch.compile wrapper prefix from all keys."""
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def _detect_config(state_dict: dict) -> TalkieConfig:
    cfg = TalkieConfig()
    for k, v in state_dict.items():
        if not hasattr(v, "shape"):
            continue
        if k == "embed.weight":
            cfg.vocab_size, cfg.n_embd = v.shape
        elif "mlp_gate.weight" in k:
            cfg.intermediate_size = v.shape[0]
        elif "head_gain.head_g" in k:
            cfg.n_head = v.shape[0]

    layer_ids = set()
    for k in state_dict:
        if k.startswith("blocks."):
            idx = k.split(".")[1]
            if idx.isdigit():
                layer_ids.add(int(idx))
    if layer_ids:
        cfg.n_layer = max(layer_ids) + 1
    cfg.head_dim = cfg.n_embd // cfg.n_head

    print(f"[detect] n_layer={cfg.n_layer}  n_embd={cfg.n_embd}  n_head={cfg.n_head}  "
          f"head_dim={cfg.head_dim}  intermediate={cfg.intermediate_size}  vocab={cfg.vocab_size}")
    return cfg


def inspect_checkpoint(ckpt_path):
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for w in ["model", "state_dict"]:
            if w in raw and isinstance(raw[w], dict):
                raw = raw[w]
                print(f"[inspect] Unwrapped '{w}'")
                break
    print(f"\n{'Key':<70} {'Shape':<25} Dtype")
    print("-" * 110)
    for k, v in sorted(raw.items()):
        s = tuple(v.shape) if hasattr(v, "shape") else "scalar"
        d = v.dtype if hasattr(v, "dtype") else type(v).__name__
        print(f"{k:<70} {str(s):<25} {d}")
    print(f"\nTotal keys: {len(raw)}")


# ═══════════════════════════════════════════════════════════════════════
#  Tokeniser
# ═══════════════════════════════════════════════════════════════════════

def load_tiktoken_tokeniser(repo_dir: Path, expected_vocab: int = 65536) -> tiktoken.Encoding:
    """Load tiktoken tokeniser from vocab.txt (base64+rank format)."""
    vocab_path = repo_dir / "vocab.txt"
    if not vocab_path.exists():
        print("[tokeniser] WARNING: No vocab.txt, using cl100k_base fallback.")
        return tiktoken.get_encoding("cl100k_base")

    # Parse base64+rank format: each line is "BASE64_TOKEN RANK"
    mergeable_ranks = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    token_bytes = base64.b64decode(parts[0])
                    rank = int(parts[1])
                    if rank < expected_vocab:
                        mergeable_ranks[token_bytes] = rank
                except Exception:
                    continue

    print(f"[tokeniser] Loaded {len(mergeable_ranks)} tokens "
          f"(max rank {max(mergeable_ranks.values()) if mergeable_ranks else 0})")
    return _build_encoding(repo_dir.name, mergeable_ranks)


def _build_encoding(name: str, mergeable_ranks: dict) -> tiktoken.Encoding:
    return tiktoken.Encoding(
        name=f"talkie-{name}",
        pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        mergeable_ranks=mergeable_ranks,
        special_tokens={"<|endoftext|>": len(mergeable_ranks)},
    )


# ═══════════════════════════════════════════════════════════════════════
#  Download
# ═══════════════════════════════════════════════════════════════════════

def download_model_repo(model_id: str) -> Path:
    local_dir = config.CACHE_DIR / model_id.replace("/", "--")
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"[model_loader] Using cached repo: {local_dir}")
        return local_dir
    print(f"[model_loader] Downloading {model_id} ...")
    snapshot_download(repo_id=model_id, local_dir=str(local_dir))
    return local_dir


def _find_checkpoint(repo_dir: Path) -> Path:
    for pattern in ["*.ckpt", "*.pt", "*.pth", "*.bin"]:
        matches = sorted(repo_dir.glob(pattern), key=lambda p: p.stat().st_size, reverse=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No checkpoint found in {repo_dir}")


# ═══════════════════════════════════════════════════════════════════════
#  ModelWrapper
# ═══════════════════════════════════════════════════════════════════════

class ModelWrapper:
    def __init__(self, model: TalkieModel, tokeniser: tiktoken.Encoding, model_id: str):
        self.model = model
        self.tokeniser = tokeniser
        self.model_id = model_id
        self.name = config.MODEL_NAMES.get(model_id, model_id)
        self.device = config.DEVICE
        self.dtype = config.DTYPE

    @torch.no_grad()
    def log_likelihood(self, text: str) -> float:
        token_ids = self.tokeniser.encode(text)
        if not token_ids:
            return 0.0
        input_ids = torch.tensor([token_ids], device=self.device)
        logits = self.model(input_ids)[0]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        return log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1).sum().item()

    @torch.no_grad()
    def log_likelihood_per_token(self, text: str) -> tuple[float, int]:
        token_ids = self.tokeniser.encode(text)
        n = len(token_ids)
        if n <= 1:
            return 0.0, max(n, 1)
        input_ids = torch.tensor([token_ids], device=self.device)
        logits = self.model(input_ids)[0]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        return log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1).sum().item(), n

    def score_pair(self, good: str, bad: str) -> dict:
        ll_good, n_good = self.log_likelihood_per_token(good)
        ll_bad, n_bad = self.log_likelihood_per_token(bad)
        return {
            "ll_good": ll_good, "ll_bad": ll_bad,
            "n_tokens_good": n_good, "n_tokens_bad": n_bad,
            "ll_good_norm": ll_good / n_good, "ll_bad_norm": ll_bad / n_bad,
            "correct": ll_good > ll_bad,
            "correct_norm": (ll_good / n_good) > (ll_bad / n_bad),
        }

    @torch.no_grad()
    def batch_log_likelihood(self, texts: list[str],
                             batch_size: int = 256) -> list[tuple[float, int]]:
        all_token_ids = [self.tokeniser.encode(t) for t in texts]
        results = [None] * len(texts)

        for start in range(0, len(texts), batch_size):
            batch_ids = all_token_ids[start:start + batch_size]
            lengths = [len(ids) for ids in batch_ids]
            max_len = max(lengths)

            padded = torch.zeros(len(batch_ids), max_len, dtype=torch.long,
                                 device=self.device)
            for j, ids in enumerate(batch_ids):
                padded[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)

            logits = self.model(padded)[0]
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            targets = padded[:, 1:].unsqueeze(-1)
            token_lps = log_probs.gather(2, targets).squeeze(-1)

            for j, n in enumerate(lengths):
                if n <= 1:
                    results[start + j] = (0.0, max(n, 1))
                else:
                    results[start + j] = (token_lps[j, :n - 1].sum().item(), n)

        return results

    @torch.no_grad()
    def batch_conditional_ll(self, prompts: list[str], completions: list[str],
                             batch_size: int = 32) -> list[tuple[float, int]]:
        """Log P(completion | prompt) for each (prompt, completion) pair.

        Returns list of (conditional_ll, n_completion_tokens).
        """
        prompt_ids = [self.tokeniser.encode(p) for p in prompts]
        full_ids = [self.tokeniser.encode(p + c) for p, c in zip(prompts, completions)]
        results = [None] * len(prompts)

        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_full = full_ids[start:end]
            batch_prompt_lens = [len(prompt_ids[start + j]) for j in range(end - start)]
            lengths = [len(ids) for ids in batch_full]
            max_len = max(lengths)
            B = len(batch_full)

            padded = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
            for j, ids in enumerate(batch_full):
                padded[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)

            logits = self.model(padded)[0]
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            targets = padded[:, 1:].unsqueeze(-1)
            token_lps = log_probs.gather(2, targets).squeeze(-1)

            for j in range(B):
                prompt_len = batch_prompt_lens[j]
                seq_len = lengths[j]
                n_completion = seq_len - prompt_len
                if n_completion <= 0:
                    results[start + j] = (0.0, 0)
                else:
                    cond_ll = token_lps[j, prompt_len - 1:seq_len - 1].sum().item()
                    results[start + j] = (cond_ll, n_completion)

        return results

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64,
                 temperature: float = 0.0, stop_tokens: Optional[list[str]] = None) -> str:
        token_ids = self.tokeniser.encode(prompt)
        generated = list(token_ids)
        eos = self.tokeniser.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
        eos_id = eos[0] if eos else None

        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated], device=self.device)
            logits = self.model(input_ids)[0]
            next_logits = logits[:, -1, :]
            if temperature <= 0:
                tok = next_logits.argmax(-1).item()
            else:
                tok = torch.multinomial(F.softmax(next_logits / temperature, -1), 1).item()
            if tok == eos_id:
                break
            generated.append(tok)
            if stop_tokens:
                decoded = self.tokeniser.decode(generated[len(token_ids):])
                if any(s in decoded for s in stop_tokens):
                    break
        return self.tokeniser.decode(generated[len(token_ids):])

    @torch.no_grad()
    def extract_hidden_states(self, text: str,
                              layer_indices: Optional[list[int]] = None) -> dict[int, torch.Tensor]:
        if layer_indices is None:
            layer_indices = config.PROBE_LAYERS
        token_ids = self.tokeniser.encode(text)[:config.PROBE_MAX_SEQ_LEN]
        input_ids = torch.tensor([token_ids], device=self.device)
        logits, hidden_states = self.model(input_ids, output_hidden_states=True)
        result = {}
        for idx in layer_indices:
            if idx < len(hidden_states):
                result[idx] = hidden_states[idx].squeeze(0).mean(dim=0).cpu().float()
        return result

    @torch.no_grad()
    def get_next_token_probs(self, prompt: str, target_tokens: list[str]) -> dict[str, float]:
        token_ids = self.tokeniser.encode(prompt)
        input_ids = torch.tensor([token_ids], device=self.device)
        logits = self.model(input_ids)[0]
        probs = F.softmax(logits[:, -1, :], dim=-1).squeeze(0)
        result = {}
        for tok_str in target_tokens:
            ids = self.tokeniser.encode(tok_str)
            result[tok_str] = probs[ids[0]].item() if ids else 0.0
        return result

    @torch.no_grad()
    def batch_next_token_probs(self, prompts: list[str],
                               target_token_ids: list[int],
                               batch_size: int = 32) -> torch.Tensor:
        """Batched next-token probabilities. Returns [n_prompts, n_targets]."""
        all_ids = [self.tokeniser.encode(p) for p in prompts]
        target_idx = torch.tensor(target_token_ids, device=self.device)
        out = torch.zeros(len(prompts), len(target_token_ids))

        for start in range(0, len(prompts), batch_size):
            batch_ids = all_ids[start:start + batch_size]
            lengths = [len(ids) for ids in batch_ids]
            max_len = max(lengths)
            B = len(batch_ids)

            padded = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
            for j, ids in enumerate(batch_ids):
                padded[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)

            last_pos = torch.tensor([l - 1 for l in lengths], device=self.device)
            logits = self.model(padded, last_positions=last_pos)[0]
            probs = F.softmax(logits, dim=-1)
            out[start:start + B] = probs[:, target_idx].cpu()

        return out


# ═══════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════

def load_model(model_id: str) -> ModelWrapper:
    repo_dir = download_model_repo(model_id)
    ckpt_path = _find_checkpoint(repo_dir)
    print(f"[model_loader] Checkpoint: {ckpt_path.name} ({ckpt_path.stat().st_size / 1e9:.1f} GB)")

    # Load & unwrap checkpoint
    print("[model_loader] Loading checkpoint into RAM ...")
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = raw
    for wrapper in ["model", "state_dict", "model_state_dict"]:
        if isinstance(state_dict, dict) and wrapper in state_dict and isinstance(state_dict[wrapper], dict):
            state_dict = state_dict[wrapper]
            break

    # Strip _orig_mod. prefix from torch.compile
    state_dict = _strip_prefix(state_dict, "_orig_mod.")

    # Detect architecture and build model
    arch_cfg = _detect_config(state_dict)
    model = TalkieModel(arch_cfg)

    # Load weights
    model_sd = model.state_dict()
    ckpt_keys = set(state_dict.keys())
    model_keys = set(model_sd.keys())

    matched = ckpt_keys & model_keys
    missing_in_ckpt = model_keys - ckpt_keys
    extra_in_ckpt = ckpt_keys - model_keys

    for k in matched:
        if model_sd[k].shape == state_dict[k].shape:
            model_sd[k] = state_dict[k]
        else:
            print(f"  Shape mismatch: {k}  model={model_sd[k].shape}  ckpt={state_dict[k].shape}")

    model.load_state_dict(model_sd, strict=False)
    del raw, state_dict

    print(f"[model_loader] Loaded {len(matched)}/{len(model_keys)} params")
    if missing_in_ckpt:
        # Filter out non-persistent buffers (rotary caches)
        real_missing = {k for k in missing_in_ckpt if "rotary" not in k and "cached" not in k}
        if real_missing:
            print(f"  Missing in ckpt: {sorted(real_missing)[:5]}")
    if extra_in_ckpt:
        print(f"  Extra in ckpt:   {sorted(extra_in_ckpt)[:5]}")

    model = model.to(dtype=torch.float32).to(config.DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model_loader] {n_params/1e9:.2f}B params on {config.DEVICE} (float32)")

    # Tokeniser
    tokeniser = load_tiktoken_tokeniser(repo_dir, expected_vocab=arch_cfg.vocab_size)
    print(f"[model_loader] Tokeniser: vocab={tokeniser.n_vocab}")

    return ModelWrapper(model, tokeniser, model_id)