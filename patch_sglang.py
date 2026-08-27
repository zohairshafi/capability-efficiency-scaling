"""
Patches SGLang's sampler at install time so that spawned TP workers load the
modified code. Must be run AFTER `pip install sglang` during Docker image build.

What the patch does:
  - Replaces the top-k logprob block in `_attach_logprobs_to_output` with:
      1. Exact Shannon entropy (bits) from the full vocabulary distribution.
      2. Number of tokens needed to cover each probability mass threshold (p* count).
      3. Shannon entropy of the renormalized truncated distribution at each threshold
         (bulk entropy — entropy computed only from tokens within that mass).
      4. Sampled-token surprisal in nats: -log p(sampled_token).
      5. Varentropy (bits²): Var[-log₂ p] = E[(log₂ p)²] - H².
      6. p_max: probability of the most likely token.
      7. Rank of the sampled token (1-indexed, 1 = most likely).

  Layout per token in meta_info["output_top_logprobs"][token_idx]
  (thresholds = [1.0, 0.99, 0.90], N = len(thresholds) = 3):
      [0]:     (entropy_full,       -1, None)   exact entropy, all vocab tokens
      [1]:     (n_tokens_1.00,      -2, None)   \
      [2]:     (n_tokens_0.99,      -3, None)    > token counts per threshold
      [3]:     (n_tokens_0.90,      -4, None)   /
      [4]:     (entropy_bulk_1.00,  -5, None)   \
      [5]:     (entropy_bulk_0.99,  -6, None)    > bulk entropy per threshold
      [6]:     (entropy_bulk_0.90,  -7, None)   /
      [7]:     (surprisal_sampled,  -8, None)   when top_logprobs_num >= 8
      [8]:     (varentropy,         -9, None)   when top_logprobs_num >= 9
      [9]:     (p_max,             -10, None)   when top_logprobs_num >= 10
     [10]:     (rank_sampled,      -11, None)   when top_logprobs_num >= 11

    Baseline entries per token: 1 + 2*N = 7.
    Set top_logprobs_num=11 to include all metrics.

  All computations use fp32 for numerical stability.
  Thresholds here MUST match P_STAR_THRESHOLDS in the calling scripts.

Why source-level patching and not monkey-patching at runtime:
  SGLang TP workers are launched with `spawn`, not `fork`. A runtime
  monkey-patch in the parent process is invisible to the workers. Patching
  the installed .py file means every spawned worker loads the patched version.

Usage:
    python3 patch_sglang.py
"""
import re
import sys
import inspect
from pathlib import Path

# ── Locate the installed sampler.py ───────────────────────────────────────────
try:
    import sglang.srt.layers.sampler as _sampler_mod
except ImportError:
    print("ERROR: cannot import sglang.srt.layers.sampler — is sglang installed?")
    sys.exit(1)

src_path = Path(inspect.getfile(_sampler_mod))
print(f"Patching: {src_path}")
src = src_path.read_text()

# ── Guard: don't double-patch ─────────────────────────────────────────────────
if "ENTROPY PATCH" in src:
    print("Already patched — skipping.")
    sys.exit(0)

# ── Replacement block ─────────────────────────────────────────────────────────
# Replaces the `if any(x > 0 ...) get_top_logprobs(...)` block with:
#   - exact entropy from full vocab
#   - p* token counts (how many tokens cover each threshold)
#   - bulk entropy (entropy of renormalized truncated distribution per threshold)
#   - sampled-token surprisal, varentropy, p_max, and rank
#
# Thresholds [1.0, 0.99, 0.90] MUST match P_STAR_THRESHOLDS in the calling script.
ENTROPY_BLOCK = (
    "        # [ENTROPY PATCH] Exact entropy, p* token counts, bulk entropy,\n"
    "        # sampled-token surprisal, varentropy, p_max, and token rank.\n"
    "        # Layout per token (thresholds = [1.0, 0.99, 0.90], N=3):\n"
    "        #   [0]: (entropy_full,      -1, None)  exact entropy, all vocab\n"
    "        #   [1]: (n_tokens_1.00,     -2, None)  }\n"
    "        #   [2]: (n_tokens_0.99,     -3, None)  } token counts\n"
    "        #   [3]: (n_tokens_0.90,     -4, None)  }\n"
    "        #   [4]: (entropy_bulk_1.00, -5, None)  }\n"
    "        #   [5]: (entropy_bulk_0.99, -6, None)  } bulk entropies\n"
    "        #   [6]: (entropy_bulk_0.90, -7, None)  }\n"
    "        #   [7]: (surprisal_sampled, -8, None)  when top_logprobs_num >= 8\n"
    "        #   [8]: (varentropy,        -9, None)  when top_logprobs_num >= 9\n"
    "        #   [9]: (p_max,            -10, None)  when top_logprobs_num >= 10\n"
    "        #  [10]: (rank_sampled,     -11, None)  when top_logprobs_num >= 11\n"
    "        _logprobs_fp32 = logprobs.float()  # use fp32 for stable entropy math\n"
    "        _probs = _logprobs_fp32.exp()  # [batch, vocab_size]\n"
    "        _ln2 = torch.log(torch.tensor(2.0, device=logprobs.device, dtype=torch.float32))\n"
    "        # ── Full entropy ──\n"
    "        _entropy = -(_probs * _logprobs_fp32).sum(dim=-1) / _ln2  # [batch], bits\n"
    "        # ── Varentropy: Var[-log2 p] = E[(log2 p)^2] - H^2  (bits^2) ──\n"
    "        _surprisal_bits = -_logprobs_fp32 / _ln2  # [batch, vocab]\n"
    "        _varentropy = (_probs * _surprisal_bits.pow(2)).sum(dim=-1) - _entropy.pow(2)  # [batch]\n"
    "        # ── p_max: probability of the most likely token ──\n"
    "        _p_max = _probs.max(dim=-1).values  # [batch]\n"
    "        # ── Surprisal of the sampled token (nats) ──\n"
    "        _sampled_lp = _logprobs_fp32[\n"
    "            torch.arange(_logprobs_fp32.shape[0], device=logprobs.device),\n"
    "            batch_next_token_ids.to(torch.long),\n"
    "        ]\n"
    "        _sampled_surprisal = -_sampled_lp\n"
    "        # ── Sort for p* calculations ──\n"
    "        _probs_sorted, _sort_idx = torch.sort(_probs, dim=-1, descending=True)  # [batch, vocab]\n"
    "        _lp_sorted = _logprobs_fp32.gather(-1, _sort_idx)  # logprobs reordered to match sorted probs\n"
    "        # ── Rank of sampled token (1-indexed, 1 = most likely) ──\n"
    "        _rank = (_sort_idx == batch_next_token_ids.to(torch.long).unsqueeze(1)).int().argmax(dim=1) + 1  # [batch]\n"
    "        _cumsum = torch.cumsum(_probs_sorted, dim=-1)  # [batch, vocab]\n"
    "        _thresholds = torch.tensor(\n"
    "            [1.0, 0.99, 0.90],\n"
    "            device=logprobs.device, dtype=torch.float32,\n"
    "        )  # [N]\n"
    "        # ── p* token counts ──\n"
    "        _n_tokens = torch.searchsorted(\n"
    "            _cumsum.contiguous(),\n"
    "            _thresholds.unsqueeze(0).expand(_probs.shape[0], -1).contiguous(),\n"
    "            right=False,\n"
    "        ) + 1  # [batch, N], 1-indexed\n"
    "        _n_tokens.clamp_(max=_probs.shape[-1])\n"
    "        # ── Bulk entropy per threshold ──\n"
    "        # Uses sorted logprobs to derive H_bulk without renormalising probabilities.\n"
    "        # Identity: H_bulk = (-sum_top(p*lnp) / S + ln(S)) / ln2\n"
    "        # where S = sum_top(p).  For p*=1.0, S->1 so H_bulk -> entropy_full exactly.\n"
    "        _positions = torch.arange(_probs_sorted.shape[1], device=logprobs.device).unsqueeze(0)\n"
    "        _entropy_bulk = []\n"
    "        for _k in range(len(_thresholds)):\n"
    "            _n_k = _n_tokens[:, _k].unsqueeze(1)  # [batch, 1]\n"
    "            _mask = (_positions < _n_k)  # [batch, vocab]\n"
    "            _p  = _probs_sorted * _mask  # zero out beyond cutoff\n"
    "            _lp = _lp_sorted   * _mask  # matching logprobs (p=0 positions contribute 0)\n"
    "            _S  = _p.sum(dim=-1).clamp(min=1e-10)  # [batch], total mass in top-n\n"
    "            _h  = (-(_p * _lp).sum(dim=-1) / _S + torch.log(_S)) / _ln2  # bits\n"
    "            _entropy_bulk.append(_h.tolist())\n"
    "        # ── Pack into pipeline format ──\n"
    "        # Baseline payload width is 7 (entropy/p*/bulk). Extra metrics:\n"
    "        #   [7]: surprisal (sentinel -8),  [8]: varentropy (sentinel -9),\n"
    "        #   [9]: p_max (sentinel -10),     [10]: rank (sentinel -11).\n"
    "        # Any remaining requested width is filled with raw top-k logprobs.\n"
    "        _e_list  = _entropy.tolist()        # [batch]\n"
    "        _n_list  = _n_tokens.tolist()       # [batch, N]\n"
    "        _s_list  = _sampled_surprisal.tolist()  # [batch], nats\n"
    "        _v_list  = _varentropy.tolist()     # [batch], bits^2\n"
    "        _p_list  = _p_max.tolist()          # [batch], probability\n"
    "        _r_list  = _rank.tolist()           # [batch], 1-indexed\n"
    "        _N = len(_thresholds)\n"
    "        _base_width = 1 + 2 * _N\n"
    "        _vals = []\n"
    "        _idxs = []\n"
    "        for _i, _k in enumerate(top_logprobs_nums):\n"
    "            _k_int = int(_k)\n"
    "            _row_vals = [_e_list[_i]] + _n_list[_i] + [_entropy_bulk[_j][_i] for _j in range(_N)]\n"
    "            _row_idxs = [-(_j + 1) for _j in range(_base_width)]\n"
    "            if _k_int > _base_width:\n"
    "                _row_vals.append(_s_list[_i])\n"
    "                _row_idxs.append(-(_base_width + 1))\n"
    "            if _k_int > _base_width + 1:\n"
    "                _row_vals.append(_v_list[_i])\n"
    "                _row_idxs.append(-(_base_width + 2))\n"
    "            if _k_int > _base_width + 2:\n"
    "                _row_vals.append(_p_list[_i])\n"
    "                _row_idxs.append(-(_base_width + 3))\n"
    "            if _k_int > _base_width + 3:\n"
    "                _row_vals.append(_r_list[_i])\n"
    "                _row_idxs.append(-(_base_width + 4))\n"
    "            if _k_int > len(_row_vals):\n"
    "                _extra_k = min(_k_int - len(_row_vals), _logprobs_fp32.shape[-1])\n"
    "                _extra_vals, _extra_idx = _logprobs_fp32[_i].topk(_extra_k, dim=-1)\n"
    "                _row_vals.extend(_extra_vals.tolist())\n"
    "                _row_idxs.extend(_extra_idx.to(torch.int32).tolist())\n"
    "            _vals.append(_row_vals[:_k_int])\n"
    "            _idxs.append(_row_idxs[:_k_int])\n"
    "        logits_output.next_token_top_logprobs_val = _vals\n"
    "        logits_output.next_token_top_logprobs_idx = _idxs\n"
)

# Match the topk block regardless of exact indentation or whitespace.
pattern = re.compile(
    r"[ \t]+if any\(x > 0 for x in top_logprobs_nums\):.*?"
    r"= get_top_logprobs\(logprobs, top_logprobs_nums\)",
    re.DOTALL,
)

match = pattern.search(src)
if not match:
    print(f"ERROR: could not locate the top_logprobs block in {src_path}")
    print("The SGLang version may have changed. Inspect the file manually.")
    print()
    idx = src.find("get_top_logprobs")
    if idx >= 0:
        print("Context around 'get_top_logprobs':")
        print(src[max(0, idx - 300) : idx + 200])
    sys.exit(1)

new_src = pattern.sub(ENTROPY_BLOCK, src)
src_path.write_text(new_src)
print(f"OK — patched {src_path}")
print(f"    Replaced {len(match.group(0))} chars with {len(ENTROPY_BLOCK)} chars.")
