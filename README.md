# Slepian-Wolf Coding Simulation

Replication of the central claim (Theorem 2, case 0011) of:

D. Slepian and J. Wolf, "Noiseless Coding of Correlated Information
Sources," *IEEE Transactions on Information Theory*, vol. 19, no. 4,
pp. 471-480, July 1973.

## What this does

Simulates two correlated binary sources X and Y (Y = X XOR Z, with Z
controlling correlation strength), compresses X using random
syndrome/binning coding **without letting the encoder see Y**, and
decodes X at the receiver using Y as side information. This tests
whether X can be compressed down to the theoretical limit H(X|Y)
(rather than the naive H(X)) while still being recovered losslessly.

See `slepian_wolf.py` for the full explanation of the method in the
module docstring.

## Setup

1. Clone the repo and `cd` into it.
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
3. Activate it:
   - macOS/Linux: `source venv/bin/activate`
   - Windows (PowerShell): `venv\Scripts\Activate.ps1`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running

Quick sanity check (error rate vs. rate, printed to terminal):
```bash
python3 -c "
import numpy as np
from slepian_wolf import binary_entropy, run_trial_batch

rng = np.random.default_rng(0)
p = 0.1
print('H(X|Y) =', binary_entropy(p))

n = 12
for m in range(2, 9):
    err, H, table = run_trial_batch(n, m, p, num_trials=500, rng=rng)
    print(f'R_X={m/n:.3f} -> error={err:.3f}')
"
```

Full experiment sweep + plots: run `experiments.py` (see repo once
added) to generate:
- Error rate vs. R_X, with the H(X|Y) threshold marked
- Error rate vs. block length n at fixed rate (convergence plot)
- Recreated admissible rate region (Fig. 8 style)

## Repo structure

```
.
├── slepian_wolf.py     # core simulation: source generation,
│                        # syndrome coding, ML decoding
├── experiments.py      # experiment sweeps + plot generation
├── requirements.txt
└── README.md
```
