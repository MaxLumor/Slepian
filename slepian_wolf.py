"""
Slepian-Wolf coding simulation.

"""

import numpy as np


def binary_entropy(p):
    """Hb(p) in bits. Returns 0 at p=0 or p=1."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def generate_source_pairs(n, p, num_trials, rng):
    """
    Generate num_trials independent (X, Y) pairs, each of block length n.

    X: (num_trials, n) array of 0/1, i.i.d. fair bits.
    Y = X XOR Z, Z ~ Bernoulli(p) i.i.d.

    Returns X, Y as int arrays.
    """
    X = rng.integers(0, 2, size=(num_trials, n))
    Z = (rng.random(size=(num_trials, n)) < p).astype(int)
    Y = X ^ Z
    return X, Y


def random_parity_check_matrix(m, n, rng):
    """
    A random binary m x n parity check matrix H (over GF(2)).
    The code rate for X is R_X = m / n bits/symbol (syndrome length
    per source symbol) -- NOT the usual "code rate" convention, since
    here we are directly compressing X (source coding), not adding
    redundancy to protect it (channel coding). Smaller m means fewer
    bits sent, i.e. lower R_X.
    """
    return rng.integers(0, 2, size=(m, n))


def build_syndrome_table(H):
    """
    Precompute, for a given H (m x n), a dictionary mapping each
    syndrome (as a tuple of m bits) to the list of all x in {0,1}^n
    that produce that syndrome (the coset/"bin").

    Only feasible for small n (n <= ~18 or so). This mirrors the
    paper's construction of bins/cosets, but built explicitly instead
    of relying on an abstract existence argument.
    """
    m, n = H.shape
    table = {}
    for i in range(2 ** n):
        x = np.array([(i >> k) & 1 for k in range(n)][::-1])
        s = tuple((H @ x) % 2)
        table.setdefault(s, []).append(x)
    return table


def ml_decode(y, syndrome, syndrome_table):
    """
    Given side information y and the received syndrome, return the
    x in the corresponding bin/coset that is closest to y in Hamming
    distance (the ML decoder for a BSC(p) with p < 0.5 between X
    and Y).
    """
    candidates = syndrome_table[tuple(syndrome)]
    if len(candidates) == 1:
        return candidates[0]
    dists = [np.sum(c != y) for c in candidates]
    return candidates[int(np.argmin(dists))]


def run_trial_batch(n, m, p, num_trials, rng, H=None, syndrome_table=None):
    """
    Run num_trials independent Slepian-Wolf coding trials at block
    length n, syndrome length m (so R_X = m/n), and correlation p.

    Returns:
        error_rate: fraction of trials where decoded X != true X
        H, syndrome_table: reusable across calls for the same (n, m)
    """
    if H is None:
        H = random_parity_check_matrix(m, n, rng)
    if syndrome_table is None:
        syndrome_table = build_syndrome_table(H)

    X, Y = generate_source_pairs(n, p, num_trials, rng)

    errors = 0
    for i in range(num_trials):
        x, y = X[i], Y[i]
        s = (H @ x) % 2
        x_hat = ml_decode(y, s, syndrome_table)
        if not np.array_equal(x_hat, x):
            errors += 1

    return errors / num_trials, H, syndrome_table


if __name__ == "__main__":
    # Quick sanity check: as R_X = m/n crosses H(X|Y), error rate
    # should drop sharply. This reproduces Theorem 2's claim directly.
    rng = np.random.default_rng(0)
    p = 0.1
    Hb = binary_entropy(p)
    print(f"Correlation p={p}, H(X|Y) = {Hb:.4f} bits/symbol")

    n = 12
    for m in range(2, 9):
        err, H, table = run_trial_batch(n, m, p, num_trials=500, rng=rng)
        print(f"n={n}, m={m}, R_X={m/n:.3f}  ->  error rate = {err:.3f}")