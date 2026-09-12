
import numpy as np
import matplotlib.pyplot as plt

from slepian_wolf import binary_entropy, run_trial_batch


def experiment_error_vs_rate(n=12, p=0.1, num_trials=300, num_H=5, seed=0):
    """
    Sweep R_X = m/n for m = 1..n-1, at fixed block length n and
    correlation p. For each m, average the error rate over several
    random parity-check matrices H (num_H) to smooth out the noise
    from any single random code choice.
    """
    rng = np.random.default_rng(seed)
    Hb = binary_entropy(p)

    rates, errors = [], []
    for m in range(1, n):
        trial_errors = [
            run_trial_batch(n, m, p, num_trials, rng)[0]
            for _ in range(num_H)
        ]
        rates.append(m / n)
        errors.append(float(np.mean(trial_errors)))

    return rates, errors, Hb


def experiment_error_vs_blocklength(
    p=0.1, margin=0.15, n_values=(6, 9, 12, 15, 18, 20),
    num_trials=500, num_H=30, seed=1,
):
    """
    Fix the rate at H(X|Y) + margin (just above the theoretical
    limit), and sweep block length n. Error should trend toward 0 as
    n grows, illustrating the "large n" asymptotics the paper's proof
    relies on but never plots numerically.
    """
    rng = np.random.default_rng(seed)
    Hb = binary_entropy(p)

    errors = []
    for n in n_values:
        m = int(np.ceil((Hb + margin) * n))
        m = max(1, min(m, n - 1))
        trial_errors = [
            run_trial_batch(n, m, p, num_trials, rng)[0]
            for _ in range(num_H)
        ]
        errors.append(float(np.mean(trial_errors)))

    return list(n_values), errors, Hb


def experiment_rate_region(
    p_values=(0.02, 0.05, 0.1, 0.15, 0.2, 0.3),
    n=14, num_trials=250, num_H=5, error_threshold=0.12, seed=2,
):
    """
    For each correlation strength p, scan R_X = m/n upward and find
    the smallest rate at which average error drops below
    error_threshold. Compare that empirical threshold against the
    theoretical H(X|Y) for the same p. This reconstructs the shape of
    the R_X boundary of the admissible rate region (paper's Fig. 8)
    across a range of correlations, rather than a single fixed p.
    """
    rng = np.random.default_rng(seed)

    theoretical_RX, empirical_RX = [], []
    for p in p_values:
        Hb = binary_entropy(p)
        theoretical_RX.append(Hb)

        # Start the search near the theoretical rate instead of from
        # m=1: low m means huge decoding cosets (slow) and is very
        # unlikely to meet the threshold anyway, so skip straight to
        # the interesting region just below/at H(X|Y).
        start_m = max(1, int(np.floor(Hb * n)) - 1)

        found_rate = 1.0  # fallback if nothing meets the threshold
        for m in range(start_m, n):
            trial_errors = [
                run_trial_batch(n, m, p, num_trials, rng)[0]
                for _ in range(num_H)
            ]
            if np.mean(trial_errors) < error_threshold:
                found_rate = m / n
                break
        empirical_RX.append(found_rate)

    return list(p_values), theoretical_RX, empirical_RX


def plot_error_vs_rate(rates, errors, Hb, path):
    plt.figure(figsize=(6, 4))
    plt.plot(rates, errors, marker="o", label="Empirical error rate")
    plt.axvline(Hb, color="red", linestyle="--", label=f"H(X|Y) = {Hb:.3f}")
    plt.xlabel("R_X (bits/symbol)")
    plt.ylabel("Decoding error probability")
    plt.title("Error rate vs. compression rate R_X")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_error_vs_blocklength(n_values, errors, Hb, margin, path):
    plt.figure(figsize=(6, 4))
    plt.plot(n_values, errors, marker="o")
    plt.xlabel("Block length n")
    plt.ylabel("Decoding error probability")
    plt.title(f"Error rate vs. block length (R_X = H(X|Y)+{margin})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_rate_region(p_values, theoretical_RX, empirical_RX, path):
    plt.figure(figsize=(6, 4))
    plt.plot(p_values, theoretical_RX, marker="o", label="Theoretical H(X|Y)")
    plt.plot(p_values, empirical_RX, marker="x", label="Empirical minimum R_X")
    plt.xlabel("Correlation strength p (crossover probability)")
    plt.ylabel("R_X (bits/symbol)")
    plt.title("Achievable R_X boundary vs. correlation strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    print("Running experiment 1: error vs. rate...")
    rates, errors, Hb = experiment_error_vs_rate()
    plot_error_vs_rate(rates, errors, Hb, "error_vs_rate.png")
    print("Saved error_vs_rate.png")

    print("Running experiment 2: error vs. block length...")
    margin = 0.15
    n_values, bl_errors, Hb2 = experiment_error_vs_blocklength(margin=margin)
    plot_error_vs_blocklength(n_values, bl_errors, Hb2, margin, "error_vs_blocklength.png")
    print("Saved error_vs_blocklength.png")

    print("Running experiment 3: rate region vs. correlation...")
    p_values, theo_RX, emp_RX = experiment_rate_region()
    plot_rate_region(p_values, theo_RX, emp_RX, "rate_region.png")
    print("Saved rate_region.png")

    print("Done. Check the .png files in this folder.")
