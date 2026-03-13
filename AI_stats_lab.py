import numpy as np


def exponential_pdf(x, lam=1):
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, lam * np.exp(-lam * x), 0.0)


def exponential_interval_probability(a, b, lam=1):
    if a < 0:
        a = 0.0
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    samples = np.random.exponential(scale=1.0, size=n)
    return np.mean((samples > a) & (samples < b))


def gaussian_pdf(x, mu, sigma):
    coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coefficient * np.exp(exponent)


def posterior_probability(time):
    prior_A, prior_B = 0.3, 0.7
    likelihood_A = np.exp(-(time - 40) ** 2 / 4)
    likelihood_B = np.exp(-(time - 45) ** 2 / 4)
    evidence = likelihood_A * prior_A + likelihood_B * prior_B
    return (likelihood_B * prior_B) / evidence


def simulate_posterior_probability(time, n=100000, tolerance=0.5):
    sigma = np.sqrt(2)
    groups = np.random.choice(['A', 'B'], size=n, p=[0.3, 0.7])
    times = np.where(
        groups == 'A',
        np.random.normal(loc=40, scale=sigma, size=n),
        np.random.normal(loc=45, scale=sigma, size=n)
    )
    mask = np.abs(times - time) < tolerance
    filtered_groups = groups[mask]
    if len(filtered_groups) == 0:
        raise ValueError("No samples near the given time. Increase n or tolerance.")
    return np.mean(filtered_groups == 'B')


if __name__ == "__main__":
    print("=" * 55)
    print("QUESTION 1 — Exponential Distribution (λ=1)")
    print("=" * 55)

    lam, a, b = 1, 2, 5
    pdf_val    = exponential_pdf(1, lam)
    analytical = exponential_interval_probability(a, b, lam)
    simulated  = simulate_exponential_probability(a, b)
    expected   = np.exp(-2) - np.exp(-5)

    print(f"PDF at x=1:          f(1) = {pdf_val:.6f}  (expected {np.exp(-1):.6f})")
    print(f"Analytical P(2<X<5)       = {analytical:.6f}")
    print(f"Formula    e^-2 - e^-5    = {expected:.6f}")
    print(f"Simulated  P(2<X<5)       = {simulated:.6f}")

    print()
    print("=" * 55)
    print("QUESTION 2 — Bayesian Classification")
    print("=" * 55)

    time = 42
    gauss_val = gaussian_pdf(40, 40, 2)
    print(f"gaussian_pdf(40,40,2)      = {gauss_val:.6f}  (expected {1/(np.sqrt(2*np.pi)*2):.6f})")

    post_analytical = posterior_probability(time)
    post_simulated  = simulate_posterior_probability(time)

    num = 0.7 * np.exp(-(42 - 45) ** 2 / 4)
    den = 0.3 * np.exp(-(42 - 40) ** 2 / 4) + num
    expected_post = num / den

    print(f"Analytical P(B|X=42)       = {post_analytical:.6f}  (expected {expected_post:.6f})")
    print(f"Simulated  P(B|X=42)       = {post_simulated:.6f}")
