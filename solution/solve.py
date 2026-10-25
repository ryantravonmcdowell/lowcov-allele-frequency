"""Reference solution.

Reads the joint-call VCF and the sample table, estimates per-batch miscall
rates by marginal likelihood across single-copy autosomal sites, estimates
per-population alt allele frequencies with genotype likelihoods under
ploidy-aware Hardy-Weinberg priors, and labels collapsed paralogs with a
combination of depth ratio and a free-class likelihood-ratio statistic.
"""
import gzip
import sys

import numpy as np

VCF = "/root/data/calls.vcf.gz"
SAMPLES = "/root/data/samples.tsv"
OUT = "/root/results/allele_frequencies.tsv"
PAR_END = 800_000

# tuning constants for this solution (fitted on development seeds, never on
# the shipped cohort)
OUTER_ITER = 30
INNER_ITER = 5
FINAL_ITER = 150
FREE_ITER = 30
EPS_INIT = 0.01
RHO_EPS_CUT = 1.6      # depth-ratio ceiling for the first error-rate pass
T_DELTA_HARD = 25.0    # free-class likelihood gain that alone flags a paralog
T_RHO = 1.5            # depth ratio that supports a weaker likelihood gain
T_DELTA_SOFT = 8.0
P_FLOOR = 1e-6
EXP_CLIP = 600.0


def read_inputs():
    meta = []
    with open(SAMPLES) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("sample_id"):
                continue
            meta.append(line.rstrip("\n").split("\t"))
    sample_id = [m[0] for m in meta]
    pop = np.array([m[1] for m in meta])
    sex = np.array([m[2] for m in meta])
    batch = np.array([m[3] for m in meta])
    chrom, pos, R, A = [], [], [], []
    header = None
    with gzip.open(VCF, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            f = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                header = f[9:]
                continue
            chrom.append(f[0])
            pos.append(int(f[1]))
            r_row, a_row = [], []
            for cell in f[9:]:
                ad = cell.split(":")[0].split(",")
                r_row.append(int(ad[0]))
                a_row.append(int(ad[1]))
            R.append(r_row)
            A.append(a_row)
    R = np.array(R, dtype=np.int32)
    A = np.array(A, dtype=np.int32)
    if header != sample_id:
        order = [header.index(s) for s in sample_id]
        R, A = R[:, order], A[:, order]
    return np.array(chrom), np.array(pos), R, A, pop, sex, batch


class Cohort:
    """Arrays split by population, with ploidy and haploid masks."""

    def __init__(self, A, R, pop_idx, ploidy):
        self.A = A.astype(float)
        self.R = R.astype(float)
        self.D = self.A + self.R
        self.pop_idx = pop_idx
        self.K = int(pop_idx.max()) + 1
        self.cols = [np.where(pop_idx == k)[0] for k in range(self.K)]
        self.ploidy = ploidy
        self.hap = (ploidy == 1).astype(float)
        self.haps = np.stack([ploidy[:, c].sum(axis=1) for c in self.cols], 1)

    def subset(self, rows):
        c = Cohort.__new__(Cohort)
        c.A, c.R, c.D = self.A[rows], self.R[rows], self.D[rows]
        c.pop_idx, c.K, c.cols = self.pop_idx, self.K, self.cols
        c.ploidy, c.hap = self.ploidy[rows], self.hap[rows]
        c.haps = self.haps[rows]
        return c


def ratios(co, eps):
    """exp(l1 - l0) and exp(l2 - l0) for the three read-fraction classes
    (0, 1/2, 1) given per-sample miscall rates."""
    e = eps[None, :]
    l0 = co.A * np.log(e) + co.R * np.log1p(-e)
    l1 = co.D * np.log(0.5)
    l2 = co.A * np.log1p(-e) + co.R * np.log(e)
    return (np.exp(np.clip(l1 - l0, -EXP_CLIP, EXP_CLIP)),
            np.exp(np.clip(l2 - l0, -EXP_CLIP, EXP_CLIP)))


def weights(q, E1, E2, hap):
    """Posterior class weights for allele frequency column q (S,1).
    hap is a 0/1 float array marking haploid samples."""
    dip = 1.0 - hap
    r = 1.0 - q
    u0 = r * (r * dip + hap)
    u1 = 2.0 * q * r * E1 * dip
    u2 = q * (q * dip + hap) * E2
    z = u0 + u1 + u2
    return u0 / z, u1 / z, u2 / z


def em_step(co, E1, E2, p):
    """One EM update of per-population frequencies; returns new p and the
    class weights of the update."""
    p_new = np.empty_like(p)
    W = []
    for k, c in enumerate(co.cols):
        w0, w1, w2 = weights(p[:, k:k + 1], E1[:, c], E2[:, c], co.hap[:, c])
        e_alt = w1 + co.ploidy[:, c] * w2
        p_new[:, k] = np.clip(e_alt.sum(axis=1) / co.haps[:, k], P_FLOOR,
                              1 - P_FLOOR)
        W.append((w0, w1, w2))
    return p_new, W


def init_p(co):
    p = np.empty((co.A.shape[0], co.K))
    for k, c in enumerate(co.cols):
        p[:, k] = np.clip(co.A[:, c].sum(axis=1) /
                          np.maximum(co.D[:, c].sum(axis=1), 1), 0.01, 0.99)
    return p


def fit_error_rates(co, batch_idx, nb, eps0):
    """Marginal-likelihood EM for per-batch miscall rates and frequencies."""
    eps = eps0.copy()
    p = init_p(co)
    for outer in range(OUTER_ITER):
        E1, E2 = ratios(co, eps)
        for _ in range(INNER_ITER):
            p, W = em_step(co, E1, E2, p)
        num = np.zeros(nb)
        den = np.zeros(nb)
        for k, c in enumerate(co.cols):
            w0, w1, w2 = W[k]
            A, R, D = co.A[:, c], co.R[:, c], co.D[:, c]
            miss = w0 * A + w2 * R + w1 * eps[c][None, :] * D
            for b in range(nb):
                sel = batch_idx[c] == b
                num[b] += miss[:, sel].sum()
                den[b] += D[:, sel].sum()
        rate = np.clip(num / den, 1e-4, 0.2)
        eps = rate[batch_idx]
    return eps, p


def fit_frequencies(co, eps, p_init=None, iters=FINAL_ITER):
    E1, E2 = ratios(co, eps)
    p = init_p(co) if p_init is None else p_init.copy()
    for _ in range(iters):
        p, _w = em_step(co, E1, E2, p)
    return p, E1, E2


def marginal_loglik(co, eps, p):
    """Sum over samples of the log marginal likelihood (up to the class-0
    read term that cancels against the free-class value)."""
    E1, E2 = ratios(co, eps)
    e = eps[None, :]
    base = co.A * np.log(e) + co.R * np.log1p(-e)
    tot = np.zeros(co.A.shape[0])
    for k, c in enumerate(co.cols):
        q = p[:, k:k + 1]
        hap = co.hap[:, c]
        dip = 1.0 - hap
        r = 1.0 - q
        u0 = r * (r * dip + hap)
        u1 = 2.0 * q * r * E1[:, c] * dip
        u2 = q * (q * dip + hap) * E2[:, c]
        tot += (base[:, c] + np.log(u0 + u1 + u2)).sum(axis=1)
    return tot


def free_class_loglik(co, eps, p):
    """Maximised log-likelihood with a free read fraction for each genotype
    class, fitted separately in each population."""
    S = co.A.shape[0]
    total = np.zeros(S)
    for k, c in enumerate(co.cols):
        A, R, D = co.A[:, c], co.R[:, c], co.D[:, c]
        e = eps[c][None, :]
        hap = co.hap[:, c]
        ploidy = co.ploidy[:, c]
        f = np.tile(np.array([0.0, 0.5, 1.0]), (S, 1))
        q = p[:, k:k + 1].copy()

        def ll_of(fc):
            qq = np.clip(fc[:, None] * (1 - 2 * e) + e, 1e-6, 1 - 1e-6)
            return A * np.log(qq) + R * np.log1p(-qq)

        for _ in range(FREE_ITER):
            l = [ll_of(f[:, j]) for j in range(3)]
            E1 = np.exp(np.clip(l[1] - l[0], -EXP_CLIP, EXP_CLIP))
            E2 = np.exp(np.clip(l[2] - l[0], -EXP_CLIP, EXP_CLIP))
            w = weights(q, E1, E2, hap)
            for j in range(3):
                num = (w[j] * (A - e * D)).sum(axis=1)
                den = (w[j] * D * (1 - 2 * e)).sum(axis=1)
                ok = den > 1e-9
                f[ok, j] = np.clip(num[ok] / den[ok], 0.0, 1.0)
            q = np.clip(((w[1] + ploidy * w[2]).sum(axis=1) /
                         ploidy.sum(axis=1))[:, None], P_FLOOR, 1 - P_FLOOR)
        l = [ll_of(f[:, j]) for j in range(3)]
        E1 = np.exp(np.clip(l[1] - l[0], -EXP_CLIP, EXP_CLIP))
        E2 = np.exp(np.clip(l[2] - l[0], -EXP_CLIP, EXP_CLIP))
        dip = 1.0 - hap
        r = 1.0 - q
        u0 = r * (r * dip + hap)
        u1 = 2.0 * q * r * E1 * dip
        u2 = q * (q * dip + hap) * E2
        total += (l[0] + np.log(u0 + u1 + u2)).sum(axis=1)
    return total


def solve_arrays(chrom, pos, R, A, pop, sex, batch):
    pops = sorted(set(pop))
    pop_idx = np.array([pops.index(x) for x in pop])
    batches = sorted(set(batch))
    batch_idx = np.array([batches.index(x) for x in batch])
    nb = len(batches)
    n = A.shape[1]
    auto = chrom != "chrX"
    nonpar = (chrom == "chrX") & (pos > PAR_END)
    haploid = nonpar[:, None] & (sex == "M")[None, :]
    ploidy = np.where(haploid, 1, 2)
    co = Cohort(A, R, pop_idx, ploidy)

    # depth ratio per site relative to the single-copy expectation
    lam = co.D[auto].mean(axis=0)
    rho = co.D.sum(axis=1) / (lam[None, :] * ploidy / 2.0).sum(axis=1)
    rho = rho / np.median(rho[auto])

    eps0 = np.full(n, EPS_INIT)
    rows1 = np.where(auto & (rho <= RHO_EPS_CUT))[0]
    eps1, _ = fit_error_rates(co.subset(rows1), batch_idx, nb, eps0)
    p2, _, _ = fit_frequencies(co, eps1)
    delta = np.maximum(free_class_loglik(co, eps1, p2) -
                       marginal_loglik(co, eps1, p2), 0.0)
    para = (delta >= T_DELTA_HARD) | ((rho >= T_RHO) & (delta >= T_DELTA_SOFT))

    rows2 = np.where(auto & ~para)[0]
    eps2, _ = fit_error_rates(co.subset(rows2), batch_idx, nb, eps1)
    p3, _, _ = fit_frequencies(co, eps2, p_init=p2, iters=60)
    delta = np.maximum(free_class_loglik(co, eps2, p3) -
                       marginal_loglik(co, eps2, p3), 0.0)
    para = (delta >= T_DELTA_HARD) | ((rho >= T_RHO) & (delta >= T_DELTA_SOFT))
    info = dict(eps={b: float(eps2[batch_idx == i][0])
                     for i, b in enumerate(batches)},
                rho=rho, delta=delta, pops=pops)
    return p3, para, info


def main():
    chrom, pos, R, A, pop, sex, batch = read_inputs()
    p, para, info = solve_arrays(chrom, pos, R, A, pop, sex, batch)
    order = [info["pops"].index("P1"), info["pops"].index("P2")]
    with open(OUT, "w") as fh:
        fh.write("chrom\tpos\tlocus_class\taf_P1\taf_P2\n")
        for j in range(len(pos)):
            if para[j]:
                fh.write(f"{chrom[j]}\t{pos[j]}\tcollapsed_paralog\tNA\tNA\n")
            else:
                fh.write(f"{chrom[j]}\t{pos[j]}\tsingle_copy\t"
                         f"{p[j, order[0]]:.6f}\t{p[j, order[1]]:.6f}\n")
    print("batch miscall rates:", info["eps"], file=sys.stderr)


if __name__ == "__main__":
    main()
