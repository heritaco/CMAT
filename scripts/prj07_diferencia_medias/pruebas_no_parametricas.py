import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats





def pruebas_no_parametricas(group1, group2, split, estudiante_o_calificacion='Salón', PATH=None, MATERIA=None):
    n1, n2 = len(group1), len(group2)

    # 1) Mann–Whitney U (Wilcoxon rank-sum)
    U, p_mw = stats.mannwhitneyu(group1, group2, alternative='two-sided', method='auto')
    U_star = max(U, n1*n2 - U)                       # for CL
    CL = U_star / (n1*n2)                            # common-language effect size
    r_rb = 2*CL - 1                                  # rank-biserial correlation

    # 2) Brunner–Munzel
    bm_stat, p_bm = stats.brunnermunzel(group1, group2, alternative='two-sided')

    # 3) Robust median shift with bootstrap CI
    def bootstrap_diff_median(x, y, B=10000, seed=0):
        rng = np.random.default_rng(seed)
        diffs = np.empty(B, float)
        n1, n2 = len(x), len(y)
        for b in range(B):
            xb = rng.choice(x, n1, replace=True)
            yb = rng.choice(y, n2, replace=True)
            diffs[b] = np.median(xb) - np.median(yb)
        ci = np.quantile(diffs, [0.025, 0.975])
        return np.median(x) - np.median(y), ci, diffs

    dmed, ci_med, _ = bootstrap_diff_median(group1, group2, B=8000, seed=42)

    # 4) Permutation test on the median difference
    def stat_median(x, y):
        return np.median(x) - np.median(y)

    perm = stats.permutation_test((group1, group2), stat_median,
                                n_resamples=5000, alternative='two-sided',
                                random_state=42)
    p_perm_med = perm.pvalue

    # 5) Cliff's delta (approx if very large)
    def cliffs_delta(x, y, max_pairs=5_000_000, seed=0):
        x = np.asarray(x); y = np.asarray(y)
        n1, n2 = len(x), len(y)
        rng = np.random.default_rng(seed)
        if n1*n2 > max_pairs:
            m = int(np.sqrt(max_pairs))
            X = x[rng.integers(0, n1, m)]
            Y = y[rng.integers(0, n2, m)]
        else:
            X, Y = x, y
        cmp = X[:, None] - Y[None, :]
        gt = np.sum(cmp > 0)
        lt = np.sum(cmp < 0)
        return (gt - lt) / (X.size * Y.size)

    delta = cliffs_delta(group1, group2)

    def bootstrap_ci_two_sample(x, y, stat_fn, B=5000, seed=0, alpha=0.05):
        """
        Generic bootstrap CI for a two-sample statistic.

        x, y      : 1D arrays
        stat_fn   : function taking (x, y) and returning a scalar
        B         : number of bootstrap resamples
        alpha     : 1 - CI level (0.05 -> 95% CI)
        """
        rng = np.random.default_rng(seed)
        x = np.asarray(x)
        y = np.asarray(y)
        n1, n2 = len(x), len(y)

        boot_stats = np.empty(B, float)
        for b in range(B):
            xb = rng.choice(x, n1, replace=True)
            yb = rng.choice(y, n2, replace=True)
            boot_stats[b] = stat_fn(xb, yb)

        lower, upper = np.quantile(boot_stats, [alpha/2, 1 - alpha/2])
        return (lower, upper), boot_stats

    U, p_mw = stats.mannwhitneyu(group1, group2, alternative='two-sided', method='auto')
    U_star = max(U, n1*n2 - U)                       # for CL
    CL = U_star / (n1*n2)                            # common-language effect size
    r_rb = 2*CL - 1                                  # rank-biserial correlation


    def stat_CL(x, y):
        n1, n2 = len(x), len(y)
        U, _ = stats.mannwhitneyu(x, y, alternative='two-sided', method='auto')
        U_star = max(U, n1*n2 - U)
        CL = U_star / (n1*n2)
        return CL

    # Bootstrap CI for CL
    ci_CL, _ = bootstrap_ci_two_sample(group1, group2, stat_CL, B=4000, seed=123)

    # From CL we derive r_rb, so we can bootstrap it directly as well:
    def stat_r_rb(x, y):
        cl = stat_CL(x, y)
        return 2*cl - 1

    ci_r_rb, _ = bootstrap_ci_two_sample(group1, group2, stat_r_rb, B=4000, seed=123)

    def stat_delta(x, y):
        return cliffs_delta(x, y)

    ci_delta, _ = bootstrap_ci_two_sample(group1, group2, stat_delta, B=400, seed=456)

    dmed, ci_med, _ = bootstrap_diff_median(group1, group2, B=800, seed=42)

    def stat_median_diff(x, y):
        return np.median(x) - np.median(y)

    ci_med2, diffs = bootstrap_ci_two_sample(group1, group2, stat_median_diff, B=800, seed=42)
    dmed2 = stat_median_diff(group1, group2)

    m1, m2 = np.median(group1), np.median(group2)
    text = (
        f"Mediana (más de {split} visitas) = {m1:.3f}\n"
        f"Mediana ({split} o menos visitas) = {m2:.3f}\n"
        f"Mann–Whitney U p={p_mw:.4g}, "
        f"CL={CL:.3f} CI95% [{ci_CL[0]:.3f}, {ci_CL[1]:.3f}], "
        f"r_rb={r_rb:.3f} CI95% [{ci_r_rb[0]:.3f}, {ci_r_rb[1]:.3f}]\n"
        f"Brunner–Munzel p={p_bm:.4g}\n"
        f"Δ mediana = {dmed:.3f}  CI95% [{ci_med[0]:.3f}, {ci_med[1]:.3f}]  "
        f"Permutación p_mediana={p_perm_med:.4g}\n"
        f"Cliff’s δ={delta:.3f} CI95% [{ci_delta[0]:.3f}, {ci_delta[1]:.3f}]"
    )
    print(text)

    # get percentages
    total = len(group1) + len(group2)
    perc1 = (len(group1) / total) * 100
    perc2 = (len(group2) / total) * 100

    # ---- Plot with medians and robust stats

    if split == 1:
        keyword = 'visita'
    else:
        keyword = 'visitas'
    sns.kdeplot(group1, cut=0, bw_adjust=0.5, fill=True, alpha=0.3, label=f'Más de {split} {keyword} ({perc1:.2f}%)')
    sns.kdeplot(group2, cut=0, bw_adjust=0.5, fill=True, alpha=0.3, label=f'Menos de {split} {keyword} ({perc2:.2f}%)')
    
    plt.vlines(x=group1.mean(), ymin=0, ymax=1.2, colors='blue', linestyles='-', label=f'Mediana de más de {split} {keyword}: {m1:.2f}', alpha=0.7)
    plt.vlines(x=group2.mean(), ymin=0, ymax=1.2, colors='orange', linestyles='-', label=f'Mediana de menos de {split} {keyword}: {m2:.2f}', alpha=0.7)
    plt.title(f'Comparación robusta de Distribuciones de Calificaciones por {estudiante_o_calificacion}')
    plt.xlabel('Calificación Estandarizada (KDE, Z-score)')
    plt.ylabel('Densidad de Calificaciones')
    if estudiante_o_calificacion == 'Estudiante':
        plt.ylabel('Densidad de la Media de Calificaciones')
    plt.legend(loc='upper left')
    plt.xlim(-4, 2)
    if MATERIA is not None:
        plt.suptitle(f'Materia: {MATERIA}', y=1.02, fontsize=16)
    plt.savefig(f'{PATH}08_NoParametricos_{estudiante_o_calificacion}.pdf', bbox_inches='tight')
    plt.show()