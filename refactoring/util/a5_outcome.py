from util.packages import *

# -*- coding: utf-8 -*-
"""
Pipeline:
- Common: semopy로 CFA 적합 + factor score 추출 → pandas df에 merge
- Main analysis: 연속 outcome ~ 연속 factor (OLS; 표준화 beta, partial R^2, Cohen's f^2)
- Auxiliary:
  (1) 이산 outcome(0/1) 집단 간 factor score two-sample t-test (Welch; Hedges' g; FDR 보정)
  (2) 이산 outcome(0/1) ~ factor scores (Logistic regression; OR; pseudo-R^2; AUC optional; FDR 보정)

Requirements:
pip install semopy statsmodels scipy scikit-learn (AUC 쓰면)
"""


# -----------------------------
# 0) Effect size helpers
# -----------------------------
def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g (small sample corrected Cohen's d) for two independent samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp2 = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if sp2 <= 0:
        return np.nan
    sp = np.sqrt(sp2)
    d = (a.mean() - b.mean()) / sp
    J = 1 - (3 / (4 * (na + nb) - 9))  # small sample correction
    return float(d * J)


def partial_r2(full_model, reduced_model) -> float:
    """Partial R^2 for nested OLS models using RSS."""
    rss_full = np.sum(full_model.resid ** 2)
    rss_red = np.sum(reduced_model.resid ** 2)
    if rss_red <= 0:
        return np.nan
    return float((rss_red - rss_full) / rss_red)


def cohens_f2(r2_full: float, r2_reduced: float) -> float:
    """Cohen's f^2 for a predictor/block using nested model R^2."""
    denom = (1 - r2_full)
    if denom <= 0:
        return np.nan
    return float((r2_full - r2_reduced) / denom)


def zscore_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - mu) / sd


# -----------------------------
# 1) Common: CFA factor scores via semopy
# -----------------------------
def fit_cfa_and_get_factor_scores(
    df: pd.DataFrame,
    model_desc: str,
    observed_cols: list[str],
    *,
    factor_score_prefix: str = "FS_",
    dropna: bool = True
) -> tuple[pd.DataFrame, object]:
    """
    Fit CFA in semopy and return factor score DataFrame (aligned to df index).
    """

    work = df.copy()
    # Use only observed vars for fitting (semopy may handle extra cols, but keep clean)
    data = work[observed_cols].copy()

    if dropna:
        data = data.dropna(axis=0, how="any")

    model = Model(model_desc)
    model.fit(data)

    fs = model.predict_factors(data)  # DataFrame (index same as data)
    fs = fs.copy()
    fs.columns = [f"{factor_score_prefix}{c}" for c in fs.columns]

    return fs, model


def attach_factor_scores(df: pd.DataFrame, factor_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join factor scores into df using index intersection.
    """
    common_idx = df.index.intersection(factor_scores_df.index)
    out = df.loc[common_idx].copy()
    out = out.join(factor_scores_df.loc[common_idx], how="left")
    return out


# -----------------------------
# 2) Main analysis: OLS regression (continuous outcomes)
# -----------------------------
def run_ols_continuous_outcomes(
    df: pd.DataFrame,
    outcome_cols: list[str],
    factor_cols: list[str],
    *,
    covariates: list[str] | None = None,
    robust_se: str = "HC3",
    standardize: bool = True,
    fdr_method: str = "fdr_bh"
) -> pd.DataFrame:
    """
    For each outcome:
      outcome ~ factors (+ covariates)
    Report:
      - standardized beta (if standardize=True)
      - coef, SE, p, CI
      - R^2, adj R^2
      - per-factor partial R^2 and Cohen's f^2 (via dropping each factor)
    Multiple testing correction applied across all (outcome, factor) p-values.
    """
    covariates = covariates or []
    needed = list(set(outcome_cols + factor_cols + covariates))
    work = df[needed].copy()

    # Standardize outcome and predictors for standardized betas
    if standardize:
        for col in outcome_cols + factor_cols + covariates:
            work[col] = zscore_series(work[col])

    results = []
    pvals = []

    for y in outcome_cols:
        cols_for_model = factor_cols + covariates
        sub = work[[y] + cols_for_model].dropna()
        if sub.shape[0] < (len(cols_for_model) + 5):
            # too few rows to be meaningful
            continue

        X = sm.add_constant(sub[cols_for_model], has_constant="add")
        yv = sub[y].astype(float)

        full = sm.OLS(yv, X).fit(cov_type=robust_se)
        r2_full = float(full.rsquared)
        adjr2_full = float(full.rsquared_adj)

        # For partial R^2 / f^2, fit reduced models dropping each factor
        for f in factor_cols:
            # Reduced model without f
            red_cols = [c for c in cols_for_model if c != f]
            X_red = sm.add_constant(sub[red_cols], has_constant="add")
            red = sm.OLS(yv, X_red).fit(cov_type=robust_se)

            pr2 = partial_r2(full, red)
            f2 = cohens_f2(r2_full, float(red.rsquared))

            coef = float(full.params.get(f, np.nan))
            se = float(full.bse.get(f, np.nan))
            p = float(full.pvalues.get(f, np.nan))
            ci_low, ci_high = full.conf_int().loc[f].tolist() if f in full.params.index else (np.nan, np.nan)

            results.append({
                "analysis": "OLS_continuous",
                "outcome": y,
                "predictor": f,
                "n": int(sub.shape[0]),
                "coef_std_beta": coef if standardize else coef,  # if standardized, this is std beta
                "se": se,
                "t": float(full.tvalues.get(f, np.nan)),
                "p_value": p,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "r2_full": r2_full,
                "adj_r2_full": adjr2_full,
                "partial_r2": pr2,
                "cohens_f2": f2,
                "covariates": ",".join(covariates) if covariates else ""
            })
            pvals.append(p)

    res = pd.DataFrame(results)
    if res.empty:
        return res

    # FDR across all predictor tests
    p = res["p_value"].to_numpy(dtype=float)
    mask = np.isfinite(p)
    padj = np.full_like(p, np.nan, dtype=float)
    if mask.sum() > 0:
        _, q, _, _ = multipletests(p[mask], method=fdr_method)
        padj[mask] = q
    res["p_adj"] = padj

    res = res.sort_values(["p_adj", "p_value"], na_position="last").reset_index(drop=True)
    return res


# -----------------------------
# 3) Auxiliary (1): t-test (binary groups) comparing factor scores
# -----------------------------
def run_ttests_binary_outcomes_on_factors(
    df: pd.DataFrame,
    binary_outcome_cols: list[str],
    factor_cols: list[str],
    *,
    equal_var: bool = False,      # Welch by default
    fdr_method: str = "fdr_bh"
) -> pd.DataFrame:
    """
    For each binary outcome (0/1):
      compare factor score distributions by group using two-sample t-test.
    Report:
      - group means, mean diff, t, p
      - Hedges' g effect size
    """
    results = []
    for outc in binary_outcome_cols:
        if outc not in df.columns:
            continue

        s = df[outc].dropna()
        uniq = pd.unique(s)
        if len(uniq) != 2:
            continue

        # Prefer 0/1 ordering if possible
        if set(uniq.tolist()) == {0, 1}:
            g0, g1 = 0, 1
        else:
            g0, g1 = uniq[0], uniq[1]

        for f in factor_cols:
            sub = df[[outc, f]].dropna()
            a = sub.loc[sub[outc] == g0, f].to_numpy(dtype=float)
            b = sub.loc[sub[outc] == g1, f].to_numpy(dtype=float)

            if a.size < 2 or b.size < 2:
                t_stat, p_val, g = np.nan, np.nan, np.nan
            else:
                t_stat, p_val = ttest_ind(a, b, equal_var=equal_var)
                g = hedges_g(a, b)

            results.append({
                "analysis": "TTEST_binary_group_vs_factor",
                "binary_outcome": outc,
                "factor": f,
                "group0": g0,
                "group1": g1,
                "n0": int(a.size),
                "n1": int(b.size),
                "mean0": float(np.mean(a)) if a.size else np.nan,
                "mean1": float(np.mean(b)) if b.size else np.nan,
                "mean_diff(0-1)": (float(np.mean(a)) - float(np.mean(b))) if a.size and b.size else np.nan,
                "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
                "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
                "hedges_g": float(g) if np.isfinite(g) else np.nan
            })

    res = pd.DataFrame(results)
    if res.empty:
        return res

    # FDR across all tests
    p = res["p_value"].to_numpy(dtype=float)
    mask = np.isfinite(p)
    padj = np.full_like(p, np.nan, dtype=float)
    if mask.sum() > 0:
        _, q, _, _ = multipletests(p[mask], method=fdr_method)
        padj[mask] = q
    res["p_adj"] = padj

    res = res.sort_values(["p_adj", "p_value"], na_position="last").reset_index(drop=True)
    return res




# -----------------------------
# 3-1) Auxiliary (1-1): u-test (binary groups) comparing factor scores
# -----------------------------
def rank_biserial_from_u(U: float, n1: int, n2: int) -> float:
    """
    Rank-biserial correlation (RBC) from Mann–Whitney U.
    RBC = 2*U/(n1*n2) - 1  (방향은 group1이 더 큰 경향이면 +가 나오도록)
    """
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return 2.0 * U / (n1 * n2) - 1.0

def run_mannwhitney_binary_outcomes_on_factors(
    df: pd.DataFrame,
    binary_outcome_cols: list[str],
    factor_cols: list[str],
    *,
    alternative: str = "two-sided",   # "two-sided" | "greater" | "less"
    fdr_method: str = "fdr_bh",
) -> pd.DataFrame:
    """
    For each binary outcome (0/1) and each factor score:
      Mann–Whitney U test between group0 vs group1.

    Outputs:
      - group medians, U, p
      - rank-biserial correlation (RBC) effect size
      - FDR-adjusted p (p_adj)
    """
    results = []

    for outc in binary_outcome_cols:
        if outc not in df.columns:
            continue

        s = df[outc].dropna()
        uniq = pd.unique(s)
        if len(uniq) != 2:
            continue

        # prefer 0/1 if possible
        if set(uniq.tolist()) == {0, 1}:
            g0, g1 = 0, 1
        else:
            g0, g1 = uniq[0], uniq[1]

        for f in factor_cols:
            if f not in df.columns:
                continue

            sub = df[[outc, f]].dropna()
            a = sub.loc[sub[outc] == g0, f].to_numpy(dtype=float)
            b = sub.loc[sub[outc] == g1, f].to_numpy(dtype=float)

            n0, n1 = a.size, b.size
            if n0 < 2 or n1 < 2:
                U, p, rbc = np.nan, np.nan, np.nan
            else:
                # scipy의 MWU는 큰 표본에서 asymptotic이 기본(버전에 따라 다름)
                try:
                    mwu = mannwhitneyu(b, a, alternative=alternative)
                    U, p = float(mwu.statistic), float(mwu.pvalue)
                except TypeError:
                    # older scipy: method 인자 미지원 등
                    U, p = mannwhitneyu(b, a, alternative=alternative)

                # 아래는 group0 기준 U 그대로 쓰고 RBC 계산.
                rbc = rank_biserial_from_u(U, n1, n0)

            results.append({
                "analysis": "MWU_binary_group_vs_factor",
                "binary_outcome": outc,
                "factor": f,
                "group0": g0,
                "group1": g1,
                "n0": int(n0),
                "n1": int(n1),
                "median0": float(np.median(a)) if n0 else np.nan,
                "median1": float(np.median(b)) if n1 else np.nan,
                "median_diff(0-1)": (float(np.median(a)) - float(np.median(b))) if n0 and n1 else np.nan,
                "U_stat": float(U) if np.isfinite(U) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "rank_biserial_r": float(rbc) if np.isfinite(rbc) else np.nan
            })

    res = pd.DataFrame(results)
    if res.empty:
        return res

    # FDR across all MWU tests
    pvals = res["p_value"].to_numpy(dtype=float)
    mask = np.isfinite(pvals)
    padj = np.full_like(pvals, np.nan, dtype=float)

    if mask.sum() > 0:
        _, q, _, _ = multipletests(pvals[mask], method=fdr_method)
        padj[mask] = q

    res["p_adj"] = padj
    res = res.sort_values(["p_adj", "p_value"], na_position="last").reset_index(drop=True)
    return res




# -----------------------------
# 4) Auxiliary (2): logistic regression for binary outcomes
# -----------------------------
def run_logistic_binary_outcomes(
    df: pd.DataFrame,
    binary_outcome_cols: list[str],
    factor_cols: list[str],
    *,
    covariates: list[str] | None = None,
    robust_se: str | None = "HC3",
    standardize_predictors: bool = True,   # OR per 1 SD increase
    fdr_method: str = "fdr_bh",
    compute_auc: bool = True,
    # --- NEW: AUC evaluation strategy ---
    auc_strategy: str = "insample",        # "insample" | "holdout" | "bootstrap"
    test_size: float = 0.3,                # holdout용 (7:3이면 0.3)
    bootstrap_n: int = 1000,               # bootstrap 반복 횟수
    bootstrap_ci: tuple[float, float] = (2.5, 97.5),
    random_state: int | None = None,       # 전역 RANDOM_SEED 전달
) -> pd.DataFrame:
    """
    For each binary outcome:
      logit(outcome) ~ factors (+ covariates)

    AUC strategies
      - insample  : fit on all available rows, AUC on same rows (기존)
      - holdout   : stratified split (train/test), fit on train, AUC on test
      - bootstrap : fit on full data(계수/OR), AUC는 bootstrap OOB로 분포 추정
                    (bootstrap 샘플에 포함되지 않은 OOB에서 AUC 계산)
    """
    covariates = covariates or []
    needed = list(set(binary_outcome_cols + factor_cols + covariates))
    work = df[needed].copy()

    # standardize predictors (OR per 1 SD)
    if standardize_predictors:
        for col in factor_cols + covariates:
            if col in work.columns:
                x = work[col]
                mu = x.mean()
                sd = x.std(ddof=0)
                work[col] = (x - mu) / sd if sd and np.isfinite(sd) and sd != 0 else np.nan

    results = []

    for outc in binary_outcome_cols:
        if outc not in work.columns:
            continue

        sub = work[[outc] + factor_cols + covariates].dropna().copy()
        if sub.shape[0] < (len(factor_cols) + len(covariates) + 10):
            continue

        # ensure binary 0/1
        y_raw = sub[outc]
        uniq = pd.unique(y_raw)
        if len(uniq) != 2:
            continue
        if set(uniq.tolist()) != {0, 1}:
            mapping = {uniq[0]: 0, uniq[1]: 1}
            y = y_raw.map(mapping).astype(int).to_numpy()
        else:
            y = y_raw.astype(int).to_numpy()

        X_df = sub[factor_cols + covariates].astype(float)
        X = sm.add_constant(X_df, has_constant="add")

        # -----------------------------
        # (A) Fit model for inference (OR/p) on:
        #   - insample : full data
        #   - holdout  : train only
        #   - bootstrap: full data (AUC는 부트스트랩으로 따로)
        # -----------------------------
        fit_idx = np.arange(len(sub))

        train_idx = fit_idx
        test_idx = None

        if auc_strategy == "holdout":
            # stratified split using y
            train_idx, test_idx = train_test_split(
                fit_idx,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
            if test_idx is None or len(test_idx) < 2:
                continue

        # fit X_fit, y_fit
        X_fit = X.iloc[train_idx, :]
        y_fit = y[train_idx]

        # -----------------------------
        # DEBUG: class balance check (VERY IMPORTANT)
        # -----------------------------
        unique, counts = np.unique(y_fit, return_counts=True)
        class_counts = dict(zip(unique.tolist(), counts.tolist()))

        print(
            f"[DEBUG][{outc}] y_fit class counts: {class_counts} "
            f"(n_fit={len(y_fit)})"
        )

        # optional: hard stop if one class too small
        if len(class_counts) < 2 or min(class_counts.values()) < 5:
            print(f"[WARN][{outc}] Too few samples in one class, skipping logistic fit.")
            continue






        logit = sm.Logit(y_fit, X_fit)

        try:
            if robust_se:
                res = logit.fit(disp=0, cov_type=robust_se)
            else:
                res = logit.fit(disp=0)
        except TypeError:
            # older statsmodels: cov_type not supported in fit()
            res = logit.fit(disp=0)

        # model-level metrics (pseudo R2 from fitted sample)
        llf = float(res.llf)
        llnull = float(getattr(res, "llnull", np.nan))
        pseudo_r2 = float(1 - llf / llnull) if np.isfinite(llnull) and llnull != 0 else np.nan

        # -----------------------------
        # (B) AUC calculation depending on strategy
        # -----------------------------
        auc_point = np.nan
        auc_ci_low = np.nan
        auc_ci_high = np.nan
        auc_n_eff = np.nan

        if compute_auc:
            try:
                if auc_strategy == "insample":
                    prob = res.predict(X_fit)
                    auc_point = float(roc_auc_score(y_fit, prob))

                elif auc_strategy == "holdout":
                    # evaluate on test
                    X_test = X.iloc[test_idx, :]
                    y_test = y[test_idx]
                    # test에 양/음 둘 다 있어야 AUC 가능
                    if len(np.unique(y_test)) == 2:
                        prob_test = res.predict(X_test)
                        auc_point = float(roc_auc_score(y_test, prob_test))

                elif auc_strategy == "bootstrap":
                    # AUC는 bootstrap OOB로 분포 추정
                    rng = np.random.default_rng(random_state)
                    aucs = []

                    # bootstrap은 "전체 sub" 기준
                    n = len(sub)
                    all_idx = np.arange(n)

                    for _ in range(int(bootstrap_n)):
                        boot_idx = rng.integers(0, n, size=n)  # with replacement
                        oob_mask = np.ones(n, dtype=bool)
                        oob_mask[boot_idx] = False
                        oob_idx = all_idx[oob_mask]

                        # OOB가 너무 작거나 클래스 한쪽만 있으면 skip
                        if oob_idx.size < 10:
                            continue
                        y_oob = y[oob_idx]
                        if len(np.unique(y_oob)) != 2:
                            continue

                        X_boot = X.iloc[boot_idx, :]
                        y_boot = y[boot_idx]

                        # bootstrap sample에서 fit
                        try:
                            boot_model = sm.Logit(y_boot, X_boot)
                            boot_res = boot_model.fit(disp=0)
                        except Exception:
                            continue

                        # OOB에서 predict + AUC
                        try:
                            prob_oob = boot_res.predict(X.iloc[oob_idx, :])
                            aucs.append(float(roc_auc_score(y_oob, prob_oob)))
                        except Exception:
                            continue

                    if len(aucs) > 0:
                        auc_n_eff = int(len(aucs))
                        auc_point = float(np.mean(aucs))
                        lo, hi = np.percentile(aucs, [bootstrap_ci[0], bootstrap_ci[1]])
                        auc_ci_low = float(lo)
                        auc_ci_high = float(hi)

                else:
                    raise ValueError(f"Unknown auc_strategy: {auc_strategy}")

            except Exception:
                auc_point = np.nan

        # -----------------------------
        # (C) Collect coefficient-level stats
        # -----------------------------
        conf = res.conf_int()
        for f in factor_cols:
            if f not in res.params.index:
                continue
            b = float(res.params[f])
            se = float(res.bse[f])
            p = float(res.pvalues[f])

            ci_low, ci_high = conf.loc[f].tolist()

            OR = float(np.exp(b))
            OR_low = float(np.exp(ci_low))
            OR_high = float(np.exp(ci_high))

            results.append({
                "analysis": "LOGIT_binary",
                "binary_outcome": outc,
                "predictor": f,
                "n_fit": int(len(train_idx)),                     # fit에 쓴 표본 수
                "n_total": int(len(sub)),                         # 가능한 전체 표본 수
                "auc_strategy": auc_strategy,
                "test_size": float(test_size) if auc_strategy == "holdout" else np.nan,
                "bootstrap_n": int(bootstrap_n) if auc_strategy == "bootstrap" else np.nan,
                "bootstrap_n_eff": auc_n_eff if auc_strategy == "bootstrap" else np.nan,

                "coef_logodds": b,
                "se": se,
                "p_value": p,
                "ci_low_logodds": float(ci_low),
                "ci_high_logodds": float(ci_high),

                ("OR_per_1SD" if standardize_predictors else "OR_per_1unit"): OR,
                "OR_ci_low": OR_low,
                "OR_ci_high": OR_high,

                "pseudo_r2_mcfadden": pseudo_r2,

                # AUC outputs
                "auc": auc_point,
                "auc_ci_low": auc_ci_low,
                "auc_ci_high": auc_ci_high,

                "covariates": ",".join(covariates) if covariates else "",
                "robust_cov_type": robust_se if robust_se else ""
            })

    resdf = pd.DataFrame(results)
    if resdf.empty:
        return resdf

    # FDR across all tests (모든 outcome×factor 계수 p_value)
    pvals = resdf["p_value"].to_numpy(dtype=float)
    mask = np.isfinite(pvals)
    padj = np.full_like(pvals, np.nan, dtype=float)
    if mask.sum() > 0:
        _, q, _, _ = multipletests(pvals[mask], method=fdr_method)
        padj[mask] = q
    resdf["p_adj"] = padj

    resdf = resdf.sort_values(["p_adj", "p_value"], na_position="last").reset_index(drop=True)
    return resdf





def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def predict_factors_pinv(model: Model, data: pd.DataFrame, center=True) -> pd.DataFrame:
    """
    pseudo-inverse 기반 회귀 factor score (수치적으로 안정)
    """
    Lambda = _first_not_none(getattr(model, "mx_lambda", None), getattr(model, "lambda_", None))
    Psi    = _first_not_none(getattr(model, "mx_psi", None),    getattr(model, "psi", None))
    Theta  = _first_not_none(getattr(model, "mx_theta", None),  getattr(model, "theta", None))

    if Lambda is None or Psi is None or Theta is None:
        raise AttributeError(
            "semopy 내부 행렬(Lambda/Psi/Theta)을 찾지 못했습니다. "
            "아래를 실행해서 model.__dict__.keys() 확인 후 attribute명을 맞춰주세요:\n"
            "print([k for k in model.__dict__.keys() if 'lambda' in k.lower() or 'theta' in k.lower() or 'psi' in k.lower()])"
        )

    X = data.to_numpy(dtype=float)
    if center:
        X = X - np.nanmean(X, axis=0, keepdims=True)

    # Σ = ΛΨΛ' + Θ
    Sigma = Lambda @ Psi @ Lambda.T + Theta

    # Regression scoring weights: W = ΨΛ' Σ^{-1}
    Sigma_inv = np.linalg.pinv(Sigma)
    W = Psi @ Lambda.T @ Sigma_inv  # (Factors x Indicators)
    Fhat = (W @ X.T).T              # (N x Factors)

    cols = [f"F{i+1}" for i in range(Fhat.shape[1])]
    return pd.DataFrame(Fhat, index=data.index, columns=cols)


def fit_cfa_get_stats_and_scores(
    df: pd.DataFrame,
    desc: str,
    observed_cols: list[str] | None = None,
    *,
    fit_options: dict | None = None,
    dropna_for_scores: bool = True,
    score_prefix: str = "FS_",
    center_scores: bool = True,
    save_est_path: str | None = None,
    save_stats_path: str | None = None,
) -> tuple[Model, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    returns:
      model, estimates_df, fit_stats_df, factor_scores_df
    """
    # 1) fit에 들어갈 데이터: df 전체를 넣되, observed_cols를 주면 그 열만 사용해야 함
    fit_df = df if observed_cols is None else df[observed_cols]
    model = Model(desc)

    print("--- CFA 모델 최적화를 시작합니다 (진행 상황 출력) ---")
    model.fit(fit_df, options=(fit_options or {"disp": True}))
    print("--- CFA 모델 최적화가 완료되었습니다 ---\n")

    # 2) 추정치/적합도
    est = model.inspect()
    stats = calc_stats(model)

    if save_est_path:
        est.to_csv(save_est_path, index=False, encoding="utf-8-sig")
    if save_stats_path:
        stats.to_csv(save_stats_path, index=False, encoding="utf-8-sig")

    # 3) factor score 계산: predict_factors 시도 → 실패하면 pinv fallback
    score_df = fit_df.copy()
    if dropna_for_scores:
        score_df = score_df.dropna(axis=0, how="any")

    try:
        fs = model.predict_factors(score_df)
    except Exception as e:
        print(f"[WARN] predict_factors 실패({type(e).__name__}: {e}) → pinv 기반 factor score로 대체합니다.")
        fs = predict_factors_pinv(model, score_df, center=center_scores)

    fs = fs.copy()
    fs.columns = [f"{score_prefix}{c}" for c in fs.columns]

    return model, est, stats, fs


# factor scoring weights 추출
def extract_scoring_weights(model: Model) -> pd.DataFrame:
    Lambda = _first_not_none(getattr(model, "mx_lambda", None), getattr(model, "lambda_", None))
    Psi    = _first_not_none(getattr(model, "mx_psi", None),    getattr(model, "psi", None))
    Theta  = _first_not_none(getattr(model, "mx_theta", None),  getattr(model, "theta", None))

    if Lambda is None or Psi is None or Theta is None:
        raise AttributeError("Cannot find Lambda/Psi/Theta in semopy model.")

    Sigma = Lambda @ Psi @ Lambda.T + Theta
    Sigma_inv = np.linalg.pinv(Sigma)
    W = Psi @ Lambda.T @ Sigma_inv  # (n_factors x n_obs)

    # ✅ semopy가 인식하는 '진짜' 변수 이름/순서를 그대로 사용
    latent_names = list(model.vars["latent"])
    observed_names = list(model.vars["observed"])

    return pd.DataFrame(W, index=latent_names, columns=observed_names)



# 아래 쓰일 factor score 라벨 생성기
def make_axis_labels(n: int, prefix: str = "FS_Factor") -> list[str]:
    """
    n개 축에 대해 FS_Factor1, FS_Factor2, ... 라벨 생성.
    """
    return [f"{prefix}{i+1}" for i in range(n)]



# factor score 라벨 적용 : 분석 코드엔 상관없는데, scoring weights에서 이름 매칭이 안 되는 문제 해결용
def relabel_scores_and_weights_as_axes(
    factor_scores_df: pd.DataFrame,
    scoring_weights_df: pd.DataFrame,
    *,
    axis_prefix: str = "Axis_"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    이름 매칭을 포기하고 '순서'로만 Axis 라벨을 부여한다.
    전제: factor_scores_df는 model.predict_factors()에서 나온 것이고,
         scoring_weights_df는 같은 model에서 계산된 W이므로 factor 차원 수가 동일해야 한다.
    """

    n_fs = factor_scores_df.shape[1]
    n_w  = scoring_weights_df.shape[0]

    if n_fs != n_w:
        raise ValueError(f"Mismatch: factor_scores_df has {n_fs} factors, "
                         f"but scoring_weights_df has {n_w} rows.")

    axis_labels = make_axis_labels(n_fs, prefix=axis_prefix)

    fs = factor_scores_df.copy()
    fs.columns = axis_labels

    W = scoring_weights_df.copy()
    # ✅ weights의 행 순서를 factor_scores_df의 column 순서로 강제 정렬하고 싶다면
    #    아래처럼 '그냥 같은 순서'로 재정의 (이름 무시)
    W = W.iloc[:n_fs, :].copy()
    W.index = axis_labels

    return fs, W


def shapiro_by_binary_groups(
    df: pd.DataFrame,
    binary_outcomes: list[str],
    factor_cols: list[str],
    *,
    alpha: float = 0.05,
    group_values: tuple[int, int] = (0, 1),
    max_n: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Shapiro-Wilk normality test for factor score distributions within each binary-outcome group.

    Returns a tidy DataFrame with:
      - binary_outcome, group, factor
      - n_used, W, p_value, normal_pass(=p>=alpha)
    """
    rng = np.random.default_rng(random_state)
    rows = []

    for outc in binary_outcomes:
        if outc not in df.columns:
            continue

        for g in group_values:
            # 해당 outcome group의 행만
            grp_df = df.loc[df[outc] == g, factor_cols].copy()

            for f in factor_cols:
                if f not in df.columns:
                    continue

                x = grp_df[f].dropna().astype(float).to_numpy()
                n = x.size

                # Shapiro는 n<3이면 불가
                if n < 3:
                    rows.append({
                        "binary_outcome": outc,
                        "group": g,
                        "factor": f,
                        "n_used": n,
                        "W": np.nan,
                        "p_value": np.nan,
                        "normal_pass": np.nan
                    })
                    continue

                # 너무 크면 샘플링 (Shapiro 민감도/권장 범위)
                if n > max_n:
                    idx = rng.choice(n, size=max_n, replace=False)
                    x_use = x[idx]
                else:
                    x_use = x

                try:
                    W, p = shapiro(x_use)
                except Exception:
                    W, p = np.nan, np.nan

                rows.append({
                    "binary_outcome": outc,
                    "group": g,
                    "factor": f,
                    "n_used": int(len(x_use)),
                    "W": float(W) if np.isfinite(W) else np.nan,
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                    "normal_pass": (p >= alpha) if np.isfinite(p) else np.nan
                })

    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values(["binary_outcome", "group", "p_value"], na_position="last").reset_index(drop=True)
    return res




# ============================= 시각화 관련 =============================

def visual_task1_ols(args) : 
    pass

def visual_task1_ttest(args) : 
    pass

def visual_task1_mwu(args) : 
    pass

