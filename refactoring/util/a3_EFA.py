from util.packages import *

def scree_plot(args, df_EFA, fa_vars) : 
    # --- 정제 단계 ---
    X = df_EFA[fa_vars].copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)

    # 1) 전부 결측 열 확인
    all_nan_cols = X.columns[X.isna().all()]
    if len(all_nan_cols) > 0:
        print("[제거] 전부 결측인 열:", list(all_nan_cols))
    X = X.drop(columns=all_nan_cols)

    # 2) 중앙값 대치
    imp = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)

    # 3) 분산 0(또는 극소) 열 확인
    stds = X_imp.std(ddof=0)
    zero_var_cols = stds[stds <= 1e-5].index
    if len(zero_var_cols) > 0:
        print("[제거] 분산≈0인 열:", list(zero_var_cols))
    X_imp = X_imp.drop(columns=zero_var_cols)

    # 4) 상관행렬 NaN 발생 여부 점검
    corr = X_imp.corr()
    bad_in_corr = corr.columns[corr.isna().any()].tolist()
    if len(bad_in_corr) > 0:
        print("[경고] 상관행렬에서 NaN 발생한 열:", bad_in_corr)

    # 5) 변수 수 최종 확인
    if X_imp.shape[1] < 3:
        raise ValueError(f"[중단] 정제 후 변수 수 부족: {list(X_imp.columns)}")

    # --- 스크리 플롯 ---
    fa_scree = FactorAnalyzer(rotation='oblimin')
    fa_scree.fit(X_imp)

    eigenvals, _ = fa_scree.get_eigenvalues()
    print("[EFA] 고유값(Eigenvalues):", np.round(eigenvals[:X_imp.shape[1]], 4))

    plt.figure()
    plt.plot(range(1, len(eigenvals) + 1), eigenvals, marker='o')
    plt.axhline(1, linestyle='--')
    plt.xlabel('Factor')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot')
    global RESULTS_DIR # plot 저장하기
    plt.savefig(os.path.join(RESULTS_DIR, "EFA_scree_plot.png"))
    # plt.show()


def preprocess_for_fa(df, fa_vars, min_var=1e-12):
    """EFA용 전처리: 숫자화→Inf/NaN 처리→전부 결측 열 제거→중앙값 대치→분산≈0 열 제거"""
    X = df[fa_vars].copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)

    # 전부 결측 열 제거
    all_nan_cols = X.columns[X.isna().all()]
    if len(all_nan_cols) > 0:
        X = X.drop(columns=all_nan_cols)

    # 중앙값 대치 및 분산이 0에 근접한 열 제거는 일단 전처리에서 이미 수행했으니 여기선 일단 뺌
    # imp = SimpleImputer(strategy='median')
    # X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)

    # # 분산≈0 열 제거
    # stds = X_imp.std(ddof=0)
    # keep = stds > min_var
    # X_imp = X_imp.loc[:, keep]

    # 남은 열이 최소 3개 미만이면 EFA 불가
    if X.shape[1] < 3:
        raise ValueError(f"[중단] 정제 후 변수 수 부족: {list(X.columns)}")

    return X


def efa_task(args, X_imp, N_FACTORS) : 
    
    global RESULTS_DIR # plot 저장하기

    # 1-요인일 때 varimax 회전은 의미가 거의 없음(keep 그대로 사용해도 무방)
    fa = FactorAnalyzer(n_factors=N_FACTORS, rotation="promax")
    fa.fit(X_imp)

    # 적재치 행렬
    cols = [f"Factor{i+1}" for i in range(N_FACTORS)]
    loadings = pd.DataFrame(fa.loadings_, index=X_imp.columns, columns=cols)
    print(loadings.round(3).to_string())
    loadings.round(3).to_csv(os.path.join(RESULTS_DIR, "EFA_loadings.csv"), index=True, encoding="utf-8-sig") # 파일 저장

    # # ---- Heatmap (축 개수 제한 없음) ----
    # plt.figure(figsize=(1.5*N_FACTORS+4, 0.4*len(loadings)+3))
    # sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    # plt.title("Factor Loadings Heatmap")
    # plt.tight_layout()
    # plt.show()


    # ---- Heatmap (축 개수 제한 없음, 단 변수 많으면 단순화) ----
    MAX_ANNOT_VARS = 4000  # 이 값 넘으면 숫자 제거 + 크기 간소화     ===>>> 일단 여기선 전체 경향을 봐야 하므로 그냥 다 표시

    n_vars = len(loadings)
    annot_flag = n_vars <= MAX_ANNOT_VARS

    fig_height = min(0.4 * n_vars + 3, 25)  # 세로축 최대 크기 제한 (예: 25인치)

    plt.figure(figsize=(1.5*N_FACTORS+4, fig_height))
    sns.heatmap(
        loadings,
        annot=annot_flag,
        cmap="coolwarm",
        center=0,
        fmt=".2f" if annot_flag else "",
        cbar=True
    )
    plt.title("Factor Loadings Heatmap" + ("" if annot_flag else " (values hidden)"))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "EFA_heatmap.png"))
    # plt.show()

    # 공통성 / 특이성 계산
    communalities = pd.Series(fa.get_communalities(), index=X_imp.columns, name="Communality")
    uniqueness    = pd.Series(fa.get_uniquenesses(), index=X_imp.columns, name="Uniqueness")

    result = pd.concat([communalities, uniqueness], axis=1).round(3)
    print(result.to_string())
    result.to_csv(os.path.join(RESULTS_DIR, "EFA_communality_and_uniqueness.csv"), index=True, encoding="utf-8-sig") # 파일 저장

