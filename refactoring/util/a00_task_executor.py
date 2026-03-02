from util.packages import *
from util.a1_preprocess import *
from util.a2_kmo import *
from util.a3_EFA import *
from util.a4_CFA import *
from util.a5_outcome import *


def etc_jobs(args) :
    
    # 폰트 설정
    fp = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

    # plt.title("한글 제목 테스트", fontproperties=fp)
    # plt.xlabel("가로축", fontproperties=fp)
    # plt.ylabel("세로축", fontproperties=fp)
    # plt.show()

    fp = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    rcParams['font.family'] = fp.get_name()   # 파일에서 읽은 실제 이름 등록
    rcParams['axes.unicode_minus'] = False





def preprocess(args, RANDOM_SEED, DATA_DIR, RESULTS_DIR) : 


    print(args)
    
    DATA_FILE = args.data_file         # 설문 데이터(.xlsx)
    META_FILE = args.meta_file       # 변수 메타데이터(.xlsx)

    # ==== 데이터/메타데이터 로드 ====
    df = pd.read_excel(DATA_FILE)
    meta = pd.read_excel(META_FILE)

    print("데이터 크기:", df.shape)
    print("컬럼 예시:", df.columns[:10].tolist())

    # 메타 필수 컬럼 체크
    required_cols = {"변수명", "범주형", "연속형"}
    missing_meta_cols = required_cols - set(meta.columns)
    if missing_meta_cols:
        raise ValueError(f"메타데이터에 필요한 컬럼이 없습니다: {missing_meta_cols}")

    # 범주/연속 변수 리스트
    categorical_vars = meta.loc[meta["범주형"] == 1, "변수명"].dropna().astype(str).tolist()
    continuous_vars = meta.loc[meta["연속형"] == 1, "변수명"].dropna().astype(str).tolist()

    # 실제 df에 존재하는 컬럼만 사용
    categorical_vars = [c for c in categorical_vars if c in df.columns]
    continuous_vars = [c for c in continuous_vars if c in df.columns]

    print("범주형 변수 개수:", len(categorical_vars))
    print("연속형 변수 개수:", len(continuous_vars))



    # 누락변수 있으면 확인 후 제거 (ID는 원래 제외하는게 맞음)
    print(set(df.columns) - set(categorical_vars + continuous_vars))

    df.drop(set(df.columns) - set(categorical_vars + continuous_vars), axis=1, inplace=True)

    # 결측치 제거
    df_clean = df.dropna()
    print("원래 행 개수:", len(df))
    print("결측치 제거 후 행 개수:", len(df_clean))
    df = df_clean.copy()



    # ===== 시간 값 변환 =====
    
    report = convert_time_columns_inplace_for_efa(df, continuous_vars)

    print("[변환 완료 요약]")
    print("  변환 성공 열 수:", len(report["converted"]))
    if report["converted"]:
        print("   └", report["converted"])

    print("  변환 스킵 열 수:", len(report["skipped"]))
    if report["skipped"]:
        print("   └", report["skipped"])
        print("\n[스킵 사유(문제 값 예시)]")
        print(report["issues"].to_string(index=False))

    # EFA 투입 열 목록 (숫자형만)
    efa_cols = report["efa_cols"]
    print("\n[EFA 열 개수]:", len(efa_cols))
    # 이제 바로 EFA:
    # X_efa = df[efa_cols].copy()
    # print("EFA 입력 크기:", X_efa.shape)



    # 시간 값 평균 취한 후 상대적 차이로 변환 (예: 수면 시작 시간은 평균보다 몇 분 늦거나 빠른지)
    df = apply_relative_to_mean(df, meta)


    # 표준화 없는 데이터 미리 분리
    df_raw = df.copy()





    # (추가) 표준화/시간변환 전에 분포 요약 + plot 2장 저장
    cont_stats, cat_props = summarize_and_plot_continuous_and_categorical(
        df=df,
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
        RESULTS_DIR=RESULTS_DIR,          # 혹은 RESULTS_DIR
        top_k_categories=5
    )
    print(cont_stats.head(10))
    print(cat_props.head(10))







    # z-score 표준화 (평균 0, 표준편차 1), 동시에 임시용으로 표준화하지 않은 원본 열도 유지

    # dtype이 number면 모두 표준화 : 사실상 전부 표준화 (카테고리값도 전부 숫자이므로)
    num_cols = df.select_dtypes(include=['number']).columns

    print(len(num_cols))






    # 수치형 열 전부 표준화 (원본 df 자체에 덮어쓰기)
    df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std(ddof=0)





    # # (2018 데이터 분리용) 이대로 저장
    # df.to_csv(os.path.join(DATA_DIR, "df_2018.csv"), index=False, encoding="utf-8-sig")
    # df_raw.to_csv(os.path.join(DATA_DIR, "df_2018_raw.csv"), index=False, encoding="utf-8-sig")

    # import sys
    # sys.exit(0)





    # 데이터셋 5:5로 분할 (EFA용, CFA용)
    df_EFA, df_CFA = train_test_split(df, test_size=0.5, random_state=RANDOM_SEED)

    df_EFA_raw, df_CFA_raw = train_test_split(df_raw, test_size=0.5, random_state=RANDOM_SEED)

    print("✅ 데이터 분할 완료")
    print(f" - EFA용 데이터: {df_EFA.shape}")
    print(f" - CFA용 데이터: {df_CFA.shape}")
    print()
    print(f" - EFA용 데이터: {df_EFA_raw.shape}")
    print(f" - CFA용 데이터: {df_CFA_raw.shape}")


    print("예시 값:", df_EFA["MCTQ_GUa"].values[:5])
    print("예시 값:", df_EFA["MCTQ_MSF"].values[:5])
    print("예시 값:", df_EFA["MSF_mismatch_pristine"].values[:5])
    print("예시 값:", df_EFA["MCTQ_MBF"].values[:5])

    print("예시 값:", df_CFA["MCTQ_GUa"].values[:5])
    print("예시 값:", df_CFA["MCTQ_MSF"].values[:5])
    print("예시 값:", df_CFA["MSF_mismatch_pristine"].values[:5])
    print("예시 값:", df_CFA["MCTQ_MBF"].values[:5])


    # 나이 그룹도 제거
    df_EFA = df_EFA.drop(columns=['age_group'])
    df_CFA = df_CFA.drop(columns=['age_group'])

    # 분산 작은 변수 제거
    var_threshold = args.var_threshold
    low_var_cols = df.var()[df.var() <= var_threshold].index.tolist()
    print("[제거] 분산이 너무 작은 열:", low_var_cols)

    df = df.drop(columns=low_var_cols)
    df_EFA = df_EFA.drop(columns=low_var_cols)
    df_CFA = df_CFA.drop(columns=low_var_cols)


    # df_EFA, df_CFA, df_EFA_raw, df_CFA_raw 저장 (전처리 쪽이니 RESULT_DIR이 아닌 DATA_DIR에 저장)

    df_EFA.to_csv(os.path.join(DATA_DIR, "df_EFA.csv"), index=False)
    df_CFA.to_csv(os.path.join(DATA_DIR, "df_CFA.csv"), index=False)
    df_EFA_raw.to_csv(os.path.join(DATA_DIR, "df_EFA_raw.csv"), index=False)
    df_CFA_raw.to_csv(os.path.join(DATA_DIR, "df_CFA_raw.csv"), index=False)





def kmo_bartlett(args) :
    # 전문가 지식으로 정제한 변수 목록은 그냥 바로 json 경로 뽑아서 읽기
    with open(args.expert_checked_var_list_dir, "r", encoding="utf-8") as f:
        fa_vars = json.load(f)

    print("KMO 결과가 일정 수치 이하인 변수 제거 작업은 스킵 : 이미 해당 과정과 함께 전문가 지식 도움 받아서 변수 정제함") # 나중에 필요하면 구현할 순 있을 것

    if len(fa_vars) < 3: # 요인 수 조건 검사
        raise ValueError(f"[skip] 요인분석 변수 수 부족: {fa_vars}")

    # 여긴 이제 형식상 있는 느낌.. 딱히 저장할 데이터도 없음 (나중에 코드 완성 시 작성하긴 해야 할 것)
    pass



def efa(args, DATA_DIR, RESULTS_DIR) :
    # 데이터 불러오기
    df_EFA = pd.read_csv(os.path.join(DATA_DIR, args.df_EFA_dir))

    with open(args.expert_checked_var_list_dir, "r", encoding="utf-8") as f: # 변수 목록 그대로 가져오기
        fa_vars = json.load(f)
    
    # Scree plot 그리기
    scree_plot(args, df_EFA, fa_vars, RESULTS_DIR)

    # 원랜 요인 수 정하는 과정이 있으나, 일단 스킵하고 자동진행(스킵여부 및 자동진행시 요인수는 config에서 조정)
    if args.skip_factor_num : 
        N_FACTORS = args.factor_num_at_skip
    else : 
        print("출력된 스크리 플롯을 보고 요인 수를 입력하시오 : ")
        N_FACTORS = int(input())
    
    X_imp = preprocess_for_fa(df_EFA, fa_vars)  # 혹시 모르니 숫자화/결측/inf 작업 등 수행

    # N_FACTORS 검증: 변수 수를 넘을 수 없음
    max_factors = min(X_imp.shape[1]-1, X_imp.shape[0]-1)  # 보수적으로 제한
    if N_FACTORS > max_factors:
        raise ValueError(f"N_FACTORS={N_FACTORS}가 허용치({max_factors})를 초과")
    
    # EFA 수행
    efa_task(args, X_imp, N_FACTORS, RESULTS_DIR)



def cfa(args, RESULTS_DIR) :
    # 데이터 불러오기
    EFA_comm_and_uniq = pd.read_csv(args.EFA_comm_uniq_dir, index_col=0)
    EFA_loadings = pd.read_csv(args.EFA_loadings_dir, index_col=0)
    N_FACTORS = args.factor_num

    # 결과 바탕으로 유효한 요인 필터링
    cfa_refined = refine_for_cfa(
    RESULTS_DIR=RESULTS_DIR,
    result=EFA_comm_and_uniq,
    loadings=EFA_loadings,
    n_factors=N_FACTORS, # 위에서 설정된 값으로 진행
    communality_thr=args.communality_thr,  # 공통성 기준
    loading_thr=args.loading_thr,  # 적재치 기준
    cross_loading_thr=args.cross_loading_thr  # 1위 요인과 2위 요인 loading 차이 기준
    )

    desc = make_cfa_description(cfa_refined["factor_groups"], RESULTS_DIR) # 자동으로 CFA 구조식(desc) 생성

    if args.skip_desc : # 자동 구조식 대신 전문가 지식 쓰려는 경우 (일단 우린 이렇게 할 것)
        with open(args.expert_desc, "r", encoding="utf-8") as f:
            desc = f.read()
    
    # semopy의 Model 사용하여 CFA 모델링 수행 (df_CFA도 로드)
    df_CFA = pd.read_csv(args.df_CFA_dir)
    stats = CFA_task(args, desc, df_CFA, RESULTS_DIR)
    CFA_visualization(args, stats, RESULTS_DIR) # 시각화







# outcome 변수들을 이용한 평가
def outcome_check(args, RESULTS_DIR, RANDOM_SEED) :
    
    # 변수 할당

    
    with open(args.expert_desc, "r", encoding="utf-8") as f:
        desc = f.read()

    with open(args.observed_cols, "r", encoding="utf-8") as f:
        observed_cols = json.load(f)
    
    continuous_outcomes = args.continuous_outcomes
    binary_outcomes = args.binary_outcomes
    covariates = args.covariates

    # 데이터
    df_CFA = pd.read_csv(args.train_data_dir)
    df_CFA_raw = pd.read_csv(args.binary_train_data_dir)
    df_EFA = pd.read_csv(args.test_data_dir) # 기회 되면 2018 데이터로 갈아탈 것!
    





    # ---- CFA 재수행 후 factor score 뽑기 ----
    model, est, stats, factor_scores_df = fit_cfa_get_stats_and_scores(
        df=df_CFA,
        desc=desc,
        observed_cols=observed_cols,
        fit_options={"disp": True},
        dropna_for_scores=True,
        save_est_path="cfa_estimates.csv",
        save_stats_path="cfa_fit_stats.csv",
    )

    # ---- factor score 만드는데 쓰인 weight matrix 추출 ----
    scoring_weights_df = extract_scoring_weights(model)

    # ---- 라벨링 통일 ----
    factor_scores_axes, scoring_weights_axes = relabel_scores_and_weights_as_axes(
        factor_scores_df=factor_scores_df,
        scoring_weights_df=scoring_weights_df,
        axis_prefix="Factor_"
    )

    # 이후 분석에서 factor_scores_df를 라벨 통일된 버전으로 사용
    factor_scores_df = factor_scores_axes

    # 저장
    factor_scores_axes.to_csv(os.path.join(RESULTS_DIR, "outcome_factor_scores.csv"), encoding="utf-8-sig")
    scoring_weights_axes.to_csv(os.path.join(RESULTS_DIR, "outcome_scoring_weights.csv"), encoding="utf-8-sig")

    # 확인용 top-weight 출력 (Axis 기준)
    for ax in scoring_weights_axes.index:
        top = scoring_weights_axes.loc[ax].abs().sort_values(ascending=False).head(3)
        print(f"[CHECK] {ax} top weights:", list(zip(top.index.tolist(), top.values.tolist())))






    # --------------------factor score과 outcome 변수를 한 df에 집어넣고 관리하기-----------------------------------
    # -> 단, binary outcome은 정규화로 0/1 값을 망치면 안되므로 raw 값으로 덮어쓰기
    # -> 다른 변수들이 군더더기로 붙어있다. 어차피 함수 안쪽에서 정제되긴 하지만, 그래도 겉에서 continuous outcome만 발라내도록 하자.
    # -> 겸사겸사 로직 변경 : 괜히 df_CFA 다 집어넣지 말고, df_fs에서 연속 outcome만 추출하고, 이산 outcome 및 factor score과 합쳐서 진행?


    df_fs_continuous = df_CFA[continuous_outcomes].copy()  # 연속형 outcome은 일단 z-score 표준화 된 데이터에서 추출
    df_fs_binary = df_CFA_raw[binary_outcomes].copy()  # 이산형 outcome은 raw로 추출

    df_fs = pd.concat([df_fs_continuous, df_fs_binary, factor_scores_df], axis=1, join="inner")  # factor score과 outcome 합치기 (연속형은 표준화된 값, 이산형은 raw 값 유지)
    print("각 df 크기 :", df_fs_continuous.shape, df_fs_binary.shape, factor_scores_df.shape)
    print("inner join 후 df_fs 크기:", df_fs.shape)






    # (선택) covariates도 raw로 유지하고 싶으면 같이 덮어쓰기
    # cov_missing = [c for c in covariates if c not in df_CFA_raw.columns]
    # if cov_missing:
    #     raise KeyError(f"Covariate columns missing in df_CFA_raw: {cov_missing}")
    # df_fs.loc[:, covariates] = df_CFA_raw.loc[df_fs.index, covariates].values






    factor_cols = list(factor_scores_df.columns)


    # t-test 정규성 검정
    shapiro_res = shapiro_by_binary_groups(
        df=df_fs,                   # 보통 t-test에 쓰는 df_fs에서 검사하는 게 자연스러움
        binary_outcomes=binary_outcomes,
        factor_cols=factor_cols,
        alpha=0.05,
        max_n=5000,
        random_state=RANDOM_SEED
    )

    shapiro_res.to_csv(os.path.join(RESULTS_DIR, "outcome_shapiro_results_by_group.csv"), index=False)
    print(shapiro_res.head(20))





    # ---- 연속 outcome 대상 OLS 적합 ----
    ols_results = run_ols_continuous_outcomes(
        df_fs,
        outcome_cols=continuous_outcomes,
        factor_cols=factor_cols,
        covariates=covariates,
        robust_se="HC3",
        standardize=True,   # 표준화 β로 보고 싶으면 True, raw 단위 해석이면 False
        fdr_method="fdr_bh"
    )
    ols_results.to_csv(os.path.join(RESULTS_DIR, "outcome_task1_ols_results.csv"), index=False)

    # ---- 이산 outcome 대상 t-test (표본수 이슈가 있을 수 있음) ----
    ttest_results = run_ttests_binary_outcomes_on_factors(
        df_fs,
        binary_outcome_cols=binary_outcomes,
        factor_cols=factor_cols,
        equal_var=False,
        fdr_method="fdr_bh"
    )
    ttest_results.to_csv(os.path.join(RESULTS_DIR, "outcome_task1_ttest_results.csv"), index=False)

    # ---- 이산 outcome 대상 MHU(Mann–Whitney U tests) ----
    mwu_results = run_mannwhitney_binary_outcomes_on_factors(
        df=df_fs,
        binary_outcome_cols=binary_outcomes,
        factor_cols=factor_cols,
        alternative="two-sided",
        fdr_method="fdr_bh"
    )

    mwu_results.to_csv(os.path.join(RESULTS_DIR, "outcome_task1_mwu_results.csv"), index=False)
    # print(mwu_results.head(30))



    # logistic regression : 이 버전은 결과가 이상해서 일단 보류하기로 했음
    # # ---- (E-1) Strategy 1: Holdout 7:3 (train fit / test AUC) ----
    # logit_holdout = run_logistic_binary_outcomes(
    #     df_fs,
    #     binary_outcome_cols=binary_outcomes,
    #     factor_cols=factor_cols,
    #     covariates=covariates,
    #     robust_se="HC3",
    #     standardize_predictors=True,
    #     fdr_method="fdr_bh",
    #     compute_auc=True,
    #     auc_strategy="holdout",
    #     test_size=0.3,
    #     bootstrap_n=1000,          # 무시됨
    #     random_state=RANDOM_SEED
    # )
    # logit_holdout.to_csv("aux_logit_results_holdout.csv", index=False)

    # # ---- (E-2) Strategy 2: Bootstrap AUC (OOB, 1000회) ----
    # logit_boot = run_logistic_binary_outcomes(
    #     df_fs,
    #     binary_outcome_cols=binary_outcomes,
    #     factor_cols=factor_cols,
    #     covariates=covariates,
    #     robust_se="HC3",
    #     standardize_predictors=True,
    #     fdr_method="fdr_bh",
    #     compute_auc=True,
    #     auc_strategy="bootstrap",
    #     test_size=0.3,             # 무시됨
    #     bootstrap_n=1000,
    #     random_state=RANDOM_SEED
    # )
    # logit_boot.to_csv("aux_logit_results_bootstrap.csv", index=False)

    # pass




    # 시각화
    
    visual_task1_ols(args, ols_results, RESULTS_DIR)
    visual_task1_ttest(args, ttest_results, RESULTS_DIR)
    visual_task1_mwu(args, mwu_results, RESULTS_DIR)


    # pearson, scatter, boxplot
    visual_task2_pearson_scatter_box(args, df_fs, RESULTS_DIR)



    # 특정 변수 대상으로 한 번의 regression 수행

    # =========================================================
    # 0) User config
    # =========================================================
    TARGET = args.regression_var  # 회귀분석 종속변수 (예: PSQI_sum_WA)
    N_SPLITS = args.N_SPLITS  # 교차검증 분할 수
    # RANDOM_SEED은 이미 인자로 받음


    scoring_weights_df = pd.read_csv(args.scoring_weights_dir, index_col=0)  # factor score 만드는 데 쓰인 가중치 행렬 (열: factor score, 행: 관측변수가 돼야 함)

    W_axes = scoring_weights_df.T  # 전치: rows=observed, cols=factors (여기선 observed_cols가 행, factor score가 열인 형태로 가정하므로 뒤집어야 함)

    # factor score column names (train features)
    # - factor_scores_df.columns 가 곧 factor 축 이름이라고 가정
    factor_cols = list(factor_scores_df.columns)





    
    # =========================================================
    # 2) Build TRAIN set (df_CFA)
    #    - features: factor_scores_df
    #    - target: df_CFA[TARGET]
    # =========================================================
    # target y_train
    y_all = df_CFA[TARGET].copy()

    # align indices (intersection) to be safe
    common_idx = factor_scores_df.index.intersection(y_all.dropna().index)
    X_all = factor_scores_df.loc[common_idx, factor_cols].copy()
    y_all = y_all.loc[common_idx].astype(float)

    # =========================================================
    # 3) 5-fold CV on TRAIN set
    # =========================================================
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all), start=1):
        X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
        y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]



        model = LinearRegression()
        model.fit(X_tr, y_tr)

        # predictions
        pred_tr = model.predict(X_tr)
        pred_va = model.predict(X_va)

        # metrics
        def _rmse(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred))

        fold_rows.append({
            "fold": fold,
            "n_train": int(len(y_tr)),
            "n_valid": int(len(y_va)),
            "r2_train": float(r2_score(y_tr, pred_tr)),
            "r2_valid": float(r2_score(y_va, pred_va)),
            "mae_train": float(mean_absolute_error(y_tr, pred_tr)),
            "mae_valid": float(mean_absolute_error(y_va, pred_va)),
            "rmse_train": float(_rmse(y_tr, pred_tr)),
            "rmse_valid": float(_rmse(y_va, pred_va)),
        })

    cv_df = pd.DataFrame(fold_rows)
    print("=== 5-fold CV (per fold) ===")
    print(cv_df)

    summary = cv_df.drop(columns=["fold", "n_train", "n_valid"]).agg(["mean", "std"]).T
    print("\n=== 5-fold CV (mean ± std) ===")
    print(summary)


    # =========================================================
    # 4) Build TEST set (df_EFA)
    #    - Create factor scores for df_EFA using scoring weights from df_CFA scope
    # =========================================================
    # train stats for standardization of observed indicators
    train_means = df_CFA[observed_cols].mean()
    train_stds  = df_CFA[observed_cols].std(ddof=0)

    # factor scores on df_EFA
    efa_factor_scores = make_factor_scores_from_weights(
        df_EFA,
        observed_cols=observed_cols,
        W_axes=W_axes, # 코드에선 observed_cols가 행, factor score가 열인 형태로 가정, 난 뒤집어서 썼으므로 여기서 다시 전치
        train_means=train_means,
        train_stds=train_stds,
    )

    # align test X/y
    y_test = df_EFA[TARGET].copy()
    test_idx = efa_factor_scores.index.intersection(y_test.dropna().index)

    X_test = efa_factor_scores.loc[test_idx, W_axes.columns].copy()
    y_test = y_test.loc[test_idx].astype(float)

    # =========================================================
    # 5) Fit FINAL model on full TRAIN, evaluate once on TEST
    # =========================================================
    final_model = LinearRegression()
    final_model.fit(X_all, y_all)

    # ✅ (추가) test 컬럼명을 train 컬럼명과 동일하게 강제 + 순서도 동일하게 (넘버링 형식 불일치 수정용)
    X_test = X_test.copy()
    X_test.columns = X_all.columns
    X_test = X_test.reindex(columns=X_all.columns)

    assert X_test.shape[1] == X_all.shape[1], "Train/Test factor count mismatch"

    pred_test = final_model.predict(X_test)


    test_metrics = {
        "n_train_total": int(len(y_all)),
        "n_test": int(len(y_test)),
        "r2_test": float(r2_score(y_test, pred_test)),
        "mae_test": float(mean_absolute_error(y_test, pred_test)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred_test))),
    }
    print("\n=== TEST (df_EFA) evaluation ===")
    print(pd.Series(test_metrics))

    # (optional) coefficients
    coef_df = pd.DataFrame({
        "factor": X_all.columns,
        "coef": final_model.coef_,
    }).sort_values("coef", key=lambda s: s.abs(), ascending=False)
    print("\n=== Final model coefficients (abs-sorted) ===")
    print(coef_df)


    # =====================================================
    #  Save CSVs
    # =====================================================
    cv_df.to_csv(os.path.join(RESULTS_DIR, "cv_fold_results.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "cv_summary_mean_std.csv"))                  # index: metric, columns: mean/std
    pd.Series(test_metrics).to_csv(os.path.join(RESULTS_DIR, "cv_test_metrics.csv"), header=False)
    coef_df.to_csv(os.path.join(RESULTS_DIR, "cv_final_model_coefs.csv"), index=False)