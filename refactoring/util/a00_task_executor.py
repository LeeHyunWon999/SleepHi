from util.packages import *
from util.a1_preprocess import *
from util.a2_kmo import *
from util.a3_EFA import *
from util.a4_CFA import *
from util.a5_outcome import *
from util.a6_visual_primal_var_check import *
from util.a7_regression_temp import *
from util.a8_regression_main import *


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

    # task6에 쓰기 위해 범주/연속 변수 리스트 저장 (보류 : 이진이 아닌 ordinal은 이 분류에서 예외라 수작업으로 해야 할 듯)
    with open(os.path.join(DATA_DIR, "categorical_vars.json"), "w", encoding="utf-8") as f:
        json.dump(categorical_vars, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_DIR, "continuous_vars.json"), "w", encoding="utf-8") as f:
        json.dump(continuous_vars, f, ensure_ascii=False, indent=2)


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

    # # EFA 투입 열 목록 (숫자형만)
    # efa_cols = report["efa_cols"]
    # print("\n[EFA 열 개수]:", len(efa_cols))
    # # 이제 바로 EFA:
    # # X_efa = df[efa_cols].copy()
    # # print("EFA 입력 크기:", X_efa.shape)



    # 시간 값 평균 취한 후 상대적 차이로 변환 (예: 수면 시작 시간은 평균보다 몇 분 늦거나 빠른지)
    df = apply_relative_to_mean(df, meta)


    # 표준화 없는 데이터 미리 분리
    df_raw = df.copy()





    # (추가) 표준화/시간변환 전에 분포 요약 + plot 2장 저장
    if getattr(args, "df_summarize_opt", False): # config에서 켜는 옵션
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





    # (2018 데이터 분리용) 이대로 저장
    if getattr(args, "data_2018_make_option", False): # 데이터 분리 전용 config 파일에만 있는 옵션
        df.to_csv(os.path.join(DATA_DIR, "df_2018.csv"), index=False, encoding="utf-8-sig")
        df_raw.to_csv(os.path.join(DATA_DIR, "df_2018_raw.csv"), index=False, encoding="utf-8-sig")
        print("2018 데이터 저장 완료. 작업 종료.")
        sys.exit(0)

    # (2011 데이터 분리하기 전에 저장용)
    if getattr(args, "data_2011_make_option", False): # 데이터 분리 전용 config 파일에만 있는 옵션
        df.to_csv(os.path.join(DATA_DIR, "df_2011.csv"), index=False, encoding="utf-8-sig")
        df_raw.to_csv(os.path.join(DATA_DIR, "df_2011_raw.csv"), index=False, encoding="utf-8-sig")
        print("2011 데이터 저장 완료. 작업 종료.")
        sys.exit(0)


    # # (추가) 2011년 데이터에서 Q51과 Q91의 다름을 확인하기 위한 간단한 검정 테스트 (df_raw로 보는게 나을 듯) (ordinal 값의 비교는 wilcoxon signed-rank test로 진행)
    # if "Q51_slp_sufficient" in df_raw.columns and "Q91_ISI_4" in df_raw.columns:

    #     paired = df_raw[["Q51_slp_sufficient", "Q91_ISI_4"]].dropna()

    #     wilcoxon_test = wilcoxon(paired["Q51_slp_sufficient"], paired["Q91_ISI_4"], alternative="two-sided")
    #     print("\n[2011 데이터 Q51 vs Q91 Wilcoxon signed-rank test]")
    #     print(f"  V-statistic: {wilcoxon_test.statistic}, p-value: {wilcoxon_test.pvalue}")

    #     # median 값을 찍어야 V-stastic이 의미가 있다 함
    #     print((paired["Q51_slp_sufficient"] - paired["Q91_ISI_4"]).median())

    #     # 이번엔 경향 차이 확인 위한 spearman
    #     spearman_test = spearmanr(paired["Q51_slp_sufficient"], paired["Q91_ISI_4"])
    #     print("\n[2011 데이터 Q51 vs Q91 Spearman correlation]")
    #     print(f"  rho: {spearman_test.correlation}, p-value: {spearman_test.pvalue}")   


    #     # 이번엔 기울기 보기용 
    #     X = paired["Q51_slp_sufficient"]
    #     Y = paired["Q91_ISI_4"]

    #     X = sm.add_constant(X)
    #     model = sm.OLS(Y, X).fit()

    #     print(model.summary())

    #     # boxplot으로 분포 시각화

    #     plt.figure(figsize=(6,5))

    #     plt.boxplot(
    #         [paired["Q51_slp_sufficient"], paired["Q91_ISI_4"]],
    #         labels=["Q51_slp_sufficient", "Q91_ISI_4"]
    #     )

    #     plt.ylabel("Score")
    #     plt.title("Distribution Comparison: Q51 vs Q91")
    #     # 저장
    #     plt.savefig(os.path.join(RESULTS_DIR, "temp_Q51_vs_Q91_boxplot.png"))


    #     # 추세선 들어간 contingency heatmap
    #     table = pd.crosstab(
    #         paired["Q51_slp_sufficient"],
    #         paired["Q91_ISI_4"]
    #     )
    #     table = table.sort_index(ascending=False)
    #     plt.figure(figsize=(6,5))
    #     sns.heatmap(
    #         table,
    #         annot=True,
    #         fmt="d",
    #         cmap="Blues"
    #     )

    #     plt.xlabel("Q91_ISI_4")
    #     plt.ylabel("Q51_slp_sufficient")
    #     plt.title("Contingency Heatmap")

    #     # 저장
    #     plt.savefig(os.path.join(RESULTS_DIR, "temp_Q51_vs_Q91_contingency_heatmap.png"))

    # else : 
    #     print("검정 대상 변수 없음")
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
    # -> 겸사겸사 로직 변경 : 괜히 df_CFA 다 집어넣지 말고, df_fs에서 연속 outcome만 추출하고, 이진 outcome 및 factor score과 합쳐서 진행?


    df_fs_continuous = df_CFA[continuous_outcomes].copy()  # 연속형 outcome은 일단 z-score 표준화 된 데이터에서 추출
    df_fs_binary = df_CFA_raw[binary_outcomes].copy()  # 이진형 outcome은 raw로 추출

    df_fs = pd.concat([df_fs_continuous, df_fs_binary, factor_scores_df], axis=1, join="inner")  # factor score과 outcome 합치기 (연속형은 표준화된 값, 이진형은 raw 값 유지)
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

    # ---- 이진 outcome 대상 t-test (표본수 이슈가 있을 수 있음) ----
    ttest_results = run_ttests_binary_outcomes_on_factors(
        df_fs,
        binary_outcome_cols=binary_outcomes,
        factor_cols=factor_cols,
        equal_var=False,
        fdr_method="fdr_bh"
    )
    ttest_results.to_csv(os.path.join(RESULTS_DIR, "outcome_task1_ttest_results.csv"), index=False)

    # ---- 이진 outcome 대상 MHU(Mann–Whitney U tests) ----
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








# 번외 0_1 : EFA, CFA 데이터셋 outcome 변수만 추리도록 작업하기 -> 2018 데이터에서 EFA, CFA 다시 나누기
# TODO : PSQI 변수가 2018 데이터셋에서 발견되지 않았음, 일단 그 변수 빼고 나머지 3개 변수(Q91_ISI_4 - dissatisfation, poor_sleeper)에 대해서만 작업할 것
# 정확히는 저 변수들은 종속변수(y)임,
def data_rework(args, RESULTSDIR, RANDOM_SEED, DATA_DIR) :

    df_EFA = pd.read_csv(args.df_EFA_dir)
    df_EFA_raw = pd.read_csv(args.df_EFA_raw_dir)
    df_CFA = pd.read_csv(args.df_CFA_dir)
    df_CFA_raw = pd.read_csv(args.df_CFA_raw_dir)


    # 2011, 2018 데이터 각각 통합 : 성별만 raw에서 가져오고 나머지 10개 변수는 정규화 한 쪽으로 진행
    
    # 리스트 열기
    with open(args.expert_desc_list_ver_dir, "r", encoding="utf-8") as f:
        factor_vars_list = json.load(f)



    # 1. vars_list에서 '성별'을 제외한 변수들만 필터링
    vars_except_gender = [col for col in factor_vars_list if col != "sex"]

    # 2. 데이터프레임 결합
    # temp_train_df에서는 '성별'을 제외한 나머지 변수를, 
    # temp_train_df_raw에서는 '성별' 변수 하나만 가져와 옆으로(axis=1) 붙이기

    refined_df_EFA = pd.concat([
        df_EFA[vars_except_gender], 
        df_EFA_raw[["sex"]]
    ], axis=1)

    refined_df_CFA = pd.concat([
        df_CFA[vars_except_gender], 
        df_CFA_raw[["sex"]]
    ], axis=1)





    # 3. 결과 확인
    print(refined_df_EFA.columns) # '성별'이 리스트의 맨 뒤나 지정한 위치에 포함되었는지 확인
    print(refined_df_EFA["sex"].tail()) # 값이 1, 2(또는 0, 1)로 잘 나오는지 확인
    print(refined_df_CFA.columns) # '성별'이 리스트의 맨 뒤나 지정한 위치에 포함되었는지 확인
    print(refined_df_CFA["sex"].tail()) # 값이 1, 2(또는 0, 1)로 잘 나오는지 확인




    # 4. 연속 & 이진 outcome 생성 (이것도 2개 년도 각각 필요)

    # 연속 : 표준화된 걸로 가져오기
    refined_outcome_EFA_conti = df_EFA[["Q91_ISI_4", "PSQI_sum_WA"]] # 나중에 2018 데이터에 PSQI 총점 생기면 여기에 추가 : 어차피 이 메소드는 데이터만들기 1회용이고 안에서 직접 작업하는 부분이 있으니 하드코딩 무방
    refined_outcome_CFA_conti = df_CFA[["Q91_ISI_4", "PSQI_sum_WA"]] # 나중에 2018 데이터에 PSQI 총점 생기면 여기에 추가 : 어차피 이 메소드는 데이터만들기 1회용이고 안에서 직접 작업하는 부분이 있으니 하드코딩 무방


    # 이진 : dissatisfaction은 Q91_ISI_4(연속값)에서 직접 작업하기, 이후 poor_sleeper과 합치면 끝

    # 1. 조건에 따라 이진 변수 생성
    # 1, 2점은 0(정상), 3~5점은 1(수면불만족)로 할당
    # np.where(조건, 참일 때 값, 거짓일 때 값)

    isi_binary_EFA = np.where(df_EFA_raw["Q91_ISI_4"] <= 2, 0, 1)
    isi_binary_CFA = np.where(df_CFA_raw["Q91_ISI_4"] <= 2, 0, 1)

    # 2. 새로운 데이터프레임 구성 (총 2열)
    # 새 변수명을 'isi_status' 등으로 지정하여 poor_sleeper와 함께 추출

    refined_outcome_EFA_binary = pd.DataFrame({
        "dissatisfaction": isi_binary_EFA,
        "poor_sleeper": df_EFA_raw["poor_sleeper"]
    }, index=df_EFA_raw.index) # 기존 인덱스 유지

    refined_outcome_CFA_binary = pd.DataFrame({
        "dissatisfaction": isi_binary_CFA,
        "poor_sleeper": df_CFA_raw["poor_sleeper"]
    }, index=df_CFA_raw.index) # 기존 인덱스 유지



    # 번외 : 성별은 현재 1, 2로 되어 있음, 이것도 남성 0, 여성 1로 변경하기
    print(refined_df_EFA['sex'].value_counts())
    refined_df_EFA['sex'] = refined_df_EFA['sex'].replace({1:0, 2:1})
    print(refined_df_EFA['sex'].value_counts()) # 혹시 모르니 변경 전후 남녀 수 출력

    print(refined_df_CFA['sex'].value_counts())
    refined_df_CFA['sex'] = refined_df_CFA['sex'].replace({1:0, 2:1})
    print(refined_df_CFA['sex'].value_counts()) # 혹시 모르니 변경 전후 남녀 수 출력

    # sys.exit(0)


    # 3. 결과 확인
    print(refined_outcome_EFA_binary.head())
    print(refined_outcome_EFA_binary.value_counts()) # 각 조합별 빈도 확인
    print(refined_outcome_EFA_conti.head())
    print(refined_df_EFA.head())

    print(refined_outcome_CFA_binary.head())
    print(refined_outcome_CFA_binary.value_counts()) # 각 조합별 빈도 확인
    print(refined_outcome_CFA_conti.head())
    print(refined_df_CFA.head())





    
    # 저장 대상 : 2011, 2018 둘 다 factor score 만들기용 데이터(11개, 그중 성별만 raw에서 가져오기), 연속 outcome(Q91_ISI_4), 이진 outcome(dissatisfaction, poor_sleeper) (전체 6개)
    refined_df_EFA.to_csv(os.path.join(DATA_DIR, "refined_df_EFA.csv"), index=False)
    refined_outcome_EFA_conti.to_csv(os.path.join(DATA_DIR, "refined_outcome_EFA_conti.csv"), index=False)
    refined_outcome_EFA_binary.to_csv(os.path.join(DATA_DIR, "refined_outcome_EFA_binary.csv"), index=False)

    refined_df_CFA.to_csv(os.path.join(DATA_DIR, "refined_df_CFA.csv"), index=False)
    refined_outcome_CFA_conti.to_csv(os.path.join(DATA_DIR, "refined_outcome_CFA_conti.csv"), index=False)
    refined_outcome_CFA_binary.to_csv(os.path.join(DATA_DIR, "refined_outcome_CFA_binary.csv"), index=False)











# # 번외 0 : 2011, 2018 데이터셋 outcome 변수만 추리도록 작업하기
# # TODO : PSQI 변수가 2018 데이터셋에서 발견되지 않았음, 일단 그 변수 빼고 나머지 3개 변수(Q91_ISI_4 - dissatisfation, poor_sleeper)에 대해서만 작업할 것
# # 정확히는 저 변수들은 종속변수(y)임,
# def data_rework(args, RESULTSDIR, RANDOM_SEED, DATA_DIR) :
#     # 2011, 2018 데이터의 정규화, raw 총 4개 불러오기
#     df_2011 = pd.read_csv(args.df_2011_dir)
#     df_2011_raw = pd.read_csv(args.df_2011_raw_dir)
#     df_2018 = pd.read_csv(args.df_2018_dir)
#     df_2018_raw = pd.read_csv(args.df_2018_raw_dir)



#     # 2011, 2018 데이터 각각 통합 : 성별만 raw에서 가져오고 나머지 10개 변수는 정규화 한 쪽으로 진행
    
#     # 리스트 열기
#     with open(args.expert_desc_list_ver_dir, "r", encoding="utf-8") as f:
#         factor_vars_list = json.load(f)



#     # 1. vars_list에서 '성별'을 제외한 변수들만 필터링
#     vars_except_gender = [col for col in factor_vars_list if col != "sex"]

#     # 2. 데이터프레임 결합
#     # temp_train_df에서는 '성별'을 제외한 나머지 변수를, 
#     # temp_train_df_raw에서는 '성별' 변수 하나만 가져와 옆으로(axis=1) 붙이기
# #    refined_df_2011 =  pd.concat([
# #        df_2011[vars_except_gender], 
# #        df_2011_raw[["sex"]]
# #    ], axis=1)

#     refined_df_2018 = pd.concat([
#         df_2018[vars_except_gender], 
#         df_2018_raw[["sex"]]
#     ], axis=1)

#     # 3. 결과 확인
# #    print(refined_df_2011.columns) # '성별'이 리스트의 맨 뒤나 지정한 위치에 포함되었는지 확인
# #    print(refined_df_2011["sex"].tail()) # 값이 1, 2(또는 0, 1)로 잘 나오는지 확인
#     print(refined_df_2018.columns) # '성별'이 리스트의 맨 뒤나 지정한 위치에 포함되었는지 확인
#     print(refined_df_2018["sex"].tail()) # 값이 1, 2(또는 0, 1)로 잘 나오는지 확인




#     # 4. 연속 & 이진 outcome 생성 (이것도 2개 년도 각각 필요)

#     # 연속 : 표준화된 걸로 가져오기
# #    refined_outcome_2011_conti = df_2011[["Q33_ISI_4", "PSQI_sum_WA"]]
#     refined_outcome_2018_conti = df_2018[["Q91_ISI_4", "PSQI_sum_WA"]] # 나중에 2018 데이터에 PSQI 총점 생기면 여기에 추가 : 어차피 이 메소드는 데이터만들기 1회용이고 안에서 직접 작업하는 부분이 있으니 하드코딩 무방



#     # 이진 : dissatisfaction은 Q91_ISI_4(연속값)에서 직접 작업하기, 이후 poor_sleeper과 합치면 끝

#     # 1. 조건에 따라 이진 변수 생성
#     # 1, 2점은 0(정상), 3~5점은 1(수면불만족)로 할당
#     # np.where(조건, 참일 때 값, 거짓일 때 값)
# #    isi_binary_2011 = np.where(df_2011_raw["Q33_ISI_4"] <= 2, 0, 1)
#     isi_binary_2018 = np.where(df_2018_raw["Q91_ISI_4"] <= 2, 0, 1)

#     # 2. 새로운 데이터프레임 구성 (총 2열)
#     # 새 변수명을 'isi_status' 등으로 지정하여 poor_sleeper와 함께 추출
# #    refined_outcome_2011_binary = pd.DataFrame({
# #        "dissatisfaction": isi_binary_2011,
# #        "poor_sleeper": df_2011_raw["poor_sleeper"]
# #    }, index=df_2011_raw.index) # 기존 인덱스 유지

#     refined_outcome_2018_binary = pd.DataFrame({
#         "dissatisfaction": isi_binary_2018,
#         "poor_sleeper": df_2018_raw["poor_sleeper"]
#     }, index=df_2018_raw.index) # 기존 인덱스 유지

#     # 번외 : 성별은 현재 1, 2로 되어 있음, 이것도 남성 0, 여성 1로 변경하기
#     print(refined_df_2018['sex'].value_counts())
#     refined_df_2018['sex'] = refined_df_2018['sex'].replace({1:0, 2:1})
#     print(refined_df_2018['sex'].value_counts()) # 혹시 모르니 변경 전후 남녀 수 출력
#     # sys.exit(0)

#     # 3. 결과 확인
# #    print(refined_outcome_2011_binary.head())
# #    print(refined_outcome_2011_binary.value_counts()) # 각 조합별 빈도 확인
#     print(refined_outcome_2018_binary.head())
#     print(refined_outcome_2018_binary.value_counts()) # 각 조합별 빈도 확인
#     print(refined_outcome_2018_conti.head())
#     print(refined_df_2018.head())

    
#     # 저장 대상 : 2011, 2018 둘 다 factor score 만들기용 데이터(11개, 그중 성별만 raw에서 가져오기), 연속 outcome(Q91_ISI_4), 이진 outcome(dissatisfaction, poor_sleeper) (전체 6개)
#     refined_df_2018.to_csv(os.path.join(DATA_DIR, "refined_df_2018.csv"), index=False)
#     refined_outcome_2018_conti.to_csv(os.path.join(DATA_DIR, "refined_outcome_2018_conti.csv"), index=False)
#     refined_outcome_2018_binary.to_csv(os.path.join(DATA_DIR, "refined_outcome_2018_binary.csv"), index=False)



# 번외 1 : 정제 끝난 데이터로 임시 outcome test
def primal_var_check(args, RESULTS_DIR, RANDOM_SEED) :

    # 독립변수(x값, 근데 이제 전체 파일이라 y값도 있는)
    temp_train_df = pd.read_csv(args.train_data_dir)
    temp_test_df = pd.read_csv(args.test_data_dir)
    temp_train_df_raw = pd.read_csv(args.train_data_raw_dir)
    temp_test_df_raw = pd.read_csv(args.test_data_raw_dir)

    # 이진변수 구분용
    with open(args.expert_checked_var_list_dir, "r", encoding="utf-8") as f:
        expert_checked_vars = json.load(f)

    with open(args.manual_binary_vars_dir, "r", encoding="utf-8") as f:
        manual_binary_vars = json.load(f)

    # 연속변수는 전문가 정제 변수에서 이진변수를 뺀 변수들로 구성
    # expert_checked_vars 요소 중 manual_binary_vars에 없는 것만 추출
    continuous_vars = [v for v in expert_checked_vars if v not in manual_binary_vars]


    # KMO 및 전문가 지식으로 정제된 변수들 전체가 대상임, 여기선 이진 변수도 표준화되어 있으므로 이걸 다시 결합해야 함

    # 1. 독립변수(X) 구성: 연속형 변수는 표준화 데이터에서, 이진 변수는 원본 데이터에서 추출

    # 이건 유기 : 자동화 말고 수작업으로 해야 함
    # with open(args.continuous_vars_list, "r", encoding="utf-8") as f: # 연속, 이진 목록 불러오기
    #     conti_vars_list = json.load(f)

    # with open(args.categorical_vars_list, "r", encoding="utf-8") as f:
    #     binary_vars_list = json.load(f)

    available_conti = [col for col in continuous_vars if col in temp_train_df.columns]      # 연속/이진 변수리스트는 정제 전에 뽑은거라 데이터에 없을 수도 있어서 데이터에 존재하는 변수인 경우만 필터링
    available_binary = [col for col in manual_binary_vars if col in temp_train_df.columns]    # train, test 모두 단일 데이터셋에서 분할하고 추가 처리가 없어서 변수 리스트는 한번만 생성해도 됨

    # print(available_conti)
    # print(available_binary)

    # sys.exit(0)

    X_train = pd.concat([
        temp_train_df[available_conti], 
        temp_train_df_raw[available_binary]
    ], axis=1)
    
    X_test = pd.concat([
        temp_test_df[available_conti], 
        temp_test_df_raw[available_binary]
    ], axis=1)

    # 갯수 맞나 확인 (2011 데이터 기준 train 1182, test 1181)
    x_temp = pd.concat([temp_train_df[available_conti], temp_train_df_raw[available_binary]], axis=1, join="inner")
    print(x_temp.shape)

    # 2. 인덱스 정렬 확인 (선택 사항)
    # 두 데이터프레임의 행 순서가 동일하다는 전제하에 결합하지만, 안전을 위해 인덱스 재설정 가능
    # X_train = X_train.reset_index(drop=True)
    # X_test = X_test.reset_index(drop=True)

    # 3. 결합 결과 확인
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(X_train.head())
    print(X_test.head())

    

    # 데이터셋 저장 : 여긴 임시라 필요없을듯?


    # 라벨 데이터셋 생성
    Y_train = temp_train_df[[args.regression_var]]
    Y_test = temp_test_df[[args.regression_var]]

    # 기초적인 linear regression 수행
    temp_linear_regression(args, X_train, X_test, Y_train, Y_test, RESULTS_DIR, RANDOM_SEED)






# 아래 outcome 작업에서 factor score 데이터 뽑기용
def make_factor_score_data(df, desc) : 

    # ---- CFA 재수행 후 factor score 뽑기 ----
    model, est, stats, factor_scores_df = fit_cfa_get_stats_and_scores(
        df=df,
        desc=desc,
        observed_cols=None, # None으로 주면 변수 리스트 체크 넘어가고 desc에 적힌 변수 알아서 골라 씀
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

    return factor_scores_df



# factor score을 이용해 특정 outcome을 대상으로 regression 수행
def regression_test(args, RESULTS_DIR, RANDOM_SEED) :
    # EFA(train & valid), CFA(test) 각각 요인구조 형성에 쓸 데이터, outcome 데이터(conti/binary) 가져오기
    train_valid_data = pd.read_csv(args.refined_df_EFA_dir)
    train_valid_outcome_conti = pd.read_csv(args.refined_outcome_EFA_conti_dir)
    train_valid_outcome_binary = pd.read_csv(args.refined_outcome_EFA_binary_dir)

    test_data = pd.read_csv(args.refined_df_CFA_dir)
    test_outcome_conti = pd.read_csv(args.refined_outcome_CFA_conti_dir)
    test_outcome_binary = pd.read_csv(args.refined_outcome_CFA_binary_dir)

    # outcome의 column 명 리스트로 미리 가져오기
    outcome_conti_list = args.regression_var_conti
    outcome_binary_list = args.regression_var_binary

    # 전문가 요인구조 가져오기
    with open(args.expert_desc, "r", encoding="utf-8") as f:
        desc = f.read()

    # factor score 데이터셋 만들기 : EFA, CFA 각각 데이터에 대해 1번씩만 하고 돌려쓰면 된다.
    train_valid_factor_score_data = make_factor_score_data(train_valid_data, desc)
    print(train_valid_factor_score_data.head())
    print(train_valid_factor_score_data.shape)

    test_factor_score_data = make_factor_score_data(test_data, desc)
    print(test_factor_score_data.head())
    print(test_factor_score_data.shape)









    # tree 기반 방법 수행 (어차피 둘 다 트리 기반 방법은 동일하다는듯)



    ## regression tree + 앙상블 : 연속형 변수 대상

    ## classification tree + 앙상블 : 범주형 변수 대상






    ############################################################

    # 일반 Decision Tree 및 xgboost로 regression, classification (tree 기반)
    # 입력은 연속/범주 다 들어감, outcome 종류따라 달라짐

    # 리니어리그레션, 로지스틱 리그레션

    # 앙상블 횟수 등은 default로 진행





    random.seed(RANDOM_SEED) # 파이썬 자체 random 시드 고정
    np.random.seed(RANDOM_SEED) # 넘파이 연산 시드 고정
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED) # 해시 함수 결과 고정


    # --- 실제 실행 프로세스 ---

    # 1. 객체 생성
    analyst = MultiOutcomeAnalyst(randNum=RANDOM_SEED, n_splits=5)

    # 2. 연속형 Outcome들에 대해 실행 (Linear, DT, XGB Regressors)
    analyst.run_experiment(
        train_valid_factor_score_data, test_factor_score_data,
        train_valid_outcome_conti, test_outcome_conti, 
        is_regression=True
    )

    # 3. 이산형 Outcome들에 대해 실행 (Logistic, DT, XGB Classifiers)
    analyst.run_experiment(
        train_valid_factor_score_data, test_factor_score_data,
        train_valid_outcome_binary, test_outcome_binary, 
        is_regression=False
    )

    # 4. 결과 출력 및 저장
    report_df = analyst.get_final_report()

    # 회귀 결과만 보기 (NaN인 분류 지표 컬럼 제거)
    regression_report = report_df[report_df['Type'] == 'Regression'].dropna(axis=1, how='all')

    # 분류 결과만 보기 (NaN인 회귀 지표 컬럼 제거)
    classification_report = report_df[report_df['Type'] == 'Classification'].dropna(axis=1, how='all')

    print(regression_report)
    print(classification_report)

    # 저장
    regression_report.to_csv(os.path.join(RESULTS_DIR,"final_regression_report.csv"), index=False)
    classification_report.to_csv(os.path.join(RESULTS_DIR,"final_classification_report.csv"), index=False)


