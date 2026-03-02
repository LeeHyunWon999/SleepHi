from util.packages import *


def refine_for_cfa(
    result,
    loadings,
    n_factors,
    communality_thr=0.5,
    loading_thr=0.5,
    cross_loading_thr=0.2  # 1순위와 2순위 loading 차이 기준
    ### 여기 값들은 기본설정 값이므로 직접 지정하는건 밑에서 할 것!
):
    df_comm = result.copy()
    df_load = loadings.copy()

    # 1️⃣ 공통성 기준
    drop_comm = df_comm[df_comm["Communality"] <= communality_thr].index.tolist()

    # 2️⃣ 최대 적재량 기준
    abs_all = df_load.abs()
    max_load = abs_all.max(axis=1)
    drop_low = max_load[max_load <= loading_thr].index.tolist()

    # 3️⃣ 1차 제거 대상 (공통성 + 최대 적재량)
    base_drop_vars = set(drop_comm + drop_low)
    print(f"[제거] 공통성≤{communality_thr} 또는 최대적재량≤{loading_thr} 변수 {len(base_drop_vars)}개:")
    print(base_drop_vars, "\n")

    # 4️⃣ 1차 필터링 후 데이터프레임
    keep_vars = [v for v in df_load.index if v not in base_drop_vars]
    abs_load = abs_all.loc[keep_vars]

    # 5️⃣ 요인별 변수 구조화 + cross-loading 처리
    factor_map = {f"Factor{i}": [] for i in range(1, n_factors + 1)}
    drop_cross = []  # cross-loading 기준 미만으로 제거된 변수들

    for var in keep_vars:
        row = abs_load.loc[var]

        # loading_thr 이상인 요인들만 "강한 적재"로 간주
        strong_mask = row >= loading_thr
        strong_factors = row.index[strong_mask].tolist()

        # 강하게 적재된 요인이 없으면 (이론상 drop_low에서 이미 제거되었으므로 거의 안 나올 것)
        if len(strong_factors) == 0:
            continue

        # ① 단일 요인에만 강하게 적재 → 그대로 그 요인에 배정
        if len(strong_factors) == 1:
            factor_map[strong_factors[0]].append(var)

        # ② 둘 이상의 요인에 강하게 적재된 경우 (cross-loading)
        else:
            # strong_factors 중에서만 정렬 (loading 내림차순)
            sorted_strong = row[strong_factors].sort_values(ascending=False)
            top1_factor, top1_val = sorted_strong.index[0], sorted_strong.iloc[0]
            top2_val = sorted_strong.iloc[1]  # 2순위 값

            diff = top1_val - top2_val

            # 차이가 cross_loading_thr 이상이면 → 1순위 요인에만 배정
            if diff >= cross_loading_thr:
                factor_map[top1_factor].append(var)
            else:
                # 어떤 요인에도 배정하지 않고 CFA에서 제거
                drop_cross.append(var)

    # 6️⃣ 최종 제거 변수 집합
    drop_vars = sorted(base_drop_vars | set(drop_cross))

    print(f"[제거] 교차적재(cross-loading) 기준 미만(Δ<{cross_loading_thr})으로 제거된 변수 {len(drop_cross)}개:")
    if drop_cross:
        print(drop_cross)
    print()

    print(f"✅ 요인별 변수 구조 (cross_loading_thr={cross_loading_thr}, loading_thr={loading_thr})")
    for fname, members in factor_map.items():
        print(f" - {fname} ({len(members)}개): {members}")

    # 7️⃣ 최종 loadings_filtered (모든 drop_vars 제거 후)
    df_load_filtered = df_load.drop(index=drop_vars)

    # 8️⃣ 가로형 CSV 저장
    factor_names = [f"Factor{i}" for i in range(1, n_factors + 1)]
    factor_lists = [factor_map[f] for f in factor_names]
    df_factors_wide = pd.DataFrame(zip_longest(*factor_lists), columns=factor_names)
    global RESULTS_DIR
    df_factors_wide.to_csv(os.path.join(RESULTS_DIR, "CFA_refined_factor_assignment.csv"), index=False, encoding="utf-8-sig")

    return {
        "loadings_filtered": df_load_filtered,
        "drop_vars": drop_vars,
        "factor_groups": factor_map
    }

# ----------------------------------------
# CFA 구조식(desc) 자동 생성 (+ 요인분산=1 고정 포함)
# ----------------------------------------
def make_cfa_description(factor_map):
    """
    factor_map: {'Factor1': ['a','b'], 'Factor2': ['c','d']} 형태
    반환: lavaan-style formula string
    """
    lines = []
    
    # (1) 요인 적재식 생성
    for factor, vars_ in factor_map.items():
        if len(vars_) == 0:
            continue
        joined = " + ".join(vars_)
        line = f"{factor} =~ {joined}"
        lines.append(line)
    
    # (2) 요인분산=1 고정 구문 자동 추가
    for factor in factor_map.keys():
        lines.append(f"{factor} ~~ 1*{factor}")
    
    # (3) 하나의 문자열로 결합
    desc = "\n".join(lines)
    
    # (4) 결과 확인용 출력
    print("✅ CFA용 요인구조(desc):\n")
    print(desc)

    # 저장
    global RESULTS_DIR
    with open(os.path.join(RESULTS_DIR, "cfa_description.txt"), "w", encoding="utf-8") as f:
        f.write(desc)
    
    return desc



# CFA 수행
def CFA_task(args, desc, df_CFA) : 
    # 1. 모델 정의
    # desc 변수는 이전 단계에서 make_cfa_description 함수로 생성했다고 가정합니다.
    # 예시: desc = "Factor1 =~ Var1 + Var2\nFactor2 =~ Var3 + Var4"
    model = Model(desc)

    # 2. 모델 학습 (핵심 수정 부분)
    # options={'disp': True}를 추가하여 최적화 과정의 상세 정보를 출력합니다.
    # 각 반복(iteration)마다 현재 상태, 함수 값 등의 정보가 화면에 나타납니다.
    print("--- CFA 모델 최적화를 시작합니다 (진행 상황 출력) ---")
    model.fit(df_CFA, options={'disp': True})
    print("--- CFA 모델 최적화가 완료되었습니다 ---\n")

    # 3. 요인 부하량, 적합도 지표 확인 (기존과 동일)
    print("--- 모델 분석 결과 ---")
    est = model.inspect()
    print(est)
    # 결과를 파일로 저장
    try:
        est.to_csv(os.path.join(RESULTS_DIR, "CFA_main_estimates.csv"), index=False, encoding="utf-8-sig")
        print("\n'CFA_main_estimates.csv' 파일로 결과가 저장되었습니다.")
    except Exception as e:
        print(f"\n파일 저장 중 오류 발생: {e}")


    print("\n--- 모델 적합도 지표 ---")
    stats = calc_stats(model)
    print(stats)
    # 결과를 파일로 저장
    try:
        stats.to_csv(os.path.join(RESULTS_DIR, "CFA_main_fit_stats.csv"), index=False, encoding="utf-8-sig")
        print("\n'CFA_main_fit_stats.csv' 파일로 결과가 저장되었습니다.")
    except Exception as e:
        print(f"\n파일 저장 중 오류 발생: {e}")
    
    return stats

# CFA 적합도 결과 시각화
def CFA_visualization(args, stats) : 
    fit_indices = stats.loc["Value", ["CFI", "TLI", "GFI", "AGFI", "RMSEA"]]
    fit_indices = fit_indices.reset_index()
    fit_indices.columns = ["Metric", "Value"]

    plt.figure(figsize=(7,4))
    sns.barplot(data=fit_indices, x="Metric", y="Value")
    plt.axhline(0.9, color="green", linestyle="--", label="RMSEA 제외 기준 (0.9)")
    plt.axhline(0.08, color="orange", linestyle="--", label="RMSEA 기준 (0.08)")
    plt.title("CFA Fit Indices")
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    global RESULTS_DIR
    plt.savefig(os.path.join(RESULTS_DIR, "CFA_visualization.png"))
    # plt.show()
