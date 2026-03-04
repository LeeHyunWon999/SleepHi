from util.packages import *

# linear regression
def temp_linear_regression(args, X_train, X_test, Y_train, Y_test, RESULTS_DIR, RANDOM_SEED):
    # =========================================================
    # 0) User config
    # =========================================================
    TARGET = args.regression_var
    N_SPLITS = args.N_SPLITS

    # =========================================================
    # 2) Build TRAIN set (Alignment)
    # =========================================================
    # 전달받은 Y_train에서 결측치를 제외하고 X_train과 인덱스 맞추기 (전처리 다 해놨으니 굳이 안해도 되긴 함)
    common_idx_train = X_train.index.intersection(Y_train.dropna().index)
    
    # CV 로직에서 사용하는 변수명인 X_all, y_all로 할당
    X_all = X_train.loc[common_idx_train].copy()
    y_all = Y_train.loc[common_idx_train].astype(float)

    # =========================================================
    # 3) 5-fold CV on TRAIN set (기존 로직 유지)
    # =========================================================
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_rows = []
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all), start=1):
        X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
        y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]

        model = LinearRegression()
        model.fit(X_tr, y_tr)

        pred_tr = model.predict(X_tr)
        pred_va = model.predict(X_va)

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
    # 4) Build TEST set (Alignment)
    # =========================================================
    # 전달받은 Y_test에서 결측치를 제외하고 X_test와 인덱스 맞추기
    common_idx_test = X_test.index.intersection(Y_test.dropna().index)
    
    X_test_final = X_test.loc[common_idx_test].copy()
    y_test_final = Y_test.loc[common_idx_test].astype(float)

    # 컬럼 순서 및 이름 강제 일치 (Train 기준)
    X_test_final.columns = X_all.columns
    X_test_final = X_test_final.reindex(columns=X_all.columns)

    # =========================================================
    # 5) Fit FINAL model and Evaluate
    # =========================================================
    final_model = LinearRegression()
    final_model.fit(X_all, y_all)

    assert X_test_final.shape[1] == X_all.shape[1], "Train/Test feature count mismatch"

    pred_test = final_model.predict(X_test_final)

    test_metrics = {
        "n_train_total": int(len(y_all)),
        "n_test": int(len(y_test_final)),
        "r2_test": float(r2_score(y_test_final, pred_test)),
        "mae_test": float(mean_absolute_error(y_test_final, pred_test)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test_final, pred_test))),
    }
    print("\n=== TEST evaluation ===")
    print(pd.Series(test_metrics))

    # (optional) coefficients
    coef_df = pd.DataFrame({
        "factor": X_all.columns,
        "coef": final_model.coef_.flatten(), # 차원 삑나는 경우 대응
    }).sort_values("coef", key=lambda s: s.abs(), ascending=False)
    print("\n=== Final model coefficients (abs-sorted) ===")
    print(coef_df)


    # =====================================================
    #  Save CSVs (수정된 부분)
    # =====================================================
    # 1. 저장할 폴더 경로 확정
    save_path = os.path.join(RESULTS_DIR, TARGET)

    # 2. 폴더가 없으면 생성 (exist_ok=True 하면 이미 폴더가 있어도 무방)
    os.makedirs(save_path, exist_ok=True)

    # 3. 파일 저장
    cv_df.to_csv(os.path.join(save_path, "cv_fold_results.csv"), index=False)
    summary.to_csv(os.path.join(save_path, "cv_summary_mean_std.csv"))
    pd.Series(test_metrics).to_csv(os.path.join(save_path, "cv_test_metrics.csv"), header=False)
    coef_df.to_csv(os.path.join(save_path, "cv_final_model_coefs.csv"), index=False)





# # logistic regression
# def visual_primal_var_check_logistic(test, train, RESULTS_DIR) :
#     pass