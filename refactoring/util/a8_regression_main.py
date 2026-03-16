from util.packages import *


class MultiOutcomeAnalyst:
    def __init__(self, randNum, n_splits=5):
        self.n_splits = n_splits
        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=randNum)
        self.results_list = []
        self.randNum = randNum

    def _get_regression_models(self):
        return [
            (LinearRegression(), "Linear_Regression"),
            (DecisionTreeRegressor(random_state=self.randNum, max_depth=2), "DT_Regressor"),
            (XGBRegressor(random_state=self.randNum, n_estimators=100), "XGB_Regressor")
        ]

    def _get_classification_models(self):
        return [
            (LogisticRegression(max_iter=1000), "Logistic_Regression"),
            (DecisionTreeClassifier(random_state=self.randNum, max_depth=2), "DT_Classifier"),
            (XGBClassifier(random_state=self.randNum, n_estimators=100, eval_metric='logloss'), "XGB_Classifier")
        ]

    def run_experiment(self, X_train, X_test, Y_train_df, Y_test_df, is_regression=True):
        """특정 유형(연속/이산)의 모든 Outcome 컬럼에 대해 실험 수행"""
        models = self._get_regression_models() if is_regression else self._get_classification_models()
        metrics = ['MSE', 'R2'] if is_regression else ['Accuracy', 'F1', 'AUC']
        
        # 각 Outcome 컬럼별 루프
        for col in Y_train_df.columns:
            y_train = Y_train_df[col]
            y_test = Y_test_df[col]
            
            # 각 모델별 루프
            for model, model_name in models:
                # 1. Cross-Validation (5-fold)
                scoring = {
                    'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'
                } if is_regression else {
                    'acc': 'accuracy', 'f1': 'f1_macro', 'auc': 'roc_auc'
                }
                
                cv_results = cross_validate(model, X_train, y_train, cv=self.kf, scoring=scoring)
                
                # 결과 기록 시작
                res = {
                    'Outcome_Var': col,
                    'Model': model_name,
                    'Type': 'Regression' if is_regression else 'Classification'
                }
                
                # CV 결과 집계 (Mean & Std)
                for score_key in scoring.keys():
                    vals = cv_results[f'test_{score_key}']
                    # MAE와 RMSE는 음수(neg)로 반환되므로 부호를 반전시켜야 함
                    if score_key in ['mae', 'rmse']: 
                        vals = -vals 
                    res[f'CV_{score_key.upper()}_Mean'] = np.mean(vals)
                    res[f'CV_{score_key.upper()}_Std'] = np.std(vals)

                # 2. Test Set Evaluation
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if is_regression:
                    res['Test_MAE'] = mean_absolute_error(y_test, y_pred) # MSE 대신 MAE 계산
                    res['Test_RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred)) # RMSE는 그대로 유지 가능
                    res['Test_R2'] = r2_score(y_test, y_pred)
                else:
                    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
                    res['Test_ACC'] = accuracy_score(y_test, y_pred)
                    res['Test_F1'] = f1_score(y_test, y_pred, average='macro')
                    res['Test_AUC'] = roc_auc_score(y_test, y_prob)
                
                self.results_list.append(res)

    def get_final_report(self):
        return pd.DataFrame(self.results_list)
















# # linear regression
# def visual_primal_var_check_linear(args, train, test, RESULTS_DIR) :

#     # 특정 변수 대상으로 한 번의 regression 수행

#     # =========================================================
#     # 0) User config
#     # =========================================================
#     TARGET = args.regression_var  # 회귀분석 종속변수 (예: PSQI_sum_WA)
#     N_SPLITS = args.N_SPLITS  # 교차검증 분할 수
#     # RANDOM_SEED은 이미 인자로 받음


#     scoring_weights_df = pd.read_csv(args.scoring_weights_dir, index_col=0)  # factor score 만드는 데 쓰인 가중치 행렬 (열: factor score, 행: 관측변수가 돼야 함)

#     W_axes = scoring_weights_df.T  # 전치: rows=observed, cols=factors (여기선 observed_cols가 행, factor score가 열인 형태로 가정하므로 뒤집어야 함)

#     # factor score column names (train features)
#     # - factor_scores_df.columns 가 곧 factor 축 이름이라고 가정
#     factor_cols = list(factor_scores_df.columns)





    
#     # =========================================================
#     # 2) Build TRAIN set (train)
#     #    - features: factor_scores_df
#     #    - target: train[TARGET]
#     # =========================================================
#     # target y_train
#     y_all = train[TARGET].copy()

#     # align indices (intersection) to be safe
#     common_idx = factor_scores_df.index.intersection(y_all.dropna().index)
#     X_all = factor_scores_df.loc[common_idx, factor_cols].copy()
#     y_all = y_all.loc[common_idx].astype(float)

#     # =========================================================
#     # 3) 5-fold CV on TRAIN set
#     # =========================================================
#     kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

#     fold_rows = []
#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all), start=1):
#         X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
#         y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]



#         model = LinearRegression()
#         model.fit(X_tr, y_tr)

#         # predictions
#         pred_tr = model.predict(X_tr)
#         pred_va = model.predict(X_va)

#         # metrics
#         def _rmse(y_true, y_pred):
#             return np.sqrt(mean_squared_error(y_true, y_pred))

#         fold_rows.append({
#             "fold": fold,
#             "n_train": int(len(y_tr)),
#             "n_valid": int(len(y_va)),
#             "r2_train": float(r2_score(y_tr, pred_tr)),
#             "r2_valid": float(r2_score(y_va, pred_va)),
#             "mae_train": float(mean_absolute_error(y_tr, pred_tr)),
#             "mae_valid": float(mean_absolute_error(y_va, pred_va)),
#             "rmse_train": float(_rmse(y_tr, pred_tr)),
#             "rmse_valid": float(_rmse(y_va, pred_va)),
#         })

#     cv_df = pd.DataFrame(fold_rows)
#     print("=== 5-fold CV (per fold) ===")
#     print(cv_df)

#     summary = cv_df.drop(columns=["fold", "n_train", "n_valid"]).agg(["mean", "std"]).T
#     print("\n=== 5-fold CV (mean ± std) ===")
#     print(summary)


#     # =========================================================
#     # 4) Build TEST set (test)
#     #    - Create factor scores for test using scoring weights from train scope
#     # =========================================================
#     # train stats for standardization of observed indicators
#     train_means = train[observed_cols].mean()
#     train_stds  = train[observed_cols].std(ddof=0)

#     # factor scores on test
#     efa_factor_scores = make_factor_scores_from_weights(
#         test,
#         observed_cols=observed_cols,
#         W_axes=W_axes, # 코드에선 observed_cols가 행, factor score가 열인 형태로 가정, 난 뒤집어서 썼으므로 여기서 다시 전치
#         train_means=train_means,
#         train_stds=train_stds,
#     )

#     # align test X/y
#     y_test = test[TARGET].copy()
#     test_idx = efa_factor_scores.index.intersection(y_test.dropna().index)

#     X_test = efa_factor_scores.loc[test_idx, W_axes.columns].copy()
#     y_test = y_test.loc[test_idx].astype(float)

#     # =========================================================
#     # 5) Fit FINAL model on full TRAIN, evaluate once on TEST
#     # =========================================================
#     final_model = LinearRegression()
#     final_model.fit(X_all, y_all)

#     # ✅ (추가) test 컬럼명을 train 컬럼명과 동일하게 강제 + 순서도 동일하게 (넘버링 형식 불일치 수정용)
#     X_test = X_test.copy()
#     X_test.columns = X_all.columns
#     X_test = X_test.reindex(columns=X_all.columns)

#     assert X_test.shape[1] == X_all.shape[1], "Train/Test factor count mismatch"

#     pred_test = final_model.predict(X_test)


#     test_metrics = {
#         "n_train_total": int(len(y_all)),
#         "n_test": int(len(y_test)),
#         "r2_test": float(r2_score(y_test, pred_test)),
#         "mae_test": float(mean_absolute_error(y_test, pred_test)),
#         "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred_test))),
#     }
#     print("\n=== TEST (test) evaluation ===")
#     print(pd.Series(test_metrics))

#     # (optional) coefficients
#     coef_df = pd.DataFrame({
#         "factor": X_all.columns,
#         "coef": final_model.coef_,
#     }).sort_values("coef", key=lambda s: s.abs(), ascending=False)
#     print("\n=== Final model coefficients (abs-sorted) ===")
#     print(coef_df)


#     # =====================================================
#     #  Save CSVs
#     # =====================================================
#     cv_df.to_csv(os.path.join(RESULTS_DIR, "cv_fold_results.csv"), index=False)
#     summary.to_csv(os.path.join(RESULTS_DIR, "cv_summary_mean_std.csv"))                  # index: metric, columns: mean/std
#     pd.Series(test_metrics).to_csv(os.path.join(RESULTS_DIR, "cv_test_metrics.csv"), header=False)
#     coef_df.to_csv(os.path.join(RESULTS_DIR, "cv_final_model_coefs.csv"), index=False)

# # logistic regression
# def visual_primal_var_check_logistic(test, train, RESULTS_DIR) :
#     pass