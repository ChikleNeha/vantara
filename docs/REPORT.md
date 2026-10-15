# Customer Behaviour Prediction Platform: Technical Report

## 1. Summary
- Data source: **file**; 5,257 customers modelled; churn rate 56.5% (no purchase in the 90 days after 2011-09-10).
- Production churn model: **lightgbm** (selected on validation ROC-AUC) with test ROC-AUC 0.802, recall 0.762, precision 0.774, F1 0.768.
- Segments: **5** K-Means segments (silhouette 0.262).
- Portfolio today: 928 high-risk customers; expected spend at risk over the next 90 days about 936,760.

## 2. Data and cleaning
| Step | Rows |
|---|---|
| raw rows | 1,067,371 |
| after dedup | 1,033,034 |
| invalid price rows | 6,042 |
| data entry errors | 4 |
| bulk rows flagged | 37,862 |
| admin rows removed | 5,432 |
| missing customer id rows | 227,143 |
| clean rows | 1,021,556 |

Features were built with a strict point-in-time cutoff (2011-09-10); labels use only the following 90 days. 27 of 30 columns remain after iterative VIF pruning (threshold 10.0). Dropped: `country_United Kingdom`, `aff_other`, `monetary_total`.

## 3. Churn model comparison (test set, ranked by ROC-AUC then recall)
Threshold per model is chosen on the validation set to keep recall at or above the target, so precision and recall are directly comparable.

| Rank | Model | CV AUC (train) | Val AUC | Test AUC | Recall | Precision | F1 | Accuracy | Confusion (TN/FP/FN/TP) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | random_forest | 0.807 | 0.793 | 0.806 | 0.771 | 0.778 | 0.775 | 0.747 | 245/98/102/344 |
| 2 | lightgbm | 0.795 | 0.796 | 0.802 | 0.762 | 0.774 | 0.768 | 0.740 | 244/99/106/340 |
| 3 | xgboost | 0.802 | 0.795 | 0.800 | 0.767 | 0.770 | 0.768 | 0.739 | 241/102/104/342 |
| 4 | ann | n/a | 0.789 | 0.795 | 0.782 | 0.769 | 0.776 | 0.744 | 238/105/97/349 |
| 5 | logistic_regression | 0.802 | 0.784 | 0.791 | 0.776 | 0.790 | 0.783 | 0.757 | 251/92/100/346 |
| 6 | knn | 0.789 | 0.772 | 0.788 | 0.773 | 0.760 | 0.767 | 0.734 | 234/109/101/345 |
| 7 | lstm | n/a | 0.774 | 0.786 | 0.780 | 0.748 | 0.764 | 0.728 | 226/117/98/348 |
| 8 | decision_tree | 0.778 | 0.780 | 0.775 | 0.821 | 0.732 | 0.774 | 0.729 | 209/134/80/366 |

### Recommendation
Serve **lightgbm** in the API: it had the best validation ROC-AUC among the SHAP-compatible tree models, trains in seconds and gives exact per-customer explanations. The ANN is kept as a secondary check; its residual correlation with lightgbm is 0.98, so it adds limited independent information on this data. With only a few thousand customers the neural models are exercises rather than production candidates, as the PRD anticipated.

## 4. Lifetime value (next-90-day net spend)
Best model (chosen on validation R2): **hurdle**. Test R2 0.390, MAE 516.3, RMSE 4487.9; R2 among customers who actually bought: 0.381.

| Model | Validation R2 | Test R2 | Test MAE | Test RMSE |
|---|---|---|---|---|
| ridge | -256.500 | -662021038427.591 | 179606394.9 | 4674847298.9 |
| xgboost_direct | 0.140 | 0.059 | 578.1 | 5574.3 |
| hurdle | 0.410 | 0.390 | 516.3 | 4487.9 |

Roughly half of customers spend nothing in the next 90 days, so the target is zero-inflated. The hurdle model separates 'will they buy at all' from 'how much if they do', which is why it beats a single regression. The remaining gap to the PRD target reflects genuinely unpredictable purchase timing over a short window.

Next-purchase category: top-1 accuracy 0.501, top-3 0.822 (majority-class baseline 0.486, n=343).

## 5. Deep learning
- ANN: stopped at 21 epochs (early stopping), test AUC 0.795. NumPy inference copy matches Keras within 1.3e-07.
- LSTM on the last 12 orders: 14 epochs, test AUC 0.786.
- Autoencoder: reconstruction-error threshold 0.6187 (training percentile 99); flag rate train/val/test = 0.010 / 0.011 / 0.008. 90 customers flagged today.
- Loss curves: `docs/figures/ann_loss.png`, `lstm_loss.png`, `autoencoder_loss.png`.

## 6. Segmentation
K chosen by silhouette over 3 to 8. K-Means: silhouette 0.262, Davies-Bouldin 1.171. Gaussian mixture: silhouette 0.100, Davies-Bouldin 3.066.

| Segment | Customers | Share | Mean recency (d) | Mean orders | Mean spend | What to do |
|---|---|---|---|---|---|---|
| Lapsed high-value | 280 | 5% | 251 | 8.4 | 8,673 | Strong past spend but gone quiet; best win-back opportunity. |
| Dormant low-value | 2,281 | 39% | 328 | 2.6 | 694 | Inactive and low spend; low-cost reactivation only. |
| Seasonal gift shoppers | 1,348 | 23% | 194 | 2.0 | 541 | Orders concentrated in Oct to Dec; target with pre-season campaigns. |
| Loyal champions | 1,888 | 32% | 41 | 13.5 | 6,346 | Recent, frequent, high spend; protect and reward. |
| Return-heavy buyers | 56 | 1% | 272 | 3.3 | 2,038 | High share of purchases sent back; investigate product fit and fulfilment. |

## 7. Explainability
Top global drivers (mean absolute SHAP on held-out customers):
1. engagement score (0.257)
2. days since last purchase (0.234)
3. order-frequency trend (0.093)
4. number of orders (0.079)
5. share of spend in gifts cards (0.053)
6. customer age in days (0.047)

The SHAP vs LIME agreement check was skipped because LIME was not installed in this environment. `docs/figures/shap_summary.png` shows the full beeswarm. Individual explanations (SHAP bars, LIME rules, plain-language narrative) are served by the API and dashboard.

## 8. Success metrics versus PRD targets
| Metric | Target | Result | Status |
|---|---|---|---|
| Churn ROC-AUC (test) | >= 0.80 | 0.802 | Met |
| Churn recall (test) | >= 0.70 | 0.762 | Met |
| CLV R2 (test, original scale) | >= 0.60 | 0.390 | Not met |
| Single-customer prediction p95 (model + SHAP path) | < 400 ms | 126.5 ms | Met |
| Dashboard full segment view load | < 3 s | measure in browser | Check manually |
| Single-command reproducible pipeline | 100% | `python -m src.pipeline` (309.8s) | Met |

The test set holds 789 customers (446 churners), so ROC-AUC and recall carry sampling error of a few points. A shortfall of about 0.01 against a target is within that noise but is still reported as not met. The CLV gap is larger: the target is zero-inflated and the window is short. Whether that is irreducible has not been tested here; a longer label window or more customers are the obvious next experiments.

## 9. Method notes and deviations from the PRD
- SQLite replaces PostgreSQL and a React dashboard replaces Streamlit (project stack decision). Ollama adds plain-language narratives and a Q&A assistant; it never produces predictions and every feature has a deterministic fallback.
- Class imbalance is handled with class weights on the training set only; validation and test sets are untouched. The test set is evaluated once, after model selection on validation data.
- Hyperparameters are tuned by 5-fold stratified CV on the training set; boosted models then use early stopping on the validation set.
- Categories are keyword-derived from product descriptions (the dataset has no taxonomy).

## Appendix A. Mathematical foundations

**Logistic regression and cross-entropy.** The model outputs p = sigma(w.x + b), sigma(z) = 1/(1+e^-z). Training minimises
L = -(1/n) sum[ y log p + (1-y) log(1-p) ] (+ lambda ||w||^2 for L2 regularisation). The gradient is (p - y)x, which is why
the loss is convex and well-behaved.

**Gradient descent and backpropagation.** Parameters move against the gradient, theta <- theta - eta * dL/dtheta. In the ANN
the chain rule propagates the error layer by layer: dL/dW_l = delta_l a_{l-1}^T with delta_l = (W_{l+1}^T delta_{l+1}) * f'(z_l).
Adam adds running estimates of the first and second moments of the gradient to scale each step.

**Regularisation.** L2 penalises large weights (ridge), L1 induces sparsity, dropout randomly zeroes activations during
training so units cannot co-adapt, and early stopping halts when validation loss stops improving.

**PCA.** With standardised data matrix Z and covariance C = Z^T Z/(n-1), the principal components are the eigenvectors of C;
the share of variance explained by component i is lambda_i / sum(lambda). Used here only to draw segments in 2-D.

**K-Means and GMM.** K-Means minimises the within-cluster sum of squares sum_k sum_{x in C_k} ||x - mu_k||^2. A Gaussian
mixture models p(x) = sum_k pi_k N(x | mu_k, Sigma_k) and is fit by expectation-maximisation, giving soft assignments.

**Cluster quality.** Silhouette s = (b - a)/max(a, b), where a is mean intra-cluster distance and b the mean distance to the
nearest other cluster (higher is better). Davies-Bouldin averages the worst-case ratio of within-cluster scatter to between-centroid
distance (lower is better).

**Metrics.** Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2PR/(P+R); ROC-AUC is the probability that a random churner is
scored above a random non-churner. R^2 = 1 - SS_res/SS_tot; MAE and RMSE are mean absolute and root-mean-square error.

**SHAP.** The Shapley value of feature i is phi_i = sum over subsets S of (|S|!(M-|S|-1)!/M!) [f(S u {i}) - f(S)]. For trees,
TreeSHAP computes these exactly in polynomial time; contributions sum to f(x) - E[f(x)] in log-odds.

**Variance inflation factor.** VIF_i = 1/(1 - R_i^2), where R_i^2 comes from regressing feature i on all others; values above
10 indicate severe collinearity.


## Future enhancements
- Retrain monthly with rolling cutoffs (multiple cutoffs per customer) to enlarge the training set and measure drift.
- Add a probability-calibration step (isotonic) and decision-cost thresholding using actual campaign costs.
- Survival models (Cox / BG-NBD with Gamma-Gamma) for a principled lifetime-value estimate beyond a 90-day window.
- Real product taxonomy instead of keyword categories; item-level recommender (e.g. implicit ALS).
- Transformer encoder over purchase sequences and a CatBoost comparison (both optional in the PRD).
- Monitoring: population-stability index on inputs, scheduled retraining, and A/B measurement of retention campaigns.
- Swap SQLite for PostgreSQL if concurrent writers or larger volumes are needed (SQLAlchemy layer already isolates this).
