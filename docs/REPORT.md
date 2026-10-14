# Customer Behaviour Prediction Platform: Technical Report

> **Data warning.** This run used the built-in SYNTHETIC generator because `online_retail_II.xlsx` was not present. Every number below demonstrates that the pipeline works; none of them are findings about real customers. Place the UCI file in `data/raw/` and re-run `python -m src.pipeline` to regenerate this report on real data.

## 1. Summary
- Data source: **synthetic**; 2,384 customers modelled; churn rate 54.3% (no purchase in the 90 days after 2011-09-10).
- Production churn model: **random_forest** (selected on validation ROC-AUC) with test ROC-AUC 0.794, recall 0.696, precision 0.763, F1 0.728.
- Segments: **4** K-Means segments (silhouette 0.254).
- Portfolio today: 742 high-risk customers; expected spend at risk over the next 90 days about 304,179.

## 2. Data and cleaning
| Step | Rows |
|---|---|
| raw rows | 218,831 |
| after dedup | 217,521 |
| invalid price rows | 335 |
| data entry errors | 60 |
| bulk rows flagged | 15,336 |
| admin rows removed | 1,966 |
| missing customer id rows | 15,899 |
| clean rows | 215,160 |

Features were built with a strict point-in-time cutoff (2011-09-10); labels use only the following 90 days. 25 of 30 columns remain after iterative VIF pruning (threshold 10.0). Dropped: `aff_other`, `country_United Kingdom`, `aff_stationery`, `monetary_total`, `monetary_avg`.

## 3. Churn model comparison (test set, ranked by ROC-AUC then recall)
Threshold per model is chosen on the validation set to keep recall at or above the target, so precision and recall are directly comparable.

| Rank | Model | CV AUC (train) | Val AUC | Test AUC | Recall | Precision | F1 | Accuracy | Confusion (TN/FP/FN/TP) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | xgboost | 0.801 | 0.837 | 0.796 | 0.717 | 0.768 | 0.741 | 0.729 | 122/42/55/139 |
| 2 | random_forest | 0.804 | 0.837 | 0.794 | 0.696 | 0.763 | 0.728 | 0.718 | 122/42/59/135 |
| 3 | decision_tree | 0.804 | 0.812 | 0.792 | 0.820 | 0.660 | 0.731 | 0.673 | 82/82/35/159 |
| 4 | logistic_regression | 0.807 | 0.834 | 0.792 | 0.686 | 0.751 | 0.717 | 0.707 | 120/44/61/133 |
| 5 | ann | n/a | 0.823 | 0.791 | 0.634 | 0.783 | 0.701 | 0.707 | 130/34/71/123 |
| 6 | knn | 0.770 | 0.800 | 0.789 | 0.737 | 0.698 | 0.717 | 0.684 | 102/62/51/143 |
| 7 | lightgbm | 0.791 | 0.831 | 0.788 | 0.670 | 0.783 | 0.722 | 0.721 | 128/36/64/130 |
| 8 | lstm | n/a | 0.767 | 0.744 | 0.696 | 0.689 | 0.692 | 0.665 | 103/61/59/135 |

### Recommendation
Serve **random_forest** in the API: it had the best validation ROC-AUC among the SHAP-compatible tree models, trains in seconds and gives exact per-customer explanations. The ANN is kept as a secondary check; its residual correlation with random_forest is 0.97, so it adds limited independent information on this data. With only a few thousand customers the neural models are exercises rather than production candidates, as the PRD anticipated.

## 4. Lifetime value (next-90-day net spend)
Best model (chosen on validation R2): **hurdle**. Test R2 0.451, MAE 206.1, RMSE 629.3; R2 among customers who actually bought: 0.632.

| Model | Validation R2 | Test R2 | Test MAE | Test RMSE |
|---|---|---|---|---|
| ridge | 0.022 | -9.619 | 348.3 | 2767.0 |
| xgboost_direct | 0.065 | 0.170 | 210.9 | 773.5 |
| hurdle | 0.439 | 0.451 | 206.1 | 629.3 |

Roughly half of customers spend nothing in the next 90 days, so the target is zero-inflated. The hurdle model separates 'will they buy at all' from 'how much if they do', which is why it beats a single regression. The remaining gap to the PRD target reflects genuinely unpredictable purchase timing over a short window.

Next-purchase category: top-1 accuracy 0.427, top-3 0.732 (majority-class baseline 0.159, n=164).

## 5. Deep learning
- ANN: stopped at 26 epochs (early stopping), test AUC 0.791. NumPy inference copy matches Keras within 2.0e-07.
- LSTM on the last 12 orders: 17 epochs, test AUC 0.744.
- Autoencoder: reconstruction-error threshold 1.8517 (training percentile 99); flag rate train/val/test = 0.010 / 0.003 / 0.017. 36 customers flagged today.
- Loss curves: `docs/figures/ann_loss.png`, `lstm_loss.png`, `autoencoder_loss.png`.

## 6. Segmentation
K chosen by silhouette over 3 to 8. K-Means: silhouette 0.254, Davies-Bouldin 1.397. Gaussian mixture: silhouette 0.103, Davies-Bouldin 1.861.

| Segment | Customers | Share | Mean recency (d) | Mean orders | Mean spend | What to do |
|---|---|---|---|---|---|---|
| Dormant low-value | 1,323 | 50% | 256 | 4.6 | 340 | Inactive and low spend; low-cost reactivation only. |
| Loyal champions | 844 | 32% | 6 | 14.8 | 1,243 | Recent, frequent, high spend; protect and reward. |
| Steady mid-value | 279 | 11% | 143 | 8.8 | 9,074 | Regular but modest spend; upsell and cross-sell. |
| Return-heavy buyers | 178 | 7% | 244 | 5.6 | 581 | High share of purchases sent back; investigate product fit and fulfilment. |

## 7. Explainability
Top global drivers (mean absolute SHAP on held-out customers):
1. days since last purchase (0.101)
2. engagement score (0.054)
3. customer age in days (0.038)
4. number of orders (0.034)
5. order-frequency trend (0.029)
6. share of orders in Oct to Dec (0.017)

SHAP and LIME agree on 80% of each customer's top-5 features on a random sample, a sanity check that the two methods tell a consistent story. `docs/figures/shap_summary.png` shows the full beeswarm. Individual explanations (SHAP bars, LIME rules, plain-language narrative) are served by the API and dashboard.

## 8. Success metrics versus PRD targets
| Metric | Target | Result | Status |
|---|---|---|---|
| Churn ROC-AUC (test) | >= 0.80 | 0.794 | Not met |
| Churn recall (test) | >= 0.70 | 0.696 | Not met |
| CLV R2 (test, original scale) | >= 0.60 | 0.451 | Not met |
| Single-customer prediction p95 (model + SHAP path) | < 400 ms | 76.4 ms | Met |
| Dashboard full segment view load | < 3 s | measure in browser | Check manually |
| Single-command reproducible pipeline | 100% | `python -m src.pipeline` (101.8s) | Met |

The test set holds 358 customers (194 churners), so ROC-AUC and recall carry sampling error of a few points. A shortfall of about 0.01 against a target is within that noise but is still reported as not met. The CLV gap is larger: the target is zero-inflated and the window is short. Whether that is irreducible has not been tested here; a longer label window or more customers are the obvious next experiments.

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
