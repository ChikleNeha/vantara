# Exploratory Data Analysis

Customers analysed: **3,679**; churn rate in the 90-day window: **56.5%**.

![RFM](figures/eda_rfm_distributions.png)
![Churn by recency](figures/eda_churn_by_recency.png)
![Seasonality](figures/eda_seasonality.png)

## Strongest correlations with churn
- `engagement_score`: -0.501
- `recency_days`: +0.437
- `frequency`: -0.318

## Strongest correlations with log next-90-day spend
- `engagement_score`: +0.551
- `recency_days`: -0.454
- `frequency`: +0.403

## Multicollinearity (VIF on the final model features, max 9.8)
- `recency_days`: 9.8
- `engagement_score`: 9.1
- `tenure_days`: 4.8
- `frequency`: 4.0
- `clv_historical`: 3.1

## Hypotheses for the modelling stage
- H1. Recency is the dominant churn signal (corr with churn +0.44); models should rank it first.
- H2. Order frequency and spend protect against churn (corr -0.32 and -0.18); loyal high-spend customers need little intervention.
- H3. A declining monthly order trend (purchase_trend corr +0.01) adds signal beyond RFM.
- H4. Q4 carries 37% of revenue, so seasonal customers will look 'churned' outside the season; seasonal_concentration should help separate them.
- H5. The top market holds 85% of revenue; country features will matter little for the main market but may flag niche overseas buyers.
- H6. Return-heavy customers form a separate behavioural group; the autoencoder should surface them as anomalies.
