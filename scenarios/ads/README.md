<!--
Copyright (c) Recommenders contributors.
Licensed under the MIT License.
-->

# Recommendation systems for Advertisement

Recommender systems have become essential in modern digital advertising, helping to match the right ads with the right users at the right time. Studies show that personalized advertising can increase click-through rates by [up to 5.3 times](https://webtribunal.net/blog/targeted-advertising-statistics#gref) compared to non-personalized ads.

## Scenarios

Here are the most common advertising scenarios and key considerations when applying recommendations in advertising.

### Real-time bidding optimization

One of the primary tasks in advertising is determining the optimal bid price for an ad impression in real-time. This involves predicting the likelihood of user engagement (clicks, conversions) and the potential value of that engagement. Models like [LightGBM](../../examples/00_quick_start/lightgbm_tinycriteo.ipynb) and [Wide & Deep](../../examples/00_quick_start/wide_deep_movielens.ipynb) are commonly used for this purpose.

### Ad targeting and personalization

This scenario focuses on selecting the most relevant ads for a specific user based on their browsing history, demographics, and behavior patterns. Collaborative filtering approaches like [ALS](../../examples/00_quick_start/als_movielens.ipynb) and deep learning models like [NCF](../../examples/00_quick_start/ncf_movielens.ipynb) can be adapted for this purpose.

### Look-alike audience targeting

This involves finding users similar to those who have already engaged with or converted on ads. The goal is to expand the reach of successful campaigns to similar audiences. Similarity-based approaches and clustering techniques are commonly used here.

### Ad sequence optimization

This scenario involves determining the optimal sequence of ads to show to a user over time to maximize long-term engagement while avoiding ad fatigue. Reinforcement learning approaches like [Vowpal Wabbit](../../examples/02_model_content_based_filtering/vowpal_wabbit_deep_dive.ipynb) can be particularly effective.

## Data and evaluation

Datasets used in advertising recommendations typically include user demographics, browsing behavior, ad impression data, click data, and conversion data. Due to privacy concerns, much of this data is anonymized.

Common evaluation metrics include [CTR](../../GLOSSARY.md), [CVR](../../GLOSSARY.md) (Conversion Rate), [ROI](../../GLOSSARY.md), and [CPA](../../GLOSSARY.md) (Cost Per Acquisition). Online evaluation through [A/B testing](../../GLOSSARY.md) is crucial in the advertising domain.

## Other considerations

Ad recommendations must balance multiple objectives including advertiser ROI, user experience, and platform revenue. Additionally, considerations around ad frequency capping, budget pacing, and compliance with privacy regulations like GDPR must be taken into account.
