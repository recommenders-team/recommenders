# Recommender Systems for Retail

Recommender systems have become a key growth and revenue driver for modern retail.  For example, recommendation was estimated to [account for 35% of customer purchases on Amazon](https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers#). In addition, recommenders have been applied by retailers to delight and retain customers and improve staff productivity. 

## Scenarios

Next we will describe several most common retail scenarios and main considerations when applying recommendations in retail.

### Personalized recommendation

A major task in applying recommenations in retail is to predict which products or set of products a user is most likely to engage with or purchase, based on the shopping or viewing history of that user. This scenario is commonly shown on the personalized home page, feed or newsletter.  Most models in this repo such as [ALS](../../examples/00_quick_start/als_movielens.ipynb), [BPR](../../examples/02_model_collaborative_filtering/cornac_bpr_deep_dive.ipynb), [LightGBM](../../examples/00_quick_start/lightgbm_tinycriteo.ipynb) and [NCF](../../examples/00_quick_start/ncf_movielens.ipynb) can be used for personalization. [Azure Personalizer](https://docs.microsoft.com/en-us/azure/cognitive-services/personalizer/concept-active-learning) also provides a cloud-based personalization service using reinforcement learning based on [Vowpal Wabbit](../../examples/02_model_content_based_filtering/vowpal_wabbit_deep_dive.ipynb).

### You might also like

In this scenario, the user is already viewing a product page, and the task is to make recommendations that are relevant to it.  Personalized recommendation techniques are still applicable here, but relevance to the product being viewed is of special importance.  As such, item similarity can be useful here, especially for cold items and cold users that do not have much interaction data.

### Frequently bought together

In this task, the retailer tries to predict product(s) complementary to or bought together with a  product that a user already put in to shopping cart. This feature is great for cross-selling and is normally displayed just before checkout.  In many cases, a machine learning solution is not required for this task.

### Similar alternatives

This scenario covers down-selling or out of stock alternatives to avoid losing a sale. Similar alternatives predicts other products with similar features, like price, type, brand or visual appearance.

## Data and evaluation

Datasets used in retail recommendations usually include  [user information](../../GLOSSARY.md), [item information](../../GLOSSARY.md) and [interaction data](../../GLOSSARY.md), among others.

To measure the performance of the recommender, it is common to use [ranking metrics](../../GLOSSARY.md). In production, the business metrics used are [CTR](../../GLOSSARY.md) and [revenue per order](../../GLOSSARY.md). To evaluate a model's performance in production in an online manner, [A/B testing](../../GLOSSARY.md) is often applied.

## Other considerations

Retailers use recommendation to achieve a broad range of business objectives, such as attracting new customers through promotions, or clearing products that are at the end of their season. These objectives are often achieved by re-ranking the outputs from recommenders in scenarios above. 




