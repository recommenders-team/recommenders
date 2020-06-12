# Recommendation systems for Retail

Retail is one of the areas where recommendation systems have been more successful. According to [McKinsey](https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers#), 35% of what consumers purchase on Amazon and 75% of what they watch on Netflix come from product recommendations.

An increasing number of online retailers are utilizing recommendation systems to increase revenue, improve customer engagement and satisfaction, increase time on the page, enhance customer’s purchasing experience, gain understanding about customers, expand the shopping cart, etc.

Next we will list the most common scenarios retailers use.

## Personalized recommendation

This scenario predicts which products or set of products a user is most likely to engage with or purchase, based on the shopping or viewing history of that user. This scenario is commonly shown on the home page or in a personalized newsletter.

The kind of data these kind of scenarios need is [implicit interaction data](../GLOSSARY.md), [user information](../GLOSSARY.md) and [item information](../GLOSSARY.md).

To measure the performance of the personalized recommendation machine learning algorithm, it is common to use [ranking metrics](../GLOSSARY.md). In production, the business metrics used are [CTR](../GLOSSARY.md) and [revenue per order](../GLOSSARY.md). For being able to measure the business metrics in production, it is recommended to implement [A/B testing](../GLOSSARY.md).


## You might also like

This recommendation scenario is similar to the personalized recommendation as it predicts the next product a user is likely to engage with or purchase. However, the starting point is typically a product page, so in addition to considering the entire shopping or viewing history of the user, the relevance of the specified product in relation to other items is used to recommend additional products.

The kind of data these kind of scenarios need is...

To measure the performance ...


## Frequently bought together

This scenario is the machine learning solution for up-selling and cross-selling. Frequently bought together predicts which product or set of products are complementary or usually bought together with a specified product, as opposed as substituting it. Normally, this scenario is displayed in the shopping cart page, just before buying.

The kind of data these kind of scenarios need is...

To measure the performance ...

## Similar alternatives

This scenario covers down-selling or out of stock alternatives and its objective is to avoid loosing a sale. Similar alternatives predicts other products with similar features, like price, type, brand or visual appearance.

The kind of data these kind of scenarios need is...

To measure the performance ...

## Recommendations of product subset

In certain situations, the retailer would like to recommend products from a subset, they could be products for sale, products with a high margin or products that have a low number of units left. This scenario can be used to delimit the outputs of all previous recommendation scenarios. 

The kind of data these kind of scenarios need is...

To measure the performance ...

This scenario is tightly related to the [long tail product](../GLOSSARY.md) concept...


## Examples of end 2 end recommendation scenarios with Microsoft Recommenders

In the repository we have the following examples that can be used in retail

| Scenario | Description | Algorithm | Implementation |
|----------|-------------|-----------|----------------|
| Personalized recommendation |  Matrix factorization algorithm for explicit feedback in large datasets, optimized by Spark MLLib for scalability and distributed computing capability | Alternating Least Squares (ALS) | [pyspark notebook using Movielens dataset](https://github.com/microsoft/recommenders/blob/staging/notebooks/00_quick_start/als_movielens.ipynb) |
| Personalized recommendation | Gradient Boosting Tree algorithm for fast training and low memory usage in content-based problems | LightGBM/MMLSpark | [spark notebook using Criteo dataset](https://github.com/microsoft/recommenders/blob/staging/notebooks/02_model/mmlspark_lightgbm_criteo.ipynb) |
| You might also like | Similarity-based algorithm for implicit feedback dataset | Simple Algorithm for Recommendation (SAR)<sup>*</sup> | [python notebook using Movielens dataset](notebooks/00_quick_start/sar_movielens.ipynb) |

**NOTE**: <sup>*</sup> indicates algorithms invented/contributed by Microsoft.

