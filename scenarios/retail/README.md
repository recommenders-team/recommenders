# Recommendation systems for Retail

Retail is one of the areas where recommendation systems have been more successful. According to [McKinsey](https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers#), 35% of what consumers purchase on Amazon and 75% of what they watch on Netflix come from product recommendations.

An increasing number of online retailers are utilizing recommendation systems to increase revenue, improve customer engagement and satisfaction, increase time on the page, enhance customer’s purchasing experience, gain understanding about customers, expand the shopping cart, etc.

## Typical Business Scenarios in Recommendation Systems for Retail

The most common scenarios retailers use are:

* Personalized recommendation: This scenario predicts which products or set of products a user is most likely to engage with or purchase, based on the shopping or viewing history of that user. This scenario is commonly shown on the home page or in a personalized newsletter.

* You might also like: This recommendation scenario is similar to the personalized recommendation as it predicts the next product a user is likely to engage with or purchase. However, the starting point is typically a product page, so in addition to considering the entire shopping or viewing history of the user, the relevance of the specified product in relation to other items is used to recommend additional products.

* Frequently bought together: This scenario is the machine learning solution for up-selling and cross-selling. Frequently bought together predicts which product or set of products are complementary or usually bought together with a specified product, as opposed as substituting it. Normally, this scenario is displayed in the shopping cart page, just before buying.

* Similar alternatives: This scenario covers down-selling or out of stock alternatives and its objective is to avoid loosing a sale. Similar alternatives predicts other products with similar features, like price, type, brand or visual appearance.

* Recommendations of product subset. In certain situations, the retailer would like to recommend products from a subset, they could be products for sale, products with a high margin or products that have a low number of units left. This scenario can be used to delimit the outputs of all previous recommendation scenarios. 


## Data in Recommendation Systems

### Data types

In RS for retail there are typically the following types of data

* Explicit interactions: When a user explicitly rate an item, typically between 1-5, the user is giving a value on the likeliness of the item. In retail, this kind of data is not very common.

* Implicit interactions: Implicit interactions are views or clicks that show a certain interest of the user about a specific items. These kind of data is more common but it doesn't define the intention of the user as clearly as the explicit data.

* User features: These include all information that define the user, some examples can be name, address, email, demographics, etc. 

* Item features: These include information about the item, some examples can be SKU, description, brand, price, etc.

* Knowledge graph data: ...

### Considerations about data size

The size of the data is important when designing the system...

### Cold start scenarios

Personalized recommender systems take advantage of users past history to make predictions. The cold start problem concerns the personalized recommendations for users with no or few past history (new users). Providing recommendations to users with small past history becomes a difficult problem for CF models because their learning and predictive ability is limited. Multiple research have been conducted in this direction using hybrid models. These models use auxiliary information (multimodal information, side information, etc.) to overcome the cold start problem.

### Long tail products

Typically, the shape of items interacted in retail follow a long tail distribution [1,2]. 

## Measuring Recommendation performance

### Machine learning metrics (offline metrics)

Offline metrics in RS are based on rating, ranking, classification or diversity. For learning more about offline metrics, see the [definitions available in Recommenders repository](../../examples/03_evaluate)

### Business success metrics (online metrics)

Below are some of the various potential benefits of recommendation systems in business, and the metrics that tipically are used:

* Click-through rate (CTR): Optimizing for CTR emphasizes engagement; you should optimize for CTR when you want to maximize the likelihood that the user interacts with the recommendation.

* Revenue per order: The revenue per order optimization objective is the default optimization objective for the "Frequently bought together" recommendation model type. This optimization objective cannot be specified for any other recommendation model type.

* Conversion rate: Optimizing for conversion rate maximizes the likelihood that the user purchases the recommended item; if you want to increase the number of purchases per session, optimize for conversion rate.

### Relationship between online and offline metrics in retail

There is some literature about the relationship between offline and online metrics...


### A/B testing

### Advanced A/B testing: online learning with VW

...

## Examples of end 2 end recommendation scenarios with Microsoft Recommenders

From a technical perspective, RS can be grouped in these categories [1]:

* Collaborative filtering: This type of recommendation system makes predictions of what might interest a person based on the taste of many other users. It assumes that if person X likes Snickers, and person Y likes Snickers and Milky Way, then person X might like Milky Way as well. See the [list of examples in Recommenders repository](../../examples/02_model_collaborative_filtering).

* Content-based filtering: This type of recommendation system focuses on the products themselves and recommends other products that have similar attributes. Content-based filtering relies on the characteristics of the products themselves, so it doesn’t rely on other users to interact with the products before making a recommendation. See the [list of examples in Recommenders repository](../../examples/02_model_content_based_filtering).

* Hybrid filtering: This type of recommendation system can implement a combination fo any two of the above systems. See the [list of examples in Recommenders repository](../../examples/02_model_hybrid).

* Knowledge-base: ...

In the repository we have the following examples that can be used in retail

| Scenario | Description | Algorithm | Implementation |
|----------|-------------|-----------|----------------|
| Collaborative Filtering with explicit interactions in Spark environment |  Matrix factorization algorithm for explicit feedback in large datasets, optimized by Spark MLLib for scalability and distributed computing capability | Alternating Least Squares (ALS) | [pyspark notebook using Movielens dataset](https://github.com/microsoft/recommenders/blob/staging/notebooks/00_quick_start/als_movielens.ipynb) |
| Content-Based Filtering for content recommendation in Spark environment | Gradient Boosting Tree algorithm for fast training and low memory usage in content-based problems | LightGBM/MMLSpark | [spark notebook using Criteo dataset](https://github.com/microsoft/recommenders/blob/staging/notebooks/02_model/mmlspark_lightgbm_criteo.ipynb) |



## References and resources

[1] Aggarwal, Charu C. Recommender systems. Vol. 1. Cham: Springer International Publishing, 2016.
[2]. Park, Yoon-Joo, and Alexander Tuzhilin. "The long tail of recommender systems and how to leverage it." In Proceedings of the 2008 ACM conference on Recommender systems, pp. 11-18. 2008. [Link to paper](http://people.stern.nyu.edu/atuzhili/pdf/Park-Tuzhilin-RecSys08-final.pdf).
[3]. Armstrong, Robert. "The long tail: Why the future of business is selling less of more." Canadian Journal of Communication 33, no. 1 (2008). [Link to paper](https://www.cjc-online.ca/index.php/journal/article/view/1946/3141).


sources: [1](https://emerj.com/ai-sector-overviews/use-cases-recommendation-systems/), [2](https://cloud.google.com/recommendations-ai/docs/placements), [3](https://www.researchgate.net/post/Can_anyone_explain_what_is_cold_start_problem_in_recommender_system)