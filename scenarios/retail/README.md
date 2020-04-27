# Recommendation systems for Retail

An increasing number of online companies are utilizing recommendation systems to increase user interaction and enrich shopping potential. Use cases of recommendation systems have been expanding rapidly across many aspects of eCommerce and online media over the last 4-5 years, and we expect this trend to continue.

Companies across many different areas of enterprise are beginning to implement recommendation systems in an attempt to enhance their customer’s online purchasing experience, increase sales and retain customers. Business owners are recognizing potential in the fact that recommendation systems allow the collection of a huge amount of information relating to user’s behavior and their transactions within an enterprise. This information can then be systematically stored within user profiles to be used for future interactions.

## Typical Business Scenarios in Recommendation Systems for Retail

The most common scenarios companies use are:

* Others you may like (also called similar items): The "Others you may like" recommendation predicts the next product that a user is most likely to engage with or purchase. The prediction is based on both the entire shopping or viewing history of the user and the candidate product's relevance to a current specified product.

* Frequently bought together"(shopping cart expansion): The "Frequently bought together" recommendation predicts items frequently bought together for a specific product within the same shopping session. If a list of products is being viewed, then it predicts items frequently bought with that product list. This recommendation is useful when the user has indicated an intent to purchase a particular product (or list of products) already, and you are looking to recommend complements (as opposed to substitutes). This recommendation is commonly displayed on the "add to cart" page, or on the "shopping cart" or "registry" pages (for shopping cart expansion).

* Recommended for you: The "Recommended for you" recommendation predicts the next product that a user is most likely to engage with or purchase, based on the shopping or viewing history of that user. This recommendation is typically used on the home page.


## Types of Recommendation Systems for Retail

Typically recommendation systems in retail can be divided into three categories:

* Collaborative filtering: This type of recommendation system makes predictions of what might interest a person based on the taste of many other users. It assumes that if person X likes Snickers, and person Y likes Snickers and Milky Way, then person X might like Milky Way as well.

* Content-based filtering: This type of recommendation system focuses on the products themselves and recommends other products that have similar attributes. Content-based filtering relies on the characteristics of the products themselves, so it doesn’t rely on other users to interact with the products before making a recommendation.

* Hybrid filtering: This type of recommendation system can implement a combination fo any two of the above systems.


## Measuring Recommendation performance

### Machine learning metrics (offline metrics)

In Recommenders, offine metrics implementation for python are found on [python_evaluation.py](https://github.com/microsoft/recommenders/blob/master/reco_utils/evaluation/python_evaluation.py) and those for PySpark are found on [spark_evaluation.py](https://github.com/microsoft/recommenders/blob/master/reco_utils/evaluation/spark_evaluation.py).

Currently available metrics include:

- Root Mean Squared Error
- Mean Absolute Error
- R<sup>2</sup>
- Explained Variance
- Precision at K
- Recall at K
- Normalized Discounted Cumulative Gain at K
- Mean Average Precision at K
- Area Under Curve
- Logistic Loss

### Business success metrics (online metrics)

Below are some of the various potential benefits of recommendation systems in business, and the metrics that tipically are used:

* Click-through rate (CTR): Optimizing for CTR emphasizes engagement; you should optimize for CTR when you want to maximize the likelihood that the user interacts with the recommendation.

* Revenue per order: The revenue per order optimization objective is the default optimization objective for the "Frequently bought together" recommendation model type. This optimization objective cannot be specified for any other recommendation model type.

* Conversion rate: Optimizing for conversion rate maximizes the likelihood that the user purchases the recommended item; if you want to increase the number of purchases per session, optimize for conversion rate.

### Relationship between online and offline metrics

### A/B testing

### Advanced A/B testing: online learning with VW

## Challenges in Recommendation systems for Retail

* Cold start: Personalized recommender systems take advantage of users past history to make predictions. The cold start problem concerns the personalized recommendations for users with no or few past history (new users). Providing recommendations to users with small past history becomes a difficult problem for CF models because their learning and predictive ability is limited. Multiple research have been conducted in this direction using hybrid models. These models use auxiliary information (multimodal information, side information, etc.) to overcome the cold start problem.

* Long tail products:

## Building end 2 end recommendation scenarios with Microsoft Recommenders

In the repository we have the following examples that can be used in retail





## References and resources

sources: [1](https://emerj.com/ai-sector-overviews/use-cases-recommendation-systems/), [2](https://cloud.google.com/recommendations-ai/docs/placements), [3](https://www.researchgate.net/post/Can_anyone_explain_what_is_cold_start_problem_in_recommender_system)