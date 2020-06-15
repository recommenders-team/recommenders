# Glossary

* A/B testing: Methodology to evaluate the performance of a system in production. In the context of Recommendation Systems it is used to measure a machine learning model performance in real-time. It works by randomizing an environment response into two groups A and B, typically half of the traffic goes to the machine learning model output and the other half is left without model. By comparing the metrics from A and B branches, it is possible to evaluate whether it is beneficial the use of the model or not. A test with more than two groups it is named Multi-Variate Test.

* Click-through rate (CTR): Ratio of the number of users who click on a link over the total number of users that visited the page. CTR is a measure of the user engagement.

* Cold-start problem: The cold start problem concerns the personalized recommendations for users with no or few past history (new users). Providing recommendations to users with small past history becomes a difficult problem for CF models because their learning and predictive ability is limited. Multiple research have been conducted in this direction using hybrid models. These models use auxiliary information (multimodal information, side information, etc.) to overcome the cold start problem.

* Collaborative filtering algorithms: This type of recommendation system makes predictions of what might interest a person based on the taste of many other users. It assumes that if person X likes Snickers, and person Y likes Snickers and Milky Way, then person X might like Milky Way as well. See the [list of examples in Recommenders repository](../../examples/02_model_collaborative_filtering).

* Content-based filtering algorithms: This type of recommendation system focuses on the products themselves and recommends other products that have similar attributes. Content-based filtering relies on the characteristics of the products themselves, so it doesn’t rely on other users to interact with the products before making a recommendation. See the [list of examples in Recommenders repository](../../examples/02_model_content_based_filtering).

* Conversion rate: Optimizing for conversion rate maximizes the likelihood that the user purchases the recommended item; if you want to increase the number of purchases per session, optimize for conversion rate.

* Diversity metrics:

* Explicit interaction data: When a user explicitly rate an item, typically between 1-5, the user is giving a value on the likeliness of the item. In retail, this kind of data is not very common.

* Hybrid filtering algorithms: This type of recommendation system can implement a combination fo any two of the above systems. See the [list of examples in Recommenders repository](../../examples/02_model_hybrid).

* Implicit interaction data: Implicit interactions are views or clicks that show a certain interest of the user about a specific items. These kind of data is more common but it doesn't define the intention of the user as clearly as the explicit data.

* Item information: These include information about the item, some examples can be SKU, description, brand, price, etc.

* Knowledge-base algorithms: ...

* Knowledge graph data: ...

* Long tail products: Typically, the shape of items interacted in retail follow a long tail distribution [1,2]....

* Multi-Variate Test (MVT): Methodology to evaluate the performance of a system in production. It is similar to A/B testing, with the difference that instead of having two test groups, MVT has multiples groups. 

* Online metrics: 

* Offline metrics:

* Ranking metrics:

* Rating metrics:

* Revenue per order: The revenue per order optimization objective is the default optimization objective for the "Frequently bought together" recommendation model type. This optimization objective cannot be specified for any other recommendation model type.

* User information: These include all information that define the user, some examples can be name, address, email, demographics, etc. 

## References and resources

[1] Aggarwal, Charu C. Recommender systems. Vol. 1. Cham: Springer International Publishing, 2016.
[2]. Park, Yoon-Joo, and Alexander Tuzhilin. "The long tail of recommender systems and how to leverage it." In Proceedings of the 2008 ACM conference on Recommender systems, pp. 11-18. 2008. [Link to paper](http://people.stern.nyu.edu/atuzhili/pdf/Park-Tuzhilin-RecSys08-final.pdf).
[3]. Armstrong, Robert. "The long tail: Why the future of business is selling less of more." Canadian Journal of Communication 33, no. 1 (2008). [Link to paper](https://www.cjc-online.ca/index.php/journal/article/view/1946/3141).






