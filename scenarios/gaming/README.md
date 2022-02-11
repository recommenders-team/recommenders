# Recommendation Systems for Gaming

Gaming is one of the first industries that adopted AI, popularized by the development and adoption of arcade games during the 80s. The AI component was usually generated via the non-player characters (NPCs), that had to act intelligently and react to the player's actions. However, it is not until recently that recommendation systems have been applied to computer games. 

## Scenarios

In this section we will describe the most common scenarios in recommendation systems for gaming.

### Personalized game recommendation

A common scenario is to recommend games to a user based on their past interactions with other games. In this scenario, the user is represented by a user profile, and the game is represented by a game profile.  The user profile contains information about the user's past interactions with the game, and the game profile contains information about the game's popularity and other attributes. This scenario is typically shown on the game store homepage (for example, the store of XBox), or it can be sent to a user via a personalized newsletter. A large number of algorithms in this repository such as [SAR](../../examples/00_quick_start/sar_movielens.ipynb), [BPR](../../examples/02_model_collaborative_filtering/cornac_bpr_deep_dive.ipynb), and [NCF](../../examples/00_quick_start/ncf_movielens.ipynb) can be used for personalized game recommendation.


### Personalized item recommendation

The addition of micro-transactions inside games have produced a complete shift in the business model of games. Traditionally, most games were based on a one-off purchase, and users got all the content. Nowadays, many games allow players to buy items while they are playing via micro-transactions. Personalized item recommendations can be used to increase the player engagement, tailor the content to the player's interests, and increase the player's retention. In addition, these recommendation provide benefits for studios, which will increase their conversion rate and revenue. These systems also take into account the players past behavior and their interactions with items. The same set of algorithms used in personalized game recommendation can be used for personalized item recommendation, by just adapting the dataset.

### Similar items

A related scenario to personalized item recommendation is similar item recommendations. In this case the recommendations are based on item feature similarity. Some features that can be taken into account are price, item category, item usage, etc. The business usecase for similar items can be increasing the likelihood of buying or avoiding losing a sale by showing an alternative item with a lower price (down-selling). The set of algorithms that can be used in this scenario are those based on content-based filtering, such as [LightGBM](../../examples/00_quick_start/lightgbm_tinycriteo.ipynb), [VW](../../examples/02_model_content_based_filtering/vowpal_wabbit_deep_dive.ipynb) or even [xDeepFM](../../examples/00_quick_start/xdeepfm_criteo.ipynb).

### Next best action prediction

An interesting scenario is next best action prediction. In this scenario, what is recommended is the most beneficial next action for the player. From the technical point of view, this can be implemented using collaborative filtering algorithms, such as [SAR](../../examples/00_quick_start/sar_movielens.ipynb), [BPR](../../examples/02_model_collaborative_filtering/cornac_bpr_deep_dive.ipynb), and [NCF](../../examples/00_quick_start/ncf_movielens.ipynb).

## Data and evaluation

Datasets used in gaming recommendations usually include [user information](../../GLOSSARY.md), [item information](../../GLOSSARY.md) and [interaction data](../../GLOSSARY.md), among others. 

For evaluation of the algorithms, it is common to use [ranking metrics](../../GLOSSARY.md). For measuring the business impact, Average Revenue Per Paying User (ARPPU) is one of the main metrics used. Other business metrics are conversion rate, average revenue per user (ARPU), daily active users and monthly active users.
