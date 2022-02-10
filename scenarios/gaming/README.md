# Recommendation Systems for Gaming

Gaming is one of the first industries that adopted AI, popularized by the development and adoption of arcade games during the 80s. The AI component was usually generated via the non-player characters (NPCs), that had to act intelligently and react to the player's actions. However, it is not until recently that recommendation systems have been applied to computer games. 

## Scenarios

In this section we will describe the most common scenarios in recommendation systems for gaming.

### Personalized game recommendation

A common scenario is to recommend games to a user based on their past interactions with other games. In this scenario, the user is represented by a user profile, and the game is represented by a game profile.  The user profile contains information about the user's past interactions with the game, and the game profile contains information about the game's popularity and other attributes. This scenario is typically shown on the game store homepage (for example, the store of XBox), or it can be send to a user via a personalized newsletter. A large number of algorithms in this repository such as [SAR](../../examples/00_quick_start/sar_movielens.ipynb), [BPR](../../examples/02_model_collaborative_filtering/cornac_bpr_deep_dive.ipynb), [LightGBM](../../examples/00_quick_start/lightgbm_tinycriteo.ipynb) and [NCF](../../examples/00_quick_start/ncf_movielens.ipynb) can be used for personalized game recommendation.

### Personalized item recommendation

The addition of micro-transactions inside games have produced a complete shift in the business model of games. Traditionally, most games were based on a one-off purchase, and users got all the content. Nowadays, many games allow players to buy items via micro-transactions. Personalized item recommendations can be used to increase the player engagement, tailor the content to the player's interests, and increase the player's retention. In addition, these recommendation provide benefits for studios, which will increase their conversion rate and revenue. The same set of algorithms used in personalized game recommendation can be used for personalized item recommendation, by just adapting the dataset.

### Similar items

### Next best action prediction


[14:15, 10/02/2022] Miguel González-Fierro: These AI-powered interactive experiences are usually generated via non-player characters, or NPCs, that act intelligently or creatively -> next best action
[14:16, 10/02/2022] Miguel González-Fierro: In the coming times, it is predicted that the benefits of Artificial Intelligence on games will continue to grow and will be more positive than not, making games more immersive, realistic, and life-like. Intelligent game behavior will provide attractive features in terms of realistic movement and interactions between game characters and game players.
[14:17, 10/02/2022] Miguel González-Fierro: It is Artificial Intelligence that is building a future that is more about personalized gaming.
[14:18, 10/02/2022] Miguel González-Fierro: Game engineers use data analytics to make informed decisions by noting details like how often a player lands in a particular area, what character is being used the most, or what weapons and other items are being used frequently. It is certain that the AI developer’s decision depends majorly on the data analytics of a player’s behavior, game assets, and the environment.
[14:30, 10/02/2022] Miguel González-Fierro: Personalized marketing in gaming helps to increase the activity of the users and at the same time attract new ones. This may be achieved due to precise tailoring of the advertising message. To assure your ads are perceived correctly you need to know which players are ad responsive and which are not.