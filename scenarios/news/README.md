# Recommendation systems for News

Online news services such as Microsoft News have become popular platforms for news reading. However, the massive continuously generated news information brings heavy news information overload to users. Personalized news recommendation can help target users with their interested news. It has been applied to many online news platforms, and effectively improves the news reading experience of many users.

## Scenarios

Next we will describe several most common retail scenarios and main considerations when applying recommendations in news.

### Personalized recommendation

A major task in personalized news recommendation is to predict which news articles a user is most likely to read based on the personal interest of this user. Users’ personal interest in news is usually inferred from their behaviors on the online news platforms, such as their clicked news articles. This scenario widely exists in the personalized news websites and feeds for news items selection and ranking. The models in this repo such as [NAML](../../examples/00_quick_start/naml_synthetic.ipynb), [NRMS](../../examples/00_quick_start/nrms_synthetic.ipynb), [DKN](../../examples/00_quick_start/dkn_MIND_dataset.ipynb) and [LSTUR](../../example/00_quick_start/lstur_synthetic.ipynb) can be used for personalized news recommendation.

News recommendation has two major differences with the general recommendation.
1. News recommendation has serious cold-start problem. New news articles are continuously emerging, and existing news articles will expire quickly. Thus, news recommendation always faces new items, which are very challenging for ID-based recommender systems.  
2. In news recommendation it is not optimal to represent items (i.e., news articles) using handcrafted features like IDs. In addition, news articles have rich texts. NLP methods are important to learn news content representations from news texts. 

## Data and evaluation

Datasets used in news recommendation usually include user historical clicked news, [news information](../../GLOSSARY.md) and [interaction data](../../GLOSSARY.md), among others.

To measure the performance of the recommender, it is common to use [ranking metrics](../../GLOSSARY.md). In production, the business metrics used are [CTR](../../GLOSSARY.md), etc. To evaluate a model's performance in production in an online manner, [A/B testing](../../GLOSSARY.md) is often applied.

To measure the performance of the recommender, it is common to use [ranking metrics](../../GLOSSARY.md). In production, the business metrics used are [CTR](../../GLOSSARY.md) and [revenue per order](../../GLOSSARY.md). To evaluate a model's performance in production in an online manner, [A/B testing](../../GLOSSARY.md) is often applied.