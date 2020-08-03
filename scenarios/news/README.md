# Recommendation Systems for News

While online news services have become a major source of information for millions of people, the massive amount of continuously generated content brings heavy information overload to users. Personalized news recommendation, which predicts which news articles a user is likely to read, can help reduce the information overload and improve the user experience. 

There are several aspects to consider when developing news recommendation systems.  Especially:
1. Cold-start is a major challenge for news recommendation. New articles are continuously emerging, and existing news articles will expire quickly. Effective representation and recommendation of new articles is essential to good performance in news recommendation.  
2. In news recommendation it is not optimal to represent items (i.e., news articles) using handcrafted features like IDs. In addition, news articles have rich texts. NLP methods are important to learn news content representations from news texts.
For more details, please refer to [this ACL paper](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf).


## Data and evaluation

Datasets used in news recommendation usually include articles read by a user, which can be used as a proxy of user interest.  In addition to such [implicit interation data](../../GLOSSARY.md), news datasets could also include [news information](../../GLOSSARY.md).  

To measure the performance of the recommender, it is common to use [ranking metrics](../../GLOSSARY.md) such as MRR and nDCG. In production, business metrics used may include [CTR](../../GLOSSARY.md) and engagement time. To evaluate a model's performance in production in an online manner, [A/B testing](../../GLOSSARY.md) is often applied.

# Microsoft News Dataset (MIND) and MIND News Recommendation Competition

To support the advancement of open research in news recommendation, Microsoft has made [MIND](https://msnews.github.io/) (Microsoft News Dataset) available to the research community.  MIND is a large-scale dataset on English news that contains 161 thousand news articles, over 3 million entities and 1 million uses.  More details of the dataset can be find in [here](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf).

In conjunction with MIND, Microsoft also launched [MIND News Competition](https://msnews.github.io/competition.html).  The development phase of the competition srarts on July 20, 2020.  The final test phase starts on August 21, 2020, and the competition ends on September 4, 2020.  More details can be found on the MIND News Competition page above.

To help competition participants get started, we have made available in this repo several baselines models for MIND and MIND News Competition.  These include complete notebooks for [DKN](../../examples/00_quick_start/dkn_MIND.ipynb), [LSTUR](../../example/00_quick_start/lstur_MIND.ipynb), [NAML](../../examples/00_quick_start/naml_MIND.ipynb), [NPA](../../examples/00_quick_start/npa_MIND.ipynb) and [NRMS](../../examples/00_quick_start/nrms_MIND.ipynb) on MIND sample datasets.