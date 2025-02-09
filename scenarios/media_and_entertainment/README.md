<!--
Copyright (c) Recommenders contributors.
Licensed under the MIT License.
-->

# Recommendation systems for Media and Entertainment

Recommendation systems are fundamental to modern media and entertainment platforms. For example, Netflix reports that their recommendation system [saves them $1 billion per year](https://www.netflix.com/tudum/articles/netflix-recommendations) in customer retention.

## Scenarios

Here are the primary scenarios and considerations for media and entertainment recommendations.

### Content discovery

The main task is helping users discover relevant content from vast libraries of movies, shows, music, or games. This is typically shown on home pages and personalized feeds. Models like [ALS](../../examples/00_quick_start/als_movielens.ipynb), [NCF](../../examples/00_quick_start/ncf_movielens.ipynb), and [SAR](../../examples/00_quick_start/sar_movielens.ipynb) are commonly used.

### Next-item recommendation

This scenario predicts what content a user might want to consume next, such as the next episode, similar songs, or related games. This is particularly important for maintaining user engagement and reducing churn.

### Personalized playlists

For music and video platforms, this involves creating custom playlists based on user preferences and listening/viewing history. Both collaborative filtering and content-based approaches can be used here.

### Live content recommendations

This scenario involves recommending live content (streams, events, broadcasts) based on user interests and current popularity. Real-time processing and trending detection are crucial here.

## Data and evaluation

Key data includes user profiles, content metadata (genre, actors, duration), viewing/listening history, and engagement metrics (watch time, ratings, shares).

Common evaluation metrics include engagement time, retention rate, and user satisfaction. For music and video platforms, metrics like [stream time](../../GLOSSARY.md) and [completion rate](../../GLOSSARY.md) are particularly important. [A/B testing](../../GLOSSARY.md) is standard practice for evaluating recommendations.

## Other considerations

Media recommendations need to balance between promoting new content and leveraging known user preferences. Content licensing windows, regional availability, and device compatibility must also be considered. Additionally, recommendations should account for household sharing of accounts and varying preferences based on viewing context.

