# SARplus

Simple Algorithm for Recommendation (SAR) is a neighborhood based
algorithm for personalized recommendations based on user transaction
history. SAR recommends items that are most **similar** to the ones
that the user already has an existing **affinity** for. Two items are
**similar** if the users that interacted with one item are also likely
to have interacted with the other. A user has an **affinity** to an
item if they have interacted with it in the past.

SARplus is an efficient implementation of this algorithm for Spark.
More details can be found at
[sarplus@microsoft/recommenders](https://github.com/microsoft/recommenders/tree/main/contrib/sarplus).
