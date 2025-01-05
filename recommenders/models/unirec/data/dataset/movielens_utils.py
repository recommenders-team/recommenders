# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.


from tqdm import tqdm


def merge_category(
    data, col_item="movieId", col_category="cateId", min_item_in_cate=50
):
    """Get mapping cate2items and item2cate, and merge categories containing a small number of
        items (lower than min_item_in_cate) into one category.

    Args:
        data (pd.DataFrame): The interactions of user-item, containing 'userId', 'movieId', 'rating', 'timestamp', 'cateId'.
        col_item (str): The column name of item id.
        col_category (str): The column name of category id.
        min_item_in_cate (int): The minimum number of items in each categories. Defaults to 50.

    Returns:
        dict: The mapping from category to category index.
        dict: The mapping from item to category.
        int: The number of categories.
    """
    cate2item = {}

    for cate, item in tqdm(
        zip(data[col_category], data[col_item]), desc="get cate2items"
    ):
        for c in cate:
            if c not in cate2item.keys():
                cate2item[c] = set([])
            cate2item[c].add(item)

    large_cate, small_cate = [], []
    for cate, items in cate2item.items():
        if len(items) <= min_item_in_cate:
            small_cate.append(cate)
        else:
            large_cate.append(cate)

    cate2idx = {x[1]: x[0] + 1 for x in enumerate(large_cate)}
    num_cates = len(large_cate) + 1
    for sc in small_cate:
        cate2idx[sc] = num_cates

    item2cate = {}
    for cate, item in tqdm(
        zip(data[col_category], data[col_item]), desc="get item2cate"
    ):
        new_cate = []
        for c in cate:
            new_cate.append(cate2idx[c])
        item2cate[item] = new_cate

    return cate2idx, item2cate, num_cates
