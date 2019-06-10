import pandas as pd

def create_dict(f_set):
    f_dict = dict()
    for i, j in enumerate(f_set):
        f_dict[j] = i+1
    return f_dict

def create_userid_dict(filepath, field_num):
    # feature_dict
    f_set_list = []
    for i in range(field_num):
        f_set_list.append(set())
    with open(filepath, "r", encoding="ISO-8859-1") as user:
        for line in user:
            line = line.strip().split("|")
            for i in range(1, field_num + 1):
                f_set_list[i-1].add(line[i])
    f_dict_list = []
    feature_num = 0
    for i in range(field_num):
        f_dict_list.append(create_dict(f_set_list[i]))
        feature_num += len(f_dict_list[i])

    user_feature_dict = dict()
    with open(filepath, "r", encoding="ISO-8859-1") as user:
        for line in user:
            line = line.strip().split("|")
            feature = ""
            for i in range(1, field_num + 1):
                feature = feature + " " + str(i) + ":" + str(f_dict_list[i-1][line[i]]) + ":1"
            user_feature_dict[line[0]] = feature[1:]
    return user_feature_dict, feature_num

def create_itemid_dict(filepath, begin_fildnum):
    # feature_dict
    ym_set = set()
    with open(filepath, "r", encoding="ISO-8859-1") as user:
        for line in user:
            line = line.strip().split("|")
            release_date = line[2].split("-")
            release_year_month = "-".join(release_date[1:])
            ym_set.add(release_year_month)
    ym_dict = create_dict(ym_set)
    feature_num = len(ym_dict) + 19

    item_feature_dict = dict()
    with open(filepath, "r", encoding="ISO-8859-1") as user:
        for line in user:
            line = line.strip().split("|")
            genre = line[-19:]
            release_date = line[2].split("-")
            release_year_month = "-".join(release_date[1:])
            feature = str(begin_fildnum) + ":" + str(ym_dict[release_year_month]) + ":1"
            for i in range(len(genre)):
                if genre[i] == "1":
                    feature = feature + " " + str(begin_fildnum+1) + ":" + str(i+1) + ":1"
            item_feature_dict[line[0]] = feature
    return item_feature_dict, feature_num





def user_item_pairs(
        user_df: object,
        item_df: object,
    #user_col=DEFAULT_USER_COL,
    #item_col=DEFAULT_ITEM_COL,
        user_col: object,
        item_col: object,
        user_item_filter_df: object = None,
        shuffle: object = True
) -> object:
    """Get all pairs of users and items data.
    Args:
        user_df (pd.DataFrame): User data containing unique user ids and maybe their features.
        item_df (pd.DataFrame): Item data containing unique item ids and maybe their features.
        user_col (str): User id column name.
        item_col (str): Item id column name.
        user_item_filter_df (pd.DataFrame): User-item pairs to be used as a filter.
        shuffle (bool): If True, shuffles the result.
    Returns:
        pd.DataFrame: All pairs of user-item from user_df and item_df, excepting the pairs in user_item_filter_df
    """

    # Get all user-item pairs
    user_df['key'] = 1
    item_df['key'] = 1
    users_items = user_df.merge(item_df, on='key')

    user_df.drop('key', axis=1, inplace=True)
    item_df.drop('key', axis=1, inplace=True)
    users_items.drop('key', axis=1, inplace=True)

    # Filter
    if user_item_filter_df is not None:
        user_item_col = [user_col, item_col]
        users_items = users_items.loc[
            ~users_items.set_index(user_item_col).index.isin(user_item_filter_df.set_index(user_item_col).index)
        ]

    if shuffle:
        users_items = users_items.sample(frac=1).reset_index(drop=True)

    return users_items

def create_final_file(filepath,flag, field_num): ##scoring_flag and filed_num are only needed for top N recommendation
    with open(filepath, "r", encoding="ISO-8859-1") as train, \
         open(filepath + "."+ flag+".final", "w", encoding="utf8") as train_final:

        if flag == 'regression': ## use rating as predition value for regression
            for line in train:
                line = line.strip().split("\t")
                rating = line[2]+'.00' #changed to .00 format
                userid = line[0]
                itemid = line[1]
                final_str = rating + " " + user_feature_dict[userid] + " " + item_feature_dict[itemid] + "\n"
                train_final.write(final_str)
        ## 2-class classification  rating>3 -> class=1
        elif flag == 'classification':

            for line in train:
                line = line.strip().split("\t")
                rating = line[2]
                userid = line[0]
                itemid = line[1]

                if int(rating) > 3:
                    rating = "1.00"
                else:
                    rating = "0.00"

                final_str = rating + " " + user_feature_dict[userid] + " " + item_feature_dict[itemid] + "\n"
                train_final.write(final_str)
        elif flag =='classification_topN':
            
            USER_COL='userID'
            ITEM_COL='itemID'
            RATING_COL='rating'
            TS_COL = 'timestamp'
            
            ## top_N recommendation
            f_set_list = []
            for i in range(field_num):#{0,1}
                f_set_list.append(set())

            for line in train:
                line = line.strip().split("\t")
                for i in range(0, field_num ): # i"0,1"
                    f_set_list[i].add(line[i]) ##f_set_list[0] is user and f_set_list[1] is movie id

            user_df = pd.DataFrame(list(f_set_list[0]),columns=[USER_COL])
            item_df = pd.DataFrame(list(f_set_list[1]),columns=[ITEM_COL])
            user_items = user_item_pairs(user_df,item_df,user_col=USER_COL,item_col=ITEM_COL)

            user_items[USER_COL] = user_items[USER_COL].astype(int)
            user_items[ITEM_COL] = user_items[ITEM_COL].astype(int)

            data = pd.read_csv(filepath, sep='\t', names=[USER_COL, ITEM_COL, RATING_COL, TS_COL])

           ## merge all user_item data with movielens data
            user_item_rating= pd.merge(user_items,data, how='left', on=[USER_COL,ITEM_COL]).fillna(0).drop(columns=[TS_COL])
            
            user_item_rating.drop(columns=[RATING_COL]).to_csv('user_item.csv', sep='\t') ##order is a little weird need to check
            
            print(len(user_item_rating))
            #user_item_rating.columns = ['']*len(user_item_rating.columns) #remove header

            for line in range(len(user_item_rating)):

                rating = user_item_rating.iloc[line,2]
                #print(user_item_rating.loc[1,"userID"])
                #(user_item_rating.iloc[1,1])
                userid = user_item_rating.iloc[line,0].astype(str)
                itemid = user_item_rating.iloc[line,1].astype(str)

                if int(rating) > 3:
                    rating = "1.00"
                else:
                    rating = "0.00"

                final_str = rating + " " + user_feature_dict[userid] + " " + item_feature_dict[itemid] + "\n"
                train_final.write(final_str)



if __name__ == "__main__":
    #user_feature_dict, user_feature_num = create_userid_dict("ml-100k/ml-100k/u.user", 3)
    #item_feature_dict, item_feature_num = create_itemid_dict("ml-100k/ml-100k/u.item", 4)
    user_feature_dict, user_feature_num = create_userid_dict("./ml-100k/u.user", 3)
    item_feature_dict, item_feature_num = create_itemid_dict("./ml-100k/u.item", 4)
    
    print("feature_num: " + str(user_feature_num+item_feature_num))

    ## if do regression
    ## transform training data ua.base into ua.base.regerssion for regression
    create_final_file("./ml-100k/ua.base",'regression',0) #0 is not used for regression
    create_final_file("./ml-100k/ua.test",'regression',0) #0 is not used for regression
    
    #### if do classification
    #create_final_file("./ml-100k/ua.base",'classification',0) #0 is not used
    #create_final_file("./ml-100k/ua.test",'classification',0) #0 is not used
    
    ### if do recommendation on top N items
    #create_final_file("./ml-100k/ua.test",'classification_topN',2) #2 is number of fields
    


