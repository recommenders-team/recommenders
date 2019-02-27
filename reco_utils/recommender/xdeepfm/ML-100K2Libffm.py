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

def create_final_file(filepath):
    with open(filepath, "r", encoding="ISO-8859-1") as train, \
         open(filepath+".final", "w", encoding="utf8") as train_final:
        for line in train:
            line = line.strip().split("\t")
            rating = line[2]
            if int(rating) > 3:
                rating = "1"
            else:
                rating = "0"
            userid = line[0]
            itemid = line[1]
            final_str = rating + " " + user_feature_dict[userid] + " " + item_feature_dict[itemid] + "\n"
            train_final.write(final_str)

if __name__ == "__main__":
    user_feature_dict, user_feature_num = create_userid_dict("ml-100k/ml-100k/u.user", 3)
    item_feature_dict, item_feature_num = create_itemid_dict("ml-100k/ml-100k/u.item", 4)
    print("feature_num: " + str(user_feature_num+item_feature_num))
    create_final_file("ml-100k/ml-100k/ua.base")
    create_final_file("ml-100k/ml-100k/ua.test")
