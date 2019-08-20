import pandas as pd
import json
import time
import sys


"""
We use the 1m dataset to train the model, but the 1m dataset has no TMDB ids. To fix this,
we use the 20m dataset's corresponding TMDB ids and add them to the data corresponding to
the 1m dataset. This script generates the csvs used in the REEL app for movies and links.
"""


def translate(id):
    if id in id_to_id:
        return id_to_id[id]
    else:
        return -1


id_to_id = {}
# links.csv refers to the links file for the 20m movielens dataset
try:
    with open('links.csv') as f1:
        df1 = pd.read_csv(f1, sep=',', lineterminator='\n')
        for x in df1.iterrows():
            if x[1].isnull().values.any():
                id_to_id[int(x[1]["movieId"])] = -1
            else:
                id_to_id[int(x[1]["movieId"])] = int(x[1]["tmdbId"])
except:
    print("Can't find links.csv in current directory, now exiting...")
    time.sleep(5)
    sys.exit(0)

# movies.dat refers to the movie file for the 1m movielens dataset
try:
    with open ('movies.dat') as f:
        df = pd.read_csv(f, sep='::', lineterminator='\n', engine='python')
except:
    print("Can't find movies.dat in current directory, now exiting...")
    time.sleep(5)
    sys.exit(1)

# edited movie csv to populate the database
df.to_csv(r'new_movies.csv')

# now we just want to use the IMDB ids as each row
with open ('links.csv') as f2:
    links = pd.read_csv(f2, sep=',', lineterminator='\n')
    df2 = links[links.columns[1:2]].copy()
    df2['tmdbID'] = df.apply(lambda x : translate(x[0]), axis=1)
    df2['movieId'] = df.apply(lambda x : x[0], axis=1)

# edited links csv to populate the database
df2.to_csv(r'new_links.csv')

print("Successfully created new files!")


