import datetime

# Time since epoch in seconds
EPOCH = datetime.datetime.utcfromtimestamp(0)
# Default value for time decay parameter in SAR
TIME_DECAY_COEFFICIENT = 30
# Switch to trigger groupby in TimeDecay calculation
TIMEDECAY_FORMULA = False
# cooccurrence matrix threshold
THRESHOLD = 1
# Current time
# TIME_NOW = (datetime.datetime.now() - EPOCH).total_seconds()
TIME_NOW = None
# Default names for functions which change the item-item cooccurrence matrix
SIM_COOCCUR = "cooccurrence"
SIM_JACCARD = "jaccard"
SIM_LIFT = "lift"

HASHED_ITEMS = "hashedItems"
HASHED_USERS = "hashedUsers"

