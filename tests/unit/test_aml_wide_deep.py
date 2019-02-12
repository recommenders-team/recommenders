from reco_utils.aml.wide_deep import parse_args


USER_COL = 'UserId'
ITEM_COL = 'MovieId'
RATING_COL = 'Rating'
ITEM_FEAT_COL = 'Genres'
PARAMS = {
    'TOP_K': 3,
    'DATA_DIR': './data',
    'TRAIN_PICKLE_PATH': 'train.pkl',
    'TEST_PICKLE_PATH': 'test.pkl',
    'MODEL_TYPE': 'wide_deep',
    'BATCH_SIZE': 128,
    'EPOCHS': 50,
    'USER_COL': USER_COL,
    'ITEM_COL': ITEM_COL,
    'ITEM_FEAT_COL': ITEM_FEAT_COL,
    'RATING_COL': RATING_COL,
    'METRICS': ['rmse', 'ndcg'],
    'LINEAR_OPTIMIZER': 'Ftrl',
    'LINEAR_OPTIMIZER_LR': 0.001,
    'LINEAR_L1_REG': 0.0001,
    'DNN_OPTIMIZER': 'Adam',
    'DNN_OPTIMIZER_LR': 0.001,
    'DNN_HIDDEN_LAYER_1': 256,
    'DNN_HIDDEN_LAYER_2': 256,
    'DNN_HIDDEN_LAYER_3': 256,
    'DNN_HIDDEN_LAYER_4': 128,
    'DNN_USER_DIM': 5,
    'DNN_ITEM_DIM': 5,
    'DNN_DROPOUT': 0.1,
    'DNN_BATCH_NORM': 1,
}


def test_parse_args():
    args = [
        '--top-k', str(PARAMS['TOP_K']),
        '--datastore', PARAMS['DATA_DIR'],
        '--train-datapath', PARAMS['TRAIN_PICKLE_PATH'],
        '--test-datapath', PARAMS['TEST_PICKLE_PATH'],
        '--user-col', PARAMS['USER_COL'],
        '--item-col', PARAMS['ITEM_COL'],
        '--rating-col', PARAMS['RATING_COL'],
        '--item-feat-col', PARAMS['ITEM_FEAT_COL'],
        '--metrics', *PARAMS['METRICS'],
        '--model-type', PARAMS['MODEL_TYPE'],
        '--linear-optimizer', PARAMS['LINEAR_OPTIMIZER'],
        '--linear-optimizer-lr', str(PARAMS['LINEAR_OPTIMIZER_LR']),
        '--linear-l1-reg', str(PARAMS['LINEAR_L1_REG']),
        '--dnn-optimizer', PARAMS['DNN_OPTIMIZER'],
        '--dnn-optimizer-lr', str(PARAMS['DNN_OPTIMIZER_LR']),
        '--dnn-hidden-layer-1', str(PARAMS['DNN_HIDDEN_LAYER_1']),
        '--dnn-hidden-layer-2', str(PARAMS['DNN_HIDDEN_LAYER_2']),
        '--dnn-hidden-layer-3', str(PARAMS['DNN_HIDDEN_LAYER_3']),
        '--dnn-hidden-layer-4', str(PARAMS['DNN_HIDDEN_LAYER_4']),
        '--dnn-user-embedding-dim', str(PARAMS['DNN_USER_DIM']),
        '--dnn-item-embedding-dim', str(PARAMS['DNN_ITEM_DIM']),
        '--dnn-batch-norm', '1',
        '--dnn-dropout', str(PARAMS['DNN_DROPOUT']),
        '--epochs', str(PARAMS['EPOCHS']),
        '--batch-size', str(PARAMS['BATCH_SIZE']),
    ]
    params = parse_args(args)

    # Test if the args and returned params are the same
    for k, v in params.items():
        assert v == PARAMS[k]
