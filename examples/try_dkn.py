
"""This script parse and run train function"""

from reco_utils.recommender.deeprec.deeprec_utils import *
from reco_utils.recommender.deeprec.models.dkn import *
from reco_utils.recommender.deeprec.IO.dkn_iterator import *

if __name__ == '__main__':
    cur_dirname = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(cur_dirname, '../tests/resources/deeprec/dkn')
    yaml_file = os.path.join(data_path, r'dkn.yaml')
    train_file = os.path.join(data_path, r'final_train_with_entity.txt')
    valid_file = os.path.join(data_path, r'final_test_with_entity.txt')
    wordEmb_file = os.path.join(data_path, r'word_embeddings_100.npy')
    entityEmb_file = os.path.join(data_path, r'TransE_entity2vec_100.npy')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/deeprec/', data_path, 'dknresources.zip')

    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, entityEmb_file=entityEmb_file)
    print(hparams)
    input_creator = DKNTextIterator

    model =DKN(hparams, input_creator)

    print(model.run_eval(valid_file))
    model.fit(train_file, valid_file)
    print(model.run_eval(valid_file))
