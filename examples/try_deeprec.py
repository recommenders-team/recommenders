
"""This script parse and run train function"""

from reco_utils.recommender.deeprec.deeprec_utils import *
from reco_utils.recommender.deeprec.models.xDeepFM import *
from reco_utils.recommender.deeprec.IO.iterator import *

if __name__ == '__main__':
    cur_dirname = os.path.dirname(os.path.realpath(__file__))
    yaml_file = os.path.join(cur_dirname, r'../tests/resources/deeprec/xDeepFM.yaml')
    train_file = os.path.join(cur_dirname, r'../tests/resources/deeprec/synthetic/part_0')
    valid_file = os.path.join(cur_dirname, r'../tests/resources/deeprec/synthetic/part_1')
    test_file = os.path.join(cur_dirname, r'../tests/resources/deeprec/synthetic/part_1')
    output_file = os.path.join(cur_dirname, r'../tests/resources/deeprec/synthetic/output.txt')

    hparams = prepare_hparams(yaml_file, FEATURE_COUNT=1000, FIELD_COUNT=10, cross_l2=0.0001, embed_l2=0.0001, learning_rate=0.001)
    print(hparams)

    input_creator = FFMTextIterator

    model = XDeepFMModel(hparams, input_creator)

    #model.load_model(r'your_model_path')

    print(model.run_eval(train_file))
    model.fit(train_file, valid_file)

    model.predict(test_file, output_file)