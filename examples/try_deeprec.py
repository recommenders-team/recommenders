
"""This script parse and run train function"""

from reco_utils.recommender.deeprec.deeprec_utils import *
from reco_utils.recommender.deeprec.models.xDeepFM import *
from reco_utils.recommender.deeprec.IO.iterator import *

if __name__ == '__main__':

    hparams = prepare_hparams(r'D:\projects\TKDD\deeprec\data\FFM_simulated\max_order_2\tmp\xDeepFM.yaml')

    print(hparams)

    input_creator = FFMTextIterator

    model = XDeepFMModel(hparams, input_creator)

    model.load_model(r'D:\projects\TKDD\deeprec\data\FFM_simulated\max_order_2\tmp\epoch_8')

    print(model.run_eval(r'D:\projects\TKDD\deeprec\data\FFM_simulated\max_order_2\valid2.txt'))
    #model.fit(r'D:\projects\TKDD\deeprec\data\FFM_simulated\max_order_2\train2.txt', r'D:\projects\TKDD\deeprec\data\FFM_simulated\max_order_2\valid2.txt')

    model.predict(r'D:\projects\TKDD\deeprec\data\FFM_simulated\max_order_2\valid2.txt', r'D:\projects\TKDD\deeprec\data\FFM_simulated\max_order_2\tmp\output\valid2.pred')