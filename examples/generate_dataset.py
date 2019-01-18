
from reco_utils.dataset.data_simulator import *


def generate_data():
    cur_dirname = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(cur_dirname, r'../tests/resources/deeprec/synthetic/')
    os.makedirs(output_dir, exist_ok=True)

    generator = DataGenerator(field_num=10, feature_num=1000, dim=10)
    generator.gen_embeddings()
    generator.gen_patterns(max_pattern_num=5, max_order=2, skew=3)
    generator.write_patterns_to_file(os.path.join(output_dir, 'patterns.csv'))
    generator.write_field2featurelist_to_file(os.path.join(output_dir, 'field2featurelist.tsv'))
    generator.write_embeddings_to_file(os.path.join(output_dir, 'embedding.txt'))
    generator.gen_instances_to_file(100000, os.path.join(output_dir, 'instances.csv'))
    convert2FFMformat(os.path.join(output_dir, 'instances.csv'),
                      os.path.join(output_dir, 'instances_ffm.csv'))
    split_file(
        os.path.join(output_dir, 'instances_ffm.csv'),
        output_dir,
        [0.7, 0.85, 1]
    )



if __name__ == '__main__':
    generate_data()