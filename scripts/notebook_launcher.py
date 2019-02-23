import sys
import os
import papermill as pm
import argparse
from azureml.core import Run

os.system(sys.executable + ' -m ipykernel install --user --name ipyazureml')

def execute_notebook(notebook_path, parameters):
    pm.execute_notebook(
        notebook_path,
        'outputs/output.ipynb',
        kernel_name='ipyazureml',
        parameters=parameters,
    )
    results = pm.read_notebook('outputs/output.ipynb').dataframe.set_index("name")["value"]

    run = Run.get_context()
    for key, value in results.items():
        run.log(key, value)
            
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook_path')
    FLAGS, unparsed = parser.parse_known_args()

    parameters= {unparsed[i]: unparsed[i+1] for i in range(0, len(unparsed), 2)}
    print(FLAGS.notebook_path, parameters)
    execute_notebook(notebook_path = FLAGS.notebook_path, 
                       parameters=parameters)


