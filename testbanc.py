import os
from pathlib import Path
from pandas.io.formats.style import pd
import sys
from Drosophila_brain_model.model import run_exp
import Drosophila_brain_model.model as model
from brian2 import ms
config = {
     'path_res'  : './results',                              # directory to store results,
     'path_comp' : './data/banc_completeness.csv',        # csv of the complete list of Flywire neurons,
     'path_con'  : './data/banc_connectivity.parquet',    # connectivity data,
     'n_proc'    : -1,                                               # number of CPU cores (-1: use all),
}
 
default_params = model.default_params
default_params['t_run'] = 1000 * ms
default_params['n_run'] = 1

neu_sugar = [# banc dataset
    720575941593536704,
    720575941476677697,
    720575941535434762,
    720575941413377169,
    720575941458665887,
    720575941435032864,
    720575941500374690,
    720575941557486244,
    720575941628869414,
    720575941536716776,
    720575941600788521,
    720575941716033067,
    720575941631160556,
    720575941607298413,
    720575941412032623,
    720575941489703733,
    720575941603351030,
    720575941649119544,
    720575941590223358,
]
# neu_sugar = []

# activate sugar sensing neurons
exp_name = "testbanc"
path_save = Path(config["path_res"]) / '{}.parquet'.format(exp_name)
if not os.path.isfile(path_save):
    run_exp(exp_name=exp_name, neu_exc=neu_sugar, params=default_params, **config)

output = pd.read_parquet(path_save)

print(output)

