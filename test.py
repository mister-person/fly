import os
from pathlib import Path
import sys
from pandas.io.formats.style import pd
from Drosophila_brain_model.model import run_exp
import Drosophila_brain_model.model as model
import Drosophila_brain_model.utils as utl
from brian2 import Hz, ms
config = {
     'path_res'  : './results',                              # directory to store results,
     'path_comp' : './Drosophila_brain_model/Completeness_783.csv',        # csv of the complete list of Flywire neurons,
     'path_con'  : './Drosophila_brain_model/Connectivity_783.parquet',    # connectivity data,
     'n_proc'    : -1,                                               # number of CPU cores (-1: use all),
}
 
default_params = model.default_params
default_params['t_run'] = 1000 * ms
default_params['n_run'] = 1

neu_sugar = [
    720575940616885538,
    720575940630233916,
    720575940609645124,
    720575940611875570,
    720575940612670570,
    720575940616167218,
    720575940617000768,
    720575940621502051,
    720575940621754367,
    720575940622486922,
    720575940627490663,
    720575940629176663,
    720575940632425919,
    720575940632889389,
    720575940637568838,
    720575940638202345,
    720575940639332736,
    720575940607347634,
    720575940608305161,
    720575940609476562,
    720575940609919897,
    720575940610788069,
    720575940612010137,
    720575940612422579,
    720575940612579053,
    720575940613601698,
    720575940613996959,
    720575940616177458,
    720575940616742657,
    720575940616811265,
    720575940617181725,
    720575940617674909,
    720575940617857694,
    720575940617937543,
    720575940618601782,
    720575940619287278,
    720575940622136022,
    720575940622405708,
    720575940622413508,
    720575940622825736,
    720575940622902535,
    720575940623138485,
    720575940623172843,
    720575940623629292,
    720575940624963786,
    720575940625203504,
    720575940626674182,
    720575940627907883,
    720575940628330256,
    720575940628853239,
    720575940629025324,
    720575940629388135,
    720575940630553415,
    720575940630797113,
]
# neu_sugar = []

# activate sugar sensing neurons
exp_name = "test783"
path_save = Path(config["path_res"]) / '{}.parquet'.format(exp_name)
if not os.path.isfile(path_save):
    run_exp(exp_name=exp_name, neu_exc=neu_sugar, params=default_params, **config)

output = pd.read_parquet(path_save)

print(output)

