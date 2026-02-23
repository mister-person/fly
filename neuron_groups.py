from collections import defaultdict
import pickle
import pandas as pd

rf_trochanter_flexor = [720575941399414563,720575941505568069,720575941450561358,720575941656372304,720575941391252886,720575941504396567,720575941579753849,720575941574013085,
    720575941564818688,720575941623147978,720575941527171000]
rf_trochanter_extensor = [720575941478281440,720575941508453761,720575941593802373,720575941593843589,720575941523368779,720575941411709073,720575941568546418,720575941667380979]
rf_tibia_extensor = [720575941488287797,720575941605019582]
rf_tibia_flexor = [720575941494146176,720575941547393825,720575941526437988,720575941551010695,720575941430029449,720575941453327785,720575941630970604,720575941447846990,720575941559281679,720575941557543918,720575941452885043,720575941590014007,720575941580731001,720575941574167803,720575941623218172,720575941549819710,720575941469045111]
rf_tarsus_depressor = [720575941571652616,720575941496126793,720575941722047290,720575941560573608]
rf_tarsus_levetator = [720575941561433746]
rf_long_tendon = [720575941640958053,720575941592581671,720575941488203209,720575941511197516,720575941501926706,720575941538128626,720575941501044055,720575941524966169]
rf_femur_reductor = [720575941631837856,720575941649909489,720575941608273226,720575941448798133,
    720575941451729274,720575941566924135] # ??????

#TODO cache this
def load_leg_neuron_groups(neuron_df, filters, leg_selector_columns, leg_selector, muscle_column, muscles, id_column):
    leg_neuron_groups: dict[str, list[int]] = defaultdict(list)
    for s in neuron_df.iloc:
        cell_class = (s[leg_selector_columns[0]], s[leg_selector_columns[1]])
        cell_type = s[muscle_column]
        for f in filters:
            if s[f[0]] != f[1]:
                break
        else:
            # if s["Class"] == "leg_motor_neuron":
            # if s["superclass"] == "vnc_motor":
            if cell_class in leg_selector:
                for muscle in muscles.keys():
                    if cell_type and muscle in cell_type:
                        name = leg_selector[cell_class] + "_" + muscles[muscle]
                        leg_neuron_groups[name].append(s[id_column])

    return leg_neuron_groups

def load_banc_leg_neuron_groups(filename: str, use_cache = True):
    if use_cache:
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            pass
    banc_legs = {
        ("left", "front_leg_motor_neuron"): "lf",
        ("right", "front_leg_motor_neuron"): "rf",
        ("left", "middle_leg_motor_neuron"): "lm",
        ("right", "middle_leg_motor_neuron"): "rm",
        ("left", "hind_leg_motor_neuron"): "lh",
        ("right", "hind_leg_motor_neuron"): "rh",
    }

    banc_muscles = {
        "trochanter_flexor",
        "trochanter_extensor",
        "tibia_flexor",
        "tibia_extensor",
        "tarsus_depressor",
        "tarsus_levetator",
        "long_tendon",
    }
    banc_muscles_dict = {a: a for a in banc_muscles}

    banc = pd.read_csv("./data/banc_neurons.csv")
    banc_neuron_groups = load_leg_neuron_groups(banc, (("Class", "leg_motor_neuron"),), ("Soma side", "Sub Class"), banc_legs, "Primary Cell Type", banc_muscles_dict, "Root ID")

    with open(filename, "wb") as f:
        pickle.dump(banc_neuron_groups, f)

    return banc_neuron_groups

def load_mbanc_leg_neuron_groups(filename: str, use_cache = True):
    if use_cache:
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            pass
    mbanc_legs = {
        ("L", "fl"): "lf",
        ("R", "fl"): "rf",
        ("L", "ml"): "lm",
        ("R", "ml"): "rm",
        ("L", "hl"): "lh",
        ("R", "hl"): "rh",
    }

    """
    'Acc. ti flexor MN', 'Acc. tr flexor MN', 'Fe reductor MN',
           'MNhl01', 'MNhl02', 'MNhl29', 'MNhl59', 'MNhl60', 'MNhl62',
           'MNhl64', 'MNhl65', 'MNhl87', 'MNhl88', 'None',
           'Pleural remotor/abductor MN', 'Sternal adductor MN',
           'Sternal anterior rotator MN', 'Sternal posterior rotator MN',
           'Sternotrochanter MN', 'Tergotr. MN', 'Ti extensor MN',
           'Ti flexor MN', 'Tr extensor MN', 'Tr flexor MN', 'ltm MN',
           'ltm1-tibia MN', 'ltm2-femur MN'
    """
    mbanc_muscles = {
        "Tr flexor": "trochanter_flexor",
        "Acc. tr flexor": "trochanter_flexor",
        "Tr extensor": "trochanter_extensor",

        "Ti flexor": "tibia_flexor",
        "Acc. ti flexor": "tibia_flexor",
        "Ti extensor": "tibia_extensor",

        "no tarsus :(": "tarsus_depressor",
        "still no tarsus :(": "tarsus_levetator",
        "ltm": "long_tendon",
    }

    mbanc = pd.read_feather("../flywire/body-annotations-male-cns-v0.9-minconf-0.5.feather")

    mbanc_neuron_groups = load_leg_neuron_groups(mbanc, (("superclass", "vnc_motor"),), ("somaSide", "subclass"), mbanc_legs, "mancType", mbanc_muscles, "bodyId")
    with open(filename, "wb") as f:
        pickle.dump(mbanc_neuron_groups, f)
    return mbanc_neuron_groups

banc_leg_neuron_groups = load_banc_leg_neuron_groups("data/banc_leg_neuron_groups.pickle")
print("banc leg neuron grousp", banc_leg_neuron_groups)
mbanc_leg_neuron_groups = load_mbanc_leg_neuron_groups("data/mbanc_leg_neuron_groups.pickle")

legs = ["lf", "rm", "lh", "rf", "lm", "rh"]

def by_leg_helper(dataset_name):
    group = mbanc_leg_neuron_groups if dataset_name == "mbanc" else banc_leg_neuron_groups
    by_leg = defaultdict(list)
    for key in group.keys():
        neurons = group[key]
        for leg in legs:
            if key.startswith(leg):
                for neuron in neurons:
                    by_leg[leg].append((key, neuron))
                break
        else:
            print("extra key", key)
        # dark_color.extend([(key, neuron) for neuron in neuron_groups.mbanc_leg_neuron_groups[key]])
    return by_leg

mbanc_by_leg = by_leg_helper("mbanc")
banc_by_leg = by_leg_helper("banc")

def leg_neurons_helper(dataset_name, leg_name):
    if dataset_name == "banc":
        leg_neurons = banc_leg_neuron_groups
    elif dataset_name == "mbanc":
        leg_neurons = mbanc_leg_neuron_groups
    else:
        raise Exception("unknown dataset")

    return (leg_neurons[leg_name + "_trochanter_flexor"] + 
        leg_neurons[leg_name + "_trochanter_extensor"] + 
        leg_neurons[leg_name + "_tibia_extensor"] + 
        leg_neurons[leg_name + "_tibia_flexor"] + 
        leg_neurons[leg_name + "_tarsus_depressor"] + 
        leg_neurons[leg_name + "_tarsus_levetator"] + 
        leg_neurons[leg_name + "_long_tendon"] + 
        leg_neurons[leg_name + "_femur_reductor"])

mbanc_rf_leg_neurons = leg_neurons_helper("mbanc", "rf")
mbanc_lf_leg_neurons = leg_neurons_helper("mbanc", "lf")

banc_rf_leg_neurons = leg_neurons_helper("banc", "rf")
banc_lf_leg_neurons = leg_neurons_helper("banc", "lf")

mbanc_leg_neurons = [x for xs in mbanc_leg_neuron_groups.values() for x in xs]
banc_leg_neurons = [x for xs in banc_leg_neuron_groups.values() for x in xs]

if __name__ == "__main__":
    banc_leg_neuron_groups = load_banc_leg_neuron_groups("data/banc_leg_neuron_groups.pickle", use_cache=False)
    mbanc_leg_neuron_groups = load_mbanc_leg_neuron_groups("data/mbanc_leg_neuron_groups.pickle", use_cache=False)
    print(list(mbanc_leg_neuron_groups.keys()))
    print(set(mbanc_leg_neuron_groups["rf_trochanter_flexor"]) == set(rf_trochanter_flexor))
    print(mbanc_leg_neuron_groups["rf_trochanter_flexor"], rf_trochanter_flexor)
    print(mbanc_rf_leg_neurons)
    print(mbanc_lf_leg_neurons)
    print(mbanc_leg_neuron_groups)

# missing ones: https://codex.flywire.ai/app/search?dataset=banc&filter_string=super_class+%3D%3D+motor+%26%26+sub_class+%3D%3D+front_leg_motor_neuron+%26%26+side+%3D%3D+right+%26%26+cell_type+%21%3D+tibia_extensor_SETi+%26%26+cell_type+%21%3D+tergopleural_promotor_pleural_promotor_miller_28_30+%26%26+cell_type+%21%3D+accessory_tibia_flexor+%26%26+cell_type+%21%3D+accessory_trochanter_flexor+%26%26+cell_type+%21%3D+trochanter_extensor+%26%26+cell_type+%21%3D+tergotrochanter_extensor+%26%26+cell_type+%21%3D+sternotrochanter_extensor+%26%26+cell_type+%21%3D+tibia_flexor+%26%26+cell_type+%21%3D+tarsus_depressor_B+%26%26+cell_type+%21%3D+trochanter_flexor+%26%26+cell_type+%21%3D+tarsus_levator_D+%26%26+cell_type+%21%3D+tarsus_depressor_retro_tarsus_depressor+%26%26+cell_type+%21%3D+long_tendon_muscle_A+%26%26+cell_type+%21%3D+long_tendon_muscle_B+%26%26+cell_type+%21%3D+tibia_extensor_FETi+%26%26+cell_type+%21%3D+femur_reductor+%26%26+cell_type+%21%3D+femur_reductor_tiny+%26%26+cell_type+%21%3D+long_tendon_muscle_B_femur+%26%26+cell_type+%21%3D+tarsus_depressor_dipalpha_ventralU&sort_by=&page_size=10

rf_leg_motor_neurons = [720575941428654255,720575941605019582,720575941556297811,720575941590014007,720575941494146176,720575941564818688,720575941508453761,720575941593802373,720575941593843589,720575941551010695,720575941571652616,720575941430029449,720575941499964555,720575941559281679,720575941411709073,720575941561433746,720575941391252886,720575941504396567,720575941691007896,720575941524966169,720575941574013085,720575941573215995,720575941631837856,720575941547393825,720575941399414563,720575941592581671,720575941592225831,720575941560573608,720575941579753849,720575941453327785,720575941559978797,720575941353301552,720575941501926706,720575941452885043,720575941488287797,720575941448798133,720575941527171000,720575941722047290,720575941549819710,720575941595712448,720575941505568069,720575941496126793,720575941608273226,720575941523368779,720575941511197516,720575941488203209,720575941447846990,720575941450561358,720575941656372304,720575941623147978,720575941501044055,720575941518204888,720575941478281440,720575941526437988,720575941640958053,720575941531180261,720575941566924135,720575941630970604,720575941557543918,720575941649909489,720575941568546418,720575941538128626,720575941667380979,720575941593715569,720575941636735861,720575941469045111,720575941580731001,720575941451729274,720575941623218172,720575941625808330,720575941574167803]
lf_leg_motor_neurons = [720575941455787245,720575941524365337,720575941501929010,720575941563973174,720575941527847608,720575941481179066,720575941561977088,720575941508205185,720575941479472258,720575941494164864,720575941505660801,720575941553516167,720575941573516296,720575941548546696,720575941460703245,720575941461333648,720575941651501585,720575941435830546,720575941611361811,720575941515355795,720575941411838612,720575941464925462,720575941561204626,720575941551356570,720575941595856540,720575941639506207,720575941506261538,720575941498960546,720575941627155238,720575941668683687,720575941450546857,720575941406488494,720575941408338862,720575941720479034,720575941720492346,720575941545479868,720575941486064962,720575941570833736,720575941642303944,720575941610029642,720575941488205769,720575941524748877,720575941524746573,720575941478577231,720575941511883985,720575941464194902,720575941518133464,720575941542454876,720575941447389277,720575941583323358,720575941583222623,720575941531504357,720575941643180901,720575941489123047,720575941610059238,720575941536923624,720575941532427754,720575941590872171,720575941518851694,720575941516874094,720575941402088304,720575941667370739,720575941568724980,720575941639281525,720575941469052535,720575941621725948,720575941483515389,720575941549009663]
    
banc_e1 = [720575941669833905,720575941560438643,720575941504247575,720575941527002649,720575941554038555,720575941552245822] #e1
banc_e2 = [720575941469024064,720575941487873488,720575941545638505,720575941625121674,720575941437338412,720575941519468654] #e2
banc_i1 = [720575941496703104,720575941503572133,720575941589995319,720575941441065919,720575941544954556,720575941687554799] #i1
banc_i2 = [720575941569601650,720575941515421443,720575941555553363,720575941414465684,720575941461249747,720575941649246741] #i2
