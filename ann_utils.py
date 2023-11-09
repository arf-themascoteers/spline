from ann_simple import ANNSimple
from ann_large import ANNLarge
from ann_savi_only import ANNSAVIOnly
from ann_savi_learned_only import ANNSAVILearnedOnly
from ann_savi_learned_only_bound import ANNSAVILearnedOnlyBound


def get_ann_by_name(algorithm):
    if algorithm == "ann_simple":
        return ANNSimple
    if algorithm == "ann_large":
        return ANNLarge
    elif algorithm == "ann_savi_only":
        return ANNSAVIOnly
    elif algorithm == "ann_savi_learned_only":
        return ANNSAVILearnedOnly
    elif algorithm == "ann_savi_learned_only_bound":
        return ANNSAVILearnedOnlyBound




