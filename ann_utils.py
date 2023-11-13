from ann_simple import ANNSimple
from ann_multiple import ANNMultiple


def get_ann_by_name(algorithm):
    if algorithm == "ann_simple":
        return ANNSimple
    elif algorithm == "ann_multiple":
        return ANNMultiple




