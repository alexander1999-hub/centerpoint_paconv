import numpy as np 
import io
import sys

'''
class ids = {
    true_pedestrian id
    true_vichle id
    false_pedestrian id
    false_vichle id
}

'''
def read_dataset(path_to_dataset: str, class_ids: np.array) \
        -> (list, np.array):
    f = open(path_to_dataset, 'r')
    all_file = f.read().split('\n')

    all_file.pop()
    point_clouds = []
    labels = []
    pcl = []

    once = True

    for it in all_file:
        it = it.split(' ')
        if len(it) == 1:
            if pcl == []:
                continue
            labels.append(class_ids[int(it[0])])
            point_clouds.append(np.array(pcl))
            pcl = []
        else:
            pcl.append(np.array([ \
                float(it[0]), \
                float(it[1]), \
                float(it[2]) \
            ]))
    labels = np.array(labels)
    return point_clouds, labels