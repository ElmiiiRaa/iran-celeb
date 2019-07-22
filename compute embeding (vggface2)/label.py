# user/bin/python3
import pickle
import sys

import numpy as np


MIN_THRESHOLD = 0.7
MAX_THRESHOLD = 1.9


def save_label(argsv):
    base_directory = argsv[1]

    image_name = argsv[2].split(' ')

    with (open(base_directory+'/embeddings.pickle', "rb")) as openfile:
        image_embeding_name = pickle.load(openfile)

    try:
        image_embedings = []
        for name in image_name:
            i = image_embeding_name['names'].index(name)
            image_embedings.append(image_embeding_name['embeddings'][i])
            del image_embeding_name['embeddings'][i]
            del image_embeding_name['names'][i]

    except:
        print('image name not in pickle file')

    diff_embeding = np.zeros([
        len(image_embeding_name['names']),
        len(image_embeding_name['embeddings'][0])
    ])

    labels = []
    names = []
    for i, img_embeding in enumerate(image_embedings):
        for j, embeding_ in enumerate(image_embeding_name['embeddings']):
            diff = np.subtract(img_embeding, embeding_)
            diff_embeding[j] = np.sum(np.square(diff))

        mean = np.mean(diff_embeding)
        if mean < MIN_THRESHOLD:
            label = 1

        elif MAX_THRESHOLD < mean < MIN_THRESHOLD:
            label = 2

        else:
            label = 3

        labels.append(label)
        names.append(image_name[i])

    with open(base_directory+'/label.txt', 'w') as text_file:
        for i in range(len(names)):
            text_file.write('file_name: {}, label: {}\n'.format(names[i], labels[i]))


if __name__ == '__main__':
    save_label(sys.argv)

