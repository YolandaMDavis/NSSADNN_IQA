import os
import csv
import random

import torch
import yaml

from torch.utils.data import Dataset

from WildTrackDataset import WildTrackDataset

root_dir = 'data'
data_dir = root_dir + '/RAW'
image_reference_file_suffix = '_image_references.csv'


def generate_data_files():

    image_reference_list = []

    subdirectories = list(os.walk(data_dir, topdown=False))[:-1]
    for subdir in subdirectories:
        image_location = subdir[0]
        images = subdir[2]
        species_rating = image_location.rsplit('/', 1)[-1].replace('_', ' ')
        score = int(species_rating.rsplit(' ', 1)[-1])
        species_class = species_rating.rsplit(' ', 1)[:-1][0]
        if len(species_class.rsplit(' ', 1)) > 1:
            species = species_class.rsplit(' ')[0]
            animal_class = ' '.join(species_class.rsplit(' ')[1:])
        else:
            animal_class = 'Unkown'
            species = species_class

        for image in images:
            image_reference = (image_location, species, animal_class, image, score)
            image_reference_list.append(image_reference)

    # shuffle then split
    seed = 1234
    random.Random(seed).shuffle(image_reference_list)
    training = image_reference_list[:int(len(image_reference_list) * 0.05)]
    validation = image_reference_list[-int(len(image_reference_list) * 0.2):]
    testing = image_reference_list[-int(len(image_reference_list) * 0.2):]

    for dataset in [('training', training), ('validation', validation), ('testing', testing)]:
        ref_file_name = root_dir + '/' + dataset[0] + image_reference_file_suffix
        with open(ref_file_name, 'w', newline='') as csvfile:
            image_ref_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            image_ref_writer.writerows(dataset[1])


if __name__ == '__main__':

    generate_data_files()
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_dataset = WildTrackDataset(root_dir + '/' + 'training' + image_reference_file_suffix, config, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=16,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0)

    train_features, train_labels = next(iter(train_loader))
    print(train_labels)
