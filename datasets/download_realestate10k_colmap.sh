#!/bin/sh

DATA_PATH=data/RealEstate10K
file_list=(
    test.pickle.gz
    train.pickle.gz
    pcl.test.tar
    pcl.train.tar
    valid_seq_ids.train.pickle.gz
    SHA512SUMS
)
ROOT_URL=https://thor.robots.ox.ac.uk/flash3d
cd $DATA_PATH 
for item in "${file_list[@]}"; do
    curl -O $ROOT_URL/$item
done
sha512sum -c SHA512SUMS

