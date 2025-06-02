#!/bin/bash

# python main.py --version release

if [ $1 == agnews ]
then
    num_clusters=4
    eta=10.0
    num_updates=100
    regularization=0.1
elif [ $1 == biomedical ] || [ $1 == stackoverflow ]
then
    num_clusters=20
    eta=10.0
    num_updates=100
    regularization=0.1
elif [ $1 == googlenews-s ] || [ $1 == googlenews-t ] || [ $1 == googlenews-ts ]
then
    num_clusters=152
    eta=10.0
    num_updates=50
    regularization=0.001
elif [ $1 == searchsnippets ]
then
    num_clusters=8
    eta=10.0
    num_updates=100
    regularization=0.01
elif [ $1 == tweet ]
then
    num_clusters=89
    eta=5.0
    num_updates=100
    regularization=0.001
fi

for seed in {0..4}
do
CUDA_VISIBLE_DEVICES=$2 python main.py --version release --logger $1 --seed $seed --data_name $1 --num_clusters $num_clusters --eta $eta --pre True --model_name $1 --num_updates $num_updates --num_epochs 50 --verbose_frequency 10 --regularization $regularization
CUDA_VISIBLE_DEVICES=$2 python main.py --version release --logger $1 --seed $seed --data_name $1 --num_clusters $num_clusters --eta $eta --pre False --model_name $1 --num_updates $num_updates --num_epochs 25 --verbose_frequency 1 --regularization $regularization
done

###
