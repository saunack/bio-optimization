#!/bin/bash

# echo "Running for different initializers and pretrained/untrained models"
# python3 main.py --prefix=mnist_pretrained_glorot --t0=100 --decay=0.98 --thresh=1e-2 --load_pretrained --initializer=glorot
# python3 main.py --prefix=mnist_pretrained_gauss --t0=100 --decay=0.98 --thresh=1e-2 --load_pretrained --initializer=gaussian
# python3 main.py --prefix=mnist_glorot --t0=100 --decay=0.98 --thresh=1e-2 --initializer=glorot
# python3 main.py --prefix=mnist_gauss --t0=100 --decay=0.98 --thresh=1e-2 --initializer=gaussian
# python3 main.py --objective=poly --prefix=sa2  --t0=50
echo "Running GA for sorting function"
python3 main.py --objective=sort --prefix=sort

echo "Running GA for MNIST"
python main.py --pop 150 --prefix=ml --iterations=300 -p 0.2

echo "Running 100 instances of mnist with glorot initializer"
python3 main.py --objective=ml_100 --prefix=batch_ml --pop=300 --iterations=400 --initializer=glorot -p 0.2
echo "Running 100 instances of a univariate function"
python3 main.py --objective=sort_100 --prefix=batch_sort

