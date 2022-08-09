#!/bin/bash

# echo "Running for different initializers and pretrained/untrained models"
# python3 main.py --prefix=mnist_pretrained_glorot --t0=100 --decay=0.98 --thresh=1e-2 --load_pretrained --initializer=glorot
# python3 main.py --prefix=mnist_pretrained_gauss --t0=100 --decay=0.98 --thresh=1e-2 --load_pretrained --initializer=gaussian
# python3 main.py --prefix=mnist_glorot --t0=100 --decay=0.98 --thresh=1e-2 --initializer=glorot
# python3 main.py --prefix=mnist_gauss --t0=100 --decay=0.98 --thresh=1e-2 --initializer=gaussian
# python3 main.py --objective=poly --prefix=sa2  --t0=50
echo "Running simulated annealing for univariate function"
python3 main.py --objective=poly --prefix=sa --t0=50

echo "Running 100 instances of mnist with glorot initializer"
python3 main.py --objective=ml_100 --prefix=batch_ml_gl --t0=50 --decay=0.95 --thresh=1e-2 --initializer=glorot
echo "Running 100 instances of mnist with gaussian initializer"
python3 main.py --objective=ml_100 --prefix=batch_ml_ga --t0=50 --decay=0.95 --thresh=1e-2 --initializer=gaussian
echo "Running 100 instances of a univariate function"
python3 main.py --objective=poly_100 --prefix=batch_poly --t0=50

