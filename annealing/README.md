## Files

- batch_ml_<>.jpg which contains outputs for 100 runs of simulated annealing on a basic MNIST model
- batch_poly_<>.jpg which contains outputs for 100 runs of SA for a univariate function
- sa_<> contains outputs for a single run of the univariate function. Includes a gif to show the proposed solution at each step

## Observations

- For the univariate case, the loss tends to converge to 0 in most of the cases. Histogram of the final loss shows the same conclusion.
- For the multivariate case, the loss does not show convergence. Rather, it keeps fluctuating wildly.
- The metropolis criterion also rises too rapidly for MNIST due to low differences in loss between iterations and exponential decrease in temperature.

## Conclusions
- Simulated annealing works well for the univariate case, but fails in the case of digit recognition on MNIST. 2 major causes for this are the lack of directionality and the comparatively huge number of states to be explored in the latter case.
- Updates made at every iteration are sampled from a distribution. Knowing the probability distribution of the solution can help in getting closer to the optimal solution in a faster and more efficient way.
- Relating to the previous point, for the multivariate case, as of now, updates are taken from different samples from the same distribution. This can be detrimental, especially in the case of neural networks. For MNIST, the weight distribution for the first layer can be roughly estimated by a Gaussian, but the final classification layer has a very sparse distribution. The latter set of weights is very difficult to achieve with simulated annealing.

## Options
- initial temperature, decay rates, thresholds/iterations for SA
- pretrained/untrained model initialization for MNIST
- weight initializer for MNIST (glorot/gaussian)

Options can be listed via `python main.py --help`

## Modifications
For the univariate case, change the objective function `f()` in `functions.py`. The corresponding sampler is `sampler()`.
Similarly for MNIST.
