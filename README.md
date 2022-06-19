# Bio-inspired optimization algorithms
Handbook on optimization algorithms by MIT Press: [PDF](https://algorithmsbook.com/optimization/files/optimization.pdf) under Creative Commons License

Code for each algorithm is separated into 4 main files:
- `main.py`: running the main function and definition of user-configurable options
- `<algorithm>.py`: main algorithm for the relevant optimizer. Passing relevant samplers, objective functions along with options defined in `main.py` will make it work
- `utils.py`: graphing functions
- `function.py`: defining samplers, objective functions, crossover/mutation relevant for each sample objective

## Simulated Annealing
### Objective function(s)
Mix of sinusoidal functions. Sample output is present in `output/base.jpg`

### Output
On running main,py, graphs are generated for the objective function defined in `annealing/function.py` for the following:
- GIF of candidates and objective function as temperature changes
- Temperature vs iterations
- Acceptance probability vs iterations
- Objective function vs iterations
- Plot of the objective function

Sample outputs are present in `annealing/output/`

### Configs
Initialization and configuration options are available in `annealing/main.py`


## Genetic Algorithm
### Objective function(s)
Generating a sorted list of first K non-negative integers (can sort in increasing or decreasing order)

### Output
On running main,py, a graph for worst and best performer in an iteration is generated in `genetic_evo/output/`.

Sample outputs are present in `genetic_evo/output/`

### Configs
Initialization and configuration options are available in `genetic_evo/main.py`

