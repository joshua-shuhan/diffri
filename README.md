# DiffRI: A Diffusion Model for Relational Inference

This repository provides partial implementations of experiments in the paper for review purpose.

## Setup
First, download necessary packages.
```
conda create --name diffri --file requirements.txt 
```
Then, activate the `diffri` environment.
```
conda activate diffri
```
## Running experiments
We provides two datasets here.

First, Kuramoto model datasets can be generated by the following example code.
```
cd data;
python kuramoto_gen.py --num-node 5 --density 0.5 --seed 1 --nsample 500
```

For Spring datasets, you can generate it with the following code.
```
cd data;
python spring_gen.py --num-node 5 --density 0.5 --seed 1 --nsample 500
```

We provide independent scripts for running experiments.
Run experiments for Kuramoto datasets:

```
python exe_kura.py --seed 1 --num-node 5 --density 0.5 --T 100 --no-reg
```

Run Spring datasets: 
```
python exe_spr.py --seed 1 --num-node 5 --density 0.5 --T 49
```

After training the model, you can test it with following codes.

Kuramoto:
```
python test_kura.py --seed 1 --num-node 5 --density 0.5 --T 100 --model-path <path>
```
Spring:
```
python test_spring.py --seed 1 --num-node 5 --density 0.5 --T 49 --model-path <path>
```
After running, the terminal will display the mean and standard deviation of accuracy across test samples.

