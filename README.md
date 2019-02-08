# trader-rl - Deep Reinforcement Learning trading agent 
Deep Reinforcement Learning Trading


## Introduction
A Hierarchical Multi-Agent Deep Reinforcement Learning trading system. 

## Getting Started

### Dependencies
- Python 3.6 or higher (https://www.anaconda.com/download) or (https://www.python.org/downloads/) 
- Optional but recommended Create (and activate) a new environment with Python 3.6.
    Create (and activate) a new environment with Python 3.6.
    - __Linux__ or __Mac__: 
	```bash
	conda create --name trader-rl python=3.6
	source activate trader-rl
	```
	- __Windows__: 
	```bash
	conda create --name trader-rl python=3.6 
	activate trader-rl
	```

- Install tensorflow:
    ```bash
    conda install tensorflow-gpu
	or
	conda install tensorflow
	```

- Install requirements:
    ```bash
    pip install -r requirements.txt
	```

## Instructions

- Run:
    ```bash
	python .
	```

## Todo
- [ ] jupyter notebook
- [ ] Comment code
- [ ] remove unused code
- [ ] tests
- [ ] MarketSim
- [ ] trade record
- [ ] Live demo testing
- [ ] win/loss percentage
- [ ] change reward from balance to reward
- [ ] automatic jpy or usd spread
- [ ] none hard coded saves
- [ ] graphs
- [ ] eval counter for continous days trading
- [ ] more indicators
- [ ] time frame arg state
- [ ] pair arg
- [ ] pair config
- [ ] eval or train arg
