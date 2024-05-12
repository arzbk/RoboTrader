# RoboTrader - Deep Reinforcement Learning for Automated Portfolio Management

This project was heavily inspired by the great work outlined in this research paper: https://ieeexplore.ieee.org/abstract/document/9877940.

A Twin Deterministic Policy Gradient approach (TD3) was created and trained on a stock market simulation that I built using the Gymnasium Python library (the successor of OpenAI's Gym for reinforcement learning). The network is fed normalized values 
downloaded from Yahoo Finance on stocks belonging to the SP500 index in the US. The reward for the network is based on the difference in value from open to close of a trading day - where actions (such as buying shares and selling at a higher price) are 
reinforced.

The state-space used was identical to that which was employed by the authors of the paper. Comparable results were achieved in testing.

TensorBoard is utilized to provide live metrics of the networks performance and training progress as the simulation plays out.

#Future Areas for Improvement

- Explore using the Transformers architecture to better capture non-linear relationships in the time-series data that may impact the performance of a given stock.
- Conduct further data exploration leveraging unsupervised learning approaches such as clustering to identify additional candidates to use as input features for the network.
