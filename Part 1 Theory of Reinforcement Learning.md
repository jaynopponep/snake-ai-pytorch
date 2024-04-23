- Reinforcement Learning (RL) - Software agents ought to take actions in an environment to maximize cumulative award
- Deep Q Learning - Extends RL by using deep neural network to predict the actions (Q value = Quality of Action)
	- 0. Initialize Q value (= init model)
	- 1. Choose action (model.predict(state)) 
	- 2. Perform action
	- 3. Measure reward
	- 4. Update Q value (+ train model)
## Bellman Equation:
$NewQ(s,a) = Q(s,a) + \alpha[R(s,a)+\gamma maxQ'(s',a')-Q(s,a)]$
$NewQ(s,a)$ -> New Q value for that state and that action
$Q(s,a)$ -> Current Q value
$\alpha$ -> Learning Rate
$R(s,a)$ -> Reward for taking that action at that state
$\gamma$ -> Discount rate
$maxQ'(s',a')$ -> Max expected future reward given new s' and all possible actions at that new state
### Bellman Simplified:
$Q=model.predict(state_0)$
$Q_{new}=R+\gamma *max(Q(state_1))$
### Loss Function for optimization:
$loss = (Q_{new}-Q)^2$
