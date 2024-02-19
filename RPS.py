# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import numpy as np
import pandas as pd
from itertools import product
import tensorflow as tf

### Q-learning

# Actions to indices and vice-versa
play2idx = dict(zip(['R','P','S'],range(3)))
idx2play = dict(zip(range(3),['R','P','S']))

# Number of games, states to indices and vice-versa
N_PLAYS = 2
states = list(product('RPS',repeat=2*N_PLAYS))
state2idx = {u:i for i,u in enumerate(states)}
idx2state = {i:u for i,u in enumerate(states)}


# Useful functions
def move(Q,s,p):
    """Next move: random or according to Q-table."""
    if np.random.uniform(0,1) < p:
        a = np.random.choice(range(3))
    else:
        a = np.argmax(Q[s,:])
        if a==None:
            a = np.random.choice(range(3))
        else:
            pass
    return a

def reward(player_play,opponent_play):
    """Reward to player for play."""
    if player_play==opponent_play:
        return 0.0
    elif play2idx[player_play]%3 == (play2idx[opponent_play]+1)%3:
        return 1.0
    else:
        return -1.0


# Player function
def player(prev_play, opponent_history=[], player_history=[''], n_plays=N_PLAYS,
    Q=np.random.uniform(0,1,size=[3**(2*N_PLAYS),3]), alpha=0.65, gamma=0.5, p=0.1):
    """ This Rock-Paper-Scissors robot learns to play against an opponent
    through reinforcement learning using the Q-learning algorithm."""

    opponent_history.append(prev_play)

    # When starting game against new player...
    if '' in opponent_history[-n_plays-1:]:
        a = move(Q,None,1.0)
        Q[:,:] = np.random.uniform(0,1,size=[3**(2*N_PLAYS),3])
    else:
        s_1 = state2idx[tuple(player_history[-n_plays-1:-1]+opponent_history[-n_plays-1:-1])]
        a_1 = play2idx[player_history[-1]]
        s = state2idx[tuple(player_history[-n_plays:]+opponent_history[-n_plays:])]
        r = reward(player_history[-1],prev_play)
        Q[s_1,a_1] = Q[s_1,a_1] + alpha*(r + gamma*Q[s,:].max() - Q[s_1,a_1])
        a = move(Q,s,p)

    play = idx2play[a]
    player_history.append(play)
    return play













