# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

# RUN THIS: python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util

class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """
    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        self.state = state

class QLearnAgent(Agent):
    def __init__(self, alpha: float = 0.2, epsilon: float = 0.05, gamma: float = 0.8,
                 maxAttempts: int = 15, numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0
        self.q_values = {}
        self.play_state_counts = {}
        self.prev_state = None
        self.prev_action = None

    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining
    
     # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        """
        Computes the reward from one state to the other

        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        return endState.getScore() - startState.getScore()

    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        """
        Returns the Q value for a given state action pair

        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        # Return 0 if none found
        return self.q_values.get((state, action), 0)

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Returns the maximum Q value from a state considering all actions

        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        return max([self.getQValue(state, action) for action in state.getLegalPacmanActions()])

    def learn(self, state: GameStateFeatures, action: Directions, reward: float, nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        q_val_old = self.getQValue(state, action)
        q_val_next_max = self.maxQValue(nextState)
        # Calculate new Q-value taking everything into consideration
        q_val_updated = (1 - self.alpha) * q_val_old + self.alpha * (reward + self.gamma * q_val_next_max)
        self.q_values[(state, action)] = q_val_updated
        self.updateCount(state, action)

    def updateCount(self, state: GameStateFeatures, action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        self.play_state_counts[(state, action)] = self.getCount(state, action) + 1

    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.play_state_counts.get((state, action), 0)

    def explorationFn(self, utility: float, counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts
        If its rarely been visited, it becomes a priority to visit

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        return float('inf') if counts == 0 else utility / counts

    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()

        # Make pacman learn from previous state, action taken and reward recieved
        if self.prev_state:
            reward = self.computeReward(self.prev_state, state)
            self.learn(self.prev_state, self.prev_action, reward, state)
        
        # If pacman is infering then compare Q values for current state and pick best action
        if self.getAlpha() == 0 and self.epsilon == 0:
            action = max(legal, key=lambda act: self.getQValue(state, act))
        else:
            # If pacman is learning then take best known action if a lot of attempts have been made, otherwise explore other paths
            if self.getCount(self.prev_state, self.prev_action) >= self.getMaxAttempts():
                action = max(legal, key=lambda act: self.getQValue(state, act))
            else:
                action = max(legal, key=lambda act: self.explorationFn(self.getQValue(state, act), self.getCount(state, act)))

        self.prev_action = action
        self.prev_state = state
        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            print("Training Done (turning off epsilon and alpha)")
            self.setAlpha(0)
            self.setEpsilon(0)
        else:
            # Make pacman learn from a successful or failed game and reset for the next
            reward = state.getScore()
            self.learn(self.prev_state, self.prev_action, reward, self.prev_state)
            self.prev_action = None
            self.prev_state = None