# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import random
import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # print('runValueIteration ', self.values)
        # prevValues = self.values
        for i in range(self.iterations):
            mdpStates = self.mdp.getStates()
            newValues = {}
            for state in mdpStates:
                legalActions = self.mdp.getPossibleActions(state)
                #what if legalActions is empty
                maxValue = 0
                if legalActions:
                    values = [self.computeQValueFromValues(state, action) for action in legalActions]
                    maxValue = max(values)

                # maxIndexes = [i for i in len(values) if values[i] is maxValue]
                newValues[state] = maxValue

            for state in mdpStates:
                self.values[state] = newValues[state]

        return
    # def runValueIteration(self):
    #     # Write value iteration code here
    #     "*** YOUR CODE HERE ***"
    #     while self.iterations > 0:
    #         vals = ()
    #         allStates = self.mdp.getStates()
    #         for s in allStates:
    #             allActions = self.mdp.getPossibleActions(s)
    #             for a in allActions:
    #                 maxValues = max([self.computeQValueFromValues(s,a) for a in allActions])
    #             vals[s] = maxValues
    #         for s in allStates:
    #             self.values[s] = vals[s]
                    
    #         #         reachableStates = self.mdp.getTransitionStatesAndProbs(s, a)
    #         #         value = 0
    #         #         for nextState, prob in reachableStates:
    #         #             value += prob * (self.mdp.getReward(s, a, nextState) + self.discount*tmpVal[nextState])
    #         #         chances.append(value)
    #         #     if len(chances) != 0:
    #         #         self.values[s] = max(chances)
    #         # self.iterations -= 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        states = self.mdp.getTransitionStatesAndProbs(state, action)

        qVal = 0
        for nextState, prob in states:
            qVal += prob * (self.mdp.getReward(state, action, nextState) + self.discount*self.values[nextState])

        return qVal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # if self.mdp.isTerminal(state):
        #     return None
        # allActionsForState = self.mdp.getPossibleActions(state)
        # finalAction = ""
        # maxSum = float("-inf")
        # for action in allActionsForState:
        #     weightedAverage = self.computeQValueFromValues(state, action)
        #     if (maxSum == 0.0 and action == "") or weightedAverage >= maxSum:
        #         finalAction = action
        #         maxSum = weightedAverage

        # return finalAction
        legalActions = self.mdp.getPossibleActions(state)
        if not legalActions:
            return None
        # maxValue = 0
        # maxIndex = None
        # for i in range(len(actions)):
        #     action = legalActions[i]
        #     value = self.computeQValueFromValues(state, action)
        #     if value>max_value:
        #         max_value = value
        #         max_index = i

        #implement random choice for equal values
        values = [self.computeQValueFromValues(state, action) for action in legalActions]
        maxValue = max(values)
        maxIndexes = [i for i in range(len(values)) if values[i] is maxValue]
        actionTaken = legalActions[random.choice(maxIndexes)]

        return actionTaken


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        allStates = self.mdp.getStates()
        numStates = len(allStates)
        for i in range(self.iterations):
            state = allStates[i % numStates]
            allActions = self.mdp.getPossibleActions(state)
            maxVal = 0
            if allActions:
                val = list()
                for act in allActions:
                    val.append(self.computeQValueFromValues(state, act))
                maxVal = max(val)
                
            self.values[state] = maxVal

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = set()
        allStates = self.mdp.getStates()
        prio = util.PriorityQueue()

        for s in allStates:
            self.values[s] = 0
            predecessors = predecessors | self.getPredecessors(s)


        for s in allStates:
            if not self.mdp.isTerminal(s):
                d = abs(self.values[s] - self.maxQVal(s))
                prio.push(s, -d)

        for _ in range (self.iterations):
            if prio.isEmpty():
                return
     
            s = prio.pop()
            self.values[s] = self.maxQVal(s)

            for p in predecessors:
                d = abs(self.values[p] - self.maxQVal(p))
                if d > self.theta:
                    prio.update(p,-d)

    def maxQVal(self, state):
        v = list()
        for a in self.mdp.getPossibleActions(state):
            v.append(self.computeQValueFromValues(state,a))
        return max(v)

    def getPredecessors(self, state):
        predecessor = set()
        states = self.mdp.getStates()

        if not self.mdp.isTerminal(state):
            for s in states:
                if not self.mdp.isTerminal(s):
                    allActions = self.mdp.getPossibleActions(s)
                    for move in allActions:
                        sp = self.mdp.getTransitionStatesAndProbs(s,move)
                        
                        for secondState, probs in sp:
                            if secondState == state and probs > 0:
                                predecessor.add(s)

        return predecessor


