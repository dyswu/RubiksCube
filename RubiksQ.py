'''RubiksQ.py
Feature-based reinforcement learning

Designed by Brume Umukoro (bumukoro@uw.edu) and Derek Wu (dysw@uw.edu)
'''

import random
safe = 1
discount = 0.9
stepSize = 0.5
Q_values = {}
Policy = {}
weights = [1,3,5]


def features(state):
    values = state.b
    counter = [0,0,0]
    for i in range(len(values)):
        c = [0,0,0,0,0,0] #what colors in side
        for j in range(len(values[i])):
            c[values[i][j]] += 1
        for x in c:
            if x == 2: #number of pairs 
                counter[0] += 1
            if x == 3: #number of trios
                counter[1] += 1
            if x == 4: #number of completed faces
                counter[2] += 1
    total = 0
    for i in range(3):
        total += weights[i]*counter[i]
    return total

def bestA(state, A): #return action from set with highest Q for state
    global Q_values
    bestQ = 0.0
    currentBest = A[0]
    for i in A:
        a = i
        if (state, a) in Q_values:
            if Q_value[(state,a)] > bestQ:
                bestQ = Q_value[(state,a)]
                currentBest = a
    return [bestQ, currentBest]

def takeaction(s, A, R):
    #take action and update weights and Q value
    global Q_values
    global stepSize
    global weights
    #print(s)
    temp = bestA(s, A)
    bestQ = temp[0]
    a = temp[1] 

    if goal_test(s):
        Q_values[(s, Exit)] = 100
        return initial_state
    step = random.randrange(0, 2, 1)
    if step > safe:
        if (s,a) not in Q_values:
            Q_values[(s,a)] = 0.0
        newState = a.state_transf(s)
        reward = R(s,a)
        old = Q_values[(s,a)]
        aprime = bestA(newState, A)[1]
        if (newState, aprime) not in Q_values:
            Q_values[(newState,aprime)] = 0.0
        gamma = reward + discount*Q_values[(newState, aprime)] - old
        Q_values[(s,a)] = features(newState) #update Q value
        for i in range(len(weights)): #update weights
            weights[i] = weights[i] + stepSize*gamma*features(s)
    else:
        step = random.randrange(0, len(A), 1)
        a = A[step]
        #print(a)
        if (s,a) not in Q_values:
            Q_values[(s,a)] = 0.0
        newState = a.state_transf(s)
        reward = R(s,a)
        old = Q_values[(s,a)]
        aprime = bestA(newState, A)[1]
        if (newState, aprime) not in Q_values:
            Q_values[(newState,aprime)] = 0.0
        gamma = reward + discount*(Q_values[(newState, aprime)]) - old
        Q_values[(s,a)] = features(newState) #update Q value
        for i in range(len(weights)): #update weights
            weights[i] = weights[i] + stepSize*gamma*features(s)
    return newState

def getQValues(S, A):
    global Q_values
    if Q_values == {}:
        for states in S:
            for actions in A:
                Q_values[states, actions] = 0.0
    return Q_values

def getPolicy(S, A):
    global Q_values
    global Policy
    for state in S:
        bestAction = A[0]
        bestQ = Q_values(S[0],A[0])
        for action in A:
            temp = Q_values[state, action]
            if temp > bestQ:
                bestQ = temp
                bestAction = action
        Policy[state] = action
    return Policy
