'''RubiksCube.py

Designed by Brume Umukoro, bumukoro@uw.edu and Derek Wu, dysw@uw.edu.

To run in console:
python3 RubiksCube.py

'''
# from RubiksQ import *
import random
# INITIAL STATE
colours = list(range(7))
c_dict = {"yellow": 0, "orange": 1, "blue": 2,"red": 3, "green": 4,"white": 5}
n_dict = {0 :"yellow", 1: "orange", 2: "blue", 3: "red", 4: "green", 5: "white"}
faces_dict = {"U" : 0, "D" : 5, "F" : 1, "B" : 3, "R" : 2, "L" : 4}
faces1_dict = {0 : "U",5 : "D", 1 : "F" ,3 : "B", 2 : "R", 4 : "L"}
LIVING_REWARD = 0.0
CLOSED = []
# allStates = None
allStates = {}
current_state = []

############## RubiksQ.py START ###################
safe = 2
discount = 0.9
stepSize = 0.5
Q_values = {}
Policy = {}
weights = [0.2, 0.1, 0.5]


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



def bestA(state, A):
    """Return action from set with highest Q for state."""
    global Q_values
    bestQ = 0.0
    currentBest = A[0]
    for i in A:
        a = i
        #print("a", a)
        #print("state", state)
        #print("state, a", state, a)
        if (state, a) in Q_values.keys():
            if Q_values[(state,a)] > bestQ:
                bestQ = Q_values[(state,a)]
                currentBest = a
    return [bestQ, currentBest]

def takeaction(s, A, R):
    #take action and update weights and Q value
    global Q_values
    global stepSize
    global weights
    #print("s in takeaction", s)

    temp = bestA(s, A)
    bestQ = temp[0]
    a = temp[1]

    if goal_test(s):
        Q_values[(s, Exit)] = 100
        print(Exit)
        return initial_state
    step = random.randrange(0, 3, 1)
    if step > safe:
        if (s, a) not in Q_values:
            Q_values[(s, a)] = 0.0
        newState = a.state_transf(s)
        reward = R(s, a)
        old = Q_values[(s, a)]
        aprime = bestA(newState, A)[1]
        if (newState, aprime) not in Q_values:
            Q_values[(newState,aprime)] = 0.0
        gamma = reward + discount*Q_values[(newState, aprime)] - old
        Q_values[(s,a)] = features(newState)  #update Q value
        for i in range(len(weights)):  #update weights
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
    print(a)
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
        bestQ = Q_values[(S[0],A[0])]
        for action in A:
            temp = Q_values[(state,action)]
            if temp > bestQ:
                bestQ = temp
                bestAction = action
        Policy[state] = bestAction # TODO check that it was correct to change this from action
    return Policy

############## RubiksCube.py END ##################



def generate_all_states():
    global allStates, CLOSED
    # allStates = {} # TODO Scope seems wrong
    OPEN = [initial_state]
    CLOSED = []
    count = 0

    while OPEN != []:
        S = OPEN.pop(0)
        CLOSED.append(S)

        if goal_test(S):
            pass
        count += 1

        L = []
        adj_list = []
        for idx, op in enumerate(OPERATORS):
            new_state = op.state_transf(S)
            adj_list.append((idx, new_state))
            if not (new_state in CLOSED):
                L.append(new_state)
        allStates[S] = adj_list

        for s2 in OPEN:
            for i in range(len(L)):
                if (s2 == L[i]):
                    del L[i]; break
        OPEN = OPEN + L
#STANDARD CODE TO DEFINE STATES
class State:
    def __init__(self, b):
        """Takes list of 6 lists each length 4"""
        if len(b) == 24:
            lists_of_lists = [b[4*i:4*i + 4] for i in range(6)]
        else:
            lists_of_lists = b
        self.b = lists_of_lists

    def __eq__(self, other):
        # TODO Maybe Account for rotations of faces/colours
        # TODO makes sets hash
        # faces_set = set(self.b)
        # other_set = set(other.b)
        faces_set = set()
        other_set = set()
        for i in self.b:
            faces_set.add(tuple(i))
        for j in other.b:
            other_set.add(tuple(j))
        if faces_set - other_set == set():
            return True
        else:  # TODO potentially add condition
            return False

    def __str__(self):
        # Produces a textual description of a state.
        txt = "\n["
        for i in range(6):
            txt += str(self.b[i]) + "\n "
        return txt[:-2] + "]"

    def __hash__(self):
        return (self.__str__()).__hash__()

    def copy(self):
        # Performs an appropriately deep copy of a state,
        # for use by operators in creating new states.
        news = State({})
        news.b = [row[:] for row in self.b]
        return news

    def top_turn(self):
        """Takes a state and returns a new state with the top face rotated 180 degrees."""
        b0 = self.b
        news = self.copy()
        b1 = news.b
        for i in range(4):
            b1[0][i] = b0[0][3-i]
            if i <= 1:
                b1[1][i] = b0[3][i]
                b1[3][i] = b0[1][i]
                b1[2][i] = b0[4][i]
                b1[4][i] = b0[2][i]
        return news

    def rotation(self, face):
        """Shifts of the faces to make sure that the face given is put
         in the top position in order to perform a turn. Returns a new
        copy after this has been carried .
        """
        b0 = self.b
        news = self.copy()
        b1 = news.b
        if face == "U":
            return self.top_turn()
        if face == "D":
            b1[0] = b0[5]
            b1[5] = b0[0]
            b1[1] = b0[3]
            b1[3] = b0[1]
            b1[2][0] = b0[2][3]
            b1[2][3] = b0[2][0]
            b1[2][1] = b0[2][2]
            b1[2][2] = b0[2][1]
            b1[4][0] = b0[4][3]
            b1[4][3] = b0[4][0]
            b1[4][1] = b0[4][2]
            b1[4][2] = b0[4][1]
        if face == "R":
            b1[0] = b0[2]
            b1[4] = b0[0]
            b1[5] = b0[4]
            b1[2] = b0[5]
            # Anticlockwise 90
            b1[1][0] = b0[1][1]
            b1[1][1] = b0[1][3]
            b1[1][3] = b0[1][2]
            b1[1][2] = b0[1][0]
            # Clockwise 90
            b1[3][1] = b0[3][0]
            b1[3][0] = b0[3][2]
            b1[3][2] = b0[3][3]
            b1[3][3] = b0[3][1]
        if face == "L":
            b1[2] = b0[0]
            b1[0] = b0[4]
            b1[4] = b0[5]
            b1[5] = b0[2]
            # Anticlockwise 90
            b1[3][0] = b0[3][1]
            b1[3][1] = b0[3][3]
            b1[3][3] = b0[3][2]
            b1[3][2] = b0[3][0]
            # Clockwise 90
            b1[1][1] = b0[1][0]
            b1[1][0] = b0[1][2]
            b1[1][2] = b0[1][3]
            b1[1][3] = b0[1][1]
        if face == "F":
            b1[0] = b0[1]
            b1[3] = b0[0]
            b1[5] = b0[3]
            b1[1] = b0[5]
            # Anticlockwise
            b1[4][0] = b0[4][1]
            b1[4][1] = b0[4][3]
            b1[4][3] = b0[4][2]
            b1[4][2] = b0[4][0]
            # Clockwise
            b1[2][1] = b0[2][0]
            b1[2][0] = b0[2][2]
            b1[2][2] = b0[2][3]
            b1[2][3] = b0[2][1]
        if face == "B":
            b1[1] = b0[0]
            b1[0] = b0[3]
            b1[3] = b0[5]
            b1[5] = b0[1]
            # Anticlockwise
            b1[2][0] = b0[2][1]
            b1[2][1] = b0[2][3]
            b1[2][3] = b0[2][2]
            b1[2][2] = b0[2][0]
            # Clockwise
            b1[4][1] = b0[4][0]
            b1[4][0] = b0[4][2]
            b1[4][2] = b0[4][3]
            b1[4][3] = b0[4][1]
        return news

    def move(self, face):
        """Assuming it's legal to make the move, this computes
           the new state resulting from moving a tile in the
           given direction, into the void.
           """
        return (self.rotation(face)).top_turn()

goal_state = State([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]])

initial_state = State([[0, 0, 0, 0], [3,3, 1, 1], [4, 4, 2, 2], [1,1, 3, 3], [2, 2, 4, 4], [5, 5, 5, 5]])

def goal_test(s):
    global goal_state
    """If all the sides are the same colour, then s is a goal state."""
    # TODO Exit reward?????
    return s == goal_state # State([[k+1 for m in range(4)] for k in range(6)])

# def goal_message(s):
#   return "You have solved the easiest Rubik's cube in the WORLD!"
goal_message = "Exit move taken, You have solved the easiest Rubik's cube in the WORLD!"

class Operator:
  def __init__(self, name, state_transf):
    self.name = name
    self.state_transf = state_transf

  def apply(self, s):
    """Carries out move."""
    return self.state_transf(s)

  def __str__(self):
    # Produces a textual description of a state.
    return self.name



#<INITIAL_STATE>
  # Use default, but override if new value supplied
             # by the user on the command line.
try:
  import sys
  init_state_string = sys.argv[2]
  print("Initial state as given on the command line: "+init_state_string)
  init_state_list = eval(init_state_string)
except:
  print("Using default initial state list: "+str(initial_state))
  # print(" (To use a specific initial state, enter it on the command line, e.g.,") # TODO Maybe replace this

#<OPERATORS>
faces = ['F','B','R','L', 'U', 'D']
OPERATORS = [Operator("Perform "+face+" turn on the cube.",
                      # The default value construct is needed
                      # here to capture the value of dir
                      # in each iteration of the list comp. iteration.
                      lambda s,face1=face: s.move(face1))
             for face in faces]

Exit = Operator(goal_message, lambda s: initial_state)
#</OPERATORS>

def R(s, a):
  '''Rules: Exiting from the correct goal state yields a
  reward of +100.  Exiting from an alternative goal state
  yields a reward of +10.
  The cost of living reward is -0.1.
  '''
# Handle goal state transitions first...
  if goal_test(s):
    if a=="Exit":
        return 100.0
    else:
        return 0.0
  return LIVING_REWARD

def run():
    global CLOSED
    global current_state
    global initial_state
    global OPERATORS
    global LIVING_REWARD
    global discount
    global safe
    
    step = input('How many steps should it go for? ')
    steps = int(step)

    reward = input('Is there a living reward? 1 for yes, 2 for no')
    rewards = int(reward)
    if rewards == 1:
        LIVING_REWARD = -0.1
    else:
        LIVING_REWARD = 0

    idiscount = input('What should the discount factor be? ')
    idiscounts = float(idiscount)
    discount = idiscounts
    
    totalOps = OPERATORS + [Exit]
    current_state = initial_state
    
    generate_all_states()
    getQValues(CLOSED, totalOps)
    
    while steps > 0:
        print("in while loop. current_state:", current_state)
        current_state = takeaction(current_state, OPERATORS, R)
        steps = steps -1
        #print(steps)
        #print(current_state)
    #print(CLOSED, totalOps)
    policy = getPolicy(CLOSED, totalOps)
    #print(policy)


#print(takeaction(start, OPERATORS, R))
run()
# TODO Make sure that one can only Exit if in goal state
