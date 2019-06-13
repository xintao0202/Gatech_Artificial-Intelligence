"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys

if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
from exampleinference import inferenceExample

inferenceExample()
# Should output:
# ('The marginal probability of sprinkler=false:', 0.80102921)
#('The marginal probability of wetgrass=false | cloudy=False, rain=True:', 0.055)

'''
WRITE YOUR CODE BELOW. DO NOT CHANGE ANY FUNCTION HEADERS FROM THE NOTEBOOK.
'''


from Node import BayesNode
from Graph import BayesNet
import numpy as np
from numpy import zeros, float32
import random
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Inference import JunctionTreeEngine
from Inference import EnumerationEngine


def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    nodes = []
    # TODO: finish this function    
    #raise NotImplementedError
    # create nodes
    A_node = BayesNode(0, 2, name='alarm')
    F_A_node= BayesNode(1, 2, name='faulty alarm')
    G_node=BayesNode(2, 2, name='gauge')
    F_G_node = BayesNode(3, 2, name='faulty gauge')
    T_node = BayesNode(4, 2, name='temperature')

    #connect nodes
    T_node.add_child(G_node)
    G_node.add_parent(T_node)
    T_node.add_child(F_G_node)
    F_G_node.add_parent(T_node)

    F_G_node.add_child(G_node)
    G_node.add_parent(F_G_node)

    G_node.add_child(A_node)
    A_node.add_parent(G_node)

    F_A_node.add_child(A_node)
    A_node.add_parent(F_A_node)

    nodes=[A_node,F_A_node,G_node,F_G_node,T_node]

    return BayesNet(nodes)

def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system."""    
    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")
    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]
    # TODO: set the probability distribution for each node
    #raise NotImplementedError

    #The temperature gauge reads the correct temperature with 95% probability when it is not faulty and 20% probability when it is faulty.
    dist = zeros([T_node.size(), F_G_node.size(), G_node.size()], dtype=float32)
    dist[0, 0, :] = [0.95, 0.05]
    dist[0, 1, :] = [0.2, 0.8]
    dist[1, 0, :] = [0.05, 0.95]
    dist[1, 1, :] = [0.8, 0.2]
    G_distribution = ConditionalDiscreteDistribution(nodes=[T_node, F_G_node, G_node], table=dist)
    G_node.set_dist(G_distribution)

    #The alarm is faulty 15% of the time.
    F_A_distribution = DiscreteDistribution(F_A_node)
    index = F_A_distribution.generate_index([], [])
    F_A_distribution[index] = [0.85, 0.15]
    F_A_node.set_dist(F_A_distribution)

    #The temperature is hot (call this "true") 20% of the time.
    T_distribution = DiscreteDistribution(T_node)
    index = T_distribution.generate_index([], [])
    T_distribution[index] = [0.8, 0.2]
    T_node.set_dist(T_distribution)

    #When the temperature is hot, the gauge is faulty 80% of the time. Otherwise, the gauge is faulty 5% of the time.
    dist = zeros([T_node.size(), F_G_node.size()], dtype=float32)  # Note the order of G_node, A_node
    dist[0, :] = [0.95, 0.05]
    dist[1, :] = [0.20, 0.80]
    F_G_distribution = ConditionalDiscreteDistribution(nodes=[T_node, F_G_node], table=dist)
    F_G_node.set_dist(F_G_distribution)

    #The alarm responds correctly to the gauge 55% of the time when the alarm is faulty, and it responds correctly to the gauge 90% of the time when the alarm is not faulty.
    dist = zeros([G_node.size(), F_A_node.size(), A_node.size()], dtype=float32)
    dist[0, 0, :] = [0.9, 0.1]
    dist[0, 1, :] = [0.55, 0.45]
    dist[1, 0, :] = [0.1, 0.9]
    dist[1, 1, :] = [0.45, 0.55]
    A_distribution = ConditionalDiscreteDistribution(nodes=[G_node, F_A_node, A_node], table=dist)
    A_node.set_dist(A_distribution)

    return bayes_net

def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal 
    probability of the alarm 
    ringing (T/F) in the 
    power plant system."""
    # TODO: finish this function
    #raise NotImplementedError

    A_node = bayes_net.get_node_by_name('alarm')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([alarm_rings], range(Q.nDims))
    alarm_prob = Q[index]
    return alarm_prob


def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge 
    showing hot (T/F) in the 
    power plant system."""
    # TOOD: finish this function
    #raise NotImplementedError
    G_node = bayes_net.get_node_by_name('gauge')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(G_node)[0]
    index = Q.generate_index([gauge_hot], range(Q.nDims))
    gauge_prob = Q[index]
    return gauge_prob


def get_temperature_prob(bayes_net,temp_hot):
    """Calculate the conditional probability 
    of the temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    #raise NotImplementedError
    A_node = bayes_net.get_node_by_name('alarm')
    F_A_node = bayes_net.get_node_by_name('faulty alarm')
    F_G_node = bayes_net.get_node_by_name('faulty gauge')
    T_node = bayes_net.get_node_by_name('temperature')
    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A_node] = True
    engine.evidence[F_A_node] = False
    engine.evidence[F_G_node] = False
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([temp_hot], range(Q.nDims))
    temp_prob = Q[index]
    return temp_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    nodes = []
    # TODO: fill this out
    #raise NotImplementedError
    #Skill level Nodes: each has 0-3 four levels
    A_node=BayesNode(0, 4, name='A')
    B_node = BayesNode(1, 4, name='B')
    C_node = BayesNode(2, 4, name='C')

    # match nodes: each has win, lose or tie
    AvB_node=BayesNode(3, 3, name='AvB')
    BvC_node = BayesNode(4, 3, name='BvC')
    CvA_node=BayesNode(5, 3, name='CvA')

    nodes = [A_node, B_node, C_node, AvB_node, BvC_node, CvA_node]

    #create network
    A_node.add_child(AvB_node)
    AvB_node.add_parent(A_node)
    A_node.add_child(CvA_node)
    CvA_node.add_parent(A_node)

    B_node.add_child(AvB_node)
    AvB_node.add_parent(B_node)
    B_node.add_child(BvC_node)
    BvC_node.add_parent(B_node)

    C_node.add_child(CvA_node)
    CvA_node.add_parent(C_node)
    C_node.add_child(BvC_node)
    BvC_node.add_parent(C_node)

    #set Probability
    # each team has 4 level of skills with probability 0.15, 0.45, 0.3 0.1
    A_distribution = DiscreteDistribution(A_node)
    index = A_distribution.generate_index([], [])
    A_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    A_node.set_dist(A_distribution)

    B_distribution = DiscreteDistribution(B_node)
    index = B_distribution.generate_index([], [])
    B_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    B_node.set_dist(B_distribution)

    C_distribution = DiscreteDistribution(C_node)
    index = C_distribution.generate_index([], [])
    C_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    C_node.set_dist(C_distribution)

    #Probability of matches
    #AvB: given skill level A, B, P(AvB|A,B)
    dist = zeros([A_node.size(), B_node.size(), AvB_node.size()], dtype=float32)
    dist[0, 0, :] = [0.1, 0.1, 0.8]
    dist[0, 1, :] = [0.2, 0.6, 0.2]
    dist[0, 2, :] = [0.15, 0.75, 0.1]
    dist[0, 3, :] = [0.05, 0.9, 0.05]
    dist[1, 0, :] = [0.6, 0.2, 0.2]
    dist[1, 1, :] = [0.1, 0.1, 0.8]
    dist[1, 2, :] = [0.2, 0.6, 0.2]
    dist[1, 3, :] = [0.15, 0.75, 0.1]
    dist[2, 0, :] = [0.75, 0.15, 0.1]
    dist[2, 1, :] = [0.6, 0.2, 0.2]
    dist[2, 2, :] = [0.1, 0.1, 0.8]
    dist[2, 3, :] = [0.2, 0.6, 0.2]
    dist[3, 0, :] = [0.9, 0.05, 0.05]
    dist[3, 1, :] = [0.75, 0.15, 0.1]
    dist[3, 2, :] = [0.6, 0.2, 0.2]
    dist[3, 3, :] = [0.1, 0.1, 0.8]
    AvB_distribution = ConditionalDiscreteDistribution(nodes=[A_node, B_node, AvB_node], table=dist)
    AvB_node.set_dist(AvB_distribution)

    # BvC: given skill level B, C, P(BvC|B,C)
    dist = zeros([B_node.size(), C_node.size(), BvC_node.size()], dtype=float32)
    dist[0, 0, :] = [0.1, 0.1, 0.8]
    dist[0, 1, :] = [0.2, 0.6, 0.2]
    dist[0, 2, :] = [0.15, 0.75, 0.1]
    dist[0, 3, :] = [0.05, 0.9, 0.05]
    dist[1, 0, :] = [0.6, 0.2, 0.2]
    dist[1, 1, :] = [0.1, 0.1, 0.8]
    dist[1, 2, :] = [0.2, 0.6, 0.2]
    dist[1, 3, :] = [0.15, 0.75, 0.1]
    dist[2, 0, :] = [0.75, 0.15, 0.1]
    dist[2, 1, :] = [0.6, 0.2, 0.2]
    dist[2, 2, :] = [0.1, 0.1, 0.8]
    dist[2, 3, :] = [0.2, 0.6, 0.2]
    dist[3, 0, :] = [0.9, 0.05, 0.05]
    dist[3, 1, :] = [0.75, 0.15, 0.1]
    dist[3, 2, :] = [0.6, 0.2, 0.2]
    dist[3, 3, :] = [0.1, 0.1, 0.8]
    BvC_distribution = ConditionalDiscreteDistribution(nodes=[B_node, C_node, BvC_node], table=dist)
    BvC_node.set_dist(BvC_distribution)

    # CvA: given skill level C, A, P(CvA|C,A)
    dist = zeros([C_node.size(), A_node.size(), CvA_node.size()], dtype=float32)
    dist[0, 0, :] = [0.1, 0.1, 0.8]
    dist[0, 1, :] = [0.2, 0.6, 0.2]
    dist[0, 2, :] = [0.15, 0.75, 0.1]
    dist[0, 3, :] = [0.05, 0.9, 0.05]
    dist[1, 0, :] = [0.6, 0.2, 0.2]
    dist[1, 1, :] = [0.1, 0.1, 0.8]
    dist[1, 2, :] = [0.2, 0.6, 0.2]
    dist[1, 3, :] = [0.15, 0.75, 0.1]
    dist[2, 0, :] = [0.75, 0.15, 0.1]
    dist[2, 1, :] = [0.6, 0.2, 0.2]
    dist[2, 2, :] = [0.1, 0.1, 0.8]
    dist[2, 3, :] = [0.2, 0.6, 0.2]
    dist[3, 0, :] = [0.9, 0.05, 0.05]
    dist[3, 1, :] = [0.75, 0.15, 0.1]
    dist[3, 2, :] = [0.6, 0.2, 0.2]
    dist[3, 3, :] = [0.1, 0.1, 0.8]
    CvA_distribution = ConditionalDiscreteDistribution(nodes=[C_node, A_node, CvA_node], table=dist)
    CvA_node.set_dist(CvA_distribution)

    return BayesNet(nodes)

def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function    
    #raise NotImplementedError
    AvB = bayes_net.get_node_by_name("AvB")
    #match_table = AvB.dist.table
    CvA=bayes_net.get_node_by_name("CvA")
    BvC=bayes_net.get_node_by_name("BvC")
    posterior_engine=EnumerationEngine(bayes_net)
    #set evidence variables. 0==win, 1==loss, 2=tie
    posterior_engine.evidence[AvB]=0
    posterior_engine.evidence[CvA]=2
    Q = posterior_engine.marginal(BvC)[0]
    win = Q.generate_index([0], range(Q.nDims))
    loss=Q.generate_index([1], range(Q.nDims))
    tie=Q.generate_index([2], range(Q.nDims))
    prob_win = Q[win]
    prob_loss=Q[loss]
    prob_tie=Q[tie]
    posterior=[prob_win,prob_loss,prob_tie]
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    sample = tuple(initial_state)
    # TODO: finish this function
    #raise NotImplementedError
    # The method should just consist of a single iteration of the algorithm. If an initial value is not given, default to a state chosen uniformly at random from the possible states.
    if initial_state is None:
        #represent skills of A, B, C, each choose from number 0-4
        # represent result of match- win, loss or tie of three matches
        initial_state=np.random.randint(4,size=3).tolist() + np.random.randint(3,size=3).tolist()


    # rand_var is a random number that represent which node to change each time
    rand_var=np.random.randint(6)

    A = bayes_net.get_node_by_name("A")
    team_table = A.dist.table
    AvB = bayes_net.get_node_by_name("AvB")
    # match table stores A_skill, B_skill, AvB results
    match_table = AvB.dist.table

    if rand_var<3:
        prob_skill=np.zeros(4)
        total=0
        for i in range(4):
            A_skill_prob=team_table[i] # 4*1, could be any of A, B, C
            AvB_match_prob=match_table[i, initial_state[(rand_var + 1) % 3], initial_state[rand_var + 3]] # known AvB, B, A: 4*4*3 matrix, could be other possibility
            CvA_match_prob=match_table[initial_state[(rand_var - 1) % 3], i, initial_state[(5) if rand_var == 0 else (rand_var + 2)]] # known CvA, C, A: 4*4*3 matrix, could be other possibility
            prob_skill[i] =   A_skill_prob* AvB_match_prob * CvA_match_prob
            total += prob_skill[i]
        skill_prob = prob_skill / total
        initial_state[rand_var] = np.random.choice(4, p=skill_prob)

    else:
        match_prob = match_table[initial_state[rand_var - 3], initial_state[(rand_var -2) %3], :]
        initial_state[rand_var] = np.random.choice(3, p=match_prob)

    sample = tuple(initial_state)
    return sample

def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """

    A= bayes_net.get_node_by_name("A")
    B =bayes_net.get_node_by_name("B")
    C = bayes_net.get_node_by_name("C")
    AvB= bayes_net.get_node_by_name("AvB")
    BvC = bayes_net.get_node_by_name("BvC")
    CvA = bayes_net.get_node_by_name("CvA")
    A_team_table = A.dist.table
    B_team_table = B.dist.table
    C_team_table = C.dist.table
    AvB_match_table = AvB.dist.table
    BvC_match_table = BvC.dist.table
    CvA_match_table = CvA.dist.table


    # TODO: finish this function
    #raise NotImplementedError

    #The first step is to initialize the sample value for each random variable. if initial-state is not given, default to a state chosen uniformly at random from the possible states.
    if initial_state is None:
        #represent skills of A, B, C, each choose from number 0-4
        # represent result of match- win, loss or tie of three matches
        initial_state=np.random.randint(4,size=3).tolist() + np.random.randint(3,size=3).tolist()

    # Generate a proposal (or a candidate) sample cand_state from the proposal distribution
    cand_state=initial_state[:]
    # for skills
    for i in range(3):
        cand_state[i]=random.randint(0,3)
    # for match results
    for i in range(3,6):
        cand_state[i]=random.randint(0,2)

    Pi_initial=A_team_table[initial_state[0]] * B_team_table[initial_state[1]] * C_team_table[initial_state[2]] * \
            AvB_match_table[initial_state[0]][initial_state[1]][initial_state[3]] * BvC_match_table[initial_state[1]][initial_state[2]][initial_state[4]] *  CvA_match_table[initial_state[2]][initial_state[0]][initial_state[5]]
    pi_cand=A_team_table[cand_state[0]] * B_team_table[cand_state[1]] * C_team_table[cand_state[2]] * \
            AvB_match_table[cand_state[0]][cand_state[1]][cand_state[3]] * BvC_match_table[cand_state[1]][cand_state[2]][cand_state[4]] *  CvA_match_table[cand_state[2]][cand_state[0]][cand_state[5]]
    # calculate acceptance function
    alpha=min(1,pi_cand/Pi_initial)
    u=np.random.uniform()
    if u<alpha:
        #accept the proposal
        new_state=cand_state[:]
    else:
        #reject the proposal
        new_state=initial_state[:]
    sample=tuple(new_state)
    return sample

def MH_sampler_evidence(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value.
    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    """

    A= bayes_net.get_node_by_name("A")
    B =bayes_net.get_node_by_name("B")
    C = bayes_net.get_node_by_name("C")
    AvB= bayes_net.get_node_by_name("AvB")
    BvC = bayes_net.get_node_by_name("BvC")
    CvA = bayes_net.get_node_by_name("CvA")
    A_team_table = A.dist.table
    B_team_table = B.dist.table
    C_team_table = C.dist.table
    AvB_match_table = AvB.dist.table
    BvC_match_table = BvC.dist.table
    CvA_match_table = CvA.dist.table


    # TODO: finish this function
    #raise NotImplementedError

    #The first step is to initialize the sample value for each random variable. if initial-state is not given, default to a state chosen uniformly at random from the possible states.
    if initial_state is None:
        #represent skills of A, B, C, each choose from number 0-4
        # represent result of match- win, loss or tie of three matches
        initial_state=np.random.randint(4,size=3).tolist() + np.random.randint(3,size=3).tolist()

    # Generate a proposal (or a candidate) sample cand_state from the proposal distribution
    cand_state=initial_state[:]
    # for skills
    for i in range(3):
        cand_state[i]=random.randint(0,3)
    # for match results
    for i in range(3,6):
        if i==3 or i==5:
            continue
        cand_state[i]=random.randint(0,2)

    Pi_initial=A_team_table[initial_state[0]] * B_team_table[initial_state[1]] * C_team_table[initial_state[2]] * \
            AvB_match_table[initial_state[0]][initial_state[1]][initial_state[3]] * BvC_match_table[initial_state[1]][initial_state[2]][initial_state[4]] *  CvA_match_table[initial_state[2]][initial_state[0]][initial_state[5]]
    pi_cand=A_team_table[cand_state[0]] * B_team_table[cand_state[1]] * C_team_table[cand_state[2]] * \
            AvB_match_table[cand_state[0]][cand_state[1]][cand_state[3]] * BvC_match_table[cand_state[1]][cand_state[2]][cand_state[4]] *  CvA_match_table[cand_state[2]][cand_state[0]][cand_state[5]]
    # calculate acceptance function
    alpha=min(1,pi_cand/Pi_initial)
    u=np.random.uniform()
    if u<alpha:
        #accept the proposal
        new_state=cand_state[:]
        #print "yes"
        #print new_state,initial_state
    else:
        #reject the proposal
        #print "no"
        new_state=initial_state[:]
        #n_reject+=1
    sample=tuple(new_state)
    return sample

def calculate_converge(bayes_net,initial_state, delta, N_iter, sampling_method, burnin_count=2000):

    good_count=0
    rejection_count=0
    #samples that match AvB win and CvA time
    good_samples=0
    total_samples=0
    prev_prob=[0.0]*3
    prev_sample=(0,0,0,0,0,0)
    #game_result[0] is number of wins, [1] number of loss and [2]number of ties
    game_results=[0]*3
    sample=tuple(initial_state)
    for i in range(burnin_count):
        initial_state[3] = 0
        initial_state[5] = 2
        if sampling_method=="Gibbs":
            sample=Gibbs_sampler(bayes_net,initial_state)

        else:
            pre_initial_state = list(sample)
            sample=MH_sampler_evidence(bayes_net,pre_initial_state)
            #print list(sample), pre_initial_state
            if list(sample) == pre_initial_state:
                rejection_count+=1
        if initial_state[3] != 0 or initial_state[5] != 2:
            sample = None

    while good_count< N_iter:
        #print good_count
        initial_state[3] = 0
        initial_state[5] = 2
        if sampling_method=="Gibbs":
            sample=Gibbs_sampler(bayes_net,initial_state)

        else:
            pre_initial_state = list(sample)
            sample=MH_sampler_evidence(bayes_net,pre_initial_state)
            if list(sample) == pre_initial_state:
                rejection_count+=1
        if initial_state[3] != 0 or initial_state[5] != 2:
            sample = None
        total_samples+=1
        #print sample
        if sample is None:
            continue
        if sample[3]==0 and sample[5]==2:
            good_samples += 1
            BvC_result=sample[4]
            game_results[BvC_result]+=1
        #probablity of wins in current sample
        if  good_samples is not 0:
            current_prob=[float(x)/ good_samples for x in game_results]
        else:
            current_prob=[0.0]*3

        #print current_prob,prev_prob
        current_delta=max([abs(x - y) for x, y in zip(current_prob, prev_prob)])
        prev_prob=current_prob

        if  good_samples>500 and current_delta<delta and current_prob!=[0.0]*3:
            good_count+=1
        else:
            good_count=0

    count=burnin_count+total_samples
    if sampling_method == "Gibbs":
        #print good_samples,total_samples
        return count,prev_prob
    else:
        return count,prev_prob,rejection_count




def compare_sampling(bayes_net,initial_state, delta):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    #raise NotImplementedError
    # succesive Iteration criteria N_iter
    N_iter=10
    # define initial_state based on A beats B and A draws with C
    initial_state=np.random.randint(4,size=3).tolist() + np.random.randint(3,size=3).tolist()
    initial_state[3]=0
    initial_state[5]=2
    Gibbs_count,Gibbs_convergence=calculate_converge(bayes_net,initial_state, delta, N_iter, "Gibbs", 2000)
    MH_count,MH_convergence,MH_rejection_count=calculate_converge(bayes_net,initial_state, delta, N_iter, "MH", 2000)


    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count

def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 0
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1.2
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    #raise NotImplementedError
    return "Xin Tao"




