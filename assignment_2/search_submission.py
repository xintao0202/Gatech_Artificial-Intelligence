# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from __future__ import division
import heapq
import os
import pickle


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
        current (int): The index of the current node in the queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        #raise NotImplementedError
        if self.queue:
            return heapq.heappop(self.queue)
        return None

    def remove(self, node_id):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """
        # reference https://stackoverflow.com/questions/5484929/removing-an-item-from-a-priority-queue
        # move note to top of heap
        while node_id>0:
            top=(node_id+1)/2-1
            self.queue[node_id]=self.queue[top]
            node_id=top
        heapq.heappop(self.queue)
        return self.queue
        #raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        #raise NotImplementedError
        heapq.heappush(self.queue,node)

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n for _, n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self == other

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """
        if len(self.queue) == 0:
            return (0, None)
        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!

    # https://stackoverflow.com/questions/8922060/how-to-trace-the-path-in-a-breadth-first-search
    if start == goal:
        return []
    frontier = PriorityQueue()
    frontier.append((0, [start]))
    explored = set()
    while frontier.size() > 0:
        # print frontier, "frontier"
        depth, path = frontier.pop()
        # print new_explored,"new explore"
        # get last node in path
        last_in_path = path[-1]
        # print last_in_path,"last"
        # print path,"path"
        # print explored.queue,"explored"
        if last_in_path == goal:
            # print last_in_path
            return path
        elif last_in_path not in explored:
            # check all adjacent nodes, make new path
            explored.add(last_in_path)
            for node in graph[last_in_path]:
                new_path = list(path)
                new_path.append(node)
                depth_sum = len(new_path)
                if node not in explored:
                    # print new_path
                    frontier.append((depth_sum, new_path))
                    if node==goal:
                        return new_path
                    # print frontier


    #raise NotImplementedError


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    #raise NotImplementedError
    if start==goal:
        return []
    frontier=PriorityQueue()
    frontier.append((0,[start]))
    explored=set()
    while frontier.size()>0:
        cost,path=frontier.pop()
        last_in_path = path[-1]
        if last_in_path==goal:
            return path
        elif last_in_path not in explored:
            for node in graph[last_in_path]:
                new_path = list(path)
                new_path.append(node)
                cost_sum=cost+graph[last_in_path][node]['weight']
                #if node not in explored and node not in frontier:
                # print new_path
                frontier.append((cost_sum,new_path))

                # print frontier
            explored.add(last_in_path)

def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node.
    """

    # TODO: finish this function!
    v1=graph.node[v]['pos']
    v2=graph.node[goal]['pos']
    return  sum([(x-y)**2 for (x,y) in zip(v1,v2)])**(0.5)
    #raise NotImplementedError


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    #raise NotImplementedError
    if start==goal:
        return []
    frontier=PriorityQueue()
    frontier.append((0,[start]))
    explored=set()
    while frontier.size()>0:
        cost,path=frontier.pop()
        last_in_path = path[-1]
        if last_in_path==goal:
            return path
        elif last_in_path not in explored:
            for node in graph[last_in_path]:
                new_path = list(path)
                new_path.append(node)
                # cost contains from last in path to goal distance
                cost_sum=cost+graph[last_in_path][node]['weight']+heuristic(graph, node, goal)-heuristic(graph, last_in_path, goal)
                #if node not in explored and node not in frontier:
                # print new_path
                frontier.append((cost_sum,new_path))
                # print frontier
            explored.add(last_in_path)

def path_cost(graph,path):
    cost=0
    if len(path)>1:
        for i in range (0,len(path)-1):
            cost=cost+graph[path[i]][path[i+1]]['weight']
    return cost
#
# def sum_path(element, link_start, link_goal):
#     start_path_cost, start_path_path = link_start[element]
#     goal_path_cost, goal_path_path = link_goal[element]
#     return start_path_path[:-1] + goal_path_path[::-1]

def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    #raise NotImplementedError
    if start == goal:
        return []

    frontier_start = PriorityQueue()
    frontier_start.append([0, start])
    frontier_goal = PriorityQueue()
    frontier_goal.append([0, goal])
    explored_path = [{}, {}]
    explored_node = [set(), set()]


    best_path = []
    best_cost = float('inf')
    shared_node = set()

    while frontier_start.size() + frontier_goal.size() > 0:

        if frontier_start.size() > 0:
            path_start = frontier_start.pop()
            node_start = path_start[-1]

            if node_start not in explored_node[0]:
                explored_node[0].add(node_start)
                explored_path[0][node_start] = path_start

            elif path_start[0] < explored_path[0][node_start][0]:
                explored_path[0][ node_start] = path_start

            if not shared_node:
                for neighbor_start in graph[node_start]:
                    if neighbor_start not in explored_node[0]:
                        new_path_start = list(path_start)
                        new_path_start.append(neighbor_start)
                        new_path_start[0] += graph[node_start][neighbor_start]['weight']
                        frontier_start.append(new_path_start)

        if frontier_goal.size() > 0:
            path_goal = frontier_goal.pop()
            node_goal = path_goal[-1]

            if node_goal not in explored_path[1]:
                explored_node[1].add(node_goal)
                explored_path[1][node_goal] = path_goal

            elif path_goal[0] < explored_path[1][node_goal][0]:
                 explored_path[1][node_goal] = path_goal

            if not shared_node:
                for neighbor_goal in graph[node_goal]:
                    if neighbor_goal not in explored_node[1]:
                        new_path_goal = list(path_goal)
                        new_path_goal.append(neighbor_goal)
                        new_path_goal[0] += graph[node_goal][neighbor_goal]['weight']
                        frontier_goal.append(new_path_goal)

        if  node_start in explored_path[1].keys():
            shared_node.add(node_start)
            cost = path_start[0] +  explored_path[1][node_start][0]

            if cost < best_cost:
                best_cost = cost
                best_path = path_start[1: -1] + explored_path[1][node_start][-1: 0: -1]

        elif node_goal in explored_path[0].keys():
            shared_node.add(node_goal)
            cost = path_goal[0] + explored_path[0][node_goal][0]
            if cost < best_cost:
                best_cost = cost
                best_path = explored_path[0][node_goal][1: -1] + path_goal[-1: 0: -1]

    return best_path


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    #raise NotImplementedError
    if start == goal:
        return []

    frontier_start = PriorityQueue()
    frontier_start.append([0, start])
    frontier_goal = PriorityQueue()
    frontier_goal.append([0, goal])
    explored_path = [{}, {}]
    explored_node = [set(), set()]


    best_path = []
    best_cost = float('inf')
    shared_node = set()

    while frontier_start.size() + frontier_goal.size() > 0:

        if frontier_start.size() > 0:
            path_start = frontier_start.pop()
            node_start = path_start[-1]

            if node_start not in explored_node[0]:
                explored_node[0].add(node_start)
                explored_path[0][node_start] = path_start

            elif path_start[0] < explored_path[0][node_start][0]:
                explored_path[0][ node_start] = path_start

            if not shared_node:
                for neighbor_start in graph[node_start]:
                    if neighbor_start not in explored_node[0]:
                        new_path_start = list(path_start)
                        new_path_start.append(neighbor_start)
                        new_path_start[0] =path_cost(graph,new_path_start[1:])
                        new_path_start[0] += heuristic(graph,neighbor_start,goal)
                        frontier_start.append(new_path_start)

        if frontier_goal.size() > 0:
            path_goal = frontier_goal.pop()
            node_goal = path_goal[-1]

            if node_goal not in explored_path[1]:
                explored_node[1].add(node_goal)
                explored_path[1][node_goal] = path_goal

            elif path_goal[0] < explored_path[1][node_goal][0]:
                 explored_path[1][node_goal] = path_goal

            if not shared_node:
                for neighbor_goal in graph[node_goal]:
                    if neighbor_goal not in explored_node[1]:
                        new_path_goal = list(path_goal)
                        new_path_goal.append(neighbor_goal)
                        new_path_goal[0] = path_cost(graph, new_path_goal[1:])
                        new_path_goal[0] += heuristic(graph, neighbor_goal, start)
                        frontier_goal.append(new_path_goal)

        if  node_start in explored_path[1].keys():
            shared_node.add(node_start)
            cost = path_start[0] +  explored_path[1][node_start][0]

            if cost < best_cost:
                best_cost = cost
                best_path = path_start[1: -1] + explored_path[1][node_start][-1: 0: -1]

        elif node_goal in explored_path[0].keys():
            shared_node.add(node_goal)
            cost = path_goal[0] + explored_path[0][node_goal][0]
            if cost < best_cost:
                best_cost = cost
                best_path = explored_path[0][node_goal][1: -1] + path_goal[-1: 0: -1]

    return best_path

def tridirectional_search(graph, goals):

    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    #raise NotImplementedError
    # id any two goal are the same, then return none
    if len(goals)!=len(set(goals)):
        return []
    frontiers=[PriorityQueue(),PriorityQueue(),PriorityQueue()]
    explored_path = [{}, {},{}]
    explored_node = [set(), set(), set()]
    for i in range(0,3):
        frontiers[i].append([0, goals[i]])
    #print frontiers,"frontiers"
    best_path=[[],[],[]] #path between 01,12,20
    best_cost=[float("inf"),float("inf"),float("inf")] #cost between 01,20,12
    shared_node=[set(),set(),set()]


    def shortest_path():
        if max(best_cost) == best_cost[0]:
            #print "0 is max"
            if set(best_path[2]).issubset(set(best_path[1])):
                return best_path[1]
            elif set(best_path[1]).issubset(set(best_path[2])):
                return best_path[2]
            return best_path[1] + best_path[2][1:]
        elif max(best_cost) == best_cost[1]:
            #print "1 is max"
            if set(best_path[2]).issubset(set(best_path[0])):
                return best_path[0]
            elif set(best_path[0]).issubset(set(best_path[2])):
                return best_path[2]
            return best_path[2] + best_path[0][1:]
        elif max(best_cost) == best_cost[2]:
            #print "2 is max"
            if set(best_path[1]).issubset(set(best_path[0])):
                return best_path[0]
            elif set(best_path[0]).issubset(set(best_path[1])):
                return best_path[1]
            return best_path[0] + best_path[1][1:]

    while frontiers[0].size()+frontiers[1].size()+frontiers[2].size()>0:


        # print explored_node,"explored_nodes"
        # print shared_node,"shared_nodes"
        cost_path=[frontiers[0].top(),frontiers[1].top(),frontiers[2].top()]
        # if current path cost is less than the best cost between 0 and 1, expand 0

        if cost_path[0][0]>=best_cost[0] or frontiers[0].size()==0:
            if cost_path[1][0]>=best_cost[1] or frontiers[1].size()==0:
                if cost_path[2][0]>=best_cost[2] or frontiers[2].size()==0:
                   return shortest_path()

        if cost_path[0][0]<best_cost[0]:
            if frontiers[0].size()>0:
                path0=frontiers[0].pop()
                node0=path0[-1]
                if node0==goals[1]:
                    if best_cost[0]>path0[0]:
                        best_cost[0]=path0[0]
                        best_path[0]=path0[1:]

                # if node0==goals[2]:
                #     if best_cost[2]>path0[0]:
                #         best_cost[2]=path0[0]
                #         best_path[2]=path0[-1:0:-1]

                if node0 not in explored_node[0]:
                    explored_node[0].add(node0)
                    explored_path[0][node0]=path0
                elif path0[0]<explored_path[0][node0][0]:
                    explored_path[0][node0]=path0

                if not shared_node[0] and not shared_node[2]:
                    for neighbor0 in graph[node0]:
                        if neighbor0 not in explored_node[0]:
                            new_path0=list(path0)
                            new_path0.append(neighbor0)
                            new_path0[0]+=graph[node0][neighbor0]['weight']
                            frontiers[0].append(new_path0)

        # if current path cost is less than the best cost between 1 and 2, expand 1
        if cost_path[1][0] < best_cost[1]:
            if frontiers[1].size() > 0:
                path1 = frontiers[1].pop()
                node1 = path1[-1]
                if node1 == goals[2]:
                    if best_cost[1] > path1[0]:
                        best_cost[1] = path1[0]
                        best_path[1] = path1[1:]

                if node1 not in explored_node[1]:
                    explored_node[1].add(node1)
                    explored_path[1][node1] = path1
                elif path1[0] < explored_path[1][node1][0]:
                    explored_path[1][node1] = path1

                if not shared_node[1]and not shared_node[0]:
                    for neighbor1 in graph[node1]:
                        if neighbor1 not in explored_node[1]:
                            new_path1 = list(path1)
                            new_path1.append(neighbor1)
                            new_path1[0] += graph[node1][neighbor1]['weight']
                            frontiers[1].append(new_path1)
        # if current path cost is less than the best cost between 2 and 0, expand 2
        if cost_path[2][0] < best_cost[2]:
            if frontiers[2].size() > 0:
                path2 = frontiers[2].pop()
                node2 = path2[-1]
                if node2 == goals[0]:
                    if best_cost[2] > path2[0]:
                        best_cost[2] = path2[0]
                        best_path[2] = path2[1:]

                if node2 not in explored_node[2]:
                    explored_node[2].add(node2)
                    explored_path[2][node2] = path2
                elif path2[0] < explored_path[2][node2][0]:
                    explored_path[2][node2] = path2

                if not shared_node[2]and not shared_node[1]:
                    for neighbor2 in graph[node2]:
                        if neighbor2 not in explored_node[2]:
                            new_path2 = list(path2)
                            new_path2.append(neighbor2)
                            new_path2[0] += graph[node2][neighbor2]['weight']
                            frontiers[2].append(new_path2)
        # if any two expansion meets:

        # print node0,explored_path[1].keys(),"node0"
        # print node0, explored_path[2].keys(), "node0"
        # print node1, explored_path[2].keys(), "node1"
        # print node1, explored_path[0].keys(), "node1"
        # print node2, explored_path[0].keys(), "node2"
        # print node2, explored_path[1].keys(), "node2"

        if node0 in explored_path[1].keys():
            shared_node[0].add(node0)
            cost01=path0[0]+explored_path[1][node0][0]
            if best_cost[0]>cost01:
                best_cost[0]=cost01
                best_path[0]=path0[1:-1]+explored_path[1][node0][-1:0:-1]
        if node0 in explored_path[2].keys():
            shared_node[2].add(node0)
            cost02=path0[0]+explored_path[2][node0][0]
            if best_cost[2]>cost02:
                best_cost[2]=cost02
                best_path[2]=explored_path[2][node0][1:-1]+path0[-1:0:-1]

        if node1 in explored_path[2].keys():
            shared_node[1].add(node1)
            cost12 = path1[0] + explored_path[2][node1][0]
            if best_cost[1] > cost12:
                best_cost[1] = cost12
                best_path[1] = path1[1:-1] + explored_path[2][node1][-1:0:-1]
        if node1 in explored_path[0].keys():
            shared_node[0].add(node1)
            cost10 = path1[0] + explored_path[0][node1][0]
            if best_cost[0] > cost10:
                best_cost[0] = cost10
                best_path[0] = explored_path[0][node1][1:-1] + path1[-1:0:-1]

        if node2 in explored_path[0].keys():
            shared_node[2].add(node2)
            cost20 = path2[0] + explored_path[0][node2][0]
            if best_cost[2] > cost20:
                best_cost[2] = cost20
                best_path[2] = path2[1:-1] + explored_path[0][node2][-1:0:-1]
        if node2 in explored_path[1].keys():
            shared_node[1].add(node2)
            cost21 = path2[0] + explored_path[1][node2][0]
            if best_cost[1] > cost21:
                best_cost[1] = cost21
                best_path[1] = explored_path[1][node2][1:-1] + path2[-1:0:-1]

    return shortest_path()

    # evaluate best path based on three pathes
    # if cost of path 1 to 2 is the max, then compare the other two path



# upgraded one implemented A* for Tridirectional search
def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    """
    Exercise 3: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    #raise NotImplementedError
    if len(goals) != len(set(goals)):
        return []
    frontiers = [PriorityQueue(), PriorityQueue(), PriorityQueue()]
    explored_path = [{}, {}, {}]
    explored_node = [set(), set(), set()]
    for i in range(0, 3):
        frontiers[i].append([0, goals[i]])
    # print frontiers,"frontiers"
    best_path = [[], [], []]  # path between 01,12,20
    best_cost = [float("inf"), float("inf"), float("inf")]  # cost between 01,20,12
    shared_node = [set(), set(), set()]

    def shortest_path():

        # print best_path, "best path"
        # print best_cost, "best cost"

        min_path={}
        best_cost012=best_cost[0]+best_cost[1]
        min_path[best_cost012] = best_path[0] + best_path[1][1:]


        best_cost120=best_cost[1]+best_cost[2]
        min_path[best_cost120] = best_path[1] + best_path[2][1:]


        best_cost201=best_cost[2]+best_cost[0]
        min_path[best_cost201] = best_path[2] + best_path[0][1:]

        min_cost=min(best_cost012, best_cost120,best_cost201)
        return min_path[min_cost]


    while frontiers[0].size() + frontiers[1].size() + frontiers[2].size() > 0:

        # print explored_node,"explored_nodes"
        # print shared_node,"shared_nodes"
        cost_path = [frontiers[0].top(), frontiers[1].top(), frontiers[2].top()]
        # if current path cost is less than the best cost between 0 and 1, expand 0
        # print frontiers[0].size(),"size"
        # print cost_path[0][0],  best_cost[0],"compare"
        if cost_path[0][0] >= best_cost[0] or frontiers[0].size() == 0:
            if cost_path[1][0] >= best_cost[1] or frontiers[1].size() == 0:
                if cost_path[2][0] >= best_cost[2] or frontiers[2].size() == 0:
                    return shortest_path()

        if cost_path[0][0] < best_cost[0]:
            if frontiers[0].size() > 0:
                path0 = frontiers[0].pop()
                node0 = path0[-1]
                if node0 == goals[1]:
                    if best_cost[0] > path0[0]:
                        best_cost[0] = path0[0]
                        best_path[0] = path0[1:]

                # if node0==goals[2]:
                #     if best_cost[2]>path0[0]:
                #         best_cost[2]=path0[0]
                #         best_path[2]=path0[-1:0:-1]
                #print node0,"node"
                if node0 not in explored_node[0]:
                    explored_node[0].add(node0)
                    explored_path[0][node0] = path0
                elif path0[0] < explored_path[0][node0][0]:
                    explored_path[0][node0] = path0

                # if not shared_node[0] and not shared_node[2]:
                for neighbor0 in graph[node0]:
                    if neighbor0 not in explored_node[0]:
                        # print explored_node[0], "explored"
                        # print neighbor0,"neighbor"

                        new_path0 = list(path0)
                        new_path0.append(neighbor0)
                        #new_path0[0] += graph[node0][neighbor0]['weight']
                        new_path0[0] = path_cost(graph, new_path0[1:])
                        new_path0[0] += heuristic(graph, neighbor0, goals[1])
                        frontiers[0].append(new_path0)

        # if current path cost is less than the best cost between 1 and 2, expand 1
        if cost_path[1][0] < best_cost[1]:
            if frontiers[1].size() > 0:
                path1 = frontiers[1].pop()
                node1 = path1[-1]
                if node1 == goals[2]:
                    if best_cost[1] > path1[0]:
                        best_cost[1] = path1[0]
                        best_path[1] = path1[1:]

                if node1 not in explored_node[1]:
                    explored_node[1].add(node1)
                    explored_path[1][node1] = path1
                elif path1[0] < explored_path[1][node1][0]:
                    explored_path[1][node1] = path1

                # if not shared_node[1] and not shared_node[0]:
                for neighbor1 in graph[node1]:
                    if neighbor1 not in explored_node[1]:
                        new_path1 = list(path1)
                        new_path1.append(neighbor1)
                        #new_path1[0] += graph[node1][neighbor1]['weight']
                        new_path1[0] = path_cost(graph, new_path1[1:])
                        new_path1[0] += heuristic(graph, neighbor1, goals[2])
                        frontiers[1].append(new_path1)
        # if current path cost is less than the best cost between 2 and 0, expand 2
        if cost_path[2][0] < best_cost[2]:
            if frontiers[2].size() > 0:
                path2 = frontiers[2].pop()
                node2 = path2[-1]
                if node2 == goals[0]:
                    if best_cost[2] > path2[0]:
                        best_cost[2] = path2[0]
                        best_path[2] = path2[1:]

                if node2 not in explored_node[2]:
                    explored_node[2].add(node2)
                    explored_path[2][node2] = path2
                elif path2[0] < explored_path[2][node2][0]:
                    explored_path[2][node2] = path2

                #if not shared_node[2] and not shared_node[1]:
                for neighbor2 in graph[node2]:
                    if neighbor2 not in explored_node[2]:
                        new_path2 = list(path2)
                        new_path2.append(neighbor2)
                        # new_path2[0] += graph[node2][neighbor2]['weight']
                        new_path2[0] = path_cost(graph, new_path2[1:])
                        new_path2[0] += heuristic(graph, neighbor2, goals[0])
                        frontiers[2].append(new_path2)
        # if any two expansion meets:

        # print node0,explored_path[1].keys(),"node0"
        # print node0, explored_path[2].keys(), "node0"
        # print node1, explored_path[2].keys(), "node1"
        # print node1, explored_path[0].keys(), "node1"
        # print node2, explored_path[0].keys(), "node2"
        # print node2, explored_path[1].keys(), "node2"

        if node0 in explored_path[1].keys():
            shared_node[0].add(node0)
            cost01 = path0[0] + explored_path[1][node0][0]
            if best_cost[0] > cost01:
                best_cost[0] = cost01
                best_path[0] = path0[1:-1] + explored_path[1][node0][-1:0:-1]
        if node0 in explored_path[2].keys():
            shared_node[2].add(node0)
            cost02 = path0[0] + explored_path[2][node0][0]
            if best_cost[2] > cost02:
                best_cost[2] = cost02
                best_path[2] = explored_path[2][node0][1:-1] + path0[-1:0:-1]

        if node1 in explored_path[2].keys():
            shared_node[1].add(node1)
            cost12 = path1[0] + explored_path[2][node1][0]
            if best_cost[1] > cost12:
                best_cost[1] = cost12
                best_path[1] = path1[1:-1] + explored_path[2][node1][-1:0:-1]
        if node1 in explored_path[0].keys():
            shared_node[0].add(node1)
            cost10 = path1[0] + explored_path[0][node1][0]
            if best_cost[0] > cost10:
                best_cost[0] = cost10
                best_path[0] = explored_path[0][node1][1:-1] + path1[-1:0:-1]

        if node2 in explored_path[0].keys():
            shared_node[2].add(node2)
            cost20 = path2[0] + explored_path[0][node2][0]
            if best_cost[2] > cost20:
                best_cost[2] = cost20
                best_path[2] = path2[1:-1] + explored_path[0][node2][-1:0:-1]
        if node2 in explored_path[1].keys():
            shared_node[1].add(node2)
            cost21 = path2[0] + explored_path[1][node2][0]
            if best_cost[1] > cost21:
                best_cost[1] = cost21
                best_path[1] = explored_path[1][node2][1:-1] + path2[-1:0:-1]

    return shortest_path()


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    #raise NotImplementedError
    return "Xin Tao"

# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    #raise NotImplementedError
    return bidirectional_ucs(graph, start, goal)


def load_data():
    """
    Loads data from data.pickle and return the data object that is passed to
    the custom_search method.

    Will be called only once. Feel free to modify.

    Returns:
         The data loaded from the pickle file.
    """

    dir_name = os.path.dirname(os.path.realpath(__file__))
    pickle_file_path = os.path.join(dir_name, "data.pickle")
    data = pickle.load(open(pickle_file_path, 'rb'))
    return data
