# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 21:22:03 2017

@author: Megamindo_0
"""

__author__ = 'Peeyush Yadav'

from collections import OrderedDict
import copy
import sys

# Where 0 denotes the blank tile or space.
initial_state = "312645780"
goal_state =    "012345678"

state_info = OrderedDict()

def checkSolvability(initial_state):
    
    inversion = 0;
    for i in range(0,len(initial_state)-1):
        for j in range(i+1,len(initial_state)):
            if initial_state[i] != '0' and initial_state[j] != '0' and initial_state[i] > initial_state[j] :
                inversion += 1    

    if inversion % 2 == 0 :
        return True
    else:
        return False

def DFS_Search(initial_state, goal_state) :
    
    global state_id
    explored_states = OrderedDict()
    open_states = OrderedDict()
    #stack = Stack()
    open_states[initial_state] = True
    state_info[initial_state] = (None,0)
    
    while len(open_states) > 0 :
        current_state,exist = open_states.popitem()
        parent,depth = state_info[current_state]
        explored_states[current_state] = True
        if current_state == goal_state:
            printPathtoGoal(current_state)
            return True;
              
        neighbour_states = explore_next_states(current_state, explored_states,open_states)
        for states in neighbour_states:
            open_states[states] = True
            state_info[states] = (current_state,depth+1)
              
    return False
    
    
def explore_next_states(current_state, explored_states, open_states):    
    new_states = []
    
    new_state = move('right', current_state)
    if new_state[0] == True and explored_states.get(new_state[1]) == None and open_states.get(new_state[1]) == None:
        new_states.append(new_state[1])
        
    new_state = move('down', current_state)
    if new_state[0] == True and explored_states.get(new_state[1]) == None and open_states.get(new_state[1]) == None:
        new_states.append(new_state[1])
             
    new_state = move('up', current_state)
    if new_state[0] == True and explored_states.get(new_state[1]) == None and open_states.get(new_state[1]) == None:
        new_states.append(new_state[1])
        
    new_state = move('left', current_state)    
    if new_state[0] == True and explored_states.get(new_state[1]) == None and open_states.get(new_state[1]) == None: 
        new_states.append(new_state[1])
        
    return new_states;    

    
def move(direction,state):
    
    index = state.index('0')
    state_list = list(state)
    
    if direction == 'up':
        if index not in [0,1,2]:
            state_list[index - 3],state_list[index] = state_list[index],state_list[index-3]
    elif direction == 'down':
        if index not in [6,7,8]:
            state_list[index + 3],state_list[index] = state_list[index],state_list[index + 3]  
    elif direction == 'left':
        if index not in [0,3,6]:
            state_list[index - 1],state_list[index] = state_list[index],state_list[index - 1]  
    else:
        if index not in [2,5,8]:
            state_list[index + 1],state_list[index] = state_list[index],state_list[index + 1]

    new_state = ''.join(state_list)
    return (new_state != state, new_state)

    
def printPathtoGoal(current_state):
    
    stackTrace = Stack()
    while True:
        stackTrace.push(current_state)
        parent, depth = state_info[current_state]
        if parent == None :
            break
        current_state = parent
        
    while stackTrace.isEmpty() == False:
        state = stackTrace.pop()
        print(state[0],state[1],state[2])
        print(state[3],state[4],state[5])
        print(state[6],state[7],state[8])
        print()       

        
class Stack:
    
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()
        
     def size(self):
         return len(self.items)
    

class State_info :
    
    def __init__(self, config, parent, depth):
        self.config  = config
        self.parent = parent
        self.depth = depth
     
        
if __name__ == "__main__":
    
    if checkSolvability(initial_state) == True :
        if DFS_Search(initial_state, goal_state) == True:
            print("Solved")
        else:
            print("DFS Failed")
    else:
        print ("Can't be Solved")
    