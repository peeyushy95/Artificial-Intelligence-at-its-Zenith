# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 21:22:03 2017

@author: Megamindo_0
"""

__author__ = 'Peeyush Yadav'

# Where 0 denotes the blank tile or space.
initial_state = [3,1,2,0,4,5,6,7,8]
goal_state = [0,1,2,3,4,5,6,7,8]

def checkSolvability(initial_state):
    inversion = 0;
    for i in range(0,len(initial_state)-1):
        for j in range(i+1,len(initial_state)):
            if initial_state[i] != 0 and initial_state[j] != 0 and initial_state[i] > initial_state[j] :
                inversion += 1    

    if inversion % 2 == 0 :
        return True
    else:
        return False

def DFS_Search(initial_state, goal_state) :
    explored_states = []
    stack = Stack()
    stack.push(initial_state)
    
    if stack.isEmpty() == False :
        current_state = stack.pop()
        explored_states.append(current_state)
        
        if current_state == goal_state:
            print ("done")
            return True;
        
        neighbour_states = explore_next_states(current_state, explored_states)
        for states in neighbour_states:
            stack.push(states)
            
    return False
    
def explore_next_states(current_state, explored_states):
    
    new_states = []
    print(current_state)
    print(move('up', current_state))
    new_state = move('up', current_state)
    if new_state[0] == True and new_state not in explored_states:
        new_states.append[new_state[1]]

    new_state = move('down', current_state)
    if new_state[0] == True and new_state not in explored_states:
        new_states.append[new_state[1]]

    new_state = move('left', current_state)
    if new_state[0] == True and new_state not in explored_states:
        new_states.append[new_state[1]]

    new_state = move('right', current_state)
    if new_state[0] == True and new_state not in explored_states :
        new_states.append[new_state[1]]

    return new_states;    

def move(direction,state):
    new_state = state[:]
    index = new_state.index( 0 )
    
    if direction == 'up':
        if index not in [0,1,2]:
           new_state[index - 3],new_state[index] = new_state[index],new_state[index-3]          
    elif direction == 'down':
        if index not in [6,7,8]:
            new_state[index + 3],new_state[index] = new_state[index],new_state[index + 3]  
    elif direction == 'left':
        if index not in [0,3,6]:
            new_state[index - 1],new_state[index] = new_state[index],new_state[index - 1]  
    else:
        if index not in [2,5,8]:
            new_state[index + 1],new_state[index] = new_state[index],new_state[index + 1]

    return (new_state == state,new_state)
    
class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

        
if __name__ == "__main__":
    if checkSolvability(initial_state) == True :
        DFS_Search(initial_state, goal_state)
    else:
        print ("Can't be Solved")
    