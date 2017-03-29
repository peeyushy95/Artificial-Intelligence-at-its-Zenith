# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 21:22:03 2017

@author: Megamindo_0
"""

__author__ = 'Peeyush Yadav'

# Where 0 denotes the blank tile or space.
initial_state = [3,1,2,0,4,5,6,7,8]
goal_state = [0,1,2,3,4,5,6,7,8]

def checkSolvalibility(initial_state):
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
    
        
if __name__ == "__main__":
    if checkSolvability(initial_state) == True :
        DFS_Search(initial_state, goal_state)
    else:
        print ("Can't be Solved")
    