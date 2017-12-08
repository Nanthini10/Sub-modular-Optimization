# -*- coding: utf-8 -*-
"""
author: harshat, nanthini
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_table("ml-1m/ratings.dat", header = None)

# Part 1.1 creating data representation in matrix form
H = np.zeros((6050,4000)) # Create Matrix to store the data
# Fill the matrix with the ratings
for i in range(len(df)):
    E = df[0][i].split("::") 
    H[int(E[0])][int(E[1])] = int(E[2])

# Reference matrix which will not be changed
Href = H

# Will use to divide by later
n = 0
for i in range(np.shape(H)[0]):
    H1 = H[i][:]
    if (np.count_nonzero(H1) != 0):
        n+=1

# Lazy Greedy Algorithm
# To store the objective values and Execution time
Fvals = []
Tvals = []

start_t = time.time()
# Sorted list of marginal gain
Deltarg = np.argsort(-np.sum(Href,axis=0)) # Indicies
Deltval = np.sum(Href,axis = 0)[Deltarg] # Values

# Maxvec contains the highest ratings of each of the users for a given set
# The average of maxvec is the value of F(A)
maxvec = Href[:,Deltarg[0]] 

# Add the movie with the higest ratings as in Greedy
Fvals.append(np.sum(maxvec))
Tvals.append(time.time()-start_t)
ival = 0 #Ival is what was just added

# Number of iterations
k = 100
for i in range(k-1):
    # Calculate the Marginal gain of adding the movie with the second best marginal gain (ival + 1) (the best was already added)
    maxvecGuess = Href[:,Deltarg[ival+1]] - maxvec 
    # Compare with the marginal gain of adding the third movie (ival + 2)
    if (np.sum(maxvecGuess*(maxvecGuess > 0)) >= Deltval[ival +2]):
        # If it is greater, then by submodularity, it is the best choice
        # Update maxvec
        maxvec = np.maximum(maxvec, Href[:,Deltarg[ival+1]])
        ival +=1
    else:
        # It is not better so now we must resort to find best marginal gain as in regular greedy
        H1 = Href - np.reshape(maxvec,(len(maxvec),1))
        H1 = H1*(H1>0)
        # Resort the marginal gains
        Deltarg = np.argsort(-np.sum(H1,axis = 0))
        Deltval = -np.sort(-np.sum(H1,axis=0))
        maxvec = np.maximum(maxvec,Href[:,Deltarg[0]])
        ival = 0
    # Append the sum of maxvec to the objective value list
    Fvals.append(np.sum(maxvec))
    Tvals.append(time.time()-start_t)

# Greedy Algorithm
H = Href
maxvec = np.zeros((np.shape(H)[0],))
# To store the objective values and Execution time
Fvals1 = []
Tvals1 = []
# Number of iterations
k = 100
start_t = time.time()
for i in range(k):
    # Find the best marginal gain (in first iteration, this is the the movie with the highest ratings)
    maxi = np.argmax(np.sum(H,axis =0))
    # Update maxvec according to selected movie
    maxvec = np.maximum(maxvec,Href[:,maxi])
    # Add the F(A) to the objective value list
    Fvals1.append(np.sum(maxvec))
    # Recalcualte the Marginal gains
    H = Href - np.reshape(maxvec, (len(maxvec),1))
    # We don't want negative values because marginal gains cannot be negative. So we zero them out
    H[np.where(H<0)] = 0
    Tvals1.append(time.time() - start_t)

# Plots
newList = [float(i/float(n)) for i in Fvals]
newList1 = [float(i/float(n)) for i in Fvals1]
plt.plot(newList[0:50], label = 'Lazy Greedy')
plt.plot(newList1[0:50], label = 'Greedy')
plt.xlabel("Iteration k")
plt.ylabel("F(A)")
plt.title("Objective Value: k = 50")
plt.legend()
plt.show()
plt.plot(Tvals[0:50], label = 'Lazy Greedy')
plt.plot(Tvals1[0:50], label = 'Greedy')
plt.legend()
plt.xlabel("Iteration k")
plt.ylabel("Execution Time (s)")
plt.title("Execution time: k = 50")
plt.show()

