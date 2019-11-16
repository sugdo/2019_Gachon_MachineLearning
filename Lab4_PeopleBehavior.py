import numpy as np

def Markov (mat, observation) :
    result =1
    for i in range(len(observation)) :
        if(i==0) :
            result = result * mat[observation[i]-1][observation[i]-1]
        else :
            result = result * mat[observation[i-1]-1][observation[i]-1]

    return result

def main() :
    mat = np.array([[0.6,0.15,0.05,0.2],[0.05,0.8,0.1,0.05],[0.05,0.15,0.5,0.3],[0.2,0.15,0.15,0.5]])
    print(mat)

    #study=1, rest=2, walk=3, eat=4
    observation = list((2,4,1,1,3,2))
    print(observation)
    result = Markov(mat,observation)
    print("Probability of the sequence of observations : ")
    print(result)






main()