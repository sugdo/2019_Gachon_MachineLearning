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
    mat = np.array([[0.9,0.07,0.03],[0.15,0.8,0.05],[0.35,0.15,0.5]])
    print(mat)

    #bull=1, bear=2, stagnant=3
    observation = list((1,2,3,1,3,2,2,1))
    print(observation)
    result = Markov(mat,observation)
    print("Probability of the sequence of observations : ")
    print(result)






main()