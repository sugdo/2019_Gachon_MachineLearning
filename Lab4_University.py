import pandas as pd


def Naive_Bayesian(dad, mom, child, university,instance) :

    yes = 0
    no = 0

    total = len(dad)


    # enter
    for i in range(total) :
        if(university[i]=='enter') :
            yes = yes +1
        else :
            no = no +1

    prob = 1
    prob = prob * auxiliary('yes',total,dad,instance[0],university)
    prob = prob * auxiliary('yes', total, mom, instance[1], university)
    prob = prob * auxiliary('yes', total, child, instance[2], university)
    prob = prob * (yes/total)
    result_yes = prob

    print("Probability of University-enter")
    print(result_yes)

    # No
    prob = 1
    prob = prob * auxiliary('no',total,dad,instance[0],university)
    prob = prob * auxiliary('no', total, mom, instance[1], university)
    prob = prob * auxiliary('no', total, child, instance[2], university)
    prob = prob * (no/total)
    result_no = prob

    print("Probability of University-None")
    print(result_no)

    if(result_yes > result_no) :
        print("Conclusion : enter")
    else :
        print("Conclusion : none")









def auxiliary(str,total,array,target,play) :

    all = 0
    part = 0

    if(str == 'yes') :
        for i in range(total) :
            if(play[i]=='enter') :
                all = all +1
                if(array[i]==target) :
                    part = part +1
    else :
        for i in range(total) :
            if(play[i]=='none') :
                all = all +1
                if(array[i]==target) :
                    part = part +1

    return float(part/all)



def main () :
    data = pd.read_csv('go_university.csv')

    dad = data['dad']
    mom = data['mom']
    child = data['child']
    university = data['university']

    instance = list(('university','university','girl'))

    Naive_Bayesian(dad, mom, child, university,instance)



main()