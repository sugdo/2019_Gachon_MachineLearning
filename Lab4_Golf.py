from sklearn import preprocessing
import pandas as pd

def Naive_Bayesian(outlook,temp,humidity,wind,play,instance) :

    yes = 0
    no = 0


    total = len(outlook)


    # Yes
    for i in range(total) :
        if(play[i]=='yes') :
            yes = yes +1
        else :
            no = no +1

    prob = 1
    prob = prob * auxiliary('yes',total,outlook,instance[0],play)
    prob = prob * auxiliary('yes',total,temp,instance[1],play)
    prob = prob * auxiliary('yes', total, humidity, instance[2], play)
    prob = prob * auxiliary('yes', total, wind, instance[3], play)
    prob = prob * (yes/total)
    result_yes = prob

    print("Probability of PlayGolf-yes")
    print(result_yes)

    # No
    prob = 1
    prob = prob * auxiliary('no',total,outlook,instance[0],play)
    prob = prob * auxiliary('no',total,temp,instance[1],play)
    prob = prob * auxiliary('no', total, humidity, instance[2], play)
    prob = prob * auxiliary('no', total, wind, instance[3], play)
    prob = prob * (no/total)
    result_no = prob

    print("Probability of PlayGolf-No")
    print(result_no)

    if(result_yes > result_no) :
        print("Conclusion : PlayGolf-yes")
    else :
        print("Conclusion : playGolf-no")









def auxiliary(str,total,array,target,play) :

    all = 0
    part = 0

    if(str == 'yes') :
        for i in range(total) :
            if(play[i]=='yes') :
                all = all +1
                if(array[i]==target) :
                    part = part +1
    else :
        for i in range(total) :
            if(play[i]=='no') :
                all = all +1
                if(array[i]==target) :
                    part = part +1

    return float(part/all)











def main () :
    data = pd.read_csv('play_golf.csv')

    #print(data)
    outlook = data['Outlook']
    temp = data['Temp']
    humidity = data['Humidity']
    wind = data['Wind']
    play = data['play Golf']

    le = preprocessing.LabelEncoder()
    wind = le.fit_transform(wind)

    # Flase : 0 , True 1
    instance = list(('sunny','cool','high',0))

    Naive_Bayesian(outlook,temp,humidity,wind,play,instance)


main()