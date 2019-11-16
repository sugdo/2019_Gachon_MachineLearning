import numpy as np
from scipy import stats

N = 10
a = np.array([45,38,52,48,25,39,51,46,55,46])
b = np.array([34,22,15,27,37,41,24,19,26,36])

var_a = a.var(ddof=1)
var_b = b.var(ddof=1)

print(a)
print(b)

s = np.sqrt((var_a + var_b)/2)

##Calculate the t-statistics
t = (a.mean() - b.mean())/(s*np.sqrt(2/N))

##Compare with the critical t-value
#degrees of freedom
df = (2*N) - 2

#p-value after comparison with the t
p = 1 - stats.t.cdf(t,df=df)

print('p-value after comparison with the t')
print("t = " + str(t))
print("p = " + str(2*p))

## Cross check with the internal SciPy function
print("Cross check with the internal SciPy function ( ttest_ind )")
t2, p2 = stats.ttest_ind(a,b)
print("t = " + str(t2))
print("p = " + str(p2))


print("Cross check with the internal SciPy function ( ttest_rel )")
t2, p2 = stats.ttest_rel(a,b)
print("t = " + str(t2))
print("p = " + str(p2))

print("----------------------------------------------")
print("Direct Calculation ( Paired Sample T test )")
a = np.array([45,38,52,48,25,39,51,46,55,46])
b = np.array([34,22,15,27,37,41,24,19,26,36])
sum_of_difference = 0
sum_of_squared_difference =0


for i in range(len(a)) :
    sum_of_difference += a[i] - b[i]
    sum_of_squared_difference += pow(a[i] - b[i],2)

print("Sum of the differences : {} ".format(sum_of_difference))
print("Sum of the squared differences : {}".format(sum_of_squared_difference))

t_score = (sum_of_difference/N) / pow((sum_of_squared_difference - (pow(sum_of_difference,2)/N))/((N-1)*N),0.5)
print("t score : {}".format(t_score))

# alpha is 0.05
# degree of freedom is 18 because of 2N - 2 ( independent sample t test )

print("in T table,18 df and 0.05 alpha has 2.101(T value)")
t_value = 2.101

if( t_value < t_score) :
    print(" t_score > t_value : H0 reject : dependent")
else :
    print(" t_score < t_value : H0 accept : independent")


p = 1 - stats.t.cdf(t_score,df=df)

print("t = " + str(t_score))
print("p = " + str(2*p))

# alpha is 0.05
if( p > 0.05) :
    print(" p > alpha : H0 accept : independent")
else :
    print(" p < alpha : H0 reject : dependent")
print("---------------------------------------------------------------------------")
print( "important : this also applies to independent sample T test")
print("What is your computed answer? : ")
print("it is dependent !!!")
print("What would be the null hypothesis in this study? :")
print("Young or old does not affect satisfaction")
print("What is your t_crit? :")
print("2.101")
print("Is there a significant difference between the two groups? :")
print("Older people tend to have higher satisfaction.")