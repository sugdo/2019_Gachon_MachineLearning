from scipy.stats import chi2_contingency
from scipy.stats import  chi2
import copy

def sum_list(lst, res = 0):
    for i in lst:
        if type(i) == list:
            res += sum_list(i)
        else:
            res += i
    return res

#contingency table
print("First, Use Library !!!")
table =[[200, 150, 50] , [250,   300,  50]]
print(table)
stat, p, dof, expected = chi2_contingency(table)

print('dof=%d' % dof)
print(expected)

prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')

#interpret p-value

alpha = 1.0 - prob
print('significance=%.3f' % (alpha))
if p <= alpha:
    print('Dependent (reject H0)')

else:
    print('Independent (fail to reject H0)')

print("--------------------------------------------------------------")
print("Second, Direct Calculation !!!")
print(table)
expected_table = copy.deepcopy(table)

print("expected value array :")

total = sum_list(table)

chi_val = 0

for i in range(len(table)) :
    for j in range (len(table[0])) :
        temp1 = table[i][j] + table[(i+1)%2][j]
        temp2 = table[i][j] + table[i][(j+1)%3] + table[i][(j+2)%3]
        expected_table[i][j] = (temp1 * temp2) / total
        chi_val += (pow(table[i][j] - expected_table[i][j],2)) / expected_table[i][j]

print(expected_table)

print("Subtract 'expected' from 'actual', square it, then divide by 'expected' ")
print("So, Chi-Square is {}".format(chi_val))

print("Degree of freedom ( row-1 ) x (columns -1 ) = 3")
print("level of significance is 0.05")
print("In Chi-square table, df=3 , level of significance = 0.05 indicate 7.815 ")


if(chi_val > 7.815) :
    print("finally, that means it is dependent")
else :
    print("finally, that means it is independent")
