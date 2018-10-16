opimport numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import functions as fun


#------------------------------


#print(fun.zerofun(fun.start))
#print('P')
#print(fun.P)


#test
nzAlpha, corrInput, corrTarget = fun.extractNZ(alpha)
#--------------

if (1):
    print ('indicator')
    print (fun.inputs)
    print (fun.targets)
    print (alpha)
    print (fun.indicator(alpha,corrInput[0][0],corrInput[0][1]))
    print (fun.indicator(alpha,corrInput[1][0],corrInput[1][1]))
    print (alpha)
