import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy import linalg as LA



def objective(alphaVec):
    #import main as m
    sum = 0
    sumA = 0
    for i in range(len(alphaVec)):
        sumA += alphaVec[i]
        for j in range(len(alphaVec)):
            sum += alphaVec[i]*alphaVec[j]*P[i][j]
    scalar = 0.5*sum-sumA
    return scalar

#calculates the value which should be constrained to zero
def zerofun(alphaVec):
    #import main as m
    #this product should be zero
    scalar = numpy.dot(alphaVec,targets)
    return scalar

#----------------Kernals--------------------
def linearKernal(xVec, yVec):
    K = numpy.dot(xVec,yVec)
    return K

def polyKernal(xVec, yVec):
    K = numpy.dot(xVec,yVec)
    K = (K+1)**p
    return K

def radKernal(xVec, yVec):
    K = numpy.e**((-1*LA.norm(xVec-yVec)**2)/(2*sigma**2))
    return K
#--------------------------------------------


def calcB(alphaVec, newPoints):
    #import main as m
    b = 0
    for i in range(len(alphaVec)):
        b += alphaVec[i]*corrTarget[i]*radKernal(newPoints,corrInput[i])
    #använd godtycklig corrTarget
    b -= corrTarget[0]
    #print ('inside b')
    #print (b)
    #print ('----')
    return b

def extractNZ(alphaVec):
    #vektor med nollskilda alpha-värden
    nzAlpha = []
    corrInput = []
    corrTarget = []
    for i in range(len(alphaVec)):
        if abs(alphaVec[i]) > 10**(-5):
            nzAlpha.append(alphaVec[i])
            corrInput.append(inputs[i])
            corrTarget.append(targets[i])
    return nzAlpha, corrInput, corrTarget

def indicator(alphaVec,x,y):
    newPoints = [x,y]
    scalar = 0
    count = 0
    #beräknar nollskilda värden på alpha, sparar dem i nzAlpha och resp. input plus target.
    
    
    #print (newPoints)
    #print (corrInput)
    #print (corrTarget)
    #print (nzAlpha)
    
    #nzAlpha = [i for i in nzAlpha]
    #corrInput = corrInput
    #corrTarget = corrTarget

    #print(b)
    #bb.append(b)
    #print ('new stuff')
    #print (len(nzAlpha))
    #print ('compared to:')
    #print (len(alphaVec))
    #print (b)
    for i in range(len(nzAlpha)):
        scalar += nzAlpha[i]*corrTarget[i]*radKernal(newPoints,corrInput[i])
    scalar -= b
    return scalar

def genData():
    #trubleshooting help, will keep the random, "not random"
    #numpy.random.seed(100)
    #------------------------------------------------------
    classA = numpy.concatenate((numpy.random.randn(9,2)*std+[1.5, 0.5],numpy.random.randn(10,2) * std+[-1.5, -0.5]))
    l = numpy.array([[0,0]])
    print ('classA')
    print (classA)
    print ('classA')
    classA = numpy.concatenate((classA,l))
    print ('classA')
    print (classA)
    print ('classA')
    l = numpy.array([[0,1.5]])
    classB = numpy.concatenate((numpy.random.randn(19, 2)*std+[0.0, -0.5] , l ))
    
    inputs = numpy.concatenate((classA,classB))
    targets = numpy.concatenate((numpy.ones(classA.shape[0]),-numpy.ones(classB.shape[0])))
    
    N = inputs.shape[0] #numper of rows (samples)
    
    premute = list(range(N))
    random.shuffle(premute)
    inputs = inputs[premute, :]
    targets = targets[premute]
    
    #debugging
    #classA = [[-2,-1],[-2,-2],[-3,-1],[-3,-2]]
    #classB = [[1,1],[1,2],[2,1],[2,2]]
    #inputs = numpy.concatenate((classA,classB))
    #targets = numpy.concatenate((numpy.ones(len(classA)),-numpy.ones(len(classB))))
    #N = len(inputs)
    #------------------------------------------------------
    
    return classA, classB, inputs, targets, N


def plotz(classA, classB, alphaVec):
    plt.plot([pkt[0] for pkt in classA], [pkt[1] for pkt in classA], 'b.')
    plt.plot([pkt[0] for pkt in classB], [pkt[1] for pkt in classB], 'r.')
    
    print ('len alpha')
    print ((classB))
    
    xgrid = numpy.linspace(-5,5)
    ygrid = numpy.linspace(-4,4)
    print (len(inputs))
    grid = numpy.array([[indicator(alphaVec,x,y) for x in xgrid] for y in ygrid])
    
    print ('grid')
    print (grid)
    plt.contour(xgrid,ygrid,grid,(-1.0,0,1), colors=('red', 'black', 'blue'), linewidths=(1,3,1), linestyles = ('dashed', 'solid', 'dashed'))
    
    
    plt.plot([cI[0] for cI in corrInput], [cI[1] for cI in corrInput], 'g+')

    labels = ('classA', 'classB', 'nonzero')
    plt.axis('equal')   #force same scale on both axes
    plt.title('Cluster with some spread')
    plt.legend(labels)
    plt.savefig('plots/svmplot_rad_rand'+str(sigma)+'_spread_with'+str(C)+'.png')  #save a copy in a file
    plt.show() # show the plot on the screen



#bb = []
#namea bilder utifrån 
#svårt med 0.3, [1.5, 0.5]
std = 0.6
p = 5
sigma = 0.5
classA, classB, inputs, targets, N = genData()

#N = 10 #number of training samples
#print(inputs)
#print(targets)
start = numpy.zeros(N) #inital points for alphaVec
#global variabel array for equation 4
#global matrix that calculates every element and palce it in a N by N matrix
s = (N,N)
P = numpy.zeros(s)
for i in range(len(inputs)):
    for j in range(len(inputs)):
        P[i][j] = targets[i]*targets[j]*radKernal(inputs[i], inputs[j])
#print (linearKernal(inputs[0], inputs[0]))
#print(linearKernal(inputs[0][0], inputs[0][1]))
#print(P[0][0])
C = 1 # upper bound for b
B=[(0, C) for b in range(N)]
XC = {'type': 'eq', 'fun':zerofun}
ret = minimize(objective , start , bounds=B, constraints=XC)
alpha = ret['x']
print ('len alpha')
print ((alpha))
#print (ret['success'])
#print (fun.bb)
#print(fun.nzAlpha)

global nzAlpha
global corrInput
global corrTarget
global bb
#nonzero alphas, with there corrInput, or support vecotrs, and target
nzAlpha, corrInput, corrTarget = extractNZ(alpha)

b = calcB(nzAlpha, corrInput[0])


plotz(classA,classB,alpha)

