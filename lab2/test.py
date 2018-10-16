import functions as fun
import numpy, random, math
def calc():
	return 2


consraint ={'type':'eq', 'fun':calc}
print (consraint.get('fun'))

a = [1, 2, 3]
b = [1, 2, 3]

print(numpy.dot(a,b))

#trubleshooting help, will keep the random, "not random"
numpy.random.seed(100)
classA = numpy.concatenate((numpy.random.randn(3,2)*0.2+[1.5, 0.5],numpy.random.randn(3,2) * 0.2+[-1.5, 0.5]))
classB = numpy.random.randn(6, 2)*0.2+[0.0, -0.5]

inputs = numpy.concatenate((classA,classB))
targets = numpy.concatenate((numpy.ones(classA.shape[0]),-numpy.ones(classB.shape[0])))

N = inputs.shape[0] #numper of rows (samples)

premute = list(range(N))
random.shuffle(premute)
#inputs = inputs[premute, :]
targets = targets[premute]

classA = [[1,-1],[1,-2],[2,-1],[2,-2]]
classB = [[1,1],[1,2],[2,1],[2,2]]
inputs = numpy.concatenate((classA,classB))
targets = numpy.concatenate((numpy.ones(len(classA)),-numpy.ones(len(classB))))
N = len(inputs)
print('classA')
print(classA)
print('-------')
print('classB')
print(classB)
print('-------')
print('inputs')
print(inputs)
print('-------')
print('targets')
print(targets)
print('-------')
print([p[0] for p in classA])
print('-------')
iii = [[1,2],[0,0],[2,3]]
print([p[0] for p in iii])
ii = [i for i in targets]

print(fun.P)

print('hej')
print (numpy.e)

#fun.plotz(classA,classB)
