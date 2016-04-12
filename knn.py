import math
from abc import *
THRESHOLD = 10 #TODO change

class Metric:
    
    @abstractmethod
    def __call__(self, X):
        pass
        
    def dist(self, X, Y):
        Z = [X[i] - Y[i] for i in range(len(X))]
        return self.__call__(Z)
        
class PNorm (Metric):
    def __init__(self, p):
        self.p = p
    
    def __call__(self, X, rooted=False):
        if isinstance(X, Point): 
            X = X.coords    
            
        d = sum([abs(x)**self.p for x in X])
        if rooted:
            return d**(1.0 / self.p)
        return d
        
class WeightedPNorm(PNorm):
    
    def __init__(self, p, W):
        assert(sum(W) == 1)
        super(WeightedPNorm, self).__init__(p)
        self.W = W
        
    def __call__(self, X, rooted=False):
        d = sum([w[i]*abs(x[i])**self.p for i in range(len(X))])
        if rooted:
            return d**(1.0 / self.p)
        return d
        
class Norms: #norms enum
    TaxicabNorm = PNorm(1)
    EuclideanNorm = PNorm(2)
    ChebysevNorm = staticmethod(lambda X: max([abs(x) for x in X]))
        

class Point:
    def __init__(self, symbol, *coords):
        self.coords = [coords[i] for i in range(len(coords))]
		self.Symbol = symbol

    def __str__(self):
        string = "( "
        for c in self.coords:
            string += str(c) + " , "

        string = string[: len(string) - 2] + ")"
        return string

    @staticmethod
    def DistanceSquared(first, second):
        return Norms.EuclideanNorm.dist(first, second)
        
        
    def __add__(self, other):
        coords = [self.coords[i] + other.coords[i] for i in range(len(self.coords))] 
        return Point(coords)
        
    def __sub__(self, other):
        coords = [self.coords[i] - other.coords[i] for i in range(len(self.coords))] 
        return Point(coords)
       
    def __mul__(self, other):
        return sum([self.coords[i]*other.coords[i] for i in range(len(self.coords))])
        
    def __len__(self):
        return len(self.coords)
        
    def __getitem__(self, I):
        return self.coords[I]
        
    def __setitem__(self, I, V):
        self.coords[I] = V        

class KNNClassifier:
    def __init__(self, sampleData = [], norm=Norms.EuclideanNorm):
        self.sampleData = sampleData

        self.norm = norm

    def GetClosest(self, p, K = 1):
        # distances = array [ [distance, point] ]
        #distances = sorted([Point.DistanceSquared(p, self.sampleData[i]) for i in range(len(self.sampleData))], key=lambda x: x[0])
        distances = []
        for y in self.sampleData:
            distances.append([self.norm.dist(p,y), y])
        distances = sorted(distances, key=lambda x: x[0])

        closePoints = {}
		for i in range(min(K, len(distances))):
            if distances[i][0] <= THRESHOLD:
				#closePoints.append(distances[i][1]) # grab the ith point
				try:
					closePoints[distances[i][1].Symbol] += 1 
				except KeyError:
					closePoints[distances[i][1].Symbol] = 1			
        return closePoints
    
	def Vote(self, p, K = 1, f = lambda x: x + 1):
		closePoints = self.GetClosest(p, K)
		keys = closePoints.keys()
		candidateSymbol = keys[0]
		for symbol in keys:
			if closePoints[candidateSymbol] < closePoints[symbol]:
				candidateSymbol = symbol
		
		for symbol in keys:
			if closePoints[candidateSymbol] == closePoints[symbol] and candidateSymbol != symbol:
				return self.Vote(p, f(K)) 
		
		return symbol
	

x = KNNClassifier([Point(2, 2), Point(4, 4), Point(6, 6)], 2, Norms.TaxicabNorm)

y = x.GetClosest(Point(2, 2))

for c in y:
    print c


A = Point(-4,2)
B = Point(3,3)
print Norms.EuclideanNorm(A)
print Norms.ChebysevNorm(A.coords)
print Norms.TaxicabNorm(A)
print Norms.EuclideanNorm.dist(A,B)
print A*B
        
