import math,string,random,time
#import numpy as np
#from abc import *
THRESHOLD = 1000 #TODO change
global VALID_GESTURE_TIME
VALID_GESTURE_TIME = 0.01
SYMBOL_COUNT = 6

#Metrics and norms
class Metric:
	
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
		
#geometry
class Point:
	def __init__(self, *coords):
		self.coords = [coords[i] for i in range(len(coords))]
		self._Symbol = ''
		
	@property
	def Symbol(self):
		return self._Symbol
	
	@Symbol.setter	
	def Symbol(self, s):
		assert(isinstance(s, str))
		self._Symbol = s

	def __str__(self):
		string = "( "
		for c in self.coords:
			string += str(c) + " , "

		string = string[: len(string) - 2] + ")"
		return string

	def __repr__(self):
		return self.__str__()

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
		
class PointCluster (dict): #inherits from hashmap
	
	def __init__(self,ndims=2):
		self.cluster = {}
		self._centroids = {}
		self.ndims = ndims
	
	def keys(self):
		return self.cluster.keys()	
		
	def append_point(self, P):
		try:
			self.cluster[P.Symbol].append(P)
		except KeyError:
			self.cluster[P.Symbol] = [P]
	
	def __str__(self):
		u = ''
		for s in self.cluster.keys():
			u += s + '\n'
			for p in self.cluster[s]:
				u+= p + '\n'
		return u
		
	def __getitem__(self, key):
		return self.cluster[key]
		
	def __setitem__(self, key,value):
		self.cluster[key] = value		
		
	@staticmethod
	def GenerateRandomPointCluster(n,dims = 2, number_of_classes = -1):
		result = []
		for i in range(n):
			P = []
			for d in range(dims):
				P.append(random.randrange(0,100))
			P = Point(*P)
			if number_of_classes == -1:
				P.Symbol = random.choice(string.letters)
			else:
				P.Symbol = random.choice(string.letters[:number_of_classes])
			result.append(P)
		return result
	
	@property	
	def centroids(self):
		return self._centroids
		
	@centroids.setter
	def centroids(self): 
		pass
		
	@centroids.getter
	def centroids(self):
		for s in self.clusters.keys():
			C = Point(*([0]*self.ndims))
			m = len(self.clusters[s])
			for p in self.clusters[s]:
				C = C + p
			for i in range(len(C)):
				C[i] = C[i]*1.0 / m
			self._centroids[s] = C
		return self._centroids
		
	@staticmethod
	def ToCluster(pts):	
		cluster = PointCluster(ndims=len(pts[0]))
		for p in pts:
			cluster.append_point(p)
		return cluster
		
	@staticmethod
	def ToList(cluster):
		l = []
		for s in cluster.cluster.keys(): 
			l += cluster.cluster[s]
			
		return l
		
	def lengths(self):
		length = {}
		for k in self.cluster.keys():
			length[k] = len(self.cluster[k])
					

class Classifier:
	"""ABC for Classifier"""
	pass

class KNNClassifier (Classifier):
	"""KNN classifier"""
	
	def __init__(self, sampleData = [], norm=Norms.EuclideanNorm):
		#self.sampleData = sampleData
		self.clusters = PointCluster.ToCluster(sampleData)
		self.N = len(sampleData)
		self.norm = norm

	def GetClosest(self, p, K = 1):
		distances = []
		for cluster in self.clusters.keys():
			for y in self.clusters[cluster]:
				distances.append([self.norm.dist(p,y), y])
		distances = sorted(distances, key=lambda x: x[0])
		closePoints = {}
		for i in range(min(K, len(distances))):
			if distances[i][0] <= THRESHOLD:
				
				try:
					closePoints[distances[i][1].Symbol] += 1 
				except KeyError:
					closePoints[distances[i][1].Symbol] = 1         
		return closePoints
	
	def Vote(self, p, K = 1, f = lambda x: x + 1, append_to_sample_data=False):
		closePoints = self.GetClosest(p, K)
		#print closePoints, K
		keys = closePoints.keys()
		try:
			candidateSymbol = [keys[0]]
		except:
			return None
		for symbol in keys:
			if closePoints[candidateSymbol[0]] < closePoints[symbol]:
				candidateSymbol = [symbol]
		
		for symbol in keys:
			if closePoints[candidateSymbol[0]] == closePoints[symbol] and candidateSymbol[0] != symbol:
				#K_prime = int(f(K))
				#if K_prime <= self.N: #avoid depth exceeded problem
					#return self.Vote(p, K_prime) 
				candidateSymbol.append(symbol)
		
		if append_to_sample_data:
			p.Symbol = symbol
			#self.sampleData.append(p)
			self.clusters[symbol].append(p) #cluster always exist here
			serlf.N += 1
		return candidateSymbol
					
	def __repr__(self):
		return self.clusters
		
	def GetStandardDeviation(self, p):
		self.centroids = self.clusters.centoids
		sigma = 0
		for s in self.centroids.keys():
			Q = self.centroids[s] - p
			sigma = sigma + Norms.EuclideanNorm(Q)
		return math.sqrt(sigma*1.0 / len(self.centroids))		

class KMeansClassifier (Classifier): #some stack overflow stuff
	"""KMeans Classifier"""

	@staticmethod
	def kmeans(data, k, c):
		centroids = []
	
		centroids = KMeansClassifier.randomize_centroids(data, centroids, k)  
	
		old_centroids = [[] for i in range(k)] 
	
		iterations = 0
		while not (KMeansClassifier.has_converged(centroids, old_centroids, iterations)):
			iterations += 1
	
			clusters = [[] for i in range(k)]
	
			# assign data points to clusters
			clusters = KMeansClassifier.dist(data, centroids, clusters, norm=Norms.EuclideanNorm) #generalize norm
	
			# recalculate centroids
			index = 0
			for cluster in clusters:
				old_centroids[index] = centroids[index]
				centroids[index] = np.mean(cluster, axis=0).tolist()
				index += 1
	
	
		print("The total number of data instances is: " + str(len(data)))
		print("The total number of iterations necessary is: " + str(iterations))
		print("The means of each cluster are: " + str(centroids))
		print("The clusters are as follows:")
		for cluster in clusters:
			print("Cluster with a size of " + str(len(cluster)) + " starts here:")
			print(np.array(cluster).tolist())
			print("Cluster ends here.")	
	
	
	# Calculates euclidean distance between
	# a data point and all the available cluster
	# centroids.   
	@staticmethod   
	def dist(data, centroids, clusters, norm = Norms.EuclideanNorm):
		for instance in data:  
			# Find which centroid is the closest
			# to the given data point.
			mu_index = min([(i[0], norm.dist(list(instance),list(centroids[i[0]]))) \
								for i in enumerate(centroids)], key=lambda t:t[1])[0]
			try:
				clusters[mu_index].append(instance)
			except KeyError:
				clusters[mu_index] = [instance]
	
		# If any cluster is empty then assign one point
		# from data set randomly so as to not have empty
		# clusters and 0 means.        
		for cluster in clusters:
			if not cluster:
				cluster.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
	
		return clusters
	
	
	# randomize initial centroids
	@staticmethod
	def randomize_centroids(data, centroids, k):
		for cluster in range(0, k):
			centroids.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
		return centroids
	
	
	# check if clusters have converged
	@staticmethod    
	def has_converged(centroids, old_centroids, iterations):
		MAX_ITERATIONS = 1000
		if iterations > MAX_ITERATIONS:
			return True
		return old_centroids == centroids

class SymbolState:
	IDLE, VALID_GESTURE = range(2)

	def __init__(self):
		self.CurrentState = SymbolState.IDLE    
		self.TimeInside = 0                      # time inside current state

	def SetState(self, newState, dt):
		
		if self.CurrentState != newState:        # enter new state => reset time
			self.TimeInside = 0
		else:                                    # already in new state => accumulate time
			self.TimeInside += dt
				
		self.CurrentState = newState

class SymbolManager:
	
	def __init__(self, train_data):
		self.State = SymbolState()
		self.ActivationData = None
		self.StartTime = time.time()
		self.Subscribers = []
		self.knn = KNNClassifier(train_data, norm=Norms.EuclideanNorm)
	
	def Update(self, values):
		if isinstance(values, Point):
			P = values
		else:
			P = Point(*values)
		
		dt = time.time() - self.StartTime
		K = int(math.sqrt(self.knn.N))

		candidate_class = self.knn.Vote(P, K)
		print candidate_class
		
		if candidate_class != None and len(candidate_class) == 1:
			self.OnActivationChanged(candidate_class[0])
				
		#if candidate_class == None or len(candidate_class) > 1:
			#self.State.SetState(SymbolState.IDLE, dt)
			#self.ActivationData = None
		#else:
			#self.State.SetState(SymbolState.VALID_GESTURE, dt)
			
			#if self.State.TimeInside >= VALID_GESTURE_TIME:         # gesture identified:
				#identifiedGesture = candidate_class[0]                      # the only element of actives array

				#if self.ActivationData is None:                     # first time data change after IDLE state
					#self.OnActivationChanged(identifiedGesture.Data)                    

				#self.ActivationData = identifiedGesture.Data 
				
		self.StartTime = time.time()
				
	def OnActivationChanged(self, data):  
		for f in self.Subscribers:
			f(data)
	
	#replaced by vote	 
	def GetActivated(self):
		print 'obsolete'
		
	def __str__(self):
		return self.knn.clusters.__str__()	
		
	@staticmethod
	def DummySub(data):
		print data

def GenerateRandomPoints(N, d):
	l = []
	for i in range(N):
		s = []
		for j in range(d):
			s.append(random.randint(0,100))
		l.append(Point(*s))
	return l

def populate_all(filename):
	cluster = PointCluster(SYMBOL_COUNT)
	symbol = ''
	with open(filename, 'r') as f:
		while True:
			line = f.readline()
			if not line:
				break
			line = line.strip('\n')	
			if line[0] != '[' and line[0] in string.letters: #f[0] is alpha
				symbol = f[0]
			elif line[0] == '[' and line[len(line)-1] == ']':
				line = ((line.strip('[')).strip(']')).strip('\n')
				line = line.strip(']')
				line = line.split(',')
				P = Point(*map( lambda x: int(x), line))
				P.Symbol = symbol
				cluster.append_point(P)
	return SymbolManager(cluster)

	
if __name__ == '__main__':	
	N = 2500 #realistic approximation? 
	M = 50
	K = int(math.sqrt(N))
	
	sample_pts = PointCluster.GenerateRandomPointCluster(N, 6, 26)
	train_pts = GenerateRandomPoints(M, 6)
	
	for p in train_pts:
		print p
		
	symbol_manager = SymbolManager(sample_pts)
	symbol_manager.Subscribers.append(SymbolManager.DummySub)
	
	for Q in train_pts:
		time.sleep(0.02)
		symbol_manager.Update(Q)
	
	
	
