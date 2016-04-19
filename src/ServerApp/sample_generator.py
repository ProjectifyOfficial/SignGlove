from __init__ import *

arch_to_str = lambda arch: sum([str(x) + ' ' for x in arch])
rule_of_thumb = lambda Ni, No, Ns, alpha: int(Ns*1.0 / (alpha*(Ni + No)))

def generate_file(archstr, datafilename):
	
	datafile = open(datafilename, 'r')
	outfile = open(datafilename, 'w')
	
	outfile.write(arch_str + '\n')

	while 1 == 1:
		line = datafile.readline()
		if not line:
			break
		outfile.write(line)
		
	datafile.close()
	outfile.close()

def generate_arch_file(archstrlist, outfilename):
	outfilename = open(outfilename, 'w')
	for s in archstr_list:
		outfilename.write(s + '\n')
	outfilename.close()
	
def populate_data_files(n_inputs = 6, n_outputs = 5):
		n_hidden = int(raw_input('Give hidden layers #:'))
		n_min = int(raw_input('Give infimum: '))
		n_max = int(raw_input('Give supremum: '))
		arch_list = []
		assert (n_min >= 0 and n_max >= n_min)		
		for i in range(n_min, n_max + 1):
			arch = [n_inputs]
			arch += n_hidden * [i]
			arch.append(n_outputs)			
			s = arch_to_str(arch)
			#generate_file(s, 'data' + s + '.txt') 
			arch_list.append(s)
		generate_arch_file(arch_list, 'archs.txt')
		
def populate_data_files_rule_of_thumb(n_inputs = 6, n_outputs = 5):
	n_samples = int(raw_input('Give number of samples: '))
	n_hidden = int(raw_input('Give hidden layers #:'))
	arch_list = []
	for n in range(1, n_hidden):
		for a in range(2,11):
			arch = [n_inputs]
			arch += n * [rule_of_thumb(n_inputs, n_outputs, n_samples,a)]
			arch.append(n_outputs)			
			s = arch_to_str(arch)
			#generate_file(s, 'data' + s + '.txt') 
			arch_list.append(s)
	generate_arch_file(arch_list, 'archs.txt')
	
class NeuralNetworkFactory:

	@staticmethod
	def GenerateNN(n, min_error = 0.01, datafile = './data.txt', archfile = '/archs.txt'):
		nn = Network()
		nn.InitializeFromFileWithSymbolsAndArch(datafile, archfile, n)
		nn.Train(min_error)
		with open(archfile) as f:
			for i in range(n):
				line = f.readline()
		nn.SaveWeights('weights' + line + '.txt')
		return nn
		
if __name__ == '__main__':
	pass
		