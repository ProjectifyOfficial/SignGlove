from __init__ import *


def arch_to_str(a):
    s = ''
    for x in a:
        s += str(x) + ' '
    return s


def rule_of_thumb(Ni, No, Ns, alpha): return int(
    Ns * 1.0 / (alpha * (Ni + No)))


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
    for s in archstrlist:
        outfilename.write(s + '\n')
    outfilename.close()


def populate_data_files(n_inputs=6, n_outputs=5):
    n_hidden = int(raw_input('Give hidden layers #: '))
    n_min = int(raw_input('Give infimum: '))
    n_max = int(raw_input('Give supremum: '))
    arch_list = []
    assert (n_min >= 0 and n_max >= n_min)

    for j in range(1, n_hidden + 1):
        for i in range(n_min, n_max + 1):
            arch = [n_inputs]
            arch += j * [i]
            arch.append(n_outputs)
            s = arch_to_str(arch)
        #generate_file(s, 'data' + s + '.txt')
            arch_list.append(s)
    generate_arch_file(arch_list, 'archs.txt')
    return n_inputs + n_outputs + n_hidden * (n_max - n_min)


def populate_data_files_rule_of_thumb(n_inputs=6, n_outputs=5):
    n_samples = int(raw_input('Give number of samples: '))
    n_hidden = int(raw_input('Give hidden layers #: '))
    arch_list = []
    for n in range(1, n_hidden):
        for a in range(2, 11):
            arch = [n_inputs]
            arch += n * [rule_of_thumb(n_inputs, n_outputs, n_samples, a)]
            arch.append(n_outputs)
            s = arch_to_str(arch)
            #generate_file(s, 'data' + s + '.txt')
            arch_list.append(s)
    generate_arch_file(arch_list, 'archs.txt')


class NeuralNetworkFactory:

    def GenerateAndTrainNN(
            self,
            n,
            min_error=0.01,
            datafile='../.././data.txt',
            archfile='./archs.txt'):
        with open(archfile) as f:
            for i in range(n):
                line = f.readline()
        print line

        nn = Network()
        nn.InitializeFromFileWithSymbolsAndArch(datafile, archfile, n)
        nn.Train(min_error)
        line = ''

        nn.SaveWeights('./weights/weights' + line + '.txt')
        return nn


if __name__ == '__main__':
    #N_i = 6; N_o = 5;
    #N = populate_data_files(N_i, N_o)
    #ans = raw_input('Train NN with current arch?: ')
    NNF = NeuralNetworkFactory()
    # print 'Total archs ' + str(N)
    n = 75
    NNF.GenerateAndTrainNN(n)

    # if ans == 'y' or ans == 'Y':
    #	ans2 = raw_input('Train all?: ')
    #	if ans2 == 'y' or ans2 == 'Y':
    #		for i in range(N):
    #			NNF.GenerateAndTrainNN(i)
    #		n = int(raw_input('Give n: '))
    #		NNF.GenerateAndTrainNN(n)
