import matplotlib.pyplot
import pylab

x = []
y = []

with open('./cneural/error.txt') as f:
    while 1 == 1:
        line = f.readline()
        if not line:
            break

        line = line.strip('\n').split(' ')[:len(line) - 2]
        x.append(int(line[0]))
        y.append(float(line[1]))

matplotlib.pyplot.scatter(x, y)

matplotlib.pyplot.show()
