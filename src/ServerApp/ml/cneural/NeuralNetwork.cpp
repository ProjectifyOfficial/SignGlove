#include "NeuralNetwork.h"
#include <iostream>
#include <ctime>
#define DBL_MAX 1.79769e+308


using namespace std;

Neuron::Neuron()
{
	output = e = 0;
}

Neuron::Neuron(int prevNum, int index)
{
	
	output = e = 0;
	weights.resize(prevNum);
	for (int i = 0; i < prevNum; i++)
		weights[i] =  (double)rand() / RAND_MAX;
	this->index = index;
}

Neuron::Neuron(int prevNum, int index, double output)
{
	e = 0;
	weights.resize(prevNum);
	for (int i = 0; i < prevNum; i++)
		weights[i] = (double)rand() / RAND_MAX;
	this->index = index;
	this->output = output;
}

double Neuron::Transfer(double x)
{
	return (1 / (1 + exp(-x)));
}

void Neuron::FeedForward(Layer& prev)
{
	// output = f(Swiouti)
	double sum = 0;
	for (int i = 0; i < prev.size(); i++)
		sum += prev[i].output * weights[i];

	output = Transfer(sum);
}

void Neuron::BackPropagateHidden(Layer& next, Layer& prev)
{
	double sum = 0;
	for (int a = 0; a < next.size(); a++)
	{
		sum += next[a].e * next[a].weights[index];
	}

	e = output * (1 - output) * sum;
	for (int w = 0; w < weights.size(); w++)
	{
		weights[w] = weights[w] - L_RATE * e * prev[w].output;
	}
}

void Neuron::BackPropagateLast(double expected, Layer& prev)
{
	e = (output - expected) * output * (1 - output);

	for (int w = 0; w < weights.size(); w++)
	{
		// w' = w - de/dw
		weights[w] = weights[w] - L_RATE * e * prev[w].output;
	}
}

Network::Network(string fmt)
{
	//srand(time(NULL));
	error = DBL_MAX;
	std::stringstream sstream(fmt);
	string line;
	vector<Neuron> layer;

	//Create input layer
	std::getline(sstream, line, ' ');
	int n = atoi(line.c_str());
	for (int i = 0; i < n; i++)
		layer.push_back(Neuron(0, i));

	//Create bias neuron
	layer.push_back(Neuron(0, n, 1));
	layers.push_back(layer);

	while (std::getline(sstream, line, ' '))
	{
		layer.clear();
		n = atoi(line.c_str());
		for (int i = 0; i < n; i++)
			layer.push_back(Neuron(layers.back().size(), i));

		layer.push_back(Neuron(layers.back().size(), n, 1));
		layers.push_back(layer);
	}
}

void Network::Initiate(vector<vector<double> >& input, vector<vector<double> >& expected)
{
	for (int i = 0; i < input.size(); i++)
	{
		inputs.push_back(input[i]);
		this->expected.push_back(expected[i]);
	}
}

void Network::Train()
{
	error = 0;

	for (int i = 0; i < inputs.size(); i++)
	{
		// Initialize input layer
		for (int j = 0; j < inputs[i].size(); j++)
			layers[0][j].output = inputs[i][j];

		// FeedForward from second to last layer
		for (int l = 1; l < layers.size(); l++)
		{
			Layer& prev = layers[l - 1];
			for (int n = 0; n < layers[l].size() - 1; n++)	// exclude bias. Bias.output = 1 always.
			{
				layers[l][n].FeedForward(prev);
			}
		}

		// Error
		for (int n = 0; n < layers.back().size() - 1; n++)
			error += pow(layers.back()[n].output - expected[i][n], 2);

		//error = sqrt(error);

		// Backpropagate last layer
		for (int n = 0; n < layers.back().size() - 1; n++) // last layer has a bias that i not needed
		{
			Layer& prev = layers[layers.size() - 2];
			layers.back()[n].BackPropagateLast(expected[i][n], prev);
		}

		// Backpropagate from n - 1 to second layer not including input layer
		for (int l = layers.size() - 2; l > 0; l--)
		{
			for (int n = 0; n < layers[l].size(); n++)
			{
				Layer& next = layers[l + 1];
				Layer& prev = layers[l - 1];
				layers[l][n].BackPropagateHidden(next, prev);
			}
		}
	}
}

vector<double> Network::Feed(vector<double> input)
{
	assert(input.size() == layers.front().size() - 1);

	// Initialize input layer for feed forward
	for (int i = 0; i < input.size(); i++)
		layers[0][i].output = input[i];

	// Feed forward
	for (int l = 1; l < layers.size(); l++)
	{
		Layer& prev = layers[l - 1];
		for (int n = 0; n < layers[l].size() - 1; n++)	// exclude bias
		{
			layers[l][n].FeedForward(prev);
		}
	}

	int length = layers.back().size() - 1;
	vector<double> outputs(length);

	// Get final outputs
	for (int i = 0; i < length; i++)
		outputs[i] = layers.back()[i].output;

	return outputs;
}

void Network::SaveWeights(string dir)
{
	ofstream fout(dir.c_str());

	for (int i = 0; i < layers.size(); i++)
		for (int j = 0; j < layers[i].size(); j++)
			for (int w = 0; w < layers[i][j].weights.size(); w++)
				fout << layers[i][j].weights[w] << " ";
}

void Network::LoadWeights(string dir)
{
	ifstream fin(dir.c_str());

	for (int i = 0; i < layers.size(); i++)
		for (int j = 0; j < layers[i].size(); j++)
			for (int w = 0; w < layers[i][j].weights.size(); w++)
				fin >> layers[i][j].weights[w];
}

string Network::GetArchitectureAsString()
{
	string arch;
	
	for(int i = 1; i < layers.size() - 1; i++)
	{
		string temp;
		stringstream stream;
		stream << layers[i].size();
		stream >> temp;
		arch += temp;
	}
	
	return arch;
}

void Print(vector<double> vec)
{
	for (int i = 0; i < vec.size(); i++)
		cout << vec[i] << " ";

	cout << endl;
}

extern "C" void* InitializeFromFile(char* dir, int* numOutputs)
{
	ifstream fin(dir);

	if (fin.is_open() == false)
	{
		cout << "File not found!" << endl;
		return 0;
	}

	string arch;
	std::getline(fin, arch);

	Network* net = new Network(arch);

	vector<vector<double> > inputs;
	vector<vector<double> > expected;

	*numOutputs = net->layers.back().size() - 1;

	while (!fin.eof())
	{
		string temp;
		std::getline(fin, temp);
		if (temp == "") continue;

		stringstream ss(temp);
		double val;
		vector<double> vec;

		for (int i = 0; i < net->layers.front().size() - 1; i++)		// exclude bias
		{
			ss >> val;
			vec.push_back(val);
		}
		inputs.push_back(vec);
		vec.clear();

		for (int i = 0; i < net->layers.back().size() - 1; i++)
		{
			ss >> val;
			vec.push_back(val);
		}
		expected.push_back(vec);
	}

	net->Initiate(inputs, expected);
	return net;
}


extern "C" void* Initialize(int n, int *data)
{
	
	string fmt;
	stringstream stream;

	for (int i = 0; i < n; i++)
	{
  string temp;
  stream << data[i];
  stream >> temp;

  fmt += temp;

	if (i < n - 1)
	fmt += " ";
	stream.clear();
	}

	Network* net = new Network(fmt);
	
	vector<vector<double> > inputs;
	vector<double> temp;
	temp.push_back(0);
	temp.push_back(0);
	inputs.push_back(temp);

	temp[0] = 1;
	temp[1] = 1;
	inputs.push_back(temp);

	vector<vector<double> > outputs;
	vector<double> temp2;
	temp2.push_back(1);
	outputs.push_back(temp2);

	temp2[0] = 0;
	outputs.push_back(temp2);

	cout << "vector ready" << endl;

	net->Initiate(inputs, outputs);

	cout << "Network initialized" << endl;

	for (int i = 0; net->error > 0.1; i++)
	{
		if (i % 1000 == 0)
			cout << "foo "<< net->error << endl;

		net->Train();
	}

	return net;
}

extern "C" double* Feed(void* network, double* params, double* outputs)
{
	Network& net = *(Network*)network;

	vector<double> temp;

	int n = net.layers[0].size() - 1;	// exclude bias neuron

	for (int i = 0; i < n; i++)
		temp.push_back(params[i]);

	cout << "End print" << endl;

	vector<double> d = net.Feed(temp);

	for (int i = 0; i < d.size(); i++)
		outputs[i] = d[i];

	return outputs;
}

extern "C" int SaveWeights(void *network, const char *dir)
{
	Network& net = *(Network*)network;
	ofstream fout(dir);
	
	 if (fout.is_open() == false)
	  return 0;
	
	 for (int i = 0; i < net.layers.size(); i++)
		  for (int j = 0; j < net.layers[i].size(); j++)
			for (int w = 0; w < net.layers[i][j].weights.size(); w++)
	    fout << net.layers[i][j].weights[w] << " ";
	
	 return 1;
}

extern "C" int LoadWeights(void *network, const char *dir)
{
	Network& net = *(Network*)network;
	ifstream fin(dir);

	if (fin.is_open() == false)
  return 0;

 for (int i = 0; i < net.layers.size(); i++)
  for (int j = 0; j < net.layers[i].size(); j++)
   for (int w = 0; w < net.layers[i][j].weights.size(); w++)
    fin >> net.layers[i][j].weights[w];    
return 1;
}

extern "C" void* InitializeFromFileWithSymbolsAndArch(char* dir, int* numOutputs, char* archDir, int n)
{
	ifstream fin(dir);
	ifstream farch(archDir);

	if (fin.is_open() == false || farch.is_open() == false)
	{
		cout << "File not found!" << endl;
		return 0;
	}
	
	string arch;
	for(int i = 0; i < n; i++)
		getline(farch, arch);
	
	//std::getline(fin, arch);

	Network* net = new Network(arch);

	vector<vector<double> > inputs;
	vector<vector<double> > expected;

	*numOutputs = net->layers.back().size() - 1;

	while (!fin.eof())
	{
		string temp;
		std::getline(fin, temp);
		
		if (temp == "") continue;

		stringstream ss(temp);
		double val;
		vector<double> vec;

		for (int i = 0; i < net->layers.front().size() - 1; i++)		// exclude bias
		{
			ss >> val;
			vec.push_back(val);
		}
		inputs.push_back(vec);
		vec.clear();

		for (int i = 0; i < net->layers.back().size() - 1; i++)
		{
			while (ss >> val)
			{
				vec.push_back(val);
				i++;
			}

			if (i == net->layers.back().size() - 1)
				break;

			vec.push_back(0);
		}

		expected.push_back(vec);
	}


	net->Initiate(inputs, expected);
	return net;
}

extern "C" void* InitializeFromFileWithSymbols(char* dir, int* numOutputs)
{
	ifstream fin(dir);

	if (fin.is_open() == false)
	{
		cout << "File not found!" << endl;
		return 0;
	}

	string arch;
	std::getline(fin, arch);

	Network* net = new Network(arch);

	vector<vector<double> > inputs;
	vector<vector<double> > expected;

	*numOutputs = net->layers.back().size() - 1;

	while (!fin.eof())
	{
		string temp;
		std::getline(fin, temp);
		
		if (temp == "") continue;

		stringstream ss(temp);
		double val;
		vector<double> vec;

		for (int i = 0; i < net->layers.front().size() - 1; i++)		// exclude bias
		{
			ss >> val;
			vec.push_back(val);
		}
		inputs.push_back(vec);
		vec.clear();

		for (int i = 0; i < net->layers.back().size() - 1; i++)
		{
			while (ss >> val)
			{
				vec.push_back(val);
				i++;
			}

			if (i == net->layers.back().size() - 1)
				break;

			vec.push_back(0);
		}

		expected.push_back(vec);
	}


	net->Initiate(inputs, expected);
	return net;
}

extern "C" void Train(void* network, double minError)
{
	Network& net = *(Network*)network;
	
	string arch = net.GetArchitectureAsString();
		
	ofstream fout("./errors/error" + arch + ".txt");
	
	for (int i = 1; net.error > minError; i++)
	{
		if (i % 1000 == 0)
		{
			cout << net.error << endl;
		}
		
		if (i % 500 == 0)
		{
			fout << i << " " << net.error << endl;
			fout.flush();
		}

		net.Train();
	}

	fout.close();
}

extern "C" void* InitializeFromFileWithSymbols(char* dir, int* numOutputs);
extern "C" void Train(void* network, double minError);
extern "C" void* InitializeFromFileWithSymbolsAndArch(char* dir, int* numOutputs, char* archDir, int n);
void Print(vector<double> vec);
extern "C" int SaveWeights(void *network, const char *dir);

int main()
{
	int outputs;

	{
		Network& net = *(Network*)InitializeFromFileWithSymbolsAndArch("../../data.txt", &outputs, "archs.txt", 70);

		cout << net.GetArchitectureAsString() << endl;

		Train(&net, 0.001);

		net.SaveWeights("./weights/weight" + net.GetArchitectureAsString() + ".txt");
	}

	system("pause");
	return 0;
}
