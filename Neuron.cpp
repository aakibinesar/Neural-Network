#include "Header.h"

double Neuron::learningRate = 0.15;    
double Neuron::momentum= 0.9;   ]


void Neuron::updateInputWeights(Layer &prevLayer)
{
	
	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
			
			learningRate
			* neuron.getOutputVal()
			* m_gradient
			
			+ momentum
			* oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
	

	return 1/(1+exp(-x));
}

double Neuron::transferFunctionDerivative(double x)
{
	
	return transferFunction(x)*(1-transferFunction(x));
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		sum += prevLayer[n].getOutputVal() *
			prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Link());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}
vector<double> Neuron::getForwardWeights()
{
	vector<double> weights;
	for (int i = 0;i<m_outputWeights.size();i++)
	{
		weights.push_back(m_outputWeights[i].weight);
	}
	return weights;
}
Neuron::Neuron(vector<double> forwardWeights, int id)
{
	for (int i = 0;i < forwardWeights.size();i++)
	{
		this->m_outputWeights.push_back(Link());
		this->m_outputWeights.back().weight = forwardWeights[i];

	}
	this->m_myIndex = id;
}