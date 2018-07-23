#pragma once
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;


struct Link
{
	double deltaWeight;
	double weight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************
class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	vector<double> getForwardWeights();
	Neuron(vector<double> forwardWeights, int id);
private:
	static double learningRate;   // [0.0..1.0] overall net training rate
	static double momentum; // [0.0..n] multiplier of last weight change (momentum)
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	vector<Link> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;
};


class Net
{
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }
	double normaliseY1(double y);
	double normaliseY2(double y);
	double deNormaiseY1(double o);
	double deNormaliseY2(double o);
	void train(string trainingData);
	double normaliseX1(double x);
	double normaliseX2(double x);
	double getTrainingError() { return trainingRMS; }
	double getValidationError() { return validationRMS; }
	void saveNetwork(string fileName);
	Net(vector<int> topology, vector<vector<vector<double>>> weights);
	static Net setUpNetwork(string fileName);
	vector<double> predict(vector<double> &inputs);
	vector<double> getOupts();
	void validate(string fileName);
private:
	vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
	double training_output1_RMS = 1;
	double training_output2_RMS = 1;
	double trainingRMS;
	double validation_output1_RMS = 1;
	double validation_output2_RMS = 1;
	double validationRMS;
	
	static double MIN_LEFT_MOTOR_SPEED;
	static double MAX_LEFT_MOTOR_SPEED;
	static double MIN_RIGHT_MOTOR_SPEED;
	static double MAX_RIGHT_MOTOR_SPEED;
	static double MIN_LEFT_SENSOR;
	static double MAX_LEFT_SENSOR;
	static double MIN_FRONT_SENSOR;
	static double MAX_FRONT_SENSOR;
};