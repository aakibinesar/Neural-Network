#include "Header.h"
#include <Aria.h>
#include <chrono>
#include <thread>
double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

double Net::MIN_LEFT_MOTOR_SPEED = 100;
double Net::MAX_LEFT_MOTOR_SPEED = 2309.643;
double Net::MIN_RIGHT_MOTOR_SPEED = 0;
double Net::MAX_RIGHT_MOTOR_SPEED = 270;

double Net::MIN_LEFT_SENSOR = 100;
double Net::MAX_LEFT_SENSOR = 5000;
double Net::MIN_FRONT_SENSOR = 910.539;
double Net::MAX_FRONT_SENSOR = 5000;
void Net::getResults(vector<double> &resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
		//cout << n<<endl;
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::backProp(const vector<double> &targetVals)
{
	
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		//cout << "targetVal " << targetVals[n];
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1; 
	m_error = sqrt(m_error); 

							 
	m_recentAverageError =
		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1.0);



	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	

	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// updating Weights for all neuron
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::feedForward(const vector<double> &inputVals)
{
	assert(inputVals.size() == m_layers[0].size() - 1);

	
	for (unsigned i = 0; i < inputVals.size(); ++i) {
		double input = i == 0 ? normaliseX1(inputVals[i]) : normaliseX2(inputVals[i]);
		//cout << "input = " << input<<endl;
		m_layers[0][i].setOutputVal(input);
	}

	
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

Net::Net(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "Made a Neuron!" << endl;
		}

		
		m_layers.back().back().setOutputVal(1.0);
	}
}
void Net::train(string triainingData)
{
	ifstream file(triainingData);
	vector<double> inputs;
	vector<double> correctvals;
	vector<double> results;
	int iterations = 1;
	while (!file.eof())
	{
		
		string line;
		getline(file, line, '\n');
		stringstream SS(line);
		inputs.clear();
		results.clear();
		correctvals.clear();

		if (line.empty())continue;
		int i = 0;
		while (i < 2) {
			getline(SS, line, ',');
			inputs.push_back(stod(line));
			i++;
		}
		
		feedForward(inputs);
		getResults(results);
		int j = 0;
		while (j < 2)
		{
			getline(SS, line, ',');
			double c = j == 0 ? normaliseY1(stod(line)) : normaliseY2(stod(line));
			correctvals.push_back(c);
			j++;
		}
		
		
		//calculating and printing RMS for each output neuron
		backProp(correctvals);
		training_output1_RMS = pow((correctvals[0] - results[0]), 2);
		training_output2_RMS = pow((correctvals[1] - results[1]), 2);
		iterations += 1;
	}
	//divide the RMS for each output neuron by the nymev
	training_output1_RMS /= iterations;
	training_output2_RMS /= iterations;

	training_output1_RMS = sqrt(training_output1_RMS);
	training_output2_RMS = sqrt(training_output2_RMS);

	trainingRMS = (training_output1_RMS + training_output2_RMS) / 2;
}
void Net::validate(string fileName)
{
	ifstream file(fileName);
	vector<double> inputs;
	vector<double> correctvals;
	vector<double> results;
	int iterations = 1;
	while (!file.eof())
	{

		string line;
		getline(file, line, '\n');
		stringstream SS(line);
		inputs.clear();
		results.clear();
		correctvals.clear();

		if (line.empty())continue;
		int i = 0;
		while (i < 2) {
			getline(SS, line, ',');
			inputs.push_back(stod(line));
			i++;
		}

		feedForward(inputs);
		getResults(results);
		int j = 0;
		while (j < 2)
		{
			getline(SS, line, ',');
			double c = j == 0 ? normaliseY1(stod(line)) : normaliseY2(stod(line));
			correctvals.push_back(c);
			j++;
		}


		//calculating and printing RMS for each output neuron
		
		validation_output1_RMS= pow((correctvals[0] - results[0]), 2);
		validation_output2_RMS = pow((correctvals[1] - results[1]), 2);
		iterations += 1;
	}
	//divide the RMS for each output neuron by the nymev
	validation_output1_RMS /= iterations;
	validation_output2_RMS /= iterations;

	validation_output1_RMS = sqrt(training_output1_RMS);
	validation_output2_RMS = sqrt(training_output2_RMS);

	validationRMS= (validation_output1_RMS+ validation_output2_RMS) / 2;
}

void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		cout << v[i] << " ";
	}

	cout << endl;
}
double Net::normaliseY1(double y)
{
	return((y - MIN_LEFT_MOTOR_SPEED) / (MAX_LEFT_MOTOR_SPEED - MIN_LEFT_MOTOR_SPEED));
}
double Net::normaliseY2(double y)

{
	return((y - MIN_RIGHT_MOTOR_SPEED) / (MAX_RIGHT_MOTOR_SPEED - MIN_RIGHT_MOTOR_SPEED));
}
double Net::deNormaiseY1(double o)
{
	return((o*(MAX_LEFT_MOTOR_SPEED - MIN_LEFT_MOTOR_SPEED) + MIN_LEFT_MOTOR_SPEED));
}
double Net::deNormaliseY2(double o)
{
	return (o*(MAX_RIGHT_MOTOR_SPEED - MIN_RIGHT_MOTOR_SPEED) + MIN_RIGHT_MOTOR_SPEED);
}
double Net::normaliseX1(double x)
{
	return (x - MIN_LEFT_SENSOR) / (MAX_LEFT_SENSOR - MIN_LEFT_SENSOR);
}
double Net::normaliseX2(double x) {
	return (x - MIN_FRONT_SENSOR) / (MAX_FRONT_SENSOR - MIN_FRONT_SENSOR);
}
void Net::saveNetwork(string fileName)

{
	ofstream file(fileName);
	stringstream ss;
	// saving topology
	for (int layer = 0; layer < m_layers.size();layer++) {
		if (layer == m_layers.size() - 1)ss << m_layers[layer].size() << "\n";
		else ss << m_layers[layer].size() << ",";
	}
	file << ss.rdbuf();
	for (int layer = 0;layer<m_layers.size() - 1;layer++)
	{
		stringstream ls;
		for (int neuron = 0; neuron<m_layers[layer].size();neuron++)
		{
			vector<double> weights = m_layers[layer][neuron].getForwardWeights();
			cout << "Neuron " << neuron << " in Layer " << layer << "has " << weights.size() << " Link" << endl;
			for (int w = 0; w<weights.size();w++)
			{
				if (layer == m_layers.size() - 1 && w == weights.size() - 1)ls << weights[w];
				else ls << weights[w] << ", ";
			}
		}
		ls << endl;
		file << ls.rdbuf();
	}


	file.close();
	cout << "File saved";
}
Net Net::setUpNetwork(string fileName)
{
	ifstream file(fileName);
	vector<int> topology;
	string line;
	getline(file, line, '\n');
	cout << line << endl;
	stringstream ss(line);

	while (!ss.eof()) {
		getline(ss, line, ',');
		topology.push_back(stoi(line));
	}

	vector<vector<vector<double>>> weights;
	int layer = 0;
	while (!file.eof() && layer<topology.size() - 1)
	{
		getline(file, line, '\n');
		cout << line;
		stringstream ls(line);
		vector<vector<double>> layerWeight;
		int neuron = 0;
		while (!ls.eof() && neuron<topology[layer])
		{
			int w = 0;
			vector<double> neuronweight;
			while (w<topology[layer + 1] - 1)
			{

				getline(ls, line, ',');
				cout << "weights = " << line << endl;
				neuronweight.push_back(stod(line));
				w++;
			}
			layerWeight.push_back(neuronweight);
			neuron++;
		}
		weights.push_back(layerWeight);
		layer++;
	}
	return Net(topology, weights);
}
vector<double> Net::predict(vector<double> &inputs)
{
	this->feedForward(inputs);
	getResults(inputs);
	return inputs;
}
Net::Net(vector<int> topology, vector<vector<vector<double>>> weights)
{
	//inputs + hidden layers
	for (int layer = 0;layer<topology.size() - 1;layer++)
	{
		vector<vector<double>> layerWeights = weights[layer];
		m_layers.push_back(Layer());
		for (int neuron = 0;neuron<topology[layer];neuron++)
		{
			m_layers.back().push_back(Neuron(layerWeights[neuron], neuron));
		}
		m_layers.back().back().setOutputVal(1); //set output values of biases to 1
	}

	// create output neuron
	m_layers.push_back(Layer());
	for (int i = 0; i<topology.back();i++)
	{
		m_layers.back().push_back(Neuron(0, i));
	}

}
int main(int argc, char** argv)
	{


		vector<unsigned> topology;
		topology.push_back(2);
		topology.push_back(5);
		topology.push_back(2);
		Net network = Net(topology);
		
		int epoch = 0;
		ofstream Error("Errors.csv");
		Error << "Train Error, Validation Error \n";

		while (epoch < 100)
		{
			network.train("M:/correctedData2.csv");
			//network.saveNetwork("Weights.csv");
			network.validate("M:/validationData.csv");
			double tError = network.getTrainingError();
			double vError = network.getValidationError();
			Error << tError << "," << vError << "\n";
			cout << "***********************************************************************************" << endl;
			cout << "epoch: " << epoch << " traininingError = " << tError<< " validationError = " << vError << endl;
			epoch++;


		}
		Error.close();

		ofstream sonarData("SonarData.csv");
		double PI = 3.14;
		system("pause");
		Aria::init();
		ArArgumentParser argParser(&argc, argv);
		argParser.loadDefaultArguments();
		ArRobot robot;
		ArRobotConnector robotConnector(&argParser, &robot);
		if (robotConnector.connectRobot())
		cout << "Robot Connected!" << endl;
		robot.runAsync(false);
		robot.lock();
		robot.enableMotors();
		robot.unlock();

		ArSensorReading *sonarSensor[8];
		for (int i = 0; i < 8; i++)
		{
		sonarSensor[i] = robot.getSonarReading(i);
		}
		
		vector<double> inputs;
		while (true)
		{
			double sonar0[5];
			double sonar1[5];			
			int reading = 0;
			while (reading < 5) 
			{
				sonar0[reading] = sonarSensor[0]->getRange();
				sonar1[reading]= sonarSensor[1]->getRange();
				reading += 1;
			}
			
			inputs.clear();
			double *min0 = min(begin(sonar0), end(sonar0));
			double *min1 = min(begin(sonar1), end(sonar1));
			if (*min0 >= 5) 
			{
				double o = pow(*min1*sin(40 * (PI / 180)),2);
				double h = pow(*min1, 2);
				*min0 = sqrt(o + h);
			}
			if (*min1 >= 5000) 
			{
				double o = pow(*min0*tan(40 * (PI / 180)), 2);
				double a = pow(*min0, 2);
				*min1 = sqrt(o + a);
			}
			inputs.push_back(*min0);
			inputs.push_back(*min1);
			vector<double> wheelVelocitys = network.predict(inputs);
			sonarData << *sonar0 << "," << *sonar1 << network.deNormaiseY1(wheelVelocitys[0])<<","<<network.deNormaliseY2(wheelVelocitys[1])<<endl;
			cout << "Sensor 0:"<<&inputs[0]<< endl;
			cout << "Sensor 1:" << &inputs[1] << endl;
			//cout << wheelVelocitys[0] << endl;
			//cout << wheelVelocitys[1] << endl;

			robot.setVel2(network.deNormaiseY1(wheelVelocitys[0]),network.deNormaliseY2(wheelVelocitys[1]));
			//;
	}
		sonarData.close();
	
}