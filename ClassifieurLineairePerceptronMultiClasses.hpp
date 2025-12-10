#pragma once
#include <vector>

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

class PerceptronMultiClasses
{
	private : 
		int inputSize;
		int numClasses;
		float learningRate;
		std::vector<std::vector<float>> weights;//poids pour chaque classe : weights[classe][entrée]
		std::vector<float> bias;//biais pour chaque classe : bias[classe]

	public : 
		PerceptronMultiClasses(int inputSize, int numClasses, float lr = 0.1f);

		//fonction d'activation (Heaviside : 1 si >= 0, 0 sinon)
		int activate(float sum);

		//prédiction pour une classe donnée (retourne 0 ou 1)
		int predictClass(const std::vector<float> &inputs, int classIndex);

		//entraînement (règle de Rosenblatt)
		void train(const std::vector<std::vector<float>> &X, const std::vector<int> &Y, int epochs);

		//prédiction finale (retourne l'indice de la classe avec le score le plus élevé)
		int predict(const std::vector<float> &inputs);

		int getInputSize() const
		{
			return inputSize;
		}

		int getNumClasses() const
		{
			return numClasses;
		}

		//getters pour sauvegarde
		const std::vector<std::vector<float>> &getWeights() const
		{
			return weights;
		}

		const std::vector<float> &getBias() const
		{
			return bias;
		}

		//setters pour chargement
		void setWeights(const std::vector<std::vector<float>> &w)
		{
			weights = w;
		}

		void setBias(const std::vector<float> &b)
		{
			bias = b;
		}
};

extern "C"
{
	DLLEXPORT PerceptronMultiClasses *create_perceptron_multiclasses(int inputSize, int numClasses, float lr);
	DLLEXPORT void train_perceptron_multiclasses(PerceptronMultiClasses *model, float *X, int *Y, int rows, int cols, int epochs);
	DLLEXPORT int predict_perceptron_multiclasses(PerceptronMultiClasses *model, float *input);
	DLLEXPORT void destroy_perceptron_multiclasses(PerceptronMultiClasses *model);

	//getters pour sauvegarde JSON
	DLLEXPORT int get_input_size(PerceptronMultiClasses *model);
	DLLEXPORT int get_num_classes(PerceptronMultiClasses *model);
	DLLEXPORT void get_weights(PerceptronMultiClasses *model, float *outWeights);
	DLLEXPORT void get_bias(PerceptronMultiClasses *model, float *outBias);

	//setters pour chargement JSON
	DLLEXPORT void set_weights(PerceptronMultiClasses *model, float *inWeights);
	DLLEXPORT void set_bias(PerceptronMultiClasses *model, float *inBias);
}