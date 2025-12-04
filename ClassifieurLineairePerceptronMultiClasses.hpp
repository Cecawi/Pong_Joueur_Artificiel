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
        //poids pour chaque classe : weights[classe][entrée]
        std::vector<std::vector<float>> weights;
        //biais pour chaque classe : bias[classe]
        std::vector<float> bias;

    public : 
        PerceptronMultiClasses(int inputSize, int numClasses, float lr = 0.1f);

        //fonction d'activation (Heaviside : 1 si >= 0, 0 sinon)
        int activate(float sum);
        
        //prédiction pour une classe donnée (retourne 0 ou 1)
        int predictClass(const std::vector<float>& inputs, int classIndex);

        //entraînement (règle de Rosenblatt)
        void train(const std::vector<std::vector<float>>& X, const std::vector<int>& Y, int epochs);

        //prédiction finale (retourne l'indice de la classe avec le score le plus élevé)
        int predict(const std::vector<float>& inputs);

        int getInputSize() const
        {
            return inputSize;
        }
};

extern "C"
{
    DLLEXPORT PerceptronMultiClasses* create_perceptron_multiclasses(int inputSize, int numClasses, float lr);
    DLLEXPORT void train_perceptron_multiclasses(PerceptronMultiClasses* model, float* X, int* Y, int rows, int cols, int epochs);
    DLLEXPORT int predict_perceptron_multiclasses(PerceptronMultiClasses* model, float* input);
    DLLEXPORT void destroy_perceptron_multiclasses(PerceptronMultiClasses* model);
}