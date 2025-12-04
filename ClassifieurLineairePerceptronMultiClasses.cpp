#include "ClassifieurLineairePerceptronMultiClasses.hpp"
#include <iostream>
#include <algorithm>
#include <random>

//g++ -shared -o ClassifieurLineairePerceptronMultiClasses.dll ClassifieurLineairePerceptronMultiClasses.cpp

PerceptronMultiClasses::PerceptronMultiClasses(int inputSize, int numClasses, float lr)
    : inputSize(inputSize), numClasses(numClasses), learningRate(lr)
{
    weights.resize(numClasses);
    bias.resize(numClasses);

    //initialisation aléatoire (-1 à 1)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for(int c = 0 ; c < numClasses ; ++c)
    {
        weights[c].resize(inputSize);
        for(int i = 0 ; i < inputSize ; ++i)
        {
            weights[c][i] = dis(gen);
        }
        bias[c] = dis(gen);//le biais est traité comme un poids connecté à une entrée fixe de 1
    }
}

int PerceptronMultiClasses::activate(float sum)
{
    //règle de Rosenblatt pour sorties 0 ou 1 (Heaviside)
    return (sum >= 0) ? 1 : 0;
}

int PerceptronMultiClasses::predictClass(const std::vector<float>& inputs, int classIndex)
{
    //calcule la somme pondérée pour la classe donnée
    float sum = bias[classIndex];
    for(int i = 0 ; i < inputSize ; ++i)
    {
        sum += weights[classIndex][i] * inputs[i];
    }
    return activate(sum);
}

void PerceptronMultiClasses::train(const std::vector<std::vector<float>>& X, const std::vector<int>& Y, int epochs)
{
    for(int epoch = 0 ; epoch < epochs ; ++epoch)
    {
        for(size_t k = 0 ; k < X.size() ; ++k)
        {
            //pour chaque exemple k
            const std::vector<float>& Xk = X[k];
            int targetClass = Y[k];

            //on entraîne chaque perceptron (Approche One-vs-All)
            for(int c = 0 ; c < numClasses ; ++c)
            {
                //Y_k pour ce perceptron : 1 si c'est la bonne classe, 0 sinon
                int Y_kc = (c == targetClass) ? 1 : 0;
                
                //g(X_k) pour ce perceptron
                int g_Xk = predictClass(Xk, c);

                //erreur (Y_k - g(X_k))
                //si Y_kc = 1 et g_Xk = 0 : erreur = 1
                //si Y_kc = 0 et g_Xk = 1 : erreur = -1
                //si Y_kc = g_Xk : erreur = 0
                int error = Y_kc - g_Xk;

                if(error != 0)
                {
                    //mise à jour W <- W + alpha * error * Xk
                    for(int i = 0 ; i < inputSize ; ++i)
                    {
                        weights[c][i] += learningRate * error * Xk[i];
                    }
                    //mise à jour biais (x0 = 1)
                    bias[c] += learningRate * error;
                }
            }
        }
    }
}

int PerceptronMultiClasses::predict(const std::vector<float>& inputs)
{
    //parcourt toutes les classes et trouve la classe avec le score le plus élevé
    float maxScore = -1e9f;
    int bestClass = -1;

    for(int c = 0 ; c < numClasses ; ++c)
    {
        float sum = bias[c];
        for(int i = 0 ; i < inputSize ; ++i)
        {
            sum += weights[c][i] * inputs[i];
        }
        
        if(sum > maxScore)
        {
            maxScore = sum;
            bestClass = c;
        }
    }
    return bestClass;
}

extern "C"
{
    PerceptronMultiClasses* create_perceptron_multiclasses(int inputSize, int numClasses, float lr)
    {
        return new PerceptronMultiClasses(inputSize, numClasses, lr);
    }

    void train_perceptron_multiclasses(PerceptronMultiClasses* model, float* X, int* Y, int rows, int cols, int epochs)
    {
        if(!model)
        {
            return;
        }
        
        std::vector<std::vector<float>> Xvec(rows, std::vector<float>(cols));
        std::vector<int> Yvec(rows);

        for(int i = 0 ; i < rows ; ++i)
        {
            Yvec[i] = Y[i];
            for(int j = 0 ; j < cols ; ++j)
            {
                Xvec[i][j] = X[i * cols + j];
            }
        }

        model->train(Xvec, Yvec, epochs);
    }

    int predict_perceptron_multiclasses(PerceptronMultiClasses* model, float* input)
    {
        if(!model)
        {
            return -1;
        }
        
        std::vector<float> inputVec(model->getInputSize());
        for(int i = 0 ; i < model->getInputSize() ; ++i)
        {
            inputVec[i] = input[i];
        }
        
        return model->predict(inputVec);
    }

    void destroy_perceptron_multiclasses(PerceptronMultiClasses* model)
    {
        if(model)
        {
            delete model;
        }
    }
}
