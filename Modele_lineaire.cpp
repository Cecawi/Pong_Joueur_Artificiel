#include "Modele_lineaire.hpp"
#include <cmath>
#include <iostream>
#include <vector>

//constructeur
LinearModel::LinearModel(int inputSize, float lr)
    : weights(inputSize, 0.0f), bias(0.0f), learningRate(lr) {}

//entraînement du modèle
void LinearModel::train(const std::vector<std::vector<float>>& X,
                        const std::vector<float>& y,
                        int epochs)
{
    size_t n = X.size();//nombre d'échantillons
    size_t d = weights.size();//nombre de valeurs

    for(int epoch = 0 ; epoch < epochs ; ++epoch)
    {
        //boucle sur chaque échantillon
        for(size_t i = 0 ; i < n ; ++i)
        {
            float pred = predict(X[i]);//prédiction pour l'échantillon i
            float error = y[i] - pred;//l'écart d'erreur c'est la différence entre la "réponse" qu'on connait et celle prédite

            //mise à jour des poids
            for(size_t j = 0 ; j < weights.size() ; ++j)
            {
                weights[j] += learningRate * error * X[i][j];
            }

            //mise à jour du biais
            bias += learningRate * error;
        }
    }
}

//prédiction pour un échantillon x
float LinearModel::predict(const std::vector<float>& x)
{
    float sum = bias;//commence par le biais
    //somme pondérée des valeurs : w1*x1 + w2*x2 + ...
    for(size_t i = 0 ; i < weights.size() ; ++i)
    {
        sum += weights[i] * x[i];
    }
    return sum;
}

int main()
{
    //test
    std::vector<std::vector<float>> X = { {1}, {2}, {3}, {4} };
    std::vector<float> y = { 3, 5, 7, 9 };  // y = 2x + 1 (c'est le dataset)
    
    //création du modèle
    LinearModel model(1, 0.01f);

    //entraînement du modèle (1000 "fois")
    model.train(X, y, 1000);
    
    //affichage des poids et du biais
    std::cout << "Weights : " << model.getWeights()[0] << std::endl;
    std::cout << "Bias : " << model.getBias() << std::endl;

    //affichage des tests
    std::cout << "y = 2x + 1" << std::endl;
    size_t i = 0;
    for (auto& x : X)
    {
        std::cout << "x = " << x[0] << " -> predict = " << model.predict(x) << " (y = " << y[i] << ")" << std::endl;
        i++;
    }

    //test de prédiction
    std::vector<float> Xtest = {5.0f};
    std::cout << "TEST PERSO : pour x = 5" << " -> predict = " << model.predict(Xtest) << " (NORMALEMENT : y = 11)" << std::endl;
    
    return 0;
}