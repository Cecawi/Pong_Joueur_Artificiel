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
    //on va tester y = 2x + 1 (c'est le dataset)
    std::cout << std::endl;
    std::cout << "Test pour y = 2x + 1 : " << std::endl;

    std::vector<std::vector<float>> X = { {1}, {2}, {3}, {4} };
    std::vector<float> y = { 3, 5, 7, 9 };
    
    //création du modèle
    LinearModel model(1, 0.01f);

    //entraînement du modèle (1000 "fois")
    model.train(X, y, 1000);

    //affichage des poids et du biais
    std::cout << std::endl;
    std::cout << "Weights : " << model.getWeights()[0] << std::endl;
    std::cout << "Bias : " << model.getBias() << std::endl;

    //affichage
    std::cout << std::endl;
    size_t i = 0;
    for (auto& x : X)
    {
        std::cout << "x = " << x[0] << " -> prediction = " << model.predict(x) << " (y = " << y[i] << ")" << std::endl;
        i++;
    }

    //test de prédiction
    std::vector<float> Xtest = {5.0f};
    std::cout << std::endl;
    std::cout << "L'information que pour x = 5, y = 11 n'ayant pas ete donnee, on va tester la prediction : " << std::endl;
    std::cout << "x = 5" << " -> prediction = " << model.predict(Xtest) << " (y = 11)" << std::endl;
    
    //on va tester avec le cas de test pour la regression linéaire simple 2D vu en cours
    std::cout << std::endl;
    std::cout << "Lineaire simple 2D : " << std::endl;

    std::vector<std::vector<float>> Xcours = {{1}, {2}};
    std::vector<float> Ycours = {2, 3};

    LinearModel modeleCasCours(1, 0.01f);

    modeleCasCours.train(Xcours, Ycours, 1000);

    std::cout << std::endl;
    std::cout << "Weights : " << model.getWeights()[0] << std::endl;
    std::cout << "Bias : " << model.getBias() << std::endl;

    std::cout << std::endl;
    std::cout << "x = 1.5 -> prediction = " << modeleCasCours.predict({1.5f}) << " (y ~= 2.5)" << std::endl;
    std::cout << std::endl;

    //on va tester avec le cas de test pour la regression non linéaire simple 2D vu en cours
    std::vector<std::vector<float>> XnonLineaire = {{1}, {2}, {3}};
    std::vector<float> YnonLineaire = {2, 3, 2.5};

    LinearModel modeleNonLinear(1, 0.01f);

    modeleNonLinear.train(XnonLineaire, YnonLineaire, 1000);

    std::cout << std::endl;
    std::cout << "Non lineaire simple 2D : " << std::endl;

    std::cout << std::endl;
    std::cout << "Weights : " << modeleNonLinear.getWeights()[0] << std::endl;
    std::cout << "Bias : " << modeleNonLinear.getBias() << std::endl;

    std::cout << std::endl;
    for (auto& x : XnonLineaire)
    {
        std::cout << "x = " << x[0] 
                  << " -> prediction = " << modeleNonLinear.predict(x)
                  << " (y = " << YnonLineaire[&x - &XnonLineaire[0]] << ")" << std::endl;
    }
    std::cout << std::endl;
    //le modèle linéaire fait toujours une droite, donc il ne pourra pas passer exactement par y = 2, 3, 2.5
    //on pourra apprendre ce pattern non linéaire à un PMC

    return 0;
}