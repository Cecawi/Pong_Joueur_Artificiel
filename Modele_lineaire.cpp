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

//////////on va tester y = 2x + 1
    std::cout << std::endl;
    std::cout << "Test pour y = 2x + 1 : " << std::endl;
    
    //dataset
    std::vector<std::vector<float>> X = { {1}, {2}, {3}, {4} };
    std::vector<float> y = { 3, 5, 7, 9 };
    
    //création du modèle
    LinearModel model(1, 0.01f);

    //entraînement du modèle (1000 "fois")
    model.train(X, y, 1000);
    std::cout << std::endl;
    std::cout << "Entrainement avec : " << std::endl;
    std::cout << "X = { {1}, {2}, {3}, {4} }" << std::endl;
    std::cout << "y = { 3, 5, 7, 9 }" << std::endl;

    //affichage des poids et du biais
    std::cout << std::endl;
    std::cout << "Weights : " << model.getWeights()[0] << std::endl;
    std::cout << "Bias : " << model.getBias() << std::endl;

    //affichage
    std::cout << std::endl;
    size_t i = 0;
    for(auto& x : X)
    {
        std::cout << "x = " << x[0] << " -> prediction = " << model.predict(x) << " (y = " << y[i] << ")" << std::endl;
        i++;
    }

    std::cout << std::endl;
    std::cout << "OK" << std::endl;

//////////test de prédiction
    std::vector<float> Xtest = {5.0f};
    std::cout << std::endl;
    std::cout << "L'information que pour x = 5, y = 11 n'ayant pas ete donnee, on va tester la prediction : " << std::endl;
    std::cout << "x = 5" << " -> prediction = " << model.predict(Xtest) << " (y = 11)" << std::endl;

    std::cout << std::endl;
    std::cout << "OK" << std::endl;
    
//////////on va tester avec le cas de test pour la regression linéaire simple 2D vu en cours
    std::cout << std::endl;
    std::cout << "Lineaire simple 2D : " << std::endl;
    std::cout << "(Test pour y = x + 1)" << std::endl;

    std::vector<std::vector<float>> Xcours = { {1}, {2} };
    std::vector<float> Ycours = { 2, 3 };

    LinearModel modeleCasCours(1, 0.01f);

    modeleCasCours.train(Xcours, Ycours, 1000);
    std::cout << std::endl;
    std::cout << "Entrainement avec : " << std::endl;
    std::cout << "Xcours = { {1}, {2} }" << std::endl;
    std::cout << "Ycours = { 2, 3 }" << std::endl;

    std::cout << std::endl;
    std::cout << "Weights : " << model.getWeights()[0] << std::endl;
    std::cout << "Bias : " << model.getBias() << std::endl;

    std::cout << std::endl;
    std::cout << "x = 1.5 -> prediction = " << modeleCasCours.predict({1.5f}) << " (y ~= 2.5)" << std::endl;

    std::cout << std::endl;
    std::cout << "OK" << std::endl;

//////////on va tester avec le cas de test pour la regression non linéaire simple 2D vu en cours
    std::cout << std::endl;
    std::cout << "Non lineaire simple 2D : " << std::endl;
    std::cout << "(Test pour y ~= -0.25*(x - 2)^2 + 3)" << std::endl;
    std::cout << "(Le modele lineaire ne peut qu'apprendre une droite approximative ici)" << std::endl;
    std::cout << "(On pourra apprendre ce pattern non linéaire à un PMC (Perceptron Multicouche))" << std::endl;

    std::vector<std::vector<float>> XnonLineaire = { {1}, {2}, {3} };
    std::vector<float> YnonLineaire = { 2, 3, 2.5 };

    LinearModel modeleNonLinear(1, 0.01f);

    modeleNonLinear.train(XnonLineaire, YnonLineaire, 1000);
    std::cout << std::endl;
    std::cout << "Entrainement avec : " << std::endl;
    std::cout << "XnonLineaire = { {1}, {2}, {3} }" << std::endl;
    std::cout << "YnonLineaire = { 2, 3, 2.5 }" << std::endl;

    std::cout << std::endl;
    std::cout << "Weights : " << modeleNonLinear.getWeights()[0] << std::endl;
    std::cout << "Bias : " << modeleNonLinear.getBias() << std::endl;

    std::cout << std::endl;
    for(auto& x : XnonLineaire)
    {
        std::cout << "x = " << x[0] 
                  << " -> prediction = " << modeleNonLinear.predict(x)
                  << " (y = " << YnonLineaire[&x - &XnonLineaire[0]] << ")" << std::endl;
    }
    //le modèle linéaire fait toujours une droite, donc il ne pourra pas passer exactement par y = 2, 3, 2.5
    //on pourra apprendre ce pattern non linéaire à un PMC
    //la relation n’est pas une droite (YnonLineaire monte puis redescend), le modèle linéaire ne peut pas bien représenter la courbe, donc il "échoue" légèrement

    std::cout << std::endl;
    std::cout << "Moyen : le modele lineaire ne peut pas parfaitement ajuster les donnees, parce que la relation n est plus exactement une droite" << std::endl;
    std::cout << "(Il a trouve la meilleure approximation lineaire qui minimise l erreur)" << std::endl;
    std::cout << "(Un PMC passera mieux ce test)" << std::endl;

//////////
    std::cout << std::endl;
    std::cout << "Lineaire simple 3D : " << std::endl;
    std::cout << "(Test pour y = w1 * x1 + w2 * x2 + b)" << std::endl;

    std::vector<std::vector<float>> X3D = { {1, 1}, {2, 2}, {3, 1} };
    std::vector<float> Y3D = { 2, 3, 2.5 };

    LinearModel modele3D(2, 0.01f);

    modele3D.train(X3D, Y3D, 1000);
    std::cout << std::endl;
    std::cout << "Entrainement avec : " << std::endl;
    std::cout << "X3D = { {1, 1}, {2, 2}, {3, 1} }" << std::endl;
    std::cout << "Y3D = { 2, 3, 2.5 }" << std::endl;

    std::cout << std::endl;
    std::cout << "Weights : " << modele3D.getWeights()[0] << ", "
              << modele3D.getWeights()[1] << std::endl;
    std::cout << "Bias : " << modele3D.getBias() << std::endl;

    std::cout << std::endl;
    for(size_t i = 0 ; i < X3D.size() ; ++i)
    {
        float pred = modele3D.predict(X3D[i]);
        std::cout << "x = (" << X3D[i][0] << ", " << X3D[i][1]
                  << ") -> prediction = " << pred
                  << " (y = " << Y3D[i] << ")" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "OK" << std::endl;

//////////
    std::cout << std::endl;
    std::cout << "Lineaire tricky 3D : " << std::endl;
    std::cout << "(Test pour y = a * x1 + b * x2 + c)" << std::endl;
    std::cout << "(Ce cas est 'tricky' car x1 et x2 evoluent ensemble : le modele doit repartir correctement le poids entre eux (w1 ~ w2 ~ 0.5))" << std::endl;
    std::cout << std::endl;

    std::vector<std::vector<float>> Xtricky = { {1, 1}, {2, 2}, {3, 3} };
    std::vector<float> Ytricky = { 1, 2, 3 };

    LinearModel modeleTricky(2, 0.01f);

    modeleTricky.train(Xtricky, Ytricky, 1000);
    std::cout << std::endl;
    std::cout << "Entrainement avec : " << std::endl;
    std::cout << "Xtricky = { {1, 1}, {2, 2}, {3, 3} }" << std::endl;
    std::cout << "Ytricky = { 1, 2, 3 }" << std::endl;

    std::cout << std::endl;
    std::cout << "Weights : " << modeleTricky.getWeights()[0] << ", "
              << modeleTricky.getWeights()[1] << " : OK (w1 ~ w2 ~ 0.5)" << std::endl;
    std::cout << "Bias : " << modeleTricky.getBias() << std::endl;

    std::cout << std::endl;
    for(size_t i = 0 ; i < Xtricky.size() ; ++i)
    {
        float pred = modeleTricky.predict(Xtricky[i]);
        std::cout << "x = (" << Xtricky[i][0] << ", " << Xtricky[i][1]
                  << ") -> prediction = " << pred
                  << " (y = " << Ytricky[i] << ")" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "OK" << std::endl;

//////////
    std::cout << std::endl;
    std::cout << "Non lineaire simple 3D : " << std::endl;
    std::cout << "(Test pour une relation non lineaire ou l interaction entre x1 et x2 change le signe de la sortie)" << std::endl;
    std::cout << "(Aucune combinaison lineaire y = w1 * x1 + w2 * x2 + b)" << std::endl;
    std::cout << "(Probleme XOR-like : sortie ne peut pas etre separee par un plan lineaire dans l espace (x1, x2))" << std::endl;
    std::cout << "(Ce type de probleme ne peut pas etre resolu par un modele lineaire, il faut un PMC)" << std::endl;

    std::vector<std::vector<float>> XnonLinear3D = { {1, 0}, {0, 1}, {1, 1}, {0, 0} };
    std::vector<float> YnonLinear3D = { 2, 1, -2, -1 };

    LinearModel modeleNonLinear3D(2, 0.01f);

    modeleNonLinear3D.train(XnonLinear3D, YnonLinear3D, 1000);
    std::cout << std::endl;
    std::cout << "Entrainement avec : " << std::endl;
    std::cout << "XnonLinear3D = { {1, 0}, {0, 1}, {1, 1}, {0, 0} }" << std::endl;
    std::cout << "YnonLinear3D = { 2, 1, -2, -1 }" << std::endl;

    std::cout << std::endl;
    std::cout << "Weights : " << modeleNonLinear3D.getWeights()[0] << ", "
              << modeleNonLinear3D.getWeights()[1] << std::endl;
    std::cout << "Bias : " << modeleNonLinear3D.getBias() << std::endl;

    std::cout << std::endl;
    for(size_t i = 0; i < XnonLinear3D.size(); ++i)
    {
        float pred = modeleNonLinear3D.predict(XnonLinear3D[i]);
        std::cout << "x = (" << XnonLinear3D[i][0] << ", " << XnonLinear3D[i][1]
                  << ") -> prediction = " << pred
                  << " (y = " << YnonLinear3D[i] << ")" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "KO : le modele lineaire ne parvient pas a reproduire la relation non lineaire" << std::endl;
    std::cout << "(Un PMC avec une couche cachee pourra passer ce test)" << std::endl;
    std::cout << std::endl;

    return 0;
}