#pragma once
#include <vector>

//classe représentant un modèle linéaire simple (y = w·x + b)
class LinearModel
{
    private:
        std::vector<float> weights;//vecteur des poids
        float bias;//biais
        float learningRate;//taux d'apprentissage

    public:

        //construsteur
        LinearModel(int inputSize, float lr = 0.01f);

        //entraînement
        void train(const std::vector<std::vector<float>>& X,//vecteur de vecteurs (valeurs d'un échantillon)
                   const std::vector<float>& y,//valeurs cibles ("résultats"/"réponses" qu'on connait et qu'on veut que le modèle "trouve" approximativement)
                   int epochs);//nombre de fois où on va entraîner le modèle
        
        //prédiction pour un échantillon x
        float predict(const std::vector<float>& x);

        //getters
        std::vector<float> getWeights()
        {
            return weights;
        }

        float getBias()
        {
            return bias;
        }
};