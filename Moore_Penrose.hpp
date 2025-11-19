#pragma once
#include <Eigen/Dense>
#include <vector>

//classe Moore-Penrose pour la régression linéaire par pseudo-inverse
class MoorePenrose
{
    private:
        Eigen::VectorXf weights;//vecteur des poids (inclut le biais)

    public:

/////CONSTRUCTEUR...

        //entraînement du modèle
        void train(const Eigen::MatrixXf& X, const Eigen::VectorXf& y);

        //prédiction pour un vecteur x
        float predict(const Eigen::VectorXf& x) const;

        //getter pour les poids
        Eigen::VectorXf getWeights() const { return weights; }
};