#pragma once
#include <Eigen/Dense>
#include <vector>

//classe Moore-Penrose pour la régression linéaire par pseudo-inverse
class MoorePenrose
{
    private : 
        Eigen::VectorXf weights;//vecteur des poids (inclut le biais en dernière position)

    public : 
        //entraîne le modèle sur les données fournies
        //X : matrice des entrées (doit inclure une colonne de biais = 1.0)
        //y : vecteur des sorties cibles
        void train(const Eigen::MatrixXf& X, const Eigen::VectorXf& y);

        //fait une prédiction pour un vecteur d'entrée
        //x : vecteur d'entrée (doit inclure le biais = 1.0 en dernière position)
        //retourne la valeur prédite
        float predict(const Eigen::VectorXf& x) const;
        
        //détecte s'il y a colinéarité et ajoute du bruit si besoin
        bool needUntrick(Eigen::MatrixXf& Xmat, int rows, int cols)

        //récupère les poids appris (biais inclus en dernière position)
        Eigen::VectorXf getWeights() const { return weights; }
};