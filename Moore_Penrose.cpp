#include "Moore_Penrose.hpp"
#include <iostream>

extern "C"
{
    __declspec(dllexport)
    void trainMoorePenrose(float* X_flat, float* y, int rows, int cols, float* outWeights)
    {
        //conversion du tableau 1D C# en matrice Eigen
        Eigen::MatrixXf X(rows, cols);
        for(int i = 0 ; i < rows ; ++i)
            {
                for(int j = 0 ; j < cols ; ++j)
                {
                    X(i, j) = X_flat[i * cols + j];
                }
            }

        Eigen::VectorXf yVec(rows);
        for(int i = 0 ; i < rows ; ++i)
            {
                yVec(i) = y[i];
            }

        //calcul de la pseudo-inverse : W = (X^T X)^-1 X^T y
        Eigen::VectorXf W = (X.transpose() * X).inverse() * X.transpose() * yVec;

        //copie des poids dans le buffer de sortie pour Unity
        //copie les valeurs calculées par Eigen(W) dans un tableau simple (outWeights) que Unity envoie en paramètre
        for(int i = 0 ; i < cols ; ++i)
        {
            outWeights[i] = W(i);
        }
    }

    __declspec(dllexport)
    float predictMoorePenrose(float* weights, float* x, int size)
    {
        Eigen::VectorXf W(size);
        Eigen::VectorXf X(size);
        for(int i = 0 ; i < size ; ++i)
        {
            W(i) = weights[i];
            X(i) = x[i];
        }
        return W.dot(X);
    }
}

void MoorePenrose::train(const Eigen::MatrixXf& X, const Eigen::VectorXf& y)
{
    weights = (X.transpose() * X).inverse() * X.transpose() * y;
}

float MoorePenrose::predict(const Eigen::VectorXf& x) const
{
    return weights.dot(x);
}

#ifdef _DEBUG//permet de tester Moore_Penrose.cpp directement avec g++ pour vérifier les résultats
             //ne sera pas inclus dans la version DLL envoyée à Unity
int main()
{
    std::cout << "Test pour y = 2x + 1 : " << std::endl;

//////////y = 2x + 1
    Eigen::MatrixXf X(3, 2);
    X << 1, 1,
         2, 1,
         3, 1;

    Eigen::VectorXf y(3);
    y << 3, 5, 7;

    MoorePenrose model;
    model.train(X, y);

    std::cout << "Poids : " << model.getWeights().transpose() << std::endl;

    Eigen::VectorXf x_test(2);
    x_test << 4, 1;

    std::cout << "Prédiction pour x = 4 : y = " << model.predict(x_test) << std::endl;

    return 0;
}
#endif