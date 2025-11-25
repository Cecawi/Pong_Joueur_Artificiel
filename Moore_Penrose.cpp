#include "Moore_Penrose.hpp"
#include <iostream>

//g++ -I "C:/Users/yecel/Downloads/eigen-5.0.0/eigen-5.0.0" -shared -o Moore_Penrose.dll Moore_Penrose.cpp

extern "C"
{
    __declspec(dllexport)
    void trainMoorePenrose(float* X, float* Y, int rows, int cols, float* outWeights, float* outBias)
    {
        //j'ai oublié le biais (colonne de 1)
        //du coup je l'ajoute
        //et je remplis X et Y sans refaire un for

        //conversion du tableau 1D C# en matrice Eigen
        Eigen::MatrixXf Xmat(rows, cols + 1);//+ 1 : colonne supplémentaire pour le biais
        Eigen::VectorXf Yvec(rows);

        for(int i = 0 ; i < rows ; ++i)
        {
            for(int j = 0 ; j < cols ; ++j)
            {
                Xmat(i, j) = X[i * cols + j];
            }
            Xmat(i, cols) = 1.0f; //colonne de biais
            Yvec(i) = Y[i];
        }

		bool needUpdate = MoorePenrose::needUntrick(Xmat, rows, cols);

		if(needUpdate)
		{
			for(int i = 0 ; i < rows ; ++i)
			{
    			for(int j = 0 ; j < cols ; ++j)
    			{
        			X[i * cols + j] = Xmat(i, j);
    			}
			}
		}

        //calcul de la pseudo-inverse : W = (X^T X)^-1 X^T y
        Eigen::VectorXf W = (Xmat.transpose() * Xmat).inverse() * Xmat.transpose() * Yvec;

        //copie des poids dans le buffer de sortie pour Unity
        //copie les valeurs calculées par Eigen(W) dans un tableau simple (outWeights) que Unity envoie en paramètre
        for(int i = 0 ; i < cols ; ++i)
        {
            outWeights[i] = W(i);
        }

        *outBias = W(cols);
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
    return weights.dot(x);//produit scalaire entre vecteur des poids (biais inclut) et vecteur x
}


//détecte s'il y a colinéarité et ajoute du bruit si besoin
bool MoorePenrose::needUntrick(Eigen::MatrixXf& Xmat, int rows, int cols)
{
    if(rows < 3)
	{
		return false;
	}
    
    //vérifier si les points sont colinéaires
    Eigen::MatrixXf XtX = Xmat.block(0, 0, rows, cols).transpose() * Xmat.block(0, 0, rows, cols);
    float det = XtX.determinant();
    
    if(-1e-6f < det && det < 1e-6f)//seuil de singularité
    {
        std::cout << "Points colinéaires détectés (det=" << det << "). Ajout de bruit" << std::endl;
        Xmat(rows - 1, 0) += 0.01f;
		return true;
    }

	return false;
}