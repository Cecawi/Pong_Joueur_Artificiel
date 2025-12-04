using System;
using System.Runtime.InteropServices;
using UnityEngine;
using System.IO;
using System.Collections.Generic;
using System.Globalization;

//vérifier couches cachées XOR : 3 -> 2 car on a augmenté le nombre d'epoch (ou sinon, faudra l'augmenter)

//TODO (ok) : Multi Cross
//TODO (ok) : dans moore penrose : si plusieurs réponses possibles : (perturber un des points) ajouter bruit... MAIS DANS MoorePenrose.cpp... DANS LA LIB
//TODO : courbes d'apprentissage (erreurs, over/underfitting...)
//TODO : dataset (+ joueur enemie en entrée pas (en sortie?))
//TODO : plus de points a prédire et/ou autre chose que que des points + ajouter au cas XOR...
//pour le dataset, il faudra un assez grand pour avoir une partie pour entrainer et une partie pour tester : 2 courbes pour comparer (over/underfitting...)
//ajouter si lineaire ou non (ne pas tracer droite si courbe...)
//nbrAPred enlever regession3D?... (pas utilisé)

//RBF : 
//on projette nos données dans l'espace gaussiennes
//sauf que beaucoup... (je saurais pas trop expliquer pourquoi)
//donc on va  élire des ‘représentants’ avec : 
//algo lloyd : maj centroides, maj clusters,maj centroides, maj clusters...

//SVM : support vector machine : c'est pas noté (bonus (il en parle "pour clore le sujet") mais je crois qu'on peux essayer si on a le tps :'))
//en francais : machine support vecteur ou separateur vaste marge (haha)
//solveur quadratique : on peut utiliser OSQP
//alpha : vecteur : contribution à la marge
//tous alpha : 0 sauf ceux qui contribuent à la définition de la marge

public class PontPMC : MonoBehaviour
{
    [DllImport("PMC", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr create_pmc(int[] neurons_per_layer, int layers_count);

    [DllImport("PMC", CallingConvention = CallingConvention.Cdecl)]
    private static extern void destroy_pmc(IntPtr handle);

    [DllImport("PMC", CallingConvention = CallingConvention.Cdecl)]
    private static extern int train_pmc
    (
        IntPtr handle, double[] X_flat, double[] Y_flat, int samples, int input_size,
        int output_size, int is_classification, int num_iter, double alpha
    );

    [DllImport("PMC", CallingConvention = CallingConvention.Cdecl)]
    private static extern int predict_pmc
    (
        IntPtr handle, double[] input, int input_size,
        double[] out_buffer, int output_size, int is_classification
    );

    [DllImport("PMC", CallingConvention = CallingConvention.Cdecl)]
    private static extern int get_pmc_io_sizes(IntPtr handle, out int input_size, out int output_size);

    [DllImport("PMC", CallingConvention = CallingConvention.Cdecl)]
    private static extern double get_pmc_weights(IntPtr handle, int l, int i, int j);

    void Start()
    {
        float posX = 0f;
        float decalageX = 30f;
        float posY = 18f;
        float decalageY = -6f;

        Debug.Log("CLASSIFICATION");
        Debug.Log("LINEAIRE SIMPLE");

        double[] xLineaireSimple =
        {
            1, 1,
            2, 3,
            3, 3
        };

        double[] yLineaireSimple = { 1, -1, -1 };

        int nbrXAPredire = 20;
        float tailleAxesRepere = 5f;

        int[] npcLineaireSimple = new int[] { 2, 1 };
        int tailleNpcLineaireSimple = 2;

        TestsClassification
        (
            npcLineaireSimple, tailleNpcLineaireSimple, xLineaireSimple, yLineaireSimple,
            nbrXAPredire, false, 0f, 5f, 1f, 3f,
            ref posX, posY, tailleAxesRepere, decalageX, decalageY
        );

        Debug.Log("LINEAIRE MULTIPLE");

        double[] xLineaireMultiple = new double[200];
        double[] yLineaireMultiple = new double[100];

        System.Random rnd = new System.Random();

        for(int i = 0 ; i < 50 ; i++)
        {
            xLineaireMultiple[i * 2] = rnd.NextDouble() * 0.9 + 1.0;
            xLineaireMultiple[i * 2 + 1] = rnd.NextDouble() * 0.9 + 1.0;
            yLineaireMultiple[i] = 1.0;
        }

        for(int i = 0 ; i < 50 ; i++)
        {
            xLineaireMultiple[(50 + i) * 2] = rnd.NextDouble() * 0.9 + 2.0;
            xLineaireMultiple[(50 + i) * 2 + 1] = rnd.NextDouble() * 0.9 + 2.0;
            yLineaireMultiple[50 + i] = -1.0;
        }

        nbrXAPredire = 50;
        int[] npcLineaireMultiple = new int[] { 2, 1 };
        int tailleNpcLineaireMultiple = 2;

        TestsClassification
        (
            npcLineaireMultiple, tailleNpcLineaireMultiple, xLineaireMultiple, yLineaireMultiple,
            nbrXAPredire, false, 0f, 5f, 1f, 3f,
            ref posX, posY, tailleAxesRepere, decalageX, decalageY
        );

        Debug.Log("XOR (On peut modifier pour tester)");

        double[] xXor =
        {
            1, 0,
            0, 1,
            0, 0,
            1, 1
        };

        double[] yXor = { 1, 1, -1, -1 };

        nbrXAPredire = 0;
        tailleAxesRepere = 2f;

        int[] npcXor = new int[] { 2, 3, 1 };
        int tailleNpcXor = 3;

        TestsClassification
        (
            npcXor, tailleNpcXor, xXor, yXor,
            nbrXAPredire, false, 0f, 5f, 1f, 3f,
            ref posX, posY, tailleAxesRepere, decalageX, decalageY
        );

        Debug.Log("CROIX");

        double[] xCroix = new double[1000];
        double[] yCroix = new double[500];

        for(int i = 0 ; i < 500 ; i++)
        {
            double px = rnd.NextDouble() * 2.0 - 1.0;
            double py = rnd.NextDouble() * 2.0 - 1.0;

            xCroix[2 * i] = px;
            xCroix[2 * i + 1] = py;

            yCroix[i] = (Math.Abs(px) <= 0.3 || Math.Abs(py) <= 0.3) ? 1.0 : -1.0;
        }

        nbrXAPredire = 0;
        tailleAxesRepere = 2f;

        int[] npcCroix = new int[] { 2, 6, 1 };
        int tailleNpcCroix = 3;

        TestsClassification
        (
            npcCroix, tailleNpcCroix, xCroix, yCroix,
            nbrXAPredire, false, -1f, 1f, -1f, 1f,
            ref posX, posY, tailleAxesRepere, decalageX, decalageY
        );

        Debug.Log("MULTI LINEAIRE 3 CLASSES");

        double[] xMulti3 = new double[1000];
        double[] yMulti3 = new double[1500];

        int compteur = 0;

        for(int i = 0 ; i < 500 ; i++)
        {
            double px = rnd.NextDouble() * 2.0 - 1.0;
            double py = rnd.NextDouble() * 2.0 - 1.0;

            bool c1 = (-px - py - 0.5 > 0) && (py < 0) && (px - py - 0.5 < 0);
            bool c2 = (-px - py - 0.5 < 0) && (py > 0) && (px - py - 0.5 < 0);
            bool c3 = (-px - py - 0.5 < 0) && (py < 0) && (px - py - 0.5 > 0);

            double[] ytmp;

            if(c1)
            {
                ytmp = new double[] { 1, -1, -1 };
            }
            else if(c2)
            {
                ytmp = new double[] { -1, 1, -1 };
            }
            else if(c3)
            {
                ytmp = new double[] { -1, -1, 1 };
            }
            else
            {
                continue;
            }

            xMulti3[2 * compteur] = px;
            xMulti3[2 * compteur + 1] = py;

            yMulti3[3 * compteur] = ytmp[0];
            yMulti3[3 * compteur + 1] = ytmp[1];
            yMulti3[3 * compteur + 2] = ytmp[2];

            compteur++;
        }

        double[] xFinal = new double[2 * compteur];
        double[] yFinal = new double[3 * compteur];
        Array.Copy(xMulti3, xFinal, 2 * compteur);
        Array.Copy(yMulti3, yFinal, 3 * compteur);

        nbrXAPredire = 0;
        tailleAxesRepere = 2f;

        int[] npcMulti3 = new int[] { 2, 3 };
        int tailleNpcMulti3 = 2;

        TestsClassification
        (
            npcMulti3, tailleNpcMulti3, xFinal, yFinal,
            nbrXAPredire, false, -1f, 1f, -1f, 1f,
            ref posX, posY, tailleAxesRepere, decalageX, decalageY
        );

        Debug.Log("MULTI CROSS");

        double[] xMultiCross = new double[2000];
        double[] yMultiCross = new double[3000];

        compteur = 0;

        for(int i = 0 ; i < 1000 ; i++)
        {
            double px = rnd.NextDouble() * 2.0 - 1.0;
            double py = rnd.NextDouble() * 2.0 - 1.0;

            double modX = Math.Abs(px % 0.5);
            double modY = Math.Abs(py % 0.5);

            bool c1 = modX <= 0.25 && modY > 0.25;//classe bleue
            bool c2 = modX > 0.25 && modY <= 0.25;//classe rouge
            bool c3 = !c1 && !c2;//classe verte

            double[] ytmp;

            if(c1)
            {
                ytmp = new double[] { 1, -1, -1 };
            }
            else if(c2)
            {
                ytmp = new double[] { -1, 1, -1 };
            }
            else
            {
                ytmp = new double[] { -1, -1, 1 };
            }

            xMultiCross[2 * compteur] = px;
            xMultiCross[2 * compteur + 1] = py;

            yMultiCross[3 * compteur] = ytmp[0];
            yMultiCross[3 * compteur + 1] = ytmp[1];
            yMultiCross[3 * compteur + 2] = ytmp[2];

            compteur++;
        }

        double[] xFinalCross = new double[2 * compteur];
        double[] yFinalCross = new double[3 * compteur];
        Array.Copy(xMultiCross, xFinalCross, 2 * compteur);
        Array.Copy(yMultiCross, yFinalCross, 3 * compteur);

        nbrXAPredire = 0;
        tailleAxesRepere = 2f;

        int[] npcMultiCross1 = new int[] { 2, 10, 10, 3 };
        int tailleNpcMultiCross = 4;

        TesteMultiCross
        (
            npcMultiCross1, tailleNpcMultiCross, xFinalCross, yFinalCross,
            nbrXAPredire, false, -1f, 1f, -1f, 1f,
            ref posX, posY, tailleAxesRepere, decalageX, decalageY
        );
        
        posY += decalageY;

        int[] npcMultiCross2 = new int[] { 2, 15, 15, 3 };

        TesteMultiCross
        (
            npcMultiCross2, tailleNpcMultiCross, xFinalCross, yFinalCross,
            nbrXAPredire, false, -1f, 1f, -1f, 1f,
            ref posX, posY, tailleAxesRepere, decalageX, decalageY
        );
        
        posY += decalageY;

        int[] npcMultiCross3 = new int[] { 2, 20, 20, 3 };

        TesteMultiCross
        (
            npcMultiCross3, tailleNpcMultiCross, xFinalCross, yFinalCross,
            nbrXAPredire, false, -1f, 1f, -1f, 1f,
            ref posX, posY, tailleAxesRepere, decalageX, decalageY
        );
        
        posX += decalageX;
        posY -= decalageY - decalageY;

        Debug.Log("REGRESSION");
        Debug.Log("LINEAIRE SIMPLE 2D");

        double[] xRegressionSimple = { 1, 2 };//X = [[1],[2]]
        double[] yRegressionSimple = { 2, 3 };//Y = [2,3]

        int[] npcRegressionSimple = new int[] { 1, 1 };//PMC : 1,1
        int tailleNpcRegressionSimple = 2;//2 couches

        nbrXAPredire = 20;
        tailleAxesRepere = 5f;

        //on utilise la même fonction de test classification/regression
        TestsRegression
        (
            npcRegressionSimple, tailleNpcRegressionSimple,
            xRegressionSimple, yRegressionSimple,
            nbrXAPredire, 0f, 5f, 0f, 5f,
            ref posX, posY, tailleAxesRepere, decalageX, decalageY,
            true
        );

        Debug.Log("REGRESSION NON LINEAIRE SIMPLE 2D");

        //X et Y
        double[] X = { 1, 2, 3 };
        double[] Y = { 2, 3, 2.5 };

        //TODO : retester : (75 et changer proprement le nombre d'itérations)
        //int[] npcNonLineaire = new int[] { 1, 75, 1 };
        int[] npcNonLineaire = new int[] { 1, 3, 1 };
        int tailleNpcNonLineaire = 3;

        nbrXAPredire = 50;//points aléatoires à prédire

        TestsRegression
        (
            npcNonLineaire, tailleNpcNonLineaire, X, Y,
            nbrXAPredire, 0f, 5f, 0f, 5f,
            ref posX, posY, tailleAxesRepere, decalageX, decalageY,
            false
        );

        Debug.Log("LINEAIRE SIMPLE 3D");

        //X (2 dimensions) aplati en tableau
        double[] xLineaireSimple3D = new double[]
        {
            1, 1,
            2, 2,
            3, 1
        };

        //Y (1 dimension)
        double[] yLineaireSimple3D = new double[]
        {
            2,
            3,
            2.5
        };

        //PMC : 2 entrées → 1 sortie
        int[] npcLineaire3D = new int[] { 2, 1 };
        int tailleNpcLineaire3D = 2;

        //nombre de points à prédire pour afficher la surface
        nbrXAPredire = 0;

        TestsRegression3D
        (
            npcLineaire3D, tailleNpcLineaire3D, xLineaireSimple3D, yLineaireSimple3D,
            nbrXAPredire, 0f, 5f, 0f, 5f, ref posX, posY, tailleAxesRepere, decalageX, decalageY
        );
        
        Debug.Log("LINEAIRE TRICKY 3D");

        //X (2 dimensions)
        double[] xLineaireTricky3D = new double[]
        {
            1, 1,
            2, 2,
            3, 3
        };

        //Y (1 dimension)
        double[] yLineaireTricky3D = new double[]
        {
            1,
            2,
            3
        };

        //PMC : 2 entrées → 1 sortie
        int[] npcLineaireTricky3D = new int[] { 2, 1 };
        int tailleNpcLineaireTricky3D = 2;

        //nombre de points à prédire
        nbrXAPredire = 80;

        float savePosY = posY;//mémorise la ligne de départ

        //on affiche 12 PMC différents pour montrer qu'il existe
        //une infinité de plans passant par ces 3 points collinéaires
        for(int affichage = 1 ; affichage <= 12 ; affichage++)
        {
            //création PMC
            IntPtr ptrPmc = create_pmc(npcLineaireTricky3D, tailleNpcLineaireTricky3D);

            try
            {
                //nombre d'itérations variable pour obtenir un plan différent
                int iter = 500 + (affichage * 400);

                train_pmc
                (
                    ptrPmc,
                    xLineaireTricky3D,
                    yLineaireTricky3D,
                    3,//3 points
                    2,//input = 2
                    1,//output = 1
                    0,//regression
                    iter,
                    0.01
                );

                //position d'affichage en grille 3x4
                float offsetX = ((affichage - 1) % 3) * 6f;
                float offsetY = -((affichage - 1) / 3) * 6f;

                //affichage - utiliser AfficherRegression3D au lieu de AfficherRegression
                AfficherRegression3D
                (
                    ptrPmc,
                    xLineaireTricky3D,
                    yLineaireTricky3D,
                    posX + offsetX,
                    savePosY + offsetY,
                    tailleAxesRepere,
                    0f,
                    5f,
                    0f,
                    5f
                );
            }
            finally
            {
                destroy_pmc(ptrPmc);
            }
        }

        posX += decalageX;

        Debug.Log("NON LINEAIRE SIMPLE 3D");

        //X (2 dimensions) aplati en tableau
        double[] xNonLineaire3D = new double[]
        {
            1, 0,
            0, 1,
            1, 1,
            0, 0
        };

        //Y (1 dimension)
        double[] yNonLineaire3D = new double[]
        {
            2,
            1,
            -2,
            -1
        };

        /*int[] npcNonLineaire3D = new int[] { 2, 2, 1 };
        int tailleNpcNonLineaire3D = 3;

        nbrXAPredire = 0;

        TestsRegression3D
        (
            npcNonLineaire3D, tailleNpcNonLineaire3D, xNonLineaire3D, yNonLineaire3D,
            nbrXAPredire, 0f, 1f, 0f, 1f, ref posX, posY, tailleAxesRepere, decalageX, decalageY
        );*/
        
        savePosY = posY;

        int[] nbNeurones = new int[] { 2, 3, 5, 10, 25, 50, 75, 100, 200 };

        for(int affichage = 0 ; affichage < 9 ; affichage++)
        {
            //PMC : 2 entrées → N cachés → 1 sortie
            int[] npcNonLineaire3D = new int[] { 2, nbNeurones[affichage], 1 };
            int tailleNpcNonLineaire3D = 3;

            IntPtr ptrPmc = create_pmc(npcNonLineaire3D, tailleNpcNonLineaire3D);

            try
            {
                train_pmc
                (
                    ptrPmc,
                    xNonLineaire3D,
                    yNonLineaire3D,
                    4,//4 points
                    2,//input = 2
                    1,//output = 1
                    0,//regression
                    10000,
                    0.01
                );

                //position d'affichage en grille 3x3
                float offsetX = (affichage % 3) * 6f;
                float offsetY = -(affichage / 3) * 6f;

                AfficherRegression3D
                (
                    ptrPmc,
                    xNonLineaire3D,
                    yNonLineaire3D,
                    posX + offsetX,
                    savePosY + offsetY,
                    tailleAxesRepere,
                    0f,
                    1f,
                    0f,
                    1f
                );
            }
            finally
            {
                destroy_pmc(ptrPmc);
            }
        }

        posX += decalageX;

        Debug.Log("PONG");
        RunPongTest(posX, decalageX);
    }

    void TestsClassification
    (
        int[] Npc, int TailleNpc, double[] DoneesX, double[] DoneesY,
        int NbrXAPred, bool BesoinDePred, float Xg, float Xd, float Yg, float Yd,
        ref float PosX, float PosY, float TailleAxesRepere, float DecalageX, float DecalageY
    )
    {
        if(NbrXAPred > 0)
        {
            BesoinDePred = true;
        }

        double[] X_flat = new double[DoneesX.Length];
        Array.Copy(DoneesX, X_flat, DoneesX.Length);

        int nbrPoints = DoneesX.Length / 2;
        int sortieParPoint = DoneesY.Length / nbrPoints;

        double[] Y_flat = new double[nbrPoints * sortieParPoint];
        for(int i = 0 ; i < nbrPoints ; i++)
        {
            for(int j = 0 ; j < sortieParPoint ; j++)
            {
                Y_flat[i * sortieParPoint + j] = DoneesY[i * sortieParPoint + j];
            }
        }

        Vector2[] donneesAleaAPredire1 = GenerePtsAlea(NbrXAPred, Xg, Xd, Yg, Yd);
        Vector2[] donneesAleaAPredire2 = GenerePtsAlea(NbrXAPred, Xg, Xd, Yg, Yd);

        for(int affichage = 1 ; affichage <= 12 ; affichage++)
        {
            IntPtr ptrPmc = create_pmc(Npc, TailleNpc);
            try
            {
                if(affichage == 1 || affichage == 7)
                {
                    Afficher
                    (
                        ptrPmc, X_flat, Y_flat, PosX + ((affichage == 7) ? 0 : 0), PosY, TailleAxesRepere,
                        (affichage <= 6) ? donneesAleaAPredire1 : donneesAleaAPredire2, BesoinDePred, sortieParPoint, affichage
                    );
                }
                else if(affichage == 2 || affichage == 8)
                {
                    train_pmc(ptrPmc, X_flat, Y_flat, nbrPoints, 2, sortieParPoint, 1, 1000, 0.01);
                    Afficher
                    (
                        ptrPmc, X_flat, Y_flat, PosX + 6f, PosY, TailleAxesRepere,
                        (affichage <= 6) ? donneesAleaAPredire1 : donneesAleaAPredire2, BesoinDePred, sortieParPoint, affichage
                    );
                }
                else
                {
                    int iterations = (affichage == 3 || affichage == 9) ? 100000 : 10000;
                    train_pmc(ptrPmc, X_flat, Y_flat, nbrPoints, 2, sortieParPoint, 1, iterations, 0.01);
                    float decX = (affichage % 3 == 2) ? 6f : (affichage % 3 == 0 ? 12f : 0f);
                    Afficher
                    (
                        ptrPmc, X_flat, Y_flat, PosX + decX, PosY, TailleAxesRepere,
                        (affichage <= 6) ? donneesAleaAPredire1 : donneesAleaAPredire2, BesoinDePred, sortieParPoint, affichage
                    );
                }
            }
            finally
            {
                destroy_pmc(ptrPmc);
            }

            if(affichage % 3 == 0)
            {
                PosY += DecalageY;
            }
        }

        PosX += DecalageX;
    }

    void TesteMultiCross
    (
        int[] Npc, int TailleNpc, double[] DoneesX, double[] DoneesY,
        int NbrXAPred, bool BesoinDePred, float Xg, float Xd, float Yg, float Yd,
        ref float PosX, float PosY, float TailleAxesRepere, float DecalageX, float DecalageY
    )
    {
        double[] X_flat = new double[DoneesX.Length];
        Array.Copy(DoneesX, X_flat, DoneesX.Length);

        int nbrPoints = DoneesX.Length / 2;
        int sortieParPoint = DoneesY.Length / nbrPoints;

        double[] Y_flat = new double[nbrPoints * sortieParPoint];
        for(int i = 0 ; i < nbrPoints ; i++)
        {
            for(int j = 0 ; j < sortieParPoint ; j++)
            {
                Y_flat[i * sortieParPoint + j] = DoneesY[i * sortieParPoint + j];
            }
        }

        Vector2[] donneesAleaAPredire1 = GenerePtsAlea(NbrXAPred, Xg, Xd, Yg, Yd);//pas besoin

        for(int affichage = 1 ; affichage <= 3 ; affichage++)
        {
            IntPtr ptrPmc = create_pmc(Npc, TailleNpc);
            try
            {
                if(affichage == 1)
                {
                    train_pmc(ptrPmc, X_flat, Y_flat, nbrPoints, 2, sortieParPoint, 1, 100000, 0.01);
                    Afficher
                    (
                        ptrPmc, X_flat, Y_flat, PosX, PosY, TailleAxesRepere,
                        donneesAleaAPredire1, BesoinDePred, sortieParPoint, affichage
                    );
                }
                else if(affichage == 2)
                {
                    train_pmc(ptrPmc, X_flat, Y_flat, nbrPoints, 2, sortieParPoint, 1, 1000000, 0.01);
                    Afficher
                    (
                        ptrPmc, X_flat, Y_flat, PosX + 6f, PosY, TailleAxesRepere,
                        donneesAleaAPredire1, BesoinDePred, sortieParPoint, affichage
                    );
                }
                else
                {
                    train_pmc(ptrPmc, X_flat, Y_flat, nbrPoints, 2, sortieParPoint, 1, 10000000, 0.01);
                    Afficher
                    (
                        ptrPmc, X_flat, Y_flat, PosX + 12f, PosY, TailleAxesRepere,
                        donneesAleaAPredire1, BesoinDePred, sortieParPoint, affichage
                    );
                }
            }
            finally
            {
                destroy_pmc(ptrPmc);
            }
        }
    }

    Vector2[] GenerePtsAlea(int NbrXAPred, float Xg, float Xd, float Yg, float Yd)
    {
        Vector2[] pts = new Vector2[NbrXAPred];
        for(int i = 0 ; i < NbrXAPred ; i++)
        {
            pts[i] = new Vector2(UnityEngine.Random.Range(Xg, Xd), UnityEngine.Random.Range(Yg, Yd));
        }
        return pts;
    }

    void Afficher
    (
        IntPtr PtrPmc, double[] DoneesX, double[] DoneesY,
        float PosX, float PosY, float TailleAxesRepere,
        Vector2[] donneesAleaAPredire, bool BesoinDePred, int sortieParPoint, int ID
    )
    {
        AfficherRepere(PosX, PosY, TailleAxesRepere, ID);
        int nbrPoints = DoneesX.Length / 2;

        for(int i = 0 ; i < nbrPoints ; i++)
        {
            double[] inp = { DoneesX[2 * i], DoneesX[2 * i + 1] };
            double[] outp = new double[sortieParPoint];

            if(predict_pmc(PtrPmc, inp, 2, outp, sortieParPoint, 1) != 0)
            {
                Debug.LogError("predict_pmc error");
            }

            int classe = 0;
            double maxVal = outp[0];
            for(int c = 1 ; c < sortieParPoint ; c++)
            {
                if(outp[c] > maxVal)
                {
                    maxVal = outp[c];
                    classe = c;
                }
            }

            Color col;
            if(sortieParPoint == 1)
            {
                col = outp[0] > 0 ? Color.blue : Color.red;
            }
            else if(sortieParPoint == 2)
            {
                col = classe == 0 ? Color.blue : Color.red;
            }
            else if(sortieParPoint == 3)
            {
                col = classe == 0 ? Color.blue : (classe == 1 ? Color.red : Color.green);
            }
            else
            {
                col = Color.HSVToRGB((float)classe / sortieParPoint, 0.8f, 0.8f);
            }

            var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sph.GetComponent<Renderer>().material.color = col;
            sph.transform.position = new Vector3((float)inp[0] + PosX, (float)inp[1] + PosY, 0);
            sph.transform.localScale = Vector3.one * 0.15f;
        }

        if(BesoinDePred)
        {
            foreach(var pt in donneesAleaAPredire)
            {
                double[] inp = { pt.x, pt.y };
                double[] outp = new double[sortieParPoint];
                if(predict_pmc(PtrPmc, inp, 2, outp, sortieParPoint, 1) != 0)
                {
                    Debug.LogError("predict_pmc error");
                }

                int classe = 0;
                double maxVal = outp[0];
                for(int c = 1 ; c < sortieParPoint ; c++)
                {
                    if(outp[c] > maxVal)
                    {
                        maxVal = outp[c];
                        classe = c;
                    }
                }

                Color col;
                if(sortieParPoint == 1)
                {
                    col = outp[0] > 0 ? Color.green : Color.yellow;
                }
                else if(sortieParPoint == 2)
                {
                    col = classe == 0 ? Color.green : Color.yellow;
                }
                else if(sortieParPoint == 3)
                {
                    col = classe == 0 ? Color.green : (classe == 1 ? Color.yellow : Color.cyan);
                }
                else
                {
                    col = Color.HSVToRGB((float)classe / sortieParPoint, 0.8f, 0.8f);
                }

                var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sph.GetComponent<Renderer>().material.color = col;
                sph.transform.position = new Vector3(pt.x + PosX, pt.y + PosY, 0);
                sph.transform.localScale = Vector3.one * 0.1f;
            }

            TracerDoite(PtrPmc, PosX, PosY);
        }
    }

    void AfficherRepere(float PosX, float PosY, float Taille, int ID)
    {
        CreerAxe(PosX, PosY, Color.red, new Vector3(0 + PosX, PosY, 0), new Vector3(Taille + PosX, PosY, 0), $"Axe x {ID}");
        CreerAxe(PosX, PosY, Color.green, new Vector3(PosX, 0 + PosY, 0), new Vector3(PosX, Taille + PosY, 0), $"Axe y {ID}");
        CreerAxe(PosX, PosY, Color.blue, new Vector3(PosX, PosY, 0), new Vector3(PosX, PosY, Taille), $"Axe z {ID}");
    }

    void CreerAxe(float PosX, float PosY, Color col, Vector3 A, Vector3 B, string name)
    {
        GameObject axis = new GameObject(name);
        var lr = axis.AddComponent<LineRenderer>();
        lr.positionCount = 2;
        lr.startWidth = lr.endWidth = 0.05f;
        lr.material = new Material(Shader.Find("Sprites/Default"));
        lr.startColor = lr.endColor = col;
        lr.SetPosition(0, A);
        lr.SetPosition(1, B);
    }

    void TracerDoite(IntPtr PtrPmc, float PosX, float PosY)
    {
        float yA = TrouverY(PtrPmc, 0f);
        float yB = TrouverY(PtrPmc, 5f);

        GameObject Droite = new GameObject("Droite");
        var lr = Droite.AddComponent<LineRenderer>();
        lr.positionCount = 2;
        lr.startWidth = lr.endWidth = 0.06f;
        lr.material = new Material(Shader.Find("Sprites/Default"));
        lr.startColor = lr.endColor = Color.purple;
        lr.SetPosition(0, new Vector3(0f + PosX, yA + PosY, 0));
        lr.SetPosition(1, new Vector3(5f + PosX, yB + PosY, 0));
    }

    float TrouverY(IntPtr handle, float x)
    {
        double[] input = new double[2];
        double[] output = new double[1];

        for(float y = -10 ; y <= 10 ; y += 0.02f)
        {
            input[0] = x;
            input[1] = y;
            if(predict_pmc(handle, input, 2, output, 1, 1) != 0)
            {
                Debug.LogError("predict_pmc error");
            }
            if(Mathf.Abs((float)output[0]) < 0.03f)
            {
                return y;
            }
        }
        return -1;
    }

    void TestsRegression
    (
        int[] Npc, int TailleNpc, double[] DoneesX, double[] DoneesY,
        int NbrXAPred, float Xg, float Xd, float Yg, float Yd,
        ref float PosX, float PosY, float TailleAxesRepere, float DecalageX, float DecalageY,
        bool FautTracerDroite
    )
    {
        double[] X_flat = new double[DoneesX.Length];
        Array.Copy(DoneesX, X_flat, DoneesX.Length);

        double[] Y_flat = new double[DoneesY.Length];
        Array.Copy(DoneesY, Y_flat, DoneesY.Length);

        Vector2[] donneesAleaAPredire = GenerePtsAlea(NbrXAPred, Xg, Xd, Yg, Yd);

        //création PMC
        IntPtr ptrPmc = create_pmc(Npc, TailleNpc);
        try
        {
            //TODO : changer proprement le nombre d'itérations (1000000 : "homogene")
            //entraînement simple
            train_pmc(ptrPmc, X_flat, Y_flat, DoneesY.Length, 1, 1, 0, 1000000, 0.01);
            AfficherRegression(ptrPmc, X_flat, Y_flat, PosX, PosY, TailleAxesRepere, donneesAleaAPredire, Xg, Xd, FautTracerDroite);
        }
        finally
        {
            destroy_pmc(ptrPmc);
        }

        PosX += DecalageX;
    }

    void AfficherRegression
    (
        IntPtr PtrPmc, double[] DoneesX, double[] DoneesY,
        float PosX, float PosY, float TailleAxesRepere,
        Vector2[] donneesAleaAPredire, float Xg, float Xd,
        bool FautTracerDroite
    )
    {
        AfficherRepere(PosX, PosY, TailleAxesRepere, 0);

        //affichage des points d'entraînement en bleu
        for(int i = 0 ; i < DoneesY.Length ; i++)
        {
            var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sph.GetComponent<Renderer>().material.color = Color.blue;
            sph.transform.position = new Vector3((float)DoneesX[i] + PosX, (float)DoneesY[i] + PosY, 0);
            sph.transform.localScale = Vector3.one * 0.15f;
        }

        //affichage des points prédits en vert
        foreach(var pt in donneesAleaAPredire)
        {
            double[] inp = { pt.x };
            double[] outp = new double[1];
            predict_pmc(PtrPmc, inp, 1, outp, 1, 0);

            var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sph.GetComponent<Renderer>().material.color = Color.green;
            sph.transform.position = new Vector3(pt.x + PosX, (float)outp[0] + PosY, 0);
            sph.transform.localScale = Vector3.one * 0.1f;
        }

        if(FautTracerDroite)
        {
            //tracer droite approximative
            double[] outMin = new double[1];
            double[] outMax = new double[1];
            predict_pmc(PtrPmc, new double[] { Xg }, 1, outMin, 1, 0);
            predict_pmc(PtrPmc, new double[] { Xd }, 1, outMax, 1, 0);

            GameObject droite = new GameObject("DroiteRegression");
            var lr = droite.AddComponent<LineRenderer>();
            lr.positionCount = 2;
            lr.startWidth = lr.endWidth = 0.05f;
            lr.material = new Material(Shader.Find("Sprites/Default"));
            lr.startColor = lr.endColor = Color.red;
            lr.SetPosition(0, new Vector3(Xg + PosX, (float)outMin[0] + PosY, 0));
            lr.SetPosition(1, new Vector3(Xd + PosX, (float)outMax[0] + PosY, 0));
        }
    }

    void AfficherRegression3D
    (
        IntPtr PtrPmc,
        double[] DoneesX, double[] DoneesY,
        float PosX, float PosY, float TailleAxesRepere,
        float Xg, float Xd, float Yg, float Yd
    )
    {
        AfficherRepere(PosX, PosY, TailleAxesRepere, 0);

        int nbrPoints = DoneesY.Length;

        //points d'entraînement
        for(int i = 0 ; i < nbrPoints ; i++)
        {
            float x = (float)DoneesX[2 * i];
            float y = (float)DoneesX[2 * i + 1];
            float z = (float)DoneesY[i];

            var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sph.GetComponent<Renderer>().material.color = Color.blue;
            sph.transform.position = new Vector3(x + PosX, y + PosY, z);
            sph.transform.localScale = Vector3.one * 0.15f;
        }

        //surface prédite → grille 20 x 20
        int N = 20;
        for(int ix = 0 ; ix < N ; ix++)
        {
            for(int iy = 0 ; iy < N ; iy++)
            {
                float x = Mathf.Lerp(Xg, Xd, ix / (float)(N - 1));
                float y = Mathf.Lerp(Yg, Yd, iy / (float)(N - 1));

                double[] inp = { x, y };
                double[] outp = new double[1];
                predict_pmc(PtrPmc, inp, 2, outp, 1, 0);

                var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sph.GetComponent<Renderer>().material.color = Color.green;
                sph.transform.position = new Vector3(x + PosX, y + PosY, (float)outp[0]);
                sph.transform.localScale = Vector3.one * 0.05f;
            }
        }
    }

    void TestsRegression3D
    (
        int[] Npc, int TailleNpc,
        double[] DoneesX, double[] DoneesY,
        int NbrXAPred, float Xg, float Xd, float Yg, float Yd,
        ref float PosX, float PosY, float TailleAxesRepere,
        float DecalageX, float DecalageY
    )
    {
        int nbrPoints = DoneesY.Length;

        IntPtr ptrPmc = create_pmc(Npc, TailleNpc);

        try
        {
            train_pmc(ptrPmc, DoneesX, DoneesY, nbrPoints, 2, 1, 0, 2000, 0.01);

            AfficherRegression3D
            (
                ptrPmc, DoneesX, DoneesY,
                PosX, PosY, TailleAxesRepere,
                Xg, Xd, Yg, Yd
            );
        }
        finally
        {
            destroy_pmc(ptrPmc);
        }

        PosX += DecalageX;
    }

    public void RunPongTest(float PosX, float DecalageX)
    {
        string filePath = @"C:\Users\yecel\Desktop\ESGI - 4A\T1\Machine Learning\Pong_Joueur_Artificiel\pong_data_test_1.csv";
        
        if(!File.Exists(filePath))
        {
            Debug.LogError("Le dataset du pong n'a pas été trouvé à l'emplacement suivant : " + filePath);
            return;
        }

        List<double[]> inputsList = new List<double[]>();
        List<double> outputsList = new List<double>();

        string[] lines = File.ReadAllLines(filePath);
        
        foreach(string line in lines)
        {
            if(string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string[] parts = line.Split(',');
            if(parts.Length < 8)
            {
                continue;
            }

            //entrées/inputs : indices 0, 1, 2, 3, 4, 5, 7
            double[] input = new double[7];
            input[0] = double.Parse(parts[0], CultureInfo.InvariantCulture); //ball x
            input[1] = double.Parse(parts[1], CultureInfo.InvariantCulture); //ball y
            input[2] = double.Parse(parts[2], CultureInfo.InvariantCulture); //ball vitesse x
            input[3] = double.Parse(parts[3], CultureInfo.InvariantCulture); //ball vitesse y
            input[4] = double.Parse(parts[4], CultureInfo.InvariantCulture); //player y
            input[5] = double.Parse(parts[5], CultureInfo.InvariantCulture); //enemy y
            input[6] = double.Parse(parts[7], CultureInfo.InvariantCulture); //enemy move (8ème donnée d'indice 7)

            //sortie/output : indice 6
            double output = double.Parse(parts[6], CultureInfo.InvariantCulture);

            inputsList.Add(input);
            outputsList.Add(output);
        }

        int rows = inputsList.Count;
        int cols = 7;
        double[] Xflat = new double[rows * cols];
        double[] Y = outputsList.ToArray();

        for(int i = 0 ; i < rows ; i++)
        {
            for(int j = 0 ; j < cols ; j++)
            {
                Xflat[i * cols + j] = inputsList[i][j];
            }
        }

        int[] npcNonLineaire3D = new int[] { 7, 1 };
        int tailleNpcNonLineaire3D = 2;

        IntPtr ptrPmc = create_pmc(npcNonLineaire3D, tailleNpcNonLineaire3D);

        train_pmc(ptrPmc, Xflat, Y, rows, cols, 1, 1, 1000, 0.01);

        //visualisation
        float scale = 10f; 
        float offsetX = PosX; //position de base pour cette visualisation

        //visualisation 1 (ce que le joueur a fait)
        //vert : Haut (1)
        //orange : Pas bougé (0)
        //rouge : Bas (-1)
        for(int i = 0; i < rows; i++)
        {
            double[] input = inputsList[i];
            double actualOutput = outputsList[i];

            float xPos = (float)input[0] * scale + offsetX;
            float yPos = (float)input[1] * scale;
            
            var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.transform.position = new Vector3(xPos, yPos, 0);
            sphere.transform.localScale = Vector3.one * 0.2f;

            if(actualOutput > 0.5f)//1
            {
                sphere.GetComponent<Renderer>().material.color = Color.green;
            }
            else if(actualOutput < -0.5f)//-1
            {
                sphere.GetComponent<Renderer>().material.color = Color.red;
            }
            else//0
            {
                sphere.transform.localScale = Vector3.one * 0.1f;
                sphere.GetComponent<Renderer>().material.color = Color.orange;
            }
        }

        //visualisation 2 (ce qui est prédit)
        //bleu : Haut (1)
        //jaune : Pas bougé (0)
        //magenta : Bas (-1)
        float offsetX2 = offsetX + DecalageX;//décalage pour la deuxième visualisation

        for(int i = 0 ; i < rows ; i++)
        {
            double[] input = inputsList[i];
            
            //prédiction avec le PMC
            double[] output_buffer = new double[1];
            predict_pmc(ptrPmc, input, cols, output_buffer, 1, 1);
            double prediction = output_buffer[0];

            float xPos = (float)input[0] * scale + offsetX2;
            float yPos = (float)input[1] * scale;

            var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.transform.position = new Vector3(xPos, yPos, 0);
            sphere.transform.localScale = Vector3.one * 0.2f;

            if(prediction > 0.33f)//haut
            {
                sphere.GetComponent<Renderer>().material.color = Color.blue;
            }
            else if(prediction < -0.33f)//bas
            {
                sphere.GetComponent<Renderer>().material.color = Color.magenta;
            }
            else//pas bougé
            {
                sphere.transform.localScale = Vector3.one * 0.1f;
                sphere.GetComponent<Renderer>().material.color = Color.yellow;
            }
        }

        PosX += DecalageX;
    }
}