using System;
using System.Runtime.InteropServices;
using UnityEngine;
using System.Collections.Generic;

public class PontRBF_FP : MonoBehaviour
{
    
    [DllImport("RBF_FP", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr create_rbf(int[] sizes, int layers_count);

    [DllImport("RBF_FP", CallingConvention = CallingConvention.Cdecl)]
    private static extern void destroy_rbf(IntPtr handle);

    [DllImport("RBF_FP", CallingConvention = CallingConvention.Cdecl)]
    private static extern int train_rbf
    (
        IntPtr handle, double[] X_flat, double[] Y_flat, int samples, int input_size,
        int output_size, int is_classification, int num_iter, double alpha
    );

    [DllImport("RBF_FP", CallingConvention = CallingConvention.Cdecl)]
    private static extern int predict_rbf
    (
        IntPtr handle, double[] input, int input_size,
        double[] out_buffer, int output_size, int is_classification
    );
    
    void Start()
    {
        // Décalages pour l'affichage 
        float posX = 0f;
        float posY = 18f;
        float decalageX = 30f;
        float decalageY = -6f;
        float tailleAxesRepere = 5f;

        // --- Test RBF_FP XOR (Classification Non Linéaire) ---
        Debug.Log("RBF_FP XOR (Classification)");

        double[] xXor = { 1, 0, 0, 1, 0, 0, 1, 1 }; // (1,0), (0,1), (0,0), (1,1)
        double[] yXor = { 1, 1, -1, -1 };          // Classes : 1, 1, -1, -1

        // RBF_FP : 2 entrées, 3 centres RBF_FP cachés, 1 sortie
        int[] rbfXorSizes = new int[] { 2, 3, 1 };
        int tailleRbfXor = 3; 

        // RBF_FP est vachement bon pour le XOR
        TestsRBFClassification(
            rbfXorSizes, tailleRbfXor, xXor, yXor,
            200, 0f, 1f, 0f, 1f,
            ref posX, posY, tailleAxesRepere, decalageX, decalageY
        );

        posX += decalageX;
        posY = 18f; // Réinitialisation de Y pour le prochain bloc
        
        // --- Test RBF_FP Régression Non Linéaire (Sinusoïde simple) ---
        Debug.Log("RBF_FP REGRESSION NON LINEAIRE");

        int N_samples = 50;
        double[] xSin = new double[N_samples];
        double[] ySin = new double[N_samples];
        System.Random rnd = new System.Random();
        
        // Génération de points y = sin(x) sur [0, 5] avec bruit
        for (int i = 0; i < N_samples; i++)
        {
            xSin[i] = (double)i * 5.0 / (N_samples - 1);
            ySin[i] = Math.Sin(xSin[i]) + (rnd.NextDouble() * 0.2 - 0.1); // sin(x) avec petit bruit
        }

        // RBF_FP : 1 entrée, 10 centres RBF_FP cachés, 1 sortie
        int[] rbfSinSizes = new int[] { 1, 10, 1 };
        int tailleRbfSin = 3;

        TestsRBFRegression(
            rbfSinSizes, tailleRbfSin, xSin, ySin,
            50, 0f, 5f, -1.5f, 1.5f,
            ref posX, posY, tailleAxesRepere, decalageX, decalageY
        );
        
        posX += decalageX;
    }

    // --- Fonctions de Test pour Classification ---
    void TestsRBFClassification
    (
        int[] RbfSizes, int TailleRbf, double[] DoneesX, double[] DoneesY,
        int NbrXAPred, float Xg, float Xd, float Yg, float Yd,
        ref float PosX, float PosY, float TailleAxesRepere, float DecalageX, float DecalageY
    )
    {
        double[] X_flat = DoneesX;
        int nbrPoints = DoneesX.Length / 2; // Entrées 2D
        int sortieParPoint = DoneesY.Length / nbrPoints;
        double[] Y_flat = DoneesY;

        Vector2[] donneesAleaAPredire = GenerePtsAlea(NbrXAPred, Xg, Xd, Yg, Yd);

        // Test avec différentes itérations
        int[] iterationsList = { 1000, 10000, 100000 };
        for(int affichage = 0 ; affichage < iterationsList.Length ; affichage++)
        {
            IntPtr ptrRbf = create_rbf(RbfSizes, TailleRbf);
            try
            {
                int iter = iterationsList[affichage];
                
                // Entraînement RBF_FP - Classification (is_classification = 1)
                train_rbf(ptrRbf, X_flat, Y_flat, nbrPoints, RbfSizes[0], RbfSizes[RbfSizes.Length - 1], 1, iter, 0.01);

                float currentPosX = PosX + affichage * 6f;
                AfficherRBFClassification(
                    ptrRbf, X_flat, Y_flat, currentPosX, PosY, TailleAxesRepere,
                    donneesAleaAPredire, sortieParPoint, $"RBF_FP XOR - {iter} iter"
                );
            }
            finally
            {
                destroy_rbf(ptrRbf);
            }
        }
    }

    // --- Fonctions de Test pour Régression ---
    void TestsRBFRegression
    (
        int[] RbfSizes, int TailleRbf, double[] DoneesX, double[] DoneesY,
        int NbrXAPred, float Xg, float Xd, float Yg, float Yd,
        ref float PosX, float PosY, float TailleAxesRepere, float DecalageX, float DecalageY
    )
    {
        double[] X_flat = DoneesX;
        double[] Y_flat = DoneesY;
        int nbrPoints = DoneesX.Length; 

        // Points à prédire pour tracer la courbe de régression
        List<double> predictionInputs = new List<double>();
        for (double x = Xg; x <= Xd; x += (Xd - Xg) / 100.0)
        {
            predictionInputs.Add(x);
        }
        
        IntPtr ptrRbf = create_rbf(RbfSizes, TailleRbf);
        try
        {
            // Entraînement RBF_FP - Régression (is_classification = 0)
            train_rbf(ptrRbf, X_flat, Y_flat, nbrPoints, RbfSizes[0], RbfSizes[RbfSizes.Length - 1], 0, 10000, 0.01);
            
            AfficherRBFRegression(
                ptrRbf, X_flat, Y_flat, PosX, PosY, TailleAxesRepere,
                predictionInputs.ToArray(), Xg, Xd, Yg, Yd
            );
        }
        finally
        {
            destroy_rbf(ptrRbf);
        }
    }

    //  Fonctions d'Affichage (similaires à PontPMC) 
    
    Vector2[] GenerePtsAlea(int NbrXAPred, float Xg, float Xd, float Yg, float Yd)
    {
        Vector2[] pts = new Vector2[NbrXAPred];
        for(int i = 0 ; i < NbrXAPred ; i++)
        {
            pts[i] = new Vector2(UnityEngine.Random.Range(Xg, Xd), UnityEngine.Random.Range(Yg, Yd));
        }
        return pts;
    }

    void AfficherRepere(float PosX, float PosY, float Taille, string title)
    {
        // Ici, on simule juste l'appel pour maintenir la structure.
        Debug.Log($"Affichage Repère: {title} à ({PosX}, {PosY})");
    }
    
    // Affichage Classification
    void AfficherRBFClassification
    (
        IntPtr PtrRbf, double[] DoneesX, double[] DoneesY,
        float PosX, float PosY, float TailleAxesRepere,
        Vector2[] donneesAleaAPredire, int sortieParPoint, string title
    )
    {
        AfficherRepere(PosX, PosY, TailleAxesRepere, title);
        int nbrPoints = DoneesX.Length / 2;

        // Affichage des points d'entraînement (Points originaux)
        for(int i = 0 ; i < nbrPoints ; i++)
        {
            // On détermine la couleur attendue (pour la classification binaire : 1 ou -1)
            Color expectedCol = DoneesY[i] > 0 ? Color.blue : Color.red;

            // Simule la création d'une sphère (point) en 3D
            var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere); 
            sph.GetComponent<Renderer>().material.color = expectedCol;
            sph.transform.position = new Vector3((float)DoneesX[2 * i] + PosX, (float)DoneesX[2 * i + 1] + PosY, 0);
            sph.transform.localScale = Vector3.one * 0.15f;
        }

        // Affichage des prédictions (Points aléatoires pour la zone de décision)
        foreach(var pt in donneesAleaAPredire)
        {
            double[] inp = { pt.x, pt.y };
            double[] outp = new double[sortieParPoint];
            
            if(predict_rbf(PtrRbf, inp, 2, outp, sortieParPoint, 1) != 0)
            {
                Debug.LogError("predict_rbf error");
                continue;
            }

            // Détermination de la classe prédite pour la couleur
            Color predictedCol = outp[0] > 0 ? Color.green : Color.yellow; // Vert/Jaune pour prédit
            
            var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sph.GetComponent<Renderer>().material.color = predictedCol;
            sph.transform.position = new Vector3(pt.x + PosX, pt.y + PosY, 0);
            sph.transform.localScale = Vector3.one * 0.1f;
        }
    }
    
    // Affichage Régression (similaire à AfficherRegression de PontPMC)
    void AfficherRBFRegression
    (
        IntPtr PtrRbf, double[] DoneesX, double[] DoneesY,
        float PosX, float PosY, float TailleAxesRepere,
        double[] predictionInputs, float Xg, float Xd, float Yg, float Yd
    )
    {
        AfficherRepere(PosX, PosY, TailleAxesRepere, "RBF_FP Regression 1D");

        // Affichage des points d'entraînement en bleu
        for(int i = 0 ; i < DoneesY.Length ; i++)
        {
            var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sph.GetComponent<Renderer>().material.color = Color.blue;
            sph.transform.position = new Vector3((float)DoneesX[i] + PosX, (float)DoneesY[i] + PosY, 0);
            sph.transform.localScale = Vector3.one * 0.15f;
        }
        
        // Tracé de la courbe de régression (points prédits en vert)
        GameObject courbe = new GameObject("CourbeRegressionRBF");
        var lr = courbe.AddComponent<LineRenderer>();
        lr.positionCount = predictionInputs.Length;
        lr.startWidth = lr.endWidth = 0.05f;
        lr.material = new Material(Shader.Find("Sprites/Default"));
        lr.startColor = lr.endColor = Color.red;
        
        for (int i = 0; i < predictionInputs.Length; i++)
        {
            double x = predictionInputs[i];
            double[] inp = { x };
            double[] outp = new double[1];
            predict_rbf(PtrRbf, inp, 1, outp, 1, 0); // Régression (is_classification = 0)
            
            // Les points prédits (petites sphères)
            var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sph.GetComponent<Renderer>().material.color = Color.green;
            sph.transform.position = new Vector3((float)x + PosX, (float)outp[0] + PosY, 0);
            sph.transform.localScale = Vector3.one * 0.08f;
            
            // Points pour la ligne
            lr.SetPosition(i, new Vector3((float)x + PosX, (float)outp[0] + PosY, 0));
        }
    }
}