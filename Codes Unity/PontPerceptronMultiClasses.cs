using System;
using System.Runtime.InteropServices;
using UnityEngine;
using System.IO;
using System.Collections.Generic;
using System.Globalization;

public class PontPerceptronMultiClasses : MonoBehaviour
{
    [DllImport("ClassifieurLineairePerceptronMultiClasses")]
    private static extern IntPtr create_perceptron_multiclasses(int inputSize, int numClasses, float lr);

    [DllImport("ClassifieurLineairePerceptronMultiClasses")]
    private static extern void train_perceptron_multiclasses(IntPtr model, float[] X, int[] Y, int rows, int cols, int epochs);

    [DllImport("ClassifieurLineairePerceptronMultiClasses")]
    private static extern int predict_perceptron_multiclasses(IntPtr model, float[] input);

    [DllImport("ClassifieurLineairePerceptronMultiClasses")]
    private static extern void destroy_perceptron_multiclasses(IntPtr model);

    void Start()
    {
        RunPongTest();
    }

    void RunPongTest()
    {
        string filePath = @"C:\Users\yecel\Desktop\ESGI - 4A\T1\Machine Learning\Pong_Joueur_Artificiel\pong_data_test_1.csv";
        
        if(!File.Exists(filePath))
        {
            Debug.LogError("Le dataset du pong n'a pas été trouvé à l'emplacement suivant : " + filePath);
            return;
        }

        List<float[]> inputsList = new List<float[]>();
        List<int> outputsList = new List<int>();

        string[] lines = File.ReadAllLines(filePath);
        
        foreach(string line in lines)
        {
            if(string.IsNullOrWhiteSpace(line)) continue;

            string[] parts = line.Split(',');
            if(parts.Length < 8) continue;

            //entrées/inputs : indices 0, 1, 2, 3, 4, 5, 7
            float[] input = new float[7];
            input[0] = float.Parse(parts[0], CultureInfo.InvariantCulture);
            input[1] = float.Parse(parts[1], CultureInfo.InvariantCulture);
            input[2] = float.Parse(parts[2], CultureInfo.InvariantCulture);
            input[3] = float.Parse(parts[3], CultureInfo.InvariantCulture);
            input[4] = float.Parse(parts[4], CultureInfo.InvariantCulture);
            input[5] = float.Parse(parts[5], CultureInfo.InvariantCulture);
            input[6] = float.Parse(parts[7], CultureInfo.InvariantCulture);

            //sortie/output : indice 6
            float rawOutput = float.Parse(parts[6], CultureInfo.InvariantCulture);
            
            //conversion en classes (0, 1, 2)
            //1.0 (haut) : 0
            //0.0 (rien) : 1
            //-1.0 (bas) : 2
            int classIndex = 1;//par defaut : rien
            if(rawOutput > 0.5f)
            {
                classIndex = 0;//haut
            }
            else if(rawOutput < -0.5f)
            {
                classIndex = 2;//bas
            }

            inputsList.Add(input);
            outputsList.Add(classIndex);
        }

        int rows = inputsList.Count;
        int cols = 7;
        int numClasses = 3;

        //aplatir les données pour le C++
        float[] Xflat = new float[rows * cols];
        int[] Y = outputsList.ToArray();

        for(int i = 0 ; i < rows ; i++)
        {
            for(int j = 0 ; j < cols ; j++)
            {
                Xflat[i * cols + j] = inputsList[i][j];
            }
        }

        //création et entraînement du modèle
        IntPtr model = create_perceptron_multiclasses(cols, numClasses, 0.01f);
        
        //entraînement
        train_perceptron_multiclasses(model, Xflat, Y, rows, cols, 1000);

        Debug.Log("Entraînement terminé.");

        //visualisation
        float xOffset = 0f;
        float yOffset = 0f;
        float scale = 10f; 

        //visualisation 1 (ce que le joueur a fait)
        for(int i = 0 ; i < rows ; i++)
        {
            float[] input = inputsList[i];
            int actualClass = outputsList[i];

            float xPos = input[0] * scale + xOffset;
            float yPos = input[1] * scale + yOffset;
            
            var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.transform.position = new Vector3(xPos, yPos, 0);
            sphere.transform.localScale = Vector3.one * 0.2f;

            if(actualClass == 0)
            {
                sphere.GetComponent<Renderer>().material.color = Color.green;//haut
            }
            else if(actualClass == 2)
            {
                sphere.GetComponent<Renderer>().material.color = Color.red;//bas
            }
            else
            {
                sphere.transform.localScale = Vector3.one * 0.1f;
                sphere.GetComponent<Renderer>().material.color = Color.orange;//rien
            }
        }

        //visualisation 2 (prédiction du perçeptron)
        float predOffset = 20f;
        int correctCount = 0;

        for(int i = 0 ; i < rows ; i++)
        {
            float[] input = inputsList[i];
            int actualClass = outputsList[i];
            
            int predictedClass = predict_perceptron_multiclasses(model, input);

            if (predictedClass == actualClass) correctCount++;

            float xPos = input[0] * scale + xOffset + predOffset;
            float yPos = input[1] * scale + yOffset;

            var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.transform.position = new Vector3(xPos, yPos, 0);
            sphere.transform.localScale = Vector3.one * 0.2f;

            if(predictedClass == 0)
            {
                sphere.GetComponent<Renderer>().material.color = Color.blue;//haut
            }
            else if(predictedClass == 2)
            {
                sphere.GetComponent<Renderer>().material.color = Color.magenta;//bas
            }
            else
            {
                sphere.transform.localScale = Vector3.one * 0.1f;
                sphere.GetComponent<Renderer>().material.color = Color.yellow;//rien
            }
        }

        float accuracy = (float)correctCount / rows * 100f;
        Debug.Log($"Précision du Perceptron Multi-classes : {accuracy:F2}%");

        destroy_perceptron_multiclasses(model);
    }
}
