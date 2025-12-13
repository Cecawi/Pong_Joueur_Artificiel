#pragma once
#include "ClassifieurLineairePerceptronMultiClasses.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

//structure pour stocker les informations de l'agent
struct AgentInfo
{
	std::string nom;
	std::string typeModele;//"linear" ou "pmc"
	int tailleEntree;
	int nombreClasses;
	float tauxApprentissage;
	int epochsEntraines;
	std::vector<std::string> datasetsUtilises;
	
	//Linear
	std::vector<std::vector<float>> poids;
	std::vector<float> biais;

	//PMC
	std::vector<int> architecture;
	std::vector<std::vector<std::vector<double>>> poidsPMC;//[layer][i][j]
};

//charge un fichier JSON d'agent
bool chargerAgentJSON(const std::string &cheminFichier, AgentInfo &agent);

//sauvegarde un agent en JSON
bool sauvegarderAgentJSON(const std::string &cheminFichier, const AgentInfo &agent);

//charge un dataset CSV
bool chargerDatasetCSV(const std::string &cheminFichier, std::vector<std::vector<float>> &X, std::vector<int> &Y, int &compteurNeutres);

//liste les fichiers CSV dans un dossier
std::vector<std::string> listerFichiersCSV(const std::string &dossier);

//déplace un fichier d'un dossier à un autre
bool deplacerFichier(const std::string &source, const std::string &destination);
