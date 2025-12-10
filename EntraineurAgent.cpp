#include "EntraineurAgent.hpp"
#include <cstring>

//g++ -o EntraineurAgent.exe EntraineurAgent.cpp ClassifieurLineairePerceptronMultiClasses.cpp; Remove-Item EntraineurAgent.o -ErrorAction SilentlyContinue
//.\EntraineurAgent.exe --tous
//.\EntraineurAgent.exe --aide

std::string trim(const std::string &str)
{
	size_t start = str.find_first_not_of(" \t\n\r");
	if(start == std::string::npos)
	{
		return "";
	}
	size_t end = str.find_last_not_of(" \t\n\r");
	return str.substr(start, end - start + 1);
}

bool chargerAgentJSON(const std::string &cheminFichier, AgentInfo &agent)
{
	std::ifstream fichier(cheminFichier);
	if(!fichier.is_open())
	{
		return false;
	}

	std::string contenu
	(
		(std::istreambuf_iterator<char>(fichier)),
		std::istreambuf_iterator<char>()
	);
	fichier.close();

	auto extraireValeurString = 
		[&contenu](const std::string &cle) -> std::string
		{
			std::string recherche = "\"" + cle + "\":";
			size_t pos = contenu.find(recherche);
			if(pos == std::string::npos)
			{
				return "";
			}
			pos = contenu.find("\"", pos + recherche.length());
			if(pos == std::string::npos)
			{
				return "";
			}
			size_t fin = contenu.find("\"", pos + 1);
			return contenu.substr(pos + 1, fin - pos - 1);
		};

	auto extraireValeurInt = 
		[&contenu](const std::string &cle) -> int
		{
			std::string recherche = "\"" + cle + "\":";
			size_t pos = contenu.find(recherche);
			if(pos == std::string::npos)
			{
				return 0;
			}
			pos += recherche.length();
			while(pos < contenu.length() && (contenu[pos] == ' ' || contenu[pos] == '\t'))
			{
				pos++;
			}
			std::string valeur;
			while(pos < contenu.length() && (isdigit(contenu[pos]) || contenu[pos] == '-'))
			{
				valeur += contenu[pos++];
			}
			return valeur.empty() ? 0 : std::stoi(valeur);
		};

	auto extraireValeurFloat = 
		[&contenu](const std::string &cle) -> float
		{
			std::string recherche = "\"" + cle + "\":";
			size_t pos = contenu.find(recherche);
			if(pos == std::string::npos)
			{
				return 0.0f;
			}
			pos += recherche.length();
			while(pos < contenu.length() && (contenu[pos] == ' ' || contenu[pos] == '\t'))
			{
				pos++;
			}
			std::string valeur;
			while(pos < contenu.length() && (isdigit(contenu[pos]) || contenu[pos] == '.' || contenu[pos] == '-'))
			{
				valeur += contenu[pos++];
			}
			return valeur.empty() ? 0.0f : std::stof(valeur);
		};

	agent.nom = extraireValeurString("nom");
	agent.tailleEntree = extraireValeurInt("tailleEntree");
	agent.nombreClasses = extraireValeurInt("nombreClasses");
	agent.tauxApprentissage = extraireValeurFloat("tauxApprentissage");
	agent.epochsEntraines = extraireValeurInt("epochsEntraines");

	//extraction des datasets utilisés
	std::string rechercheDatasets = "\"datasetsUtilises\":";
	size_t posDatasets = contenu.find(rechercheDatasets);
	if(posDatasets != std::string::npos)
	{
		size_t debutTableau = contenu.find("[", posDatasets);
		size_t finTableau = contenu.find("]", debutTableau);
		if(debutTableau != std::string::npos && finTableau != std::string::npos)
		{
			std::string tableauStr = contenu.substr(debutTableau + 1, finTableau - debutTableau - 1);
			size_t pos = 0;
			while((pos = tableauStr.find("\"", pos)) != std::string::npos)
			{
				size_t fin = tableauStr.find("\"", pos + 1);
				if(fin != std::string::npos)
				{
					agent.datasetsUtilises.push_back(tableauStr.substr(pos + 1, fin - pos - 1));
					pos = fin + 1;
				}
				else
				{
					break;
				}
			}
		}
	}

	//extraction des poids
	std::string recherchePoids = "\"poids\":";
	size_t posPoids = contenu.find(recherchePoids);
	if(posPoids != std::string::npos)
	{
		size_t debutTableauExt = contenu.find("[", posPoids);
		int niveau = 0;
		size_t finTableauExt = debutTableauExt;
		for(size_t i = debutTableauExt ; i < contenu.length() ; ++i)
		{
			if(contenu[i] == '[')
			{
				niveau++;
			}
			else if(contenu[i] == ']')
			{
				niveau--;
				if(niveau == 0)
				{
					finTableauExt = i;
					break;
				}
			}
		}

		std::string poidsStr = contenu.substr(debutTableauExt + 1, finTableauExt - debutTableauExt - 1);
		agent.poids.resize(agent.nombreClasses);

		size_t posClasse = 0;
		int classeIdx = 0;
		while
		(
			(posClasse = poidsStr.find("[", posClasse)) != std::string::npos
			&&
			classeIdx < agent.nombreClasses
		)
		{
			size_t finClasse = poidsStr.find("]", posClasse);
			std::string classeStr = poidsStr.substr(posClasse + 1, finClasse - posClasse - 1);

			std::stringstream ss(classeStr);
			std::string valeur;
			while(std::getline(ss, valeur, ','))
			{
				valeur = trim(valeur);
				if(!valeur.empty())
				{
					agent.poids[classeIdx].push_back(std::stof(valeur));
				}
			}
			posClasse = finClasse + 1;
			classeIdx++;
		}
	}

	//extraction des biais
	std::string rechercheBiais = "\"biais\":";
	size_t posBiais = contenu.find(rechercheBiais);
	if(posBiais != std::string::npos)
	{
		size_t debutTableau = contenu.find("[", posBiais);
		size_t finTableau = contenu.find("]", debutTableau);
		if(debutTableau != std::string::npos && finTableau != std::string::npos)
		{
			std::string biaisStr = contenu.substr(debutTableau + 1, finTableau - debutTableau - 1);
			std::stringstream ss(biaisStr);
			std::string valeur;
			while(std::getline(ss, valeur, ','))
			{
				valeur = trim(valeur);
				if(!valeur.empty())
				{
					agent.biais.push_back(std::stof(valeur));
				}
			}
		}
	}

	return true;
}

bool sauvegarderAgentJSON(const std::string &cheminFichier, const AgentInfo &agent)
{
	std::ofstream fichier(cheminFichier);
	if(!fichier.is_open())
	{
		return false;
	}

	fichier << "{\n";
	fichier << "\t\"nom\": \"" << agent.nom << "\",\n";
	fichier << "\t\"tailleEntree\": " << agent.tailleEntree << ",\n";
	fichier << "\t\"nombreClasses\": " << agent.nombreClasses << ",\n";
	fichier << "\t\"tauxApprentissage\": " << agent.tauxApprentissage << ",\n";
	fichier << "\t\"epochsEntraines\": " << agent.epochsEntraines << ",\n";

	fichier << "\t\"datasetsUtilises\": [";
	for(size_t i = 0 ; i < agent.datasetsUtilises.size() ; ++i)
	{
		fichier << "\"" << agent.datasetsUtilises[i] << "\"";
		if(i < agent.datasetsUtilises.size() - 1)
		{
			fichier << ", ";
		}
	}
	fichier << "],\n";

	fichier << "\t\"poids\": [\n";
	for(size_t c = 0 ; c < agent.poids.size() ; ++c)
	{
		fichier << "\t\t[";
		for(size_t i = 0 ; i < agent.poids[c].size() ; ++i)
		{
			fichier << agent.poids[c][i];
			if(i < agent.poids[c].size() - 1)
			{
				fichier << ", ";
			}
		}
		fichier << "]";
		if(c < agent.poids.size() - 1)
		{
			fichier << ",";
		}
		fichier << "\n";
	}
	fichier << "\t],\n";

	fichier << "\t\"biais\": [";
	for(size_t i = 0 ; i < agent.biais.size() ; ++i)
	{
		fichier << agent.biais[i];
		if(i < agent.biais.size() - 1)
		{
			fichier << ", ";
		}
	}
	fichier << "]\n";

	fichier << "}\n";
	fichier.close();
	return true;
}

bool chargerDatasetCSV
(
	const std::string &cheminFichier, std::vector<std::vector<float>> &X,
	std::vector<int> &Y, int &compteurNeutres
)
{
	std::ifstream fichier(cheminFichier);
	if(!fichier.is_open())
	{
		return false;
	}

	std::string ligne;
	while(std::getline(fichier, ligne))
	{
		if(ligne.empty())
		{
			continue;
		}

		std::stringstream ss(ligne);
		std::string valeur;
		std::vector<std::string> colonnes;
		while(std::getline(ss, valeur, ','))
		{
			colonnes.push_back(valeur);
		}

		if(colonnes.size() < 8)
		{
			continue;
		}

		/*
		//version 1 : sans filtre (7 entrées, toutes les données)
		{
			//format CSV : ballX, ballY, ballVX, ballVY, playerY, enemyY, playerMove, enemyMove
			//entrées : indices 0-5 + 7 (enemyMove)
			std::vector<float> input(7);
			input[0] = std::stof(colonnes[0]);//ballX
			input[1] = std::stof(colonnes[1]);//ballY
			input[2] = std::stof(colonnes[2]);//ballVX
			input[3] = std::stof(colonnes[3]);//ballVY
			input[4] = std::stof(colonnes[4]);//playerY
			input[5] = std::stof(colonnes[5]);//enemyY
			input[6] = std::stof(colonnes[7]);//enemyMove (indice 7)

			//sortie : indice 6 (playerMove)
			float rawOutput = std::stof(colonnes[6]);

			//conversion en classes (0, 1, 2)
			//rawOutput = 1.0 : haut (Up = -1) : classe 0
			//rawOutput = 0.0 : neutre (Neutral = 0) : classe 1
			//rawOutput = -1.0 : bas (Down = 1) : classe 2
			int classIndex = 1;//par défaut : neutre
			if(rawOutput > 0.5f)
			{
				classIndex = 0;//haut
			}
			else if(rawOutput < -0.5f)
			{
				classIndex = 2;//bas
			}

			X.push_back(input);
			Y.push_back(classIndex);
		}
		*/

		/*
		//version 2 : avec filtre 1000 neutres (7 entrées)
		{
			//filtrage des données neutres (limite à 1000)
			if(colonnes[6] == "0" && compteurNeutres >= 1000)
			{
				continue;
			}
			if(colonnes[6] == "0")
			{
				compteurNeutres++;
			}

			//format CSV : ballX, ballY, ballVX, ballVY, playerY, enemyY, playerMove, enemyMove
			//entrées : indices 0-5 + 7 (enemyMove)
			std::vector<float> input(7);
			input[0] = std::stof(colonnes[0]);//ballX
			input[1] = std::stof(colonnes[1]);//ballY
			input[2] = std::stof(colonnes[2]);//ballVX
			input[3] = std::stof(colonnes[3]);//ballVY
			input[4] = std::stof(colonnes[4]);//playerY
			input[5] = std::stof(colonnes[5]);//enemyY
			input[6] = std::stof(colonnes[7]);//enemyMove (indice 7)

			//sortie : indice 6 (playerMove)
			float rawOutput = std::stof(colonnes[6]);

			//conversion en classes (0, 1, 2)
			//rawOutput = 1.0 : haut (Up = -1) : classe 0
			//rawOutput = 0.0 : neutre (Neutral = 0) : classe 1
			//rawOutput = -1.0 : bas (Down = 1) : classe 2
			int classIndex = 1;//par défaut : neutre
			if(rawOutput > 0.5f)
			{
				classIndex = 0;//haut
			}
			else if(rawOutput < -0.5f)
			{
				classIndex = 2;//bas
			}

			X.push_back(input);
			Y.push_back(classIndex);
		}
		*/

		//version 3, 4 et 5 avec uniformisation up/neutre/down et 5 entrées

		/*
		//version 3 : avec équilibrage 600 neutres
		{
			//sortie : indice 6 (playerMove), valeurs directes du CSV : -1 (haut), 0 (neutre), 1 (bas)
			float rawOutput = std::stof(colonnes[6]);
			
			//conversion en classes (0, 1, 2) avec mapping uniforme
			//rawOutput = -1.0 : haut (Up) : classe 0
			//rawOutput = 0.0 : neutre (Neutral) : classe 1
			//rawOutput = 1.0 : bas (Down) : classe 2
			int classIndex = 1;//par défaut : neutre
			if(rawOutput < -0.5f)
			{
				classIndex = 0;//haut (-1)
			}
			else if(rawOutput > 0.5f)
			{
				classIndex = 2;//bas (1)
			}

			//équilibrage des classes : limiter les neutres au nombre de haut/bas
			if(classIndex == 1 && compteurNeutres >= 600)
			{
				continue;
			}
			if(classIndex == 1)
			{
				compteurNeutres++;
			}

			//format CSV : ballX, ballY, ballVX, ballVY, playerY, enemyY, playerMove, enemyMove
			//entrées simplifiées : ballX, ballY, ballVX, ballVY, playerY (5 entrées)
			std::vector<float> input(5);
			input[0] = std::stof(colonnes[0]);//ballX
			input[1] = std::stof(colonnes[1]);//ballY
			input[2] = std::stof(colonnes[2]);//ballVX
			input[3] = std::stof(colonnes[3]);//ballVY
			input[4] = std::stof(colonnes[4]);//playerY

			X.push_back(input);
			Y.push_back(classIndex);
		}
		*/

		//version 4 : avec filtre x neutres
		{
			//sortie : indice 6 (playerMove), valeurs directes du CSV : -1 (haut), 0 (neutre), 1 (bas)
			float rawOutput = std::stof(colonnes[6]);
			
			//conversion en classes (0, 1, 2) avec mapping uniforme
			//rawOutput = -1.0 : haut (Up) : classe 0
			//rawOutput = 0.0 : neutre (Neutral) : classe 1
			//rawOutput = 1.0 : bas (Down) : classe 2
			int classIndex = 1;//par défaut : neutre
			if(rawOutput < -0.5f)
			{
				classIndex = 0;//haut (-1)
			}
			else if(rawOutput > 0.5f)
			{
				classIndex = 2;//bas (1)
			}

			//filtre neutres
			if(classIndex == 1 && compteurNeutres >= 1500)
			{
				continue;
			}
			if(classIndex == 1)
			{
				compteurNeutres++;
			}

			//format CSV : ballX, ballY, ballVX, ballVY, playerY, enemyY, playerMove, enemyMove
			//entrées simplifiées : ballX, ballY, ballVX, ballVY, playerY (5 entrées)
			std::vector<float> input(5);
			input[0] = std::stof(colonnes[0]);//ballX
			input[1] = std::stof(colonnes[1]);//ballY
			input[2] = std::stof(colonnes[2]);//ballVX
			input[3] = std::stof(colonnes[3]);//ballVY
			input[4] = std::stof(colonnes[4]);//playerY

			X.push_back(input);
			Y.push_back(classIndex);
		}

		/*
		//version 5 : sans filtre
		{
			//sortie : indice 6 (playerMove), valeurs directes du CSV : -1 (haut), 0 (neutre), 1 (bas)
			float rawOutput = std::stof(colonnes[6]);
			
			//conversion en classes (0, 1, 2) avec mapping uniforme
			//rawOutput = -1.0 : haut (Up) : classe 0
			//rawOutput = 0.0 : neutre (Neutral) : classe 1
			//rawOutput = 1.0 : bas (Down) : classe 2
			int classIndex = 1;//par défaut : neutre
			if(rawOutput < -0.5f)
			{
				classIndex = 0;//haut (-1)
			}
			else if(rawOutput > 0.5f)
			{
				classIndex = 2;//bas (1)
			}

			//format CSV : ballX, ballY, ballVX, ballVY, playerY, enemyY, playerMove, enemyMove
			//entrées simplifiées : ballX, ballY, ballVX, ballVY, playerY (5 entrées)
			std::vector<float> input(5);
			input[0] = std::stof(colonnes[0]);//ballX
			input[1] = std::stof(colonnes[1]);//ballY
			input[2] = std::stof(colonnes[2]);//ballVX
			input[3] = std::stof(colonnes[3]);//ballVY
			input[4] = std::stof(colonnes[4]);//playerY

			X.push_back(input);
			Y.push_back(classIndex);
		}
		*/
	}

	fichier.close();
	return true;
}

std::vector<std::string> listerFichiersCSV(const std::string &dossier)
{
	std::vector<std::string> fichiers;

	if(!fs::exists(dossier))
	{
		return fichiers;
	}

	for(const auto &entry : fs::directory_iterator(dossier))
	{
		if(entry.is_regular_file() && entry.path().extension() == ".csv")
		{
			fichiers.push_back(entry.path().filename().string());
		}
	}

	std::sort(fichiers.begin(), fichiers.end());
	return fichiers;
}

bool deplacerFichier(const std::string &source, const std::string &destination)
{
	try
	{
		fs::rename(source, destination);
		return true;
	}
	catch(const std::exception &e)
	{
		std::cerr << "Erreur lors du déplacement : " << e.what() << std::endl;
		return false;
	}
}

void afficherAide()
{
	std::cout << "Usage: EntraineurAgent.exe [options]\n\n";
	std::cout << "Options:\n";
	std::cout << "  --premier       Entrainer avec le 1er fichier de DatasetsAEntrainer\n";
	std::cout << "  --nombre N      Entrainer avec les N premiers fichiers\n";
	std::cout << "  --tous          Entrainer avec tous les fichiers disponibles\n";
	std::cout << "  --epochs N      Nombre d'epochs (defaut: 1000)\n";
	std::cout << "  --agent NOM     Nom de l'agent (defaut: agent)\n";
	std::cout << "  --reset         Supprimer l'agent et le recreer depuis zero\n";
	std::cout << "  --aide          Afficher cette aide\n\n";
	std::cout << "Exemples:\n";
	std::cout << "  EntraineurAgent.exe --premier\n";
	std::cout << "  EntraineurAgent.exe --nombre 3 --epochs 500\n";
	std::cout << "  EntraineurAgent.exe --tous --agent mon_agent\n";
	std::cout << "  EntraineurAgent.exe --reset\n";
	std::cout << "  EntraineurAgent.exe --reset --agent mon_agent\n";
}

int main(int argc, char *argv[])
{
	//valeurs par défaut
	std::string nomAgent = "agent";//par defaut
	int epochs = 1000;
	int nombreFichiers = 0;//0 = non défini
	bool premier = false;
	bool tous = false;
	bool reset = false;

	for(int i = 1 ; i < argc ; ++i)
	{
		std::string arg = argv[i];

		if(arg == "--aide" || arg == "-h" || arg == "--help")
		{
			afficherAide();
			return 0;
		}
		else if(arg == "--premier")
		{
			premier = true;
		}
		else if(arg == "--tous")
		{
			tous = true;
		}
		else if(arg == "--reset")
		{
			reset = true;
		}
		else if(arg == "--nombre" && i + 1 < argc)
		{
			nombreFichiers = std::stoi(argv[++i]);
		}
		else if(arg == "--epochs" && i + 1 < argc)
		{
			epochs = std::stoi(argv[++i]);
		}
		else if(arg == "--agent" && i + 1 < argc)
		{
			nomAgent = argv[++i];
		}
	}

	//vérification des arguments
	if(!premier && !tous && nombreFichiers == 0 && !reset)
	{
		std::cout << "Erreur: Veuillez specifier --premier, --nombre N, --tous ou --reset\n\n";
		afficherAide();
		return 1;
	}

	//chemins
	std::string dossierBase = ".";
	std::string dossierAgents = dossierBase + "/AgentsIA";
	std::string dossierAEntrainer = dossierBase + "/DatasetsAEntrainer";
	std::string dossierUtilises = dossierBase + "/DatasetsUtilises";
	std::string cheminAgent = dossierAgents + "/" + nomAgent + ".json";

	//créer les dossiers si nécessaire
	fs::create_directories(dossierAgents);
	fs::create_directories(dossierUtilises);

	if(reset)
	{
		if(fs::exists(cheminAgent))
		{
			//charger l'agent existant pour garder les métadonnées
			AgentInfo agentReset;
			chargerAgentJSON(cheminAgent, agentReset);
			
			//réinitialiser les poids et biais à zéro
			agentReset.epochsEntraines = 0;
			agentReset.datasetsUtilises.clear();
			agentReset.poids.clear();
			agentReset.biais.clear();
			
			//sauvegarder l'agent réinitialisé
			sauvegarderAgentJSON(cheminAgent, agentReset);
			std::cout << "Agent reinitialise : " << cheminAgent << std::endl;
		}
		else
		{
			std::cout << "Agent inexistant : " << cheminAgent << std::endl;
		}
		
		if(!premier && !tous && nombreFichiers == 0)
		{
			std::cout << "\nTermine!" << std::endl;
			return 0;
		}
	}

	//lister les fichiers à entraîner
	std::vector<std::string> fichiersDisponibles = listerFichiersCSV(dossierAEntrainer);

	if(fichiersDisponibles.empty())
	{
		std::cout << "Aucun fichier CSV trouve dans " << dossierAEntrainer << std::endl;
		return 1;
	}

	std::cout << "Fichiers disponibles : " << fichiersDisponibles.size() << std::endl;

	//déterminer le nombre de fichiers à utiliser
	int nbFichiersAUtiliser = 0;
	if(premier)
	{
		nbFichiersAUtiliser = 1;
	}
	else if(tous)
	{
		nbFichiersAUtiliser = fichiersDisponibles.size();
	}
	else
	{
		nbFichiersAUtiliser = std::min(nombreFichiers, (int)fichiersDisponibles.size());
	}

	std::cout << "Fichiers a utiliser : " << nbFichiersAUtiliser << std::endl;
	std::cout << "Epochs : " << epochs << std::endl;
	std::cout << "Agent : " << nomAgent << std::endl;
	std::cout << std::endl;

	//charger ou créer l'agent
	AgentInfo agent;
	bool agentExiste = chargerAgentJSON(cheminAgent, agent);

	if(agentExiste)
	{
		std::cout << "Agent existant charge : " << agent.nom << std::endl;
		std::cout << "Epochs precedents : " << agent.epochsEntraines << std::endl;
		std::cout << "Datasets deja utilises : " << agent.datasetsUtilises.size() << std::endl;
	}
	else
	{
		std::cout << "Creation d'un nouvel agent : " << nomAgent << std::endl;
		agent.nom = nomAgent;
		agent.tailleEntree = 5;
		agent.nombreClasses = 3;
		agent.tauxApprentissage = 0.01f;
		agent.epochsEntraines = 0;
	}

	//créer le modèle
	PerceptronMultiClasses *model = create_perceptron_multiclasses(agent.tailleEntree, agent.nombreClasses, agent.tauxApprentissage);

	//charger les poids existants si l'agent existe
	if(agentExiste && !agent.poids.empty())
	{
		std::cout << "Chargement des poids existants..." << std::endl;

		//aplatir les poids
		std::vector<float> poidsFlat;
		for(const auto &classe : agent.poids)
		{
			for(float p : classe)
			{
				poidsFlat.push_back(p);
			}
		}

		set_weights(model, poidsFlat.data());
		set_bias(model, agent.biais.data());
	}

	//entraîner avec chaque fichier
	for(int i = 0 ; i < nbFichiersAUtiliser ; ++i)
	{
		std::string nomFichier = fichiersDisponibles[i];
		std::string cheminComplet = dossierAEntrainer + "/" + nomFichier;

		std::cout << "\nEntrainement avec : " << nomFichier << std::endl;

		//charger le dataset
		std::vector<std::vector<float>> X;
		std::vector<int> Y;
		int compteurNeutres = 0;

		if(!chargerDatasetCSV(cheminComplet, X, Y, compteurNeutres))
		{
			std::cerr << "Erreur lors du chargement de " << nomFichier << std::endl;
			continue;
		}

		std::cout << "Donnees chargees : " << X.size() << " exemples" << std::endl;

		//aplatir X pour le training
		int rows = X.size();
		int cols = agent.tailleEntree;
		std::vector<float> Xflat(rows * cols);
		for(int r = 0 ; r < rows ; ++r)
		{
			for(int c = 0 ; c < cols ; ++c)
			{
				Xflat[r * cols + c] = X[r][c];
			}
		}

		//entraîner
		train_perceptron_multiclasses(model, Xflat.data(), Y.data(), rows, cols, epochs);

		std::cout << "Entrainement termine (" << epochs << " epochs)" << std::endl;

		//mettre à jour les infos de l'agent
		agent.epochsEntraines += epochs;
		agent.datasetsUtilises.push_back(nomFichier);

		//déplacer le fichier vers DatasetsUtilises
		std::string destination = dossierUtilises + "/" + nomFichier;
		if(deplacerFichier(cheminComplet, destination))
		{
			std::cout << "Fichier deplace vers DatasetsUtilises" << std::endl;
		}
	}

	//récupérer les poids et biais du modèle
	int totalPoids = agent.tailleEntree * agent.nombreClasses;
	std::vector<float> poidsFlat(totalPoids);
	std::vector<float> biais(agent.nombreClasses);

	get_weights(model, poidsFlat.data());
	get_bias(model, biais.data());

	//reconstruire les poids en 2D
	agent.poids.resize(agent.nombreClasses);
	int idx = 0;
	for(int c = 0 ; c < agent.nombreClasses ; ++c)
	{
		agent.poids[c].resize(agent.tailleEntree);
		for(int i = 0 ; i < agent.tailleEntree ; ++i)
		{
			agent.poids[c][i] = poidsFlat[idx++];
		}
	}
	agent.biais = biais;

	//sauvegarder l'agent
	if(sauvegarderAgentJSON(cheminAgent, agent))
	{
		std::cout << "\nAgent sauvegarde : " << cheminAgent << std::endl;
		std::cout << "Total epochs entraines : " << agent.epochsEntraines << std::endl;
		std::cout << "Total datasets utilises : " << agent.datasetsUtilises.size() << std::endl;
	}
	else
	{
		std::cerr << "Erreur lors de la sauvegarde de l'agent" << std::endl;
	}

	//nettoyer
	destroy_perceptron_multiclasses(model);

	std::cout << "\nTermine!" << std::endl;
	return 0;
}