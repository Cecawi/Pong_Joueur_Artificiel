Début du projet de Machine Learning

1ers TODO : 
Ahmed : envoyer à Céline la clé publique de ton PC pour pouvoir avoir accès au github (Céline : ????? : first time -> je vais être "lente" -> discord? de mon côté?)
Céline : faire en sorte qu'Ahmed puisse avoir accès au github (en utilisant sa clé publique?)
Both : call discord (surement 02/10 ~16h) -> clarifier le projet et répartir les premières tâches

Potentielles 1ères tâches : 
  clarifier ce qu'on veut faire dans notre projet (pong?)
    règles
    qui code quoi?
    comment?
      chacun de son côté?
      parfois en même temps?
      en C# normalement (GDScript considéré comme désuet? qu'on n'utilise plus)
  fixer les horraires de meeting/review
  peut-être faire une fiche de référence pour mettre toutes nos idées concernant le jeu + fiche plus structurée
    classes, méthodes/fonctions, objets, liens...
  COMMENCER A CODER!

A CLARIFIER ------->!!!!!Avant d’appliquer les algorithmes et modèles vus en cours à la problématique choisie, il sera impératif de
démontrer la justesse de l’implémentation de ces derniers sur les cas de tests proposés. Ainsi, il est
proposé d’implémenter ceux-ci sur des jeux de données classiques (données linéairement séparables,
non linéairement séparables, tâches de classification, tâches de régression, etc.) tels que vus en cours de
Machine Learning!!!!!

On va devoir créer un joueur artificiel pour notre (futur) jeu de pong (Godot)
Ce projet sera réalisé en passants par plusieurs étapes
On va devoir : 
  créer notre (futur) jeu de pong
  pouvoir jouer à notre jeu de pong
  récupérer des données à partir de plusieurs partis jouées par nous même pour constituer notre dataset
     étiqueter les données par "états du jeu" ou "action choisie par l'humain" (dataset étiqueté)
  créer notre lib/bibliothèque dynamique from scratch en C++
  implémenter des modèles de différents types (Linéaire / PMC / RBF / ...)
  expérimenter pour entraîner (SUFFISAMENT LONGTEMPS ?mais pas trop?) nos modèles sur notre dataset étiqueté
  permettre l'utilisation dans Unity d'un modèle entrainé se comportant"comme un humain"
                                |---> ~Interop (pouvoir utiliser code en C++ sur Unity (C#) (bibliothèque dynamique)

Notre 1er cap à passer est d'incrémenter un modèle linéaire qui passera les cas de tests
  A CLARIFIER
    date?
    quels cas de tests?
    rapport?
