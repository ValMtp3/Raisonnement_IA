Raisonnement_IA/README.md
# Raisonnement IA

## Description

Ce projet implémente une **couche de raisonnement** autour du modèle de langage Mistral AI pour permettre à l'IA de résoudre des problèmes complexes en plusieurs étapes. Il utilise des patterns avancés comme **ReAct** (Reasoning and Acting), **Chain-of-Thought** (CoT), et **Retrieval-Augmented Generation** (RAG) pour décomposer les tâches, utiliser des outils externes (calcul, recherche), vérifier les résultats et itérer si nécessaire.

Le système est conçu pour être modulaire, extensible et robuste, avec des fonctionnalités de logging, métriques et tests unitaires.

## Fonctionnalités

- **Planner** : Décomposition automatique des tâches en étapes logiques.
- **Step Solver** : Exécution d'étapes individuelles avec appel à des outils (calcul mathématique, recherche RAG).
- **Verifier** : Vérification de la cohérence des résultats.
- **RAG avec ChromaDB** : Recherche dans une base de connaissances vectorielle pour enrichir les réponses.
- **Métriques et Logging** : Suivi des performances avec pandas et affichage élégant via rich.
- **Tests Unitaires** : Validation des composants avec pytest.
- **Intégration API Mistral** : Utilisation de l'API Mistral pour les générations de texte et embeddings.

## Technologies Utilisées

- **Langage** : Python 3.13+
- **Modèle IA** : Mistral AI (via API)
- **Base de Données Vectorielle** : ChromaDB
- **Frameworks** : LangChain, Pydantic, Tenacity
- **Outils** : httpx, pandas, rich, dotenv
- **Gestionnaire de Paquets** : UV

## Installation

### Prérequis

- Python 3.13 ou supérieur
- Clé API Mistral AI (obtenez-la sur [Mistral AI](https://mistral.ai/))

### Étapes d'Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/Raisonnement_IA.git
   cd Raisonnement_IA
   ```

2. Installez les dépendances avec UV :
   ```bash
   uv sync
   ```

3. Créez un fichier `.env` à la racine du projet et ajoutez votre clé API :
   ```
   MISTRAL_API_KEY=votre_clé_api_ici
   ```

## Configuration

Le fichier `pyproject.toml` contient les dépendances. Les paramètres principaux sont configurés dans le dictionnaire `CONFIG` du notebook `reasoning_layer.ipynb` :

- Modèle Mistral utilisé
- Endpoints API
- Paramètres de température, tokens, etc.
- Chemin de la base ChromaDB
- Documents d'exemple pour la RAG

Modifiez ces valeurs selon vos besoins.

## Utilisation

### Lancement du Notebook

Ouvrez le notebook `reasoning_layer.ipynb` dans Jupyter :

```bash
jupyter notebook reasoning_layer.ipynb
```

Exécutez les cellules dans l'ordre pour initialiser le système.

### Fonction Principale : `perform_reasoning`

Utilisez la fonction `perform_reasoning(query)` pour traiter une requête :

```python
from reasoning_layer import perform_reasoning

result = perform_reasoning("Calcule 123 * 47 et explique les étapes.")
print(result["final_answer"])
```

### Exemples

Le notebook inclut des exemples prêts à l'emploi :

- Calcul mathématique : `perform_reasoning("Calcule 123 * 47 et explique les étapes.")`
- Recherche RAG : `perform_reasoning("Trouve des informations sur Paris.")`
- Planification : `perform_reasoning("Planifie un voyage Paris-Lyon en minimisant le temps.")`

### Outils Disponibles

- **CALC:** : Pour des calculs mathématiques simples (ex: `CALC: 2+2`).
- **SEARCH:** : Pour rechercher dans la base de connaissances (ex: `SEARCH: population de Paris`).

## Tests

Exécutez les tests unitaires avec pytest :

```bash
uv run pytest
```

Les tests couvrent le planner, les outils, le verifier, etc.

## Structure du Projet

```
Raisonnement_IA/
├── .env                    # Variables d'environnement (clé API)
├── .gitignore              # Fichiers ignorés par Git
├── .python-version         # Version Python
├── pyproject.toml          # Configuration des dépendances
├── uv.lock                 # Lockfile UV
├── reasoning_layer.ipynb   # Notebook principal
├── EXPLICATION_CODE.md     # Explication détaillée du code
├── guide.md                # Guide d'implémentation
├── chroma_db/              # Base de données vectorielle
├── eval/                   # Scripts d'évaluation
└── tools/                  # Outils supplémentaires
```

## Contribution

Les contributions sont les bienvenues ! Veuillez :

1. Forker le projet.
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/nouvelle-fonction`).
3. Commiter vos changements (`git commit -am 'Ajout de nouvelle fonctionnalité'`).
4. Pousser vers la branche (`git push origin feature/nouvelle-fonction`).
5. Ouvrir une Pull Request.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Auteurs

- Valentin Fiess (développeur principal)

## Remerciements

- Mistral AI pour leur modèle de langage.
- Les communautés open-source pour les bibliothèques utilisées (ChromaDB, LangChain, etc.).