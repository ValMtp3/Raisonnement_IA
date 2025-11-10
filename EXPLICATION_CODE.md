# 🧠 Explication détaillée du notebook `reasoning_layer.ipynb`

Ce document a pour but de vous expliquer, section par section, le fonctionnement du code contenu dans le notebook
`reasoning_layer.ipynb`. L'objectif de ce notebook est de construire une **couche de raisonnement** autour d'un modèle
de langage (Mistral) pour lui permettre de résoudre des problèmes complexes en plusieurs étapes, d'utiliser des outils
externes et de vérifier ses propres résultats.

---

## 1. Imports

Cette première cellule de code importe toutes les bibliothèques Python nécessaires au projet.

```python
import json, os, re, time
from typing import List, Optional
import chromadb
import httpx
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential
```

- **Bibliothèques standard** :
    - `json` : Pour manipuler le format de données JSON, très utilisé pour communiquer avec les APIs.
    - `os` : Pour interagir avec le système d'exploitation, notamment pour récupérer des variables d'environnement (clé
      API).
    - `re` : Pour les expressions régulières, utiles pour extraire des informations précises depuis le texte généré par
      le modèle.
    - `time` : Pour gérer le temps, notamment pour ajouter des pauses (`sleep`) afin d'éviter de surcharger l'API.
    - `typing` : Pour ajouter des indications de type (`List`, `Optional`), ce qui rend le code plus lisible et robuste.

- **Bibliothèques externes** :
    - `chromadb` : Le client pour interagir avec la base de données vectorielle ChromaDB, qui stocke les connaissances
      pour le RAG.
    - `httpx` : Un client HTTP moderne pour envoyer des requêtes à l'API de Mistral.
    - `pandas` : Utilisé pour créer et gérer un DataFrame qui stockera les métriques de performance.
    - `dotenv` : Pour charger les variables d'environnement (comme la clé API) depuis un fichier `.env`.
    - `pydantic` : Pour valider les données. On définit des "schémas" et Pydantic s'assure que les données reçues
      correspondent à ces schémas.
    - `rich` : Pour afficher des tableaux et du texte formaté de manière élégante dans le terminal.
    - `tenacity` : Un outil très pratique pour ajouter une logique de "réessai" (retry) automatique aux appels API. Si
      une requête échoue, `tenacity` la relancera plusieurs fois.

---

## 2. Variables et Configuration

Cette section centralise tous les paramètres du projet. C'est une excellente pratique car elle permet de modifier le
comportement du système sans avoir à changer le code lui-même.

```python
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CONFIG = { ... }
```

- `load_dotenv()` : Charge le fichier `.env` présent à la racine du projet.
- `MISTRAL_API_KEY` : Récupère la clé API depuis les variables d'environnement.
- `CONFIG` : Un dictionnaire qui contient :
    - Le modèle Mistral à utiliser (`mistral-small-2506`).
    - L'URL de base et les points d'accès (`endpoints`) de l'API.
    - Le modèle d'embedding (`mistral-embed`).
    - Le chemin de la base de données ChromaDB.
    - Les paramètres par défaut pour les appels à l'API (température, nombre de tokens, etc.).
    - Les paramètres pour la logique de réessai (`tenacity`).
    - Des documents d'exemple pour peupler la base de connaissances.

---

## 3. Wrapper API Mistral (`MistralClient`)

Cette classe est un "emballage" (wrapper) qui simplifie et sécurise les appels à l'API de Mistral.

```python
class MistralClient:
    # ...
```

- `__init__` : Le constructeur initialise le client avec les informations du dictionnaire `CONFIG`.
- `@retry(...)` : Ce décorateur de `tenacity` est appliqué à la méthode `_request`. Il indique que si cette méthode
  échoue, elle doit être réessayée jusqu'à 3 fois (`retry_attempts`), avec un temps d'attente qui augmente entre chaque
  essai (`wait_exponential`). C'est crucial pour rendre le système robuste face aux erreurs réseau ou aux surcharges
  temporaires de l'API.
- `_request` : La méthode privée qui envoie la requête HTTP POST à l'API. Elle gère l'authentification et la structure
  de la requête.
- `chat_completion` et `embeddings` : Des méthodes publiques qui préparent la charge utile (`payload`) spécifique à
  chaque type d'appel (génération de texte ou d'embeddings) et utilisent `_request` pour faire l'appel.

---

## 4. Planner

Le `planner` est le "cerveau" initial du système. Son rôle est de prendre la question de l'utilisateur et de la
décomposer en un plan d'action logique.

```python
def planner(query):
    # ...
```

- Il construit un **prompt** spécifique demandant au modèle de se comporter comme un "planner".
- Il exige une réponse au format **JSON** (`{"plan": ["étape 1", ...]}`). C'est une technique clé pour obtenir des
  sorties structurées et fiables d'un LLM.
- Il inclut une **logique de secours** : si le modèle ne renvoie pas un JSON valide, le code essaie d'extraire le JSON
  du texte, et si tout échoue, il utilise un plan par défaut.

---

## 5. Outils Disponibles (`execute_tool`)

Cette fonction est le "bras armé" du système. Elle permet au modèle d'interagir avec le monde extérieur.

```python
def execute_tool(action):
    # ...
```

- Elle reçoit une chaîne de caractères représentant une action (ex: `"CALC: 2+2"`).
- **`CALC:`** : Si l'action est un calcul, elle extrait l'expression mathématique.
    - **Sécurité** : Elle utilise `eval()` dans un environnement sécurisé (`safe_dict`) qui n'autorise que des
      opérations mathématiques de base, empêchant ainsi l'exécution de code malveillant.
- **`SEARCH:`** : Si l'action est une recherche, elle appelle la fonction `rag_search` pour interroger la base de
  connaissances.
- Elle retourne le résultat sous forme d'une chaîne "OBSERVATION:", imitant le pattern **ReAct (Reasoning and Acting)**.

---

## 6. Implémentation RAG avec ChromaDB

Cette section met en place la **Recherche Augmentée par Génération (RAG)**. Le RAG permet au modèle d'accéder à des
connaissances externes pour répondre à des questions.

- **`MistralEmbeddingFunction`** : Une classe wrapper qui permet à ChromaDB d'utiliser directement l'API d'embedding de
  Mistral pour vectoriser les documents. C'est ici que se trouvait l'erreur principale que nous avons corrigée.
- **Initialisation de ChromaDB** : Crée ou charge une collection (une sorte de table) dans la base de données
  vectorielle.
- **`collection.add(...)`** : Ajoute les documents d'exemple (`sample_docs`) à la base de données après les avoir
  transformés en vecteurs (embeddings).
- **`rag_search(query)`** : Cette fonction prend une requête, la transforme en vecteur, et recherche les documents les
  plus similaires dans ChromaDB. Elle retourne le contenu de ces documents.

---

## 7. Exécution des Étapes (`run_step`)

Cette fonction exécute une seule étape du plan généré par le `planner`.

```python
def run_step(step, context):
    # ...
```

- Elle utilise le pattern **ReAct (Reasoning-Acting)**.
- Elle construit un prompt qui inclut :
    - La description de l'étape à réaliser.
    - Le "contexte" (ce qui a été fait dans les étapes précédentes, stocké dans le `scratchpad`).
- Elle demande au modèle de générer sa **Pensée** (`Thought`) et son **Action** (`Action`).
- Elle analyse la réponse :
    - Si l'action est un appel à un outil (`CALC:` ou `SEARCH:`), elle utilise `execute_tool`.
    - Sinon, la réponse est considérée comme finale pour cette étape.
- Elle retourne un dictionnaire structuré contenant la pensée, l'action et le résultat.

---

## 8. Verifier (`verify`)

Une fonction simple mais importante qui vérifie si une étape a produit un résultat valide.

```python
def verify(step_out):
    # ...
```

- Pour l'instant, elle vérifie simplement que le résultat n'est pas vide.
- Dans un projet plus complexe, on pourrait y ajouter des vérifications de format, de cohérence, ou même des tests
  unitaires.

---

## 9. Modèles Pydantic

Ces classes définissent la structure attendue des données.

```python
class StepOutput(BaseModel):
    # ...
class ReasoningOutput(BaseModel):
    # ...
```

- `StepOutput` : Définit à quoi doit ressembler la sortie d'une seule étape.
- `ReasoningOutput` : Définit la structure de la sortie finale de tout le processus.
- Utiliser Pydantic permet de s'assurer que les données circulant dans le système sont toujours conformes à ce qui est
  attendu, ce qui évite de nombreux bugs.

---

## 10. Logging et Métriques

Cette section met en place un système pour suivre et évaluer les performances du raisonnement.

- `metrics_df` : Un DataFrame `pandas` pour stocker les informations sur chaque exécution (requête, temps pris, etc.).
- `log_reasoning` : Ajoute une nouvelle ligne à ce DataFrame après chaque appel à `perform_reasoning`.
- `display_metrics` : Utilise `rich` pour afficher les métriques dans un tableau bien formaté.

---

## 11. `perform_reasoning` (L'Orchestrateur)

C'est la fonction principale qui orchestre tout le processus de raisonnement.

```python
def perform_reasoning(query, max_tokens=None):
    # ...
```

Elle suit le flux logique :

1. **Planning** : Appelle `planner(query)` pour obtenir le plan.
2. **Exécution** : Itère sur chaque étape du plan et appelle `run_step` pour l'exécuter.
3. **Stockage** : Sauvegarde le résultat de chaque étape dans le `scratchpad`.
4. **Vérification** : Appelle `verify` après chaque étape.
5. **Agrégation** : Combine les résultats de toutes les étapes pour formuler une réponse finale.
6. **Logging** : Appelle `log_reasoning` pour enregistrer les métriques de performance.
7. **Retourne** un dictionnaire complet avec le plan, le `scratchpad` et la réponse finale.

---

## 12. Exemples et Tests

- Les dernières cellules montrent comment utiliser `perform_reasoning` avec différents types de questions (calcul,
  recherche RAG, etc.).
- Elles incluent également des tests unitaires (`pytest`) pour valider que les composants individuels (planner, outils,
  verifier) fonctionnent comme prévu.

J'espère que cette explication détaillée vous sera utile pour mieux comprendre le code et les concepts qui le
sous-tendent !

