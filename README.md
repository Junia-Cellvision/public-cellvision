# cellvision

## MEA pris en charge 

Familles :
- MEA_60MEA_200
    - MEA_60MEA_200_30
    - MEA_60MEA_200_10
- MEA_60MEA_500
    - MEA_60MEA_500_30
    - MEA_60MEA_500_10
- MEA_60HexaMEA_40_10

## Développement

### Requirements

- git
- mamba
- ssh
- se connecter au réseau "JUNIA_STUDENTS"
- accès au serveur junia local

### Installation

Le repo contient un submodule. 

#### Cloner avec les submodules

```bash
git clone --recurse-submodules git@github.com:Junia-Cellvision/public-cellvision.git
```

Ou en https

```bash
git clone --recurse-submodules https://github.com/Junia-Cellvision/public-cellvision.git
```

#### Mettre à jour les submodules

```bash
git submodule update --init --force --remote
```

#### Ajouter les submodules si vous avez cloné sans

1. Initialiser les submodules

```bash
git submodule init
```

2. Mettre à jour les submodules

```bash
git submodule update
```


### Mise en place

1. Créer un environnement mamba

```bash
mamba env create -f env.yml
```

2. Activer l'env

```bash
mamba activate cellvision
```

3. Récupérer les données depuis le serveur

```bash
sh pull_data.sh
```

4. Lancer le script 
```bash
python3 cellvision.py
```

5. Accéder à l'interface web

### Ajouter une dépendance python

1. Installer la dépendance

```bash
pip install <dependency>
```

2. Ajouter la dépendance à l'environnement

```bash
mamba env export --from-history > env.yml
```

### Installer les dépendances dans l'environnement

```bash
mamba env update -f env.yml --prune
```