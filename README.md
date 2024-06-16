
# DRONE-RACING-SIMULATOR

Ein Simulator für Reinforcement basiertes Drohnenracing Training mit hoch akkuraten Dynamiken und OPENAI BASELINES Implementation für den ML Algorithmus.



## Installation

Dieses Projekt ist auf der Basis von STABLE BASELINES 2 geschrieben worden. Diesbezüglich wird Python <= 3.6 erfordert. Somit ist es am einfachsten eine Art von virtueller Umgebung aufzusetzen (bsp. [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/)). In Zukunft, falls Zeit übrig bleibt, werde ich die Software auf Basis von STABLE BASELINES 3 umschreiben (Pytorch anstelle von Tensorflow). In Bezug auf Leistung ist aber kein Unterschied festzustellen, weswegen der Wechsel vorerst nicht nötig ist.

Anschliessend soll eine virtuelle Umgebung mit Python=3.6 gestartet werden.

```bash
  conda create --name SIMULATOR python=3.6
  conda activate SIMULATOR
```

Folgend kann das Github Repository geklont werden in ein beliebiges Verzeichnis:
```bash
  git clone https://github.com/maurice-mxm/drone-racing-simulator.git
```

Um von dem Code Gebrauch machen zu können, müssen noch zwei Abhängigkeit von OPENAI BASELINES installiert werden:
```bash
  pip install gym==0.11, numpy, stable_baselines==2.10.1, tensorflow-gpu==1.14, scikit-build
```

Anschliessend müssen noch die zwei Verzeichnisse SIM und BASELINES zum $PYTHONPATH hinzugefügt werden (dies muss jedes Mal bei neuen Start des Environments geschehen):
```bash
  export PYTHONPATH="/.../sim/:/.../baselines
```
für (...) muss der absolute Pfad zum Verzeichnis stehen. 
## Usage/Examples

Das Repository ist in mehrere Unterverzeichnisse gegliedert:
- BASELINES: Implementation von Stable Baselines mit ein paar kleineren Abänderungen
- SIM: Grosse Übereinheit der Simulation
    * Run: Hier wird der Befehl zum Starten des Training bzw. zur Evaluation gegeben
    * vec_env: Instanziierung der Vektoren Umgebungen
    * dynamics: Dynamiken für das simulieren der Drohne. (beschleunigt mit NUMBA)

zur Benützung gehen Sie bitte ins Verzeichnis Run. Mit der Datei "learn.py" kann das Training bzw. die Evaluation gestartet werden. Dies mittels:

```bash
python3 learn.py
```

Das trainierte Produkt wird dann im Verzeichnis "models/" gespeichert. Um so eines zu testen müssen Sie den Namen des ".zip" Verzeichnisses kopieren und bei "learn.py" in den Dictonary einfügen rsp. Training auf False setzten. (hier folgt nacht, dass das direkt von der Commandozeile gemacht werden kann).

## Authors

- [@maurice-mxm](https://www.github.com/maurice-mxm)


## Acknowledgements

 - [STABLE BASELINES](https://github.com/openai/baselines)


