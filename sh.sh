rm meilleur_model.h5.keras dernier__model.h5.keras

set -e
clear

#	--- Données Binance CSV ---
#python3 binance_données.py 15m 50000 binance_btcusdt_15m.csv

#	--- Cree les données ---
#python3 cree_les_données.py

#	--- CNN ---
python3 cnn.py
python3 tester_la_validitée.py meilleur_model.h5.keras