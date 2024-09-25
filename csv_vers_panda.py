import numpy as np
import pandas as pd

def macd(l):
	ema12 = l.ewm(com=12).mean()
	ema26 = l.ewm(com=26).mean()
	return ema12 - ema26

def ligne_rouge(l):
	ema9 = l.ewm(com=9).mean()
	return l - ema9

############################################################################################################
############################################################################################################

def interv_iser(df, infos, I):
	ret = pf.DataFrame()
	for info in infos: ret[info] = df[info].ewm(com=1.0 * I).mean()
	return ret

############################################################################################################
############################################################################################################

def binance_btcusdt_15m(verbose=False):
	colonnes = "OUnix,CUnix,ODate,CDate,Symbol,Open,High,Low,Close,qaV,trades,btcVol,usdtVol"
	fichier  = './binance_btcusdt_15m.csv'
	date     = 'Date'
	close    = 'Close'
	
	df = pd.read_csv(fichier)
	print(df)

	#	========================================	#
	#	============= Extraction ===============	#
	#	========================================	#
	
	#	Si des valeurs se répètent
	for colonne in 'Open', 'High', 'Low', 'Close', 'qaV', 'trades', 'btcVol', 'usdtVol':
		print(f" ############ Verification des {colonne} ############ ")
		for i in range(1, len(df)-1):
			if df[colonne].iloc[i-1] == df[colonne].iloc[i]:
				if verbose: print(f"\033[93mDeux mêmes valeurs    (Odata={df['ODate'][i]}): {list(df[colonne].iloc[i-2:i+2])}\033[0m")
			if df[colonne].iloc[i-1] == df[colonne].iloc[i]:
				if df[colonne].iloc[i] == df[colonne].iloc[i+1]:
					if verbose: print(f"\033[91mTrop de mêmes valeurs (Odata={df['ODate'][i]}): {list(df[colonne].iloc[i-2:i+6])}\033[0m")
					plus = 1
					while not df[colonne].iloc[i+plus] != df[colonne].iloc[i]:
						plus += 1
					for j in range(1+plus):
						df.loc[i+j, colonne] = df[colonne].iloc[i-1] + (1+j)*(df[colonne].iloc[i+plus+1]-df[colonne].iloc[i-1])/(plus+1+1)
					if verbose: print(f"Nouvelle maj                                       {list(df[colonne].iloc[i-2:i+6])}")
				else:
					df.loc[i, colonne] = (df[colonne].iloc[i-1] + df[colonne].iloc[i+1])/2

	#	=== Informations complémentaires ===
	df['volume'] = df['btcVol'] - df['usdtVol'] / df['Close']
	
	#	========================================	#
	#	================ Interv ================	#
	#	========================================	#

	infos  = 'Open', 'High', 'Low', 'Close', 'qaV', 'trades', 'btcVol', 'usdtVol', 'volume'

	intervalles = 1, 4, 16, 64, 256

	interv_dfs = {
		I : interv_iser(df, infos, I=I) for I in intervalles
	}

	#	========================================	#
	#	========= Analyses Techniques ==========	#
	#	========================================	#

	for I in intervalles:
		interv_dfs[I]['macd_Close'      ] = macd(interv_dfs[I]['Close'])
		interv_dfs[I]['macd_rouge_Close'] = ligne_rouge(interv_dfs[I]['macd_Close'])

	return df, interv_dfs, close, date