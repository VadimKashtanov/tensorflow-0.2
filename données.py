import numpy as np
import pandas as pd

def macd(l):
	ema12 = l.ewm(com=12).mean()
	ema26 = l.ewm(com=26).mean()
	return ema12 - ema26

def ligne_rouge(l):
	ema9 = l.ewm(com=9).mean()
	return l - ema9

############################################################################################################""
############################################################################################################""
############################################################################################################""

def binance_btcusdt_15m(verbose=False):
	colonnes = "OUnix,CUnix,ODate,CDate,Symbol,Open,High,Low,Close,qaV,trades,btcVol,usdtVol"
	fichier  = './binance_btcusdt_15m.csv'
	date     = 'Date'
	close    = 'Close'
	
	#	Inverser le fichier
	df = pd.read_csv(fichier)
	print(df)
	
	#	Si des valeurs se répètent
	for i in range(1, len(df)-1):
		if df['Close'].iloc[i-1] == df['Close'].iloc[i]:
			if verbose: print(f"\033[93mDeux mêmes valeurs    (Odata={df['ODate'][i]}): {list(df['Close'].iloc[i-2:i+2])}\033[0m")
		if df['Close'].iloc[i-1] == df['Close'].iloc[i]:
			if df['Close'].iloc[i] == df['Close'].iloc[i+1]:
				if verbose: print(f"\033[91mTrop de mêmes valeurs (Odata={df['ODate'][i]}): {list(df['Close'].iloc[i-2:i+6])}\033[0m")
				plus = 1
				while not df['Close'].iloc[i+plus] != df['Close'].iloc[i]:
					plus += 1
				for j in range(1+plus):
					df.loc[i+j, 'Close'] = df['Close'].iloc[i-1] + (1+j)*(df['Close'].iloc[i+plus+1]-df['Close'].iloc[i-1])/(plus+1+1)
				if verbose: print(f"Nouvelle maj                                       {list(df['Close'].iloc[i-2:i+6])}")
			else:
				df.loc[i, 'Close'] = (df['Close'].iloc[i-1] + df['Close'].iloc[i+1])/2
	#
	#
	DEPART = 26
	#
	#	=== Analyses techniques ===
	#
	df['macd_Close']       = macd(df['Close'])
	df['macd_rouge_Close'] = ligne_rouge(df['macd_Close'])
	#
	#
	return df[DEPART:].reset_index(), close, date