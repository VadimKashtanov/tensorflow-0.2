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
	ret = pd.DataFrame()
	for info in infos: ret[info] = df[info].ewm(com=1.0 * I).mean()
	return ret

############################################################################################################
############################################################################################################

def binance_btcusdt_15m(verbose=False):
	#	========================================	#
	#	================ Extraire ==============	#
	#	========================================	#

	colonnes = "OUnix,CUnix,ODate,CDate,Symbol,Open,High,Low,Close,qaV,trades,btcVol,usdtVol"
	fichier  = './binance_btcusdt_15m.csv'
	date     = 'Date'
	close    = 'Close'
	
	df = pd.read_csv(fichier)
	print(df)

	#	+++
	df['volume'] = df['btcVol'] - df['usdtVol'] / df['Close']
	#	+++
	
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

	return df, (interv_dfs, infos, intervalles), (close, date)