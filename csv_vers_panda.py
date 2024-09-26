import numpy as np
import pandas as pd

ema = lambda l,K: l.ewm(com=K-1).mean()

def macd(l, I=1):
	ema12 = ema(l, 12*I)
	ema26 = ema(l, 26*I)
	return ema12 - ema26

def ligne_rouge(l, I):
	ema9 = ema(l, 9*I)
	return l - ema9

############################################################################################################
############################################################################################################

def interv_iser(df, infos, I):
	ret = pd.DataFrame()
	for info in infos: ret[info] = ema(df[info], 1*I)
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
	
	df = pd.read_csv(fichier)#[-(16 + 26*4 + 8*4 + 8*4 + 4 + 1):]
	print(df)

	#	+++
	df['volume'] = df['btcVol'] - df['usdtVol'] / df['Close']
	df['diff_M'] = (df['High'] - df['Close'])/(df['High'] - df['Low'])
	#	+++
	
	#	========================================	#
	#	================ Interv ================	#
	#	========================================	#

	infos  = 'Open', 'High', 'Low', 'Close', 'qaV', 'trades', 'btcVol', 'usdtVol', 'volume', 'diff_M'

	intervalles = 1, 4,# 16, 64, 256

	interv_dfs = {
		I : interv_iser(df, infos, I=I) for I in intervalles
	}

	#	========================================	#
	#	========= Analyses Techniques ==========	#
	#	========================================	#

	for I in intervalles:
		interv_dfs[I]['macd_Close'      ] = macd(interv_dfs[I]['Close'], I=I)
		interv_dfs[I]['macd_rouge_Close'] = ligne_rouge(interv_dfs[I]['macd_Close'], I=I)

	DEPART_analyses_techniques = max(intervalles) * 26

	for I in intervalles:
		interv_dfs[I] = interv_dfs[I][DEPART_analyses_techniques:].reset_index(drop=True)
	
	df = df[DEPART_analyses_techniques:].reset_index(drop=True)

	infos  = infos + ('macd_Close', 'macd_rouge_Close')

	return df, (interv_dfs, infos, intervalles), (close, date)