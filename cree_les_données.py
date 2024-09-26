#! /usr/bin/python3

from from_import import *

ema = lambda l,K: l.ewm(com=K-1).mean()

#	======================================================================

def montrer(_ls, noms=None):
	if len(_ls) > 1:
		A = int(1 + len(_ls)**.5)
		fig, ax = plt.subplots(1+len(_ls)//A, A)
		for i,ls in enumerate(_ls):
			for l in ls: ax[i//A][i%A].plot(l)
			if noms: ax[i//A][i%A].set_title(noms[i])
			ax[i//A][i%A].grid(True)
		plt.show()
	else:
		for l in _ls[0]: plt.plot(l)
		if noms: plt.set_title(noms[0])
		plt.grid(True)
		plt.show()

#	======================================================================

from csv_vers_panda import binance_btcusdt_15m

df, (interv_dfs, infos, intervalles), (Close, la_Date) = binance_btcusdt_15m(verbose=(__name__ == "__main__"))
print(df)

print(infos)
print(intervalles)

#	============================================================================================

if __name__ == "__main__":
	montrer([
		[interv_dfs[I][nom] for I in intervalles]
		for nom in infos
	], infos)

#	============================================================================================

intervs = (1,)#4,)
N = 8
MAX_I    = max(intervalles)
MAX_ROLL = N * MAX_I

#	la je [t]/[t-I*N], mais je pourrais, [t]/[t-1]

courbe_ematique = lambda l, I: np.array(l[MAX_I+MAX_ROLL:] / np.roll(ema(l, I)[MAX_I:], I*N)[MAX_ROLL:] - 1)
moyennage       = lambda l: l / (np.sum(np.abs(l))/len(l))
#'Open', 'High', 'Low', 'Close', 'qaV', 'trades', 'btcVol', 'usdtVol', 'volume', 'diff_M', 'macd_Close', 'macd_rouge_Close'

E = lambda I: list(map(moyennage, [
	courbe_ematique(interv_dfs[I]['Close'   ], I),
	#courbe_ematique(interv_dfs[I]['trades'  ], I),
	#courbe_ematique(interv_dfs[I]['qaV'     ], I),
	#courbe_ematique(interv_dfs[I]['btcVol'  ], I),
	#courbe_ematique(interv_dfs[I]['usdtVol' ], I),
	#courbe_ematique(interv_dfs[I]['volume'  ], I),
	#courbe_ematique(interv_dfs[I]['diff_M' ], I),
	#courbe_ematique(interv_dfs[I]['macd_Close'  ], I),
	#courbe_ematique(interv_dfs[I]['macd_rouge_Close'  ], I),
]))

Expertises = [E(I) for I in intervs]

expertises    = len(Expertises[0])
nb_expertises = len(intervs) * expertises

print(f"nb_expertises : {nb_expertises}")

#	============================================================================================

if __name__ == "__main__":
	montrer([
		[Expertises[i][e] for i in range(len(intervs))]
		for e in range(expertises)
	])

#	============================================================================================

VALIDATION = 2048
T      = len(Expertises[0][0])
DEPART = N * MAX_I

df = df[-T:].reset_index(drop=True)

print(f'DEPART={DEPART} T={T} VALIDATION={VALIDATION}')
print(f'Train T = {T-DEPART-VALIDATION}')
print(f'Test  T = {VALIDATION}')
print(f'len(df) = {len(df)}')

assert T-DEPART > VALIDATION

U = 4

CLASSES = 2

SORTIES = CLASSES + 0 #+1 (pour un h)

#	============================================================================================

if __name__ == "__main__":
	print("Création des données ...")

	X = np.zeros((T-DEPART, N, nb_expertises))

	for t in tqdm(range(DEPART, T), desc="Création des X : ", ncols=100, bar_format="{l_bar}{bar:20}{r_bar}", colour="green"):
		for i,I in enumerate(intervs):
			for e in range(expertises):
				for n in range(N):
					X[t-DEPART][N-n-1][i*expertises + e] = Expertises[i][e][t - n*I]

	#	===========================================
	
	Y = np.zeros((T-DEPART, CLASSES))

	for t in tqdm(range(DEPART, T), desc="Création des Y : ", ncols=100, bar_format="{l_bar}{bar:20}{r_bar}", colour="green"):
		if t < T-U-1:
			prochains = [df['Close'][t+1+u]/df['Close'][t]-1 for u in range(U)]
			#
			hausse = max([0.0] + [abs(p) for p in prochains if p > 0.0])
			baisse = max([0.0] + [abs(p) for p in prochains if p < 0.0])
			#
			s = np.array([hausse, baisse])
			s = s / sum(s)
			#
			Y[t-DEPART][0] = s[0]
			Y[t-DEPART][1] = s[1]
		else:
			Y[t-DEPART][0] = 0.0
			Y[t-DEPART][1] = 0.0

	#Y = np.array([[100*(df['Close'][t+1]/df['Close'][t]-1 if t!=T-1 else 0.0)] for t in range(DEPART, T)])

	#	================== Montrer ================

	plt.imshow(X[0]); plt.show()

	montrer([
		[ [X[0][n][i] for n in range(N)] ]
		for i in range(nb_expertises)
	])

	#	===========================================

	print("Séparation de train et test ...")
	X_train, X_test = X[:-VALIDATION], X[-VALIDATION:]
	Y_train, Y_test = Y[:-VALIDATION], Y[-VALIDATION:]
	print("Données écrites !")

	#	===========================================

	print(f"\nX_train[0] = \n{X_train[0]}\n\nY_train[0] = \n{Y_train[0]}\n")

	#	===========================================

	print('\n'.join([f'{i} : {eval(i).shape}' for i in ('X_train', 'Y_train', 'X_test', 'Y_test')]))

	for la_liste in 'X_train', 'Y_train', 'X_test', 'Y_test':
		with open(la_liste, 'wb') as co:
			arr = eval(la_liste).flatten()
			co.write(st.pack('f'*len(arr), *arr))