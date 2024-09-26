#! /usr/bin/python3

from from_import import *

ema = lambda l,K: l.ewm(com=K-1).mean()

#	======================================================================

def montrer(*_ls):
	A = int(1 + len(_l)**.5)
	fig, ax = plt.subplots(1+len(_l)//A, A)
	for i,ls in enumerate(_ls):
		for l in ls:
			ax[i//A][i%A].plot(l, label=f'{i}')
		ax[i//A][i%A].legend()
		ax[i//A][i%A].grid(True)
	plt.show()

#	======================================================================

from csv_vers_panda import binance_btcusdt_15m

df, (interv_dfs, infos, intervalles), (Close, la_Date) = binance_btcusdt_15m(verbose=(__name__ == "__main__"))
print(df)

print(infos)
print(intervalles)

if __name__ == "__main__":
	montrer([
		[interv_dfs[I][nom] for I in intervalles]
		for nom in infos
	])

#	============================================================================================

intervs = (1,)
N = 8
MAX_I    = 8
MAX_ROLL = N*MAX_I

courbe_ematique = lambda l, K: (l[MAX_ROLL:] / np.roll(ema(l, K), MAX_ROLL)[MAX_ROLL:] - 2)

E = lambda I: [
	courbe_ematique(interv_dfs[I]['Close'   ], I),
	courbe_ematique(interv_dfs[I]['trades'  ], I),
	courbe_ematique(interv_dfs[I]['qaV'     ], I),
	#courbe_ematique(interv_dfs[I]['btcVol' ], I),
	#courbe_ematique(interv_dfs[I]['usdtVol'], I),
	courbe_ematique(interv_dfs[I]['volume'  ], I),
]

Expertises = [expertise for I in intervs for expertise in E(I)]

nb_expertises = len(Expertises)

print(f"nb_expertises : {nb_expertises}")

#	============================================================================================

if __name__ == "__main__":
	fig,ax = plt.subplots(2,2)
	L = 300
	f = lambda x: x*(x>0)
	for j,e in enumerate(Expertises): ax[0][0].plot(e['l'][-L:], label=f'{j}')
	for j,e in enumerate(Expertises): ax[0][1].plot(np.convolve(e['l'][-L:], np.array([-1, -1, 3, 0, 0]), "same"), label=f'{j}')
	for j,e in enumerate(Expertises): ax[1][1].plot(f(np.convolve(e['l'][-L:], np.array([-1, -1, 3, 0, 0]), "same")), label=f'{j}')
	ax[0][0].legend()
	ax[1][0].plot(np.array(df['Close'][MAX_ROLL:][-L:]))
	for i in range(2):
		for j in range(2):
			ax[i][j].grid(True)
	plt.show()
	#exit(0)
	#
	A = int(1+(len(Expertises))**.5)
	fig, ax = plt.subplots(A,A)
	for i,(l,_) in enumerate(Expertises): ax[i//A][i%A].plot(l)
	plt.show()

	montrer([
		[interv_dfs[I][nom] for I in intervalles]
		for e in Expertises
	])

#	============================================================================================

VALIDATION = 2048
T      = len(Expertises[0]['l'])
DEPART = N * MAX_I

assert T-DEPART > VALIDATION

print(f'DEPART={DEPART} T={T} VALIDATION={VALIDATION}')
print(f'Train T = {T-DEPART-VALIDATION}')
print(f'Test  T = {VALIDATION}')

U = 4

CLASSES = 2

SORTIES = CLASSES + 0 #+1 (pour un h)

#	============================================================================================

if __name__ == "__main__":
	print("Création des données ...")

	X = np.zeros((T-DEPART, N, nb_expertises))

	for t in tqdm(range(DEPART, T), desc="Création des X : ", ncols=100, bar_format="{l_bar}{bar:20}{r_bar}", colour="green"):
		ie = 0
		for l,Is in Expertises:
			for I in Is:
				for n in range(N):
					X[t-DEPART][N-n-1][ie] = l[t - n*I]
				ie += 1

	#	===========================================
	
	Y = np.zeros((T-DEPART, 2))

	for t in tqdm(range(DEPART, T), desc="Création des Y : ", ncols=100, bar_format="{l_bar}{bar:20}{r_bar}", colour="green"):
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

	print(t, T, len(df['Close']))
	print(prochains)
	print(Y[-10:])

	#Y = np.array([[100*(df['Close'][t+1]/df['Close'][t]-1 if t!=T-1 else 0.0)] for t in range(DEPART, T)])

	#	================== Montrer ================

	plt.imshow(X[0]); plt.show()

	A = int(1+(nb_expertises)**.5)
	fig, ax = plt.subplots(A,A)
	for i in range(nb_expertises): ax[i//A][i%A].plot([X[0][n][i] for n in range(N)])
	plt.show()

	#	===========================================

	print("Séparation de train et test ...")
	X_train, Y_train = X[:-VALIDATION], Y[:-VALIDATION]
	X_test , Y_test  = X[-VALIDATION:], Y[-VALIDATION:]
	print("Données écrites !")

	#	===========================================

	print(f"\nX_train[0] = \n{X_train[0]}\n\nY_train[0] = \n{Y_train[0]}\n")

	#	===========================================

	print('\n'.join([f'{i} : {eval(i).shape}' for i in ('X_train', 'Y_train', 'X_test', 'Y_test')]))

	for la_liste in 'X_train', 'Y_train', 'X_test', 'Y_test':
		with open(la_liste, 'wb') as co:
			arr = eval(la_liste).flatten()
			co.write(st.pack('f'*len(arr), *arr))