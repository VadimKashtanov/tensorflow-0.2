#! /usr/bin/python3

from from_import import *

#	============================================================	#

from cnn import *

############################    Modèle     #########################

from sys import argv

model = load_model(argv[1])

############################    Tester     #########################

prixs_test  = np.array(df['Close'][-VALIDATION:])
prixs_train = np.array(df['Close'][:-VALIDATION])

Y_pred_train = model.predict(X_train)
Y_pred_test  = model.predict(X_test )

print(f'y_test = {Y_test[0:50]}')
print(f'y_pred = {Y_pred_test[0:50]}')

#	Validation[VALIDATION]
Y0_test = list(w0 for w0,w1  in Y_test)
Y1_test = list(w1 for w0,w1  in Y_test)
s0_test = list(s0 for s0,s1, in Y_pred_test)
s1_test = list(s1 for s0,s1, in Y_pred_test)

#	Train:[T-DEPART-VALIDATION]
Y0_train = list(w0 for w0,w1, in Y_train)
Y1_train = list(w1 for w0,w1  in Y_train)
s0_train = list(s0 for s0,s1, in Y_pred_train)
s1_train = list(s1 for s0,s1, in Y_pred_train)

####################################################################

fig, ax = plt.subplots(2)

#	--
ax[0].plot(Y0_test, '-',  label='Valeurs réelles Y0',  color='green')
ax[0].plot(Y1_test, '-',  label='Valeurs réelles Y1',  color='red'  )
ax[0].plot(s0_test, '--', label='Prédictions S0',      color='green')
ax[0].plot(s1_test, '--', label='Prédictions S1',      color='red'  )
ax[0].grid(True)
ax[0].legend()

#	Flèches
p = normer(prixs_test)
plt.plot(p, label='Prixs', color='black')
#
for t in range(VALIDATION):
	pred       = s0_test[t] - s1_test[t]
	point_prix = p[t]
	if pred >= 0.0: ax[1].plot([t, t+U], [point_prix, point_prix + 0.05], 'g')
	else:           ax[1].plot([t, t+U], [point_prix, point_prix - 0.05], 'r')
ax[1].grid(True)
ax[1].legend()

#	--
plt.show()

####################################################################
############################    Tester     #########################
####################################################################

moyenne = lambda x: (0 if len(list(x))==0 else sum(list(x))/len(list(x)))
eq1     = lambda x: (1 if signe(x)==1 else 0)

def informations(w,y,h):
	assert all(len(w)==len(a) for a in (w,y,h))
	#
	val_moy =   moyenne([int(signe(y[i])==signe(w[i]))        for i in range(len(w))])
	dst_moy = 1-moyenne([abs((tanh(y[i])-signe(w[i]))/2)      for i in range(len(w))])
	inv_bon = 1-moyenne([abs(eq1(y[i]*w[i])-logistique(h[i])) for i in range(len(w))])
	inv_tot =   moyenne([logistique(h[i])                     for i in range(len(w))])
	#
	return val_moy, dst_moy, inv_bon, inv_tot

prixs = fermetures = list(df['Close'][-VALIDATION:])
dates =              list(df['ODate'][-VALIDATION:])

signe = lambda x: (1 if x>=0 else -1)

print(" Y_test     Y_pred(signe)   Y_pred(invest)   Prixs     %Prochain   Date")
for t in range(VALIDATION):
	if t == 40: print("...\n...\n...")
	if 40 < t < VALIDATION-40: continue
	#
	w,y,h = Y_test[t], predictions[t], investissements[t]
	#
	p1   = (0 if t==VALIDATION-1 else prixs[t+1])
	p0   = prixs[ t ]
	#
	ch = p1/p0 - 1
	#
	print("%+3.3f | %+3.3f(%+4.3f) %+4.3f(%+4.3f) | %+7.4f$ %6s%% | %s %s" % (
		w,
		#
		tanh(y),y,
		logistique(h),h,
		#
		prixs[t],
		(str(ch*100)[:5] if t != VALIDATION-1 else '?'),
		#
		dates[t],
		(f'\033[92m +$ \033[0m' if signe(w) == signe(y) else '\033[91m -$ \033[0m'),
	))

print("\n === Validation === ")
val_moy, dst_moy, inv_bon, inv_tot = informations(
	w=Y_test,
	y=predictions,
	h=investissements
)
print(f"[Validation] (y) Validité       moyenne : {'%7s' % str(round(100*val_moy,4))}%")
print(f"[Validation] (y) Distance       moyenne : {'%7s' % str(round(100*dst_moy,4))}%")
print(f"[Validation] (h) Invest % bonne moyenne : {'%7s' % str(round(100*inv_bon,4))}%")
print(f"[Validation] (h) Investissement moyen   : {'%7s' % str(round(100*inv_tot,4))}%")
print(" ============= \n")

print("\n === Train === ")
val_moy_train, dst_moy_train, inv_bon_train, inv_tot_train = informations(
	w=Y_train,
	y=predictions_train,
	h=investissements_train,
)
print(f"[Train] (y) Validité       moyenne : {'%7s' % str(round(100*val_moy_train,4))}%")
print(f"[Train] (y) Distance       moyenne : {'%7s' % str(round(100*dst_moy_train,4))}%")
print(f"[Train] (h) Invest % bonne moyenne : {'%7s' % str(round(100*inv_bon_train,4))}%")
print(f"[Train] (h) Investissement moyen   : {'%7s' % str(round(100*inv_tot_train,4))}%")
print(" ============= \n")

##################################################################################
############################          Gains      #################################
##################################################################################

format_f = lambda f: f'\033[{(91 if f<100 else 92)}m' + ('%7s$' % str(round(float(f),2))) + '\033[0m'

fig, ax = plt.subplots(2,2)

h_max = max(np_logistique(investissements))

print("          u0     u1     u2     u3")

for L in [1,10,25,50,125]:
	u0 = 100
	u1 = 100
	u2 = 100
	u3 = 100
	_u0 = []
	_u1 = []
	_u2 = []
	_u3 = []
	for t in range(VALIDATION-1):
		w,y,h = Y_test[t], tanh(predictions[t]), logistique(investissements[t])
		p1   = prixs[t+1]
		p0   = prixs[ t ]
		#
		u0 += u0 * L *       y  * (p1/p0-1) * h# / h_max
		u1 += u1 * L *       y  * (p1/p0-1) * 1
		u2 += u2 * L * signe(y) * (p1/p0-1) * h# / h_max
		u3 += u3 * L * signe(y) * (p1/p0-1) * 1
		#
		if u0 < 0: u0 = 0
		if u1 < 0: u1 = 0
		if u2 < 0: u2 = 0
		if u3 < 0: u3 = 0
		#
		_u0 += [u0]
		_u1 += [u1]
		_u2 += [u2]
		_u3 += [u3]
	#
	ax[0][0].plot(_u0, label=f'x{L}')
	ax[0][1].plot(_u1, label=f'x{L}')
	ax[1][0].plot(_u2, label=f'x{L}')
	ax[1][1].plot(_u3, label=f'x{L}')
	#
	print(f"L{'%4s' % L} : " + ''.join(list(map(format_f, [u0,u1,u2,u3]))))

#
ax[0][0].set_title('      y *h'); ax[0][0].legend()
ax[0][1].set_title('      y *1'); ax[0][1].legend()
ax[1][0].set_title('signe(y)*h'); ax[1][0].legend()
ax[1][1].set_title('signe(y)*1'); ax[1][1].legend()
#
print(f"\nDernière prédiction : {predictions[-1]}")
#
plt.show()