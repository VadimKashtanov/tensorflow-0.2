import pandas as pd

def reparer_un_csv(fichier, colonnes):
	df = pd.read_csv(fichier)
	
	for colonne in colonnes:
		print(f" ############ Verification des {colonne} ############ ")
		for i in range(1, len(df)-1):
			if df[colonne].iloc[i-1] == df[colonne].iloc[i]:
				print(f"\033[93mDeux mêmes valeurs    (Odata={df['ODate'][i]}): {list(df[colonne].iloc[i-2:i+2])}\033[0m")
				#
				if df[colonne].iloc[i] == df[colonne].iloc[i+1]:
					print(f"\033[91mTrop de mêmes valeurs (Odata={df['ODate'][i]}): {list(df[colonne].iloc[i-2:i+6])}\033[0m")
					plus = 1
					while not df[colonne].iloc[i+plus] != df[colonne].iloc[i]:
						plus += 1
					for j in range(1+plus):
						df.loc[i+j, colonne] = df[colonne].iloc[i-1] + (1+j)*(df[colonne].iloc[i+plus+1]-df[colonne].iloc[i-1])/(plus+1+1)
					print(f"Nouvelle maj                                       {list(df[colonne].iloc[i-2:i+6])}")
				else:
					df.loc[i, colonne] = (df[colonne].iloc[i-1] + df[colonne].iloc[i+1])/2

			if df[colonne].iloc[i-1] == 0:
				print(f"\033[92mValeure nulle        (Odata={df['ODate'][i]}): {list(df[colonne].iloc[i-2:i+2])}\033[0m")

	df.to_csv(fichier, index=False)

if __name__ == "__main__":
	from sys import argv

	csv, colonnes = argv[1], argv[2].split('-')

	reparer_un_csv(csv, colonnes)