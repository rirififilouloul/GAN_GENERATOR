
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#                              Imports                              #
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

import sys
print(sys.version)
import cv2
import keras
import os
import numpy as np
from tqdm import tqdm
from glob import glob
import tensorflow as tf
import tensorflow.keras.optimizers
from tensorflow.keras.models import Sequential, load_model, model_from_yaml
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Reshape, UpSampling2D

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#                CHARGER + PREPARER IMAGE REELLES DATA              #
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
optimizer = tf.keras.optimizers.legacy.Adam(lr=0.0002, beta_1=0.5)

images_vraies =[]
path="dataSet/archive/training_set/cats/"
noms_image = glob(path)
compteur = 1

images_load = [img for img in os.listdir(path) if img.endswith(".jpg")]


for nom in images_load:

	nom = path + nom

	image = cv2.imread(nom)

	image = cv2.resize(image, (128,128))
	image = image.astype("float32")	
	image = (image-127.5)/127.5
	images_vraies.append(image)

	compteur+=1

images_vraies = np.array(images_vraies)


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#                        Créer Discriminateur                       #
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
if os.path.exists("Discriminateurs/discriminateur_epoch900.h5"):
	discriminateur = load_model("Discriminateurs/discriminateur_epoch900.h5")
else:
	discriminateur = Sequential()
	discriminateur._name="Discriminateur"

	discriminateur.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128,128,3)))
	discriminateur.add(MaxPooling2D(pool_size=(2, 2)))

	discriminateur.add(Conv2D(32, kernel_size=(3, 3), activation='relu' ))
	discriminateur.add(MaxPooling2D(pool_size=(2, 2)))

	discriminateur.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))
	discriminateur.add(MaxPooling2D(pool_size=(2, 2)))

	discriminateur.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
	discriminateur.add(MaxPooling2D(pool_size=(2, 2)))

	discriminateur.add(Flatten())

	discriminateur.add(Dense(128, activation = "relu" ))
	discriminateur.add(Dense(1, activation = "sigmoid" ))

	discriminateur.summary()

#Sauvegarde de l'architecture du model dans log.txt :
with open('log.txt','w') as log:
    discriminateur.summary(print_fn=lambda x: log.write(x + '\n'))
    log.writelines(["\n\n"])
    log.writelines(["#######################################################################\n"])
    log.writelines(["#######################################################################\n\n\n"])

discriminateur.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#                       Créer Générateur                            #
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

if os.path.isfile("Generateurs/generateur_epoch900.h5"):
	generateur = load_model("Generateurs/generateur_epoch900.h5")
else:
	generateur = Sequential()
	generateur._name="Generateur"

	generateur.add(Dense(16*16*256, activation='relu' , input_shape=(100,)))
	generateur.add(Reshape((16, 16, 256))) # 16x16

	generateur.add(UpSampling2D(size=(2, 2))) # 16x16 -> 32x32
	generateur.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))

	generateur.add(UpSampling2D(size=(2, 2))) # 32x32 -> 64x64
	generateur.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))

	generateur.add(UpSampling2D(size=(2, 2))) # 64x64 -> 128x128
	generateur.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))

	generateur.add(Conv2D(3, kernel_size=(2, 2),padding='same', activation='tanh' ))
	generateur.summary()

#Sauvegarde de l'architecture du model dans log.txt :
with open('log.txt','a') as log:
    generateur.summary(print_fn=lambda x: log.write(x + '\n'))
    log.writelines(["\n\n"])
    log.writelines(["#######################################################################\n"])
    log.writelines(["#######################################################################\n\n\n"])   
#On n'entraine pas le générateur tout seul du coup on ne le compile pas

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#                          Créer COMBO                              #
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#


combo = Sequential()
combo._name="Combo"
combo.add(generateur)
combo.add(discriminateur)

discriminateur.trainable = False
combo.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

combo.summary()

#Sauvegarde de l'architecture du model dans log.txt :
with open('log.txt','a') as log:
    combo.summary(print_fn=lambda x: log.write(x + '\n'))

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#                           Entrainer                               #
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
iterations = 1001
demi_batch= 500

#plusieurs itération
for iteration in range(iterations):
	os.system('cls' if os.name == 'nt' else 'clear')# Clean la console à chaque boucle
	print()
	print(" ##########################")
	print("   Boucle n°"+str(iteration)+"/"+str(iterations))
	print(" ##########################")
	################################################
	# créer pack de données pour le discriminateur #
	################################################

	# etape 1 : prendre des bonnes images
	# etape 2 : créer les labels (1 pour vrai) pour les bonnes images du dataset
	# etape 3 : generer des mauvaises images
	# etape 4 : créer les labels (0 pour faux) pour les mauvaises images générés

	x = []
	y = []

	# etape 1 : prendre des bonnes images
	print(images_vraies.shape)
	images_bonnes = images_vraies[np.random.randint(0, images_vraies.shape[0], size=demi_batch)]
	# etape 2 : créer les labels (1 pour vrai) pour les bonnes images du dataset
	labels_bonnes = np.ones(demi_batch) #un tableau avec 1000 fois le label 1
	# etape 3 : generer des mauvaises images
	bruit = np.random.normal(0, 1, size=[demi_batch,100]) # 1000 tableaux de 100 nombres aléatoires
	images_mauvaises = generateur.predict(bruit) # milles images générées
	# etape 4 : créer les labels (0 pour faux) pour les mauvaises images générés
	labels_mauvaises = np.zeros(demi_batch)

	x = np.concatenate([images_bonnes,images_mauvaises])
	y = np.concatenate([labels_bonnes,labels_mauvaises])

	############################
	# entrainer discriminateur #
	############################

	discriminateur.trainable = True
	print()
	print("Entrainement du discriminateur :")
	print()
	discriminateur.fit(x,y, epochs = 1, batch_size=32)


	#######################################
	# créer pack de données pour le combo #
	#######################################

	# generer du bruit
	bruit = np.random.normal(0, 1, size=[demi_batch,100]) # 1000 tableaux de 100 nombres aléatoires
	# créer les labels 1
	labels_combo = np.ones(demi_batch)


	###################
	# entrainer combo #
	###################
	print()
	print("Entrainement du Générateur :")
	print()
	discriminateur.trainable = False

	#Il y a une erreur à cette ligne corrige la

	combo.fit(bruit, labels_combo, epochs = 1, batch_size=32,validation_split=0.2)

	######################################
	# Généreration d'image et sauvegarde #
	######################################
	if iteration % 5 == 0 :
		bruit = np.random.normal(0, 1, size=[1, 100])
		print("Génération d'image...")
		print()
		image = generateur.predict(bruit)
		image = (image*127.5)+127.5
		image = image.astype("uint8")
		image = image.reshape((128,128,3))
		imname = "testchat_"+str(iteration)+".png"
		cv2.imwrite("Images/" +imname, image)


	#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
	#          Sauvegarde des Models toutes les 500 boucles             #
	#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
	if iteration % 100 == 0 and iteration != 0 :
		print()
		print("Sauvegarde des models...")
		print()
		discriminateur.save("Discriminateurs/discriminateur_epoch"+str(iteration)+".h5")
		generateur.save("Generateurs/generateur_epoch"+str(iteration)+".h5")
		#evaluer le model
		score = discriminateur.evaluate(x,y, verbose=0)

		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

