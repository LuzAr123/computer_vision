#cvlib

#Librerías
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

###INICIA MÉTODO imgview()###
def imgview(img,axis=False):
	"""Devuelve un np.array visualmente
	Args:
		img(numpy array): fuente de la imagen
	Returns:
		img (list): imagen en gráfica
	"""
	fig=plt.figure(figsize=(5,5))
	sub=fig.add_subplot(111)
	if len(img.shape)==3:
		sub.imshow(img,extent=None)
	else:
		sub.imshow(img,extent=None,cmap='gray',vmin=0,vmax=255)
	plt.show()
###FIN###

###INICIA MÉTODO hist()###
def hist(img):
	"""Devuelve la imagen y su histograma
	Args:
		img(numpy array): Fuente de la imagen
	Returns:
		img (list): Imagen en gráfica
		hist (list):Histograma de la imagen
	"""

	fig=plt.figure(figsize=(10,5))

	ax1=fig.add_subplot(121)
	if len(img.shape)==3:
		ax1.imshow(img,extent=None)
	else:
		ax1.imshow(img,extent=None,cmap='gray',vmin=0,vmax=255)

	ax2=fig.add_subplot(122)
	if len(img.shape)==3:
		colors = ['r','g','b']
		for i, color in enumerate(colors):
			histr = cv.calcHist([img],[i],None,[256],[0,256])
			ax2.plot(histr, c=color, alpha=0.9)
			x = np.arange(0.0, 256, 1)
	else:
		histr = cv.calcHist([img],[0],None,[256],[0,256])
		ax2.plot(histr, c="w", alpha=0.9)
		ax2.set_facecolor('k')
		ax2.grid(alpha=0.3)
	plt.show()
###FIN###

###INICIA MÉTODO imgcmp()###
def imgcmp(img1, img2):
	"""Devuelve dos imágenes para comparar
	Args:
		img1(numpy array): imagen 1
		img2(numpy array): imagen 2
	Returns:
		img1 (list): Imagen 1 en gráfica
		img2 (list): Imagen 2 en gráfica
	"""

	fig=plt.figure(figsize=(10,5))

	ax1=fig.add_subplot(121)
	if len(img1.shape)==3:
		ax1.imshow(img1,extent=None)
	else:
		ax1.imshow(img1,extent=None,cmap='gray',vmin=0,vmax=255)

	ax2=fig.add_subplot(122)
	if len(img2.shape)==3:
		ax2.imshow(img2,extent=None)
	else:
		ax2.imshow(img2,extent=None,cmap='gray',vmin=0,vmax=255)

	plt.show()
###FIN###

###INICIA MÉTODO split_rgb()###
def split_rgb(img):
    #Verificar que la imagen sea a color
    if len(img.shape)!= 3:
        return 0

    #Definimos cada canal
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]

    #Definir la imagen con todos los canales
    k=8
    fig= plt.figure(figsize=(k,k))
    ax1 = fig.add_subplot(221)
    ax1.set_title('RGB')
    ax1.imshow(img,cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    #Definir la imagen con el canal rojo
    ax2 = fig.add_subplot(222)
    ax2.set_title('R')
    ax2.imshow(r,cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    #Definir la imagen con el canal verde
    ax3 = fig.add_subplot(223)
    ax3.set_title('G')
    ax3.imshow(g,cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    #Definir la imagen con el canal azúl
    ax4 = fig.add_subplot(224)
    ax4.set_title('B')
    ax4.imshow(b,cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.show()
###FIN###

###INICIA MÉTODO imgnorm()###
def imgnorm(img):
    """normaliza una imagen a blanco y negro
    Args:
        img (numpy array): Recurso de la imagen
    Returns:
        normalized (numpy array): Imagen normalizada
    """
    vmin, vmax = img.min(), img.max()
    valores_normalizados = []
    delta = vmax-vmin

    for p in img.ravel():
        valores_normalizados.append(255*(p-vmin)/delta)

    normalized  = np.array(valores_normalizados).astype(np.uint8).reshape(img.shape[0],-1)
    return normalized
###FIN###
