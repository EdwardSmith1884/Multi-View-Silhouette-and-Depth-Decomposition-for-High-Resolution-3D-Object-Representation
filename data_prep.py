import os 
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR + '/scripts/')

import urllib
from multiprocessing import Pool
import binvox_rw
import scipy.io as sio
import numpy as np 
from tqdm import tqdm
from glob import glob
import random 
import shutil
from PIL import Image
from PIL import ImageOps
import argparse
from scipy import ndimage
from subprocess import call

# this is the dataset for object translation, it will download the object files, convert then into numpy matricies, and overlay them onto pictures from the sun dataset 

parser = argparse.ArgumentParser(description='Dataset prep for image to 3D object super resolution')
parser.add_argument('-o','--objects', default=['chair'], help='List of object classes to be used downloaded and converted.', nargs='+' )
parser.add_argument('-no','--num_objects', default=10000, help='number of objects to be converted', type = int)
parser.add_argument('-ni','--num_images', default=10, help='number of images to be created for each object', type = int)
parser.add_argument('-l','--low', default=32, help='Low resolution value', type = int)
parser.add_argument('-hi','--high', default=256, help='high resolution value', type = int)
args = parser.parse_args()

if args.low >= args.high: 
	print '-----------------------------------------------------'
	print 'lower resolution must be lower then higher resolution'
	print '-----------------------------------------------------'
	exit()

#labels for the union of the core shapenet classes and the ikea dataset classes 
labels = {'03001627' : 'chair', 
'04128520': 'sofa', '04379243': 'table', '02858304':'boat', '02958343':'car',  
'02691156': 'plane' }
 

wanted_classes=[]
for l in labels: 
	if labels[l] in args.objects:
		wanted_classes.append(l)


debug_mode = 0 # change to make all of the called scripts print their errors and warnings 
if debug_mode:
	io_redirect = ''
else:
	io_redirect = ' > /dev/null 2>&1'


# make data directories 
if not os.path.exists('data/voxels/'):
	os.makedirs('data/voxels/')
if not os.path.exists('data/objects/'):
	os.makedirs('data/objects/')



# download .obj obect files 
def download():
	with open('scripts/binvox_file_locations.txt','rb') as f: # location of all the binvoxes for shapenet's core classes 
		content = f.readlines()

	# make data sub-directories for each class
	for s in wanted_classes: 
		obj = 'data/objects/' + labels[s]+'/'
		if not os.path.exists(obj):
			os.makedirs(obj)
		voxes = 'data/voxels/' + labels[s]+'/'
		if not os.path.exists(voxes):
			os.makedirs(voxes)

	# search object for correct object classes
	binvox_urls = []
	obj_urls = []
	for file in content: 
		current_class = file.split('/')
		if current_class[1] in wanted_classes:  
			if '_' in current_class[3]: continue 
			if 'presolid' in current_class[3]: continue 
			obj_urls.append(['http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/'+file.split('/')[1]+'/'+file.split('/')[2]+'/model.obj', 'data/objects/'+labels[current_class[1]]+ '/'+ current_class[2]+'.obj'])
	
	# get randomized sample from each object class of correct size
	random.shuffle(obj_urls)
	final_urls = []
	dictionary = {}
	for o in obj_urls:
		obj_class = o[1].split('/')[-2]
		if obj_class in dictionary: 
			dictionary[obj_class] += 1
			if dictionary[obj_class]> args.num_objects: 
				continue
		else: 
			dictionary[obj_class] = 1
		final_urls.append(o) 
	
	# parallel downloading of object .obj files
	pool = Pool()
	pool.map(down, final_urls)


# download .mtl files for each .obj file to add textures during image processing 
def process_mtl(): 
	import requests
	from bs4 import BeautifulSoup
	location = 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/'
	for s in wanted_classes: 
		files = glob('data/objects/' + labels[s]+'/*.obj')
		commands = []
		for f in tqdm(files): 
			file = f.split('/')[-1][:-4]
			if not os.path.exists('data/objects/' + labels[s]+'/' + file + '/images/'):
				os.makedirs('data/objects/' + labels[s]+'/' + file + '/images/')
			if not os.path.exists('data/objects/' + labels[s]+'/' + file +  '/' + file + '/'):
				os.makedirs('data/objects/' + labels[s]+'/' + file +  '/' + file + '/')



			shutil.move(f,'data/objects/' + labels[s]+'/' + file + '/' + f.split('/')[-1])
			commands.append([location+s+'/'+file+'/model.mtl', 'data/objects/' + labels[s]+'/' + file + '/model.mtl'])
			
			soup = BeautifulSoup(requests.get(location+s+'/'+file+'/images/').text,  "html5lib")
			for a in soup.find_all('a', href=True):
				if 'textu' in a['href']: 
					commands.append([location+s+'/'+file+'/images/'+a['href'], 'data/objects/' + labels[s]+'/' + file + '/images/'+ a['href'] ])
			
			soup = BeautifulSoup(requests.get(location+s+'/'+file+ '/' + file + '/').text,  "html5lib")
			for a in soup.find_all('a', href=True):
				if 'jpg' in a['href']  or 'png' in a['href']: 
					commands.append([location+s+'/'+file+  '/' + file + '/'+a['href'], 'data/objects/' + labels[s]+'/' + file + '/'+ file +'/'+ a['href'] ])

			if len(commands) == 100: 
				pool = Pool()
				pool.map(down, commands)
				commands = []

		pool = Pool()
		pool.map(down, commands)



# these are two simple fucntions for parallel processing, down() downloads , and call() calls functions 
def down(url):
	urllib.urlretrieve(url[0], url[1])
def call(command):
	os.system('%s %s' % (command, io_redirect))


# converts .obj files to .binvox files, intermidiate step before converting to voxel .npy files 
def binvox():
	for s in wanted_classes: 
		dirs = glob('data/objects/' + labels[s]+'/*/*.obj')
		commands =[]
		count = 0 
		for d in tqdm(dirs):
			command = './binvox ' + d  + ' -d ' + str(args.high)+ ' -pb -cb -c -dc -aw -e'   # this executable can be found at http://www.patrickmin.com/binvox/ , 
			# -d x idicates resoltuion will be x by x by x , -pb is to stop the visualization, the rest of the commnads are to help make the object water tight 
			commands.append(command)
			if count %10 == 0  and count != 0:
				pool = Pool()
				pool.map(call, commands)
				pool.close()
				pool.join()
				commands = []
			count +=1 
		pool = Pool()
		pool.map(call, commands)
		pool.close()
		pool.join()



# splits each object classes into training, validation and test set in ration 70:10:20
def split():
	for s in wanted_classes: 
		dirs = glob('data/objects/' + labels[s]+'/*')
		dirs = [d for d in dirs if ( 'train' not in d) and ('test' not in d) and ('valid' not in d )]
		random.shuffle(dirs)
		train = dirs[:int(len(dirs)*.7)]
		valid = dirs[int(len(dirs)*.7):int(len(dirs)*.8)]
		test  = dirs[int(len(dirs)*.8):]
		if not os.path.exists('data/objects/' + labels[s]+'/train/'):
			os.makedirs('data/objects/' + labels[s]+'/train/')
		if not os.path.exists('data/objects/' + labels[s]+'/valid/'):
			os.makedirs('data/objects/' + labels[s]+'/valid/')
		if not os.path.exists('data/objects/' + labels[s]+'/test/'):
			os.makedirs('data/objects/' + labels[s]+'/test/')
		for t in train: 
			shutil.move(t , 'data/objects/' + labels[s]+'/train/' + t.split('/')[-1])
		for t in valid: 
			shutil.move(t , 'data/objects/' + labels[s]+'/valid/' + t.split('/')[-1])
		for t in test: 
			shutil.move(t , 'data/objects/' + labels[s]+'/test/' + t.split('/')[-1])
		


# extracts odms from an object 
def odm(data, high, low): 
	dim = data.shape[0] 
	down = high // low 
	a,b,c = np.where(data == 1)
	large = int(dim *1.5)
	big_list = [[[[-1,large]for j in range(dim)] for i in range(dim)] for k in range(3)]
	# over the whole object extract for each face the first and last occurance of a voxel at each pixel
	# we take highest for convinience
	for i,j,k in zip(a,b,c):
		big_list[0][i][j][0] = (max(k,big_list[0][i][j][0]))
		big_list[0][i][j][1] = (min(k,big_list[0][i][j][1]))
		big_list[1][i][k][0] = (max(j,big_list[1][i][k][0]))
		big_list[1][i][k][1] = (min(j,big_list[1][i][k][1]))
		big_list[2][j][k][0] = (max(i,big_list[2][j][k][0]))
		big_list[2][j][k][1] = (min(i,big_list[2][j][k][1]))
	faces = np.zeros((6,dim,dim)) # will hold odms 
	for i in range(dim): 
		for j in range(dim): 
			faces[0,i,j] =   dim -1 - big_list[0][i][j][0]         if    big_list[0][i][j][0]   > -1 else dim
			# we subtract from the (dimension -1) as we computed the last occurance, instead of the first for half of the faces
			faces[1,i,j] =   big_list[0][i][j][1]        		   if    big_list[0][i][j][1]   < large else dim 
			faces[2,i,j] =   dim -1 - big_list[1][i][j][0]         if    big_list[1][i][j][0]   > -1 else dim
			faces[3,i,j] =   big_list[1][i][j][1]        		   if    big_list[1][i][j][1]   < large else dim
			faces[4,i,j] =   dim -1 - big_list[2][i][j][0]         if    big_list[2][i][j][0]   > -1 else dim
			faces[5,i,j] =   big_list[2][i][j][1]         		   if    big_list[2][i][j][1]   < large else dim



	return faces


# converts .binvox files to numpy array, downsimples to the low resolution and convert odms for both resolutions 
# to obtain watertigh meshes, we  apply the high resolution odms to the filled low resolution model 	
def convert_bin():
	low = args.low 
	high = args.high
	down = high // low
	for s in wanted_classes:
		directory = 'data/voxels/'+labels[s]+'/train/' 
		if not os.path.exists(directory):
			os.makedirs(directory)
		directory = 'data/voxels/'+labels[s]+'/valid/' 
		if not os.path.exists(directory):
			os.makedirs(directory)
		directory = 'data/voxels/'+labels[s]+'/test/' 
		if not os.path.exists(directory):
			os.makedirs(directory)
	for num in wanted_classes: 
		train = glob('data/objects/'+labels[num]+'/train/*/*.binvox')
		valid = glob('data/objects/'+labels[num]+'/valid/*/*.binvox')
		test  = glob('data/objects/'+labels[num]+'/test/*/*.binvox') 
		
		for e,mods in enumerate([train, valid, test]):
		
				if e == 0: 
					print '------------'
					print 'doing the training set'
					print '------------'
					
				if e == 1: 
					print '------------'
					print 'doing the validation set'
					print '------------'

				if e == 2:
					print '------------' 
					print 'doing the test set'
					print '------------'
					
				for m in tqdm(mods):   
					
					# convert .binvox model to np array 
					with open(m, 'rb') as f:
						try: 
							model = binvox_rw.read_as_3d_array(f)
						except ValueError:
							continue
					model = model.data.astype(int)

					

					# obtain low resolution model and fill 
					a,b,c = np.where(model==1)
					low_model = np.zeros((low,low,low))
					for x,y,z in zip(a,b,c):
							low_model[ x//down, y//down, z//down] =1 
					low_model[ndimage.binary_fill_holes(low_model)] = 1

					# obtain odms 
					faces = odm(model, high,low)
					low_faces = odm(low_model, high,low)
					
					if e < 2: 
						if e == 0: 
							place = '/train/'
						else:
							place = '/valid/'

						# saving traning and validation set 
						sio.savemat('data/voxels/'+labels[num]+ place+ '/full_object_'+m.split('/')[-1][:-7], {'low_model':low_model.astype(np.uint8)})
						for i in range(6):
								sio.savemat('data/voxels/'+labels[num]+ place+ '/face_'+ str(i)+ '_' +m.split('/')[-1][:-7], {'high_odm':faces[i].astype(np.uint16), 
										'low_odm':low_faces[i].astype(np.uint8)})

					else:

						# applies high resolution odm to low resolution model to extract water tight models 
						
						# nearest neighbor upsapling of low res model to the high resolution 
						corrected = np.zeros((high,high,high))
						for i in range(low): 
							for j in range(low): 
								for k in range(low):
									corrected[i*down: (i+1)*down, j*down:(j+1)*down, k*down:(k+1)*down] = low_model[i,j,k]
						
						#carving away of voxeles using high resolution odms 
						for i in range(high): 
							for j in range(high): 
								if faces[0,i,j] >0:
									corrected[i,j,int((high - faces[0,i,j])):high]=0
								else: 
									corrected[i,j,:] =0

								if faces[1,i,j] >0: 
									corrected[i,j,0:int(faces[1,i,j])]=0
								else: 
									corrected[i,j,:] =0

								if faces[2,i,j] >0: 
									corrected[i,int((high - faces[2,i,j])):high, j] =0 
								else: 
									corrected[i,:,j] =0

								if faces[3,i,j] >0:
									corrected[i,0:int(faces[3,i,j]-1), j] =0 
								else: 
									corrected[i,:,j] =0

								if faces[4,i,j] >0:
									corrected[int((high - faces[4,i,j])):high,i,j] =0 
								else: 
									corrected[:,i,j] =0

								if faces[5,i,j] >0:
									corrected[0:int(faces[5,i,j]-1),i,j] =0 
								else: 
									corrected[:,i,j] =0


						#saving test set 
						sio.savemat('data/voxels/'+labels[num]+'/test/full_object_'+m.split('/')[-1][:-7], {'model': corrected.astype(np.uint16),'low_model':low_model.astype(np.uint8)})
						for i in range(6):
								sio.savemat('data/voxels/'+labels[num]+ '/test/face_'+ str(i)+ '_' +m.split('/')[-1][:-7], {'high_odm':faces[i].astype(np.uint16), 
										'low_odm':low_faces[i].astype(np.uint8)})


 # code for rendering the cad models  in 128 by 128 images 
def render():
	for s in wanted_classes: 
		sets = ['train', 'valid', 'test']
		for place in sets:
			print '------------' 
			print 'doing: ' + place
			print '------------'

			img_dir = 'data/images/'+labels[s]+ '/'  + place + '/'
			if not os.path.exists(img_dir):
				os.makedirs(img_dir)
			Model_dir = 'data/objects/'+labels[s]+ '/' + place
			models = glob(Model_dir+'/*/*.obj')
			l=0
			commands = []

			# for each model we by default make 10 images
			# if textures are not availible then a random colour is applied to each face 
			for model in tqdm(models): 
				model_name = model.split('/')[-1].split('.')[0]
				target = os.path.join(img_dir,model_name)

				if not os.path.exists(target):
					os.mkdir(target)
				target = target + '/' + model_name
				python_cmd = 'blender scripts/blank.blend -b -P scripts/blend.py -- %s %s %s' %(args.num_images, model, target)
				commands.append(python_cmd)			
				
				if l%50 == 49: 

					pool = Pool()
					pool.map(call, commands)
					pool.close()
					pool.join()
					commands = []
					
				l+=1
			pool = Pool()
			pool.map(call, commands)
			pool.close()
			pool.join()
			commands = []



# print '------------'
# print'downloading'
# download()
# print '------------'
# print'downloading mlts'
# process_mtl()
# print '------------'
# print'converting .obj to binvoxes'
# binvox()
# print '------------'
# print'splitting data'
# split()
# print '------------'
print'obtaining odms and models'
convert_bin()
print '------------'
# print'rendering images'
# render()
# print'finished eratin'



