import numpy as np


class imageObject:
	""" create an new object for each classification and track its centre coordinate"""
	def __init__(self, label):
		# first initialisation at label, u, v
		self.label = label 
		self.u = u
		self.v = v
	
		
def update(objects, label, centre):
	
	# updates centre position.
	status = 0 # 1 if object is found.
	# objects with the same label
	object_label_inds = [i for o,i in enumerate(objects) if o.label==label]
	object_coordinates = np.array([[objects[i].u, objects[i].v] for i in object_label_inds])
	
	distances = np.linalg.norm([object_coordinates - centre], axis=1)
	print(distances)
	
	
	
		
		
def update_all_objects(objects, labels, centres):
	
	# loop through label, centre and search for nearest object
	for label, centre in zip(labels, centres):
		# look for nearest obj
		objects, status = update(objects, label, centre)
		
	return objects		
	
