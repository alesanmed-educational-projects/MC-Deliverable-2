import numpy as np

def find_connected_components(image, n_neighbour=8):
	L = np.zeros(image.shape)
	T = {}
	curr_label = 1

	if n_neighbour == 8:
		x_vector = [-1, 0, -1, -1]
		y_vector = [0, -1, -1, 1]
	elif n_neighbour==4:
		x_vector = [-1, 0]
		y_vector = [0, -1]

	rows = image.shape[0]
	columns = image.shape[1]

	for row in range(rows):
		for column in range(columns):
			p = image[row, column]

			if not p == 0.0:
				neighbour_values = []
				neighbour_labels = []
				for movement in range(0, len(x_vector)):
					neighbour = (row+x_vector[movement], column+y_vector[movement])
					if neighbour[0] < 0 or neighbour[0] > rows-1 or neighbour[1] < 0 or neighbour[1] > columns-1:
						neighbour_values.append(0)
						neighbour_labels.append(0)
					else:
						neighbour_values.append(int(not image[neighbour]==0.0))
						neighbour_labels.append(L[neighbour])


				if sum(neighbour_values) == 0:
					L[row, column] = curr_label
					curr_label += 1
				elif sum(neighbour_values) == 1:
					L[row, column] = neighbour_labels[neighbour_values.index(1)]
				else:
					neighbour_foreground = np.where(np.array(neighbour_values)==1)[0]
					neighbour_foreground_labels = [neighbour_labels[i] for i in neighbour_foreground]
					mask = np.ma.masked_equal(neighbour_foreground_labels, 0, copy=False)

					final_label = min(mask)

					L[row, column] = final_label
					for label in neighbour_foreground_labels:
						if label != final_label and label != 0:
							if final_label in T:
								T[label] = T[final_label]
							else:
								T[label] = final_label

	for row in range(rows):
		for column in range(columns):
			p = image[row, column]

			if not p == 0.0 and L[row, column] in T:
				L[row, column] = T[L[row, column]]


	return (curr_label-1, L)