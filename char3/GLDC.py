import numpy as np

from connect.tool.LocalConnect import localConnect

def get_distance(point1, point2):
	return np.sqrt(np.sum((point1 - point2) ** 2))


def compute_density(dataset, min_points):
	densities = np.zeros(len(dataset))
	for i in range(len(dataset)):
		dist_sum = 0
		for j in range(len(dataset)):
			if i != j:
				dist_sum += np.exp(-get_distance(dataset[i], dataset[j]) ** 2)
		densities[i] = dist_sum
	return densities


def leaders_algorithm(dataset, min_points):
	densities = compute_density(dataset, min_points)
	leaders = []
	clusters = []
	
	for i in range(len(dataset)):
		if densities[i] > min_points:
			is_leader = True
			for leader in leaders:
				if get_distance(dataset[i], leader) <= min_points:
					is_leader = False
					break
			if is_leader:
				leaders.append(dataset[i])
				clusters.append([i])
	
	for i in range(len(dataset)):
		if i in [idx for cluster in clusters for idx in cluster]:
			continue
		nearest_leader_idx = \
			min([(j, get_distance(dataset[i], leader)) for j, leader in enumerate(leaders)], key=lambda x: x[1])[0]
		clusters[nearest_leader_idx].append(i)
	
	return clusters


def get_data_by_user_id(user_id):
	query = "SELECT * FROM loadtest WHERE user_id = %s ORDER BY data DESC LIMIT 10000"
	cursor = localConnect().cursor()
	cursor.execute(query, (str(user_id),))
	result = cursor.fetchall()
	sorted_result = sorted(result, key=lambda x: x[1])
	return sorted_result


