import random
import math

def generate_points(count, distribution, dimensions=2):
	"""Create points in a normalized coordinate system (-1..1)."""
	points = []

	if distribution == "Rácspontok":
		grid_size = max(2, math.ceil(count ** (1 / dimensions)))
		coordinates = [
			-1 + 2 * index / (grid_size - 1)
			for index in range(grid_size)
		]
		all_grid_points = []
		if dimensions == 3:
			all_grid_points = [
				(x, y, z)
				for x in coordinates
				for y in coordinates
				for z in coordinates
			]
		else:
			all_grid_points = [(x, y) for x in coordinates for y in coordinates]
		if len(all_grid_points) > count:
			step = len(all_grid_points) / count
			points = [all_grid_points[min(len(all_grid_points) - 1, int(index * step))] for index in range(count)]
		else:
			points = all_grid_points
	elif distribution == "Gauss-i klaszter":
		for _ in range(count):
			points.append(tuple(random.gauss(0, 0.28) for _ in range(dimensions)))
	elif distribution == "Két klaszter":
		for _ in range(count):
			center_x = -0.42 if random.random() < 0.5 else 0.42
			points.append((center_x + random.gauss(0, 0.16),) + tuple(random.gauss(0, 0.22) for _ in range(dimensions - 1)))
	elif distribution == "Zajos kör":
		for _ in range(count):
			angle = random.random() * math.tau
			radius = 0.62 + random.gauss(0, 0.06)
			points.append((radius * math.cos(angle), radius * math.sin(angle)) + tuple(random.gauss(0, 0.06) for _ in range(dimensions - 2)))
	else:
		points = [tuple(random.uniform(-1, 1) for _ in range(dimensions)) for _ in range(count)]

	return [tuple(max(-1.0, min(1.0, coordinate)) for coordinate in point) for point in points]

def squared_distance(point, other):
	"""Return squared Euclidean distance without an unnecessary square root."""
	return sum((coordinate - other_coordinate) ** 2 for coordinate, other_coordinate in zip(point, other))

def farthest_point_order(points):
	"""Order points with greedy farthest-point sampling."""
	first_index, second_index = max(
		(
			(squared_distance(point, other), index, other_index)
			for index, point in enumerate(points)
			for other_index, other in enumerate(points[index + 1:], start=index + 1)
		),
		key=lambda item: item[0],
	)[1:]
	selected_indices = {first_index, second_index}
	ordered_indices = [first_index, second_index]
	minimum_distances = [
		min(squared_distance(point, points[first_index]), squared_distance(point, points[second_index]))
		for point in points
	]
	while len(selected_indices) < len(points):
		next_index = max(
			(index for index in range(len(points)) if index not in selected_indices),
			key=minimum_distances.__getitem__,
		)
		selected_indices.add(next_index)
		ordered_indices.append(next_index)
		for index, point in enumerate(points):
			minimum_distances[index] = min(minimum_distances[index], squared_distance(point, points[next_index]))
	return ordered_indices


def calculate_farthest_values(points):
	"""Assign values so TOP N matches the first N farthest-point selections."""
	values = [0.0] * len(points)
	for rank, index in enumerate(farthest_point_order(points)):
		values[index] = float(len(points) - rank)
	return values