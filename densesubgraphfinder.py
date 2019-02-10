class DenseSubgraphFinder():
	''' keeps removing nodes from a graph until all nodes have outgoing edge count >= min_edges '''
	def __init__(self):
		self.nodes = {}
		self.total_filled = 0
		self.known_node1s = set()
		self.known_node2s = set()
	def addEdge(self, node1, node2):
		node1set = self.nodes.setdefault(node1, set())
		if node2 not in node1set:
			self.total_filled += 1
			node1set.add(node2)
		self.nodes.setdefault(node2, set()).add(node1)
		self.known_node1s.add(node1)
		self.known_node2s.add(node2)
	def removeNode(self, node):
		links = self.nodes.get(node)
		for link in links:
			linkOuts = self.nodes.get(link)
			linkOuts.remove(node)
			if len(linkOuts) == 0:
				# self.nodes.pop(link)
				self.removeNode(link)
		self.nodes.pop(node)
		self.total_filled -= len(links)
		if node in self.known_node1s:
			self.known_node1s.remove(node)
		if node in self.known_node2s:
			self.known_node2s.remove(node)
	def purge_1round(self, min_edges):
		foundany = False
		for node in self.nodes:
			if len(self.nodes.get(node, set())) < min_edges:
				foundany = True
				self.removeNode(node)
				break
		return foundany
	def purge(self, min_edges):
		ongoing = True
		while ongoing:
			ongoing = self.purge_1round(min_edges)
	def condense(self, density_target):
		sortedNodes = sorted(self.nodes.items(), key=lambda x:len(x[1]))
		density = self.total_filled / (len(self.known_node1s) * len(self.known_node2s))
		for i in range(len(sortedNodes)):
			# sortedNodes = sorted(self.nodes.items(), key=lambda x:len(x[1]))
			worst_node = sortedNodes[i]
			if worst_node[1]:
				self.removeNode(worst_node[0])
			density = self.total_filled / (len(self.known_node1s) * len(self.known_node2s))
			print("current density: " + str(density))
			if density >= density_target:
				break
		# remove nodes with least number of links until target density is reached
	def __str__(self):
		return str(self.nodes)


# dsf = DenseSubgraphFinder()
# dsf.addEdge('a','b')
# dsf.addEdge('c','b')
# dsf.addEdge('a','d')
# dsf.addEdge('c','d')
# print(dsf)
# dsf.purge_1round(2)
# print(dsf)
# dsf.purge(3)
# print(dsf)

