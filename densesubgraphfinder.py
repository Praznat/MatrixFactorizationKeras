class DenseSubgraphFinder():
	def __init__(self):
		self.nodes = {}
	def addEdge(self, node1, node2):
		self.nodes.setdefault(node1, set()).add(node2)
		self.nodes.setdefault(node2, set()).add(node1)
	def removeNode(self, node):
		links = self.nodes.get(node)
		for link in links:
			linkOuts = self.nodes.get(link)
			linkOuts.remove(node)
			if len(linkOuts) == 0:
				self.nodes.pop(link)
		self.nodes.pop(node)
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

