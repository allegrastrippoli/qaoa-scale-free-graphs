from algorithms.algofactory import AlgorithmFactory
import networkx as nx 

if __name__ == "__main__":
    p = 1 
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0)])
    rq = AlgorithmFactory.create("rqaoa", G, p)
    rq.run()
    print(rq.angles, rq.constraints)

