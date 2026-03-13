from algorithms.algofactory import AlgorithmFactory
import networkx as nx 

if __name__ == "__main__":
    p = 1 
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0)])
    q = AlgorithmFactory.create("qaoa", G, p)
    q.run()
    print(f"{q.angles=}\n", 
          f"{q.q_energy=}\n",
          f"{q.q_error=}\n",
          f"{q.f_state=}\n",
          f"{q.olap=}\n",
          f"{q.angles=}\n",
          f"{q.best_bitstring=}\n")
    print("---------------------------------------------------------")
    rq = AlgorithmFactory.create("rqaoa", G, p)
    rq.run()
    print(f"{rq.angles=}\n"
          f"{rq.mapping=}\n", 
          f"{rq.constraints=}\n", 
          f"{rq.best_bitstring=}\n",
          f"{rq.history=}\n")
    print("---------------------------------------------------------")
