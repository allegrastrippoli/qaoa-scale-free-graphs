import gurobipy as gp
from gurobipy import GRB

def maxcut_gurobi(G):
    nodes = list(G.nodes())
    edges = []
    for i, j, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        edges.append((i, j, w))
    model = gp.Model("maxcut_qubo")
    x = model.addVars(nodes, vtype=GRB.BINARY, name="x")
    obj = gp.QuadExpr()
    for i, j, w in edges:
        obj.add(w * x[i])
        obj.add(w * x[j])
        obj.add(-2 * w * x[i] * x[j])
    model.setParam("OutputFlag", 0)
    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()
    solution = {i: int(x[i].X) for i in nodes}
    print(f"{model.ObjVal=}", f"{solution=}")
    return solution, -model.ObjVal
