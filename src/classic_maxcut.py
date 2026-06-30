import gurobipy as gp
from gurobipy import GRB
import numpy as np

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
    model.setParam("TimeLimit", 60)      
    model.setParam("MIPGap", 0.01) 
    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()
    status = {9: "Time Limit", 2 : "Optimal"}
    print("Status:", status[model.Status])      
    print(f"Gap: {np.abs(model.ObjBound - model.ObjVal) /  np.abs(model.ObjVal)}%" )
    solution = {i: int(x[i].X) for i in nodes}
    return solution, -model.ObjVal


