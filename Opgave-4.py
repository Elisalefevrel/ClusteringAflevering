#Opgave 4
#

import pyomo.environ as pyomo       # Used for modelling the IP
import matplotlib.pyplot as plt     # Used to plot the instance
import numpy as np                  # Used for calculating distances
import readAndWriteJson as rwJson   # Used to read data from Json file
import math


def makeLpNormDistanceMatrix(data: dict, p: int) -> list:
    points = np.column_stack((data['Murder'], data['Assault'], data['UrbanPop'], data['Rape']))
    nrPoints = len(data['State'])
    dist = []
    for i in range(0, nrPoints):
        dist.append([])
        for j in range(0, nrPoints):
            dist[i].append(np.linalg.norm(points[i] - points[j], p))
    return dist


def readData(clusterData: str) -> dict():
    data = rwJson.readJsonFileToDictionary(clusterData)
    data['nrPoints'] = len(data['State'])
    data['dist'] = makeLpNormDistanceMatrix(data, 2)
    return data

def buildModel(data: dict) -> pyomo.ConcreteModel():
    # Create model
    model = pyomo.ConcreteModel()
    #Add data
    model.State = data["State"]
    model.Murder = data["Murder"]
    model.Assault = data["Assault"]
    model.UrbanPop = data["UrbanPop"]
    model.Rape = data["Rape"]
    model.k = data['k']
    model.nrPoints = data['nrPoints']
    model.points = range(0, data['nrPoints'])
    model.dist = data['dist']

    # Define variables
    model.y = pyomo.Var(model.points, within=pyomo.Binary)
    model.x = pyomo.Var(model.points, model.points, within=pyomo.Binary)
    model.rhoMax = pyomo.Var(within=pyomo.NonNegativeReals)
    # Add objective function
    model.obj = pyomo.Objective(expr=model.rhoMax)
    # Add "all must be represented" constraints
    model.allRep = pyomo.ConstraintList()
    for j in model.points:
        model.allRep.add(expr=sum(model.x[i, j] for i in model.points) == 1)

    # Add only represent if y[i]=1 (x[i][j]=1 => y[i]=1)
    model.GUB = pyomo.ConstraintList()
    for i in model.points:
        for j in model.points:
            model.GUB.add(expr=model.x[i, j] <= model.y[i])
    # Add cardinality constraint on number of groups
    model.cardinality = pyomo.Constraint(expr=sum(model.y[i] for i in model.points) == model.k)
    # Add correct definition of the rhoMax
    model.rhoMaxDef = pyomo.ConstraintList()
    for j in model.points:
        model.rhoMaxDef.add(expr=sum(model.dist[i][j]*model.x[i, j] for i in model.points) <= model.rhoMax)
    return model



def solveModel(model: pyomo.ConcreteModel()):
    # Set the solver
    solver = pyomo.SolverFactory('glpk')
    # Solve the model
    solver.solve(model, tee=True)

def displaySolution(model: pyomo.ConcreteModel()):
    print('Optimal sum of distances in clusters:', pyomo.value(model.obj))
    labels = [0] * model.nrPoints
    # Print the groups to promt and save coordinates for plotting
    for i in model.points:
        if pyomo.value(model.y[i]) == 1:
            print('Point', model.State[i], 'represents points:')
            for j in model.points:
                if pyomo.value(model.x[i, j]) == 1:
                    print(model.State[j], ",", end='')
                    labels[j] = i
            print('\n')

def main(clusterDataFile: str):
    data = readData(clusterDataFile)
    model = buildModel(data)
    solveModel(model)
    displaySolution(model)


if __name__ == '__main__':
    theDataFile = "USArrest"
    main(theDataFile)
