#Opgave 1: Vælg mindst́ én metrik (afstandsmål), som skal bruges til at måle hvor ens
#to stater er (I kan blandt andet se på Euklidisk afstand og/eller Taxi-cap normen).
#Brug metrikken/metrikkerne til at udregne afstandsmatricen/afstandsmatricerne
#cij (se evt. afsnittet om lokationsbaseret clustering i noterne).


import pyomo.environ as pyomo       # Used for modelling the IP
import numpy as np                  # Used for calculating distances
import readAndWriteJson as rwJson   # Used to read data from Json file


def makeLpNormDistanceMatrix(data: dict, p: int) -> list:
    points = np.column_stack((data['Murder'], data['Assault'], data['UrbanPop'], data['Rape']))
    nrPoints = len(data['State'])
    dist = []
    for i in range(0, nrPoints):
        dist.append([])
        for j in range(0, nrPoints):
            dist[i].append(np.linalg.norm(points[i] - points[j], p))
    return dist

#Opgave 2: Opstil det lokationsbaserede IP med sum-objektfunktion fra Clustering- noten for
#datasættet USArrests.csv. Implementer modellen i Python og løs det vha. en solver.

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
    # Add objective function
    model.obj = pyomo.Objective(expr=sum(model.dist[i][j] * model.x[i, j] for i in model.points for j in model.points),
                                sense=pyomo.minimize)
    # Add "all must be represented" constraints
    model.one_group = pyomo.ConstraintList()
    for j in model.points:
        model.one_group.add(expr=sum(model.x[i, j] for i in model.points) == 1.0)
    # Add only represent if y[i]=1 (x[i][j]=1 => y[i]=1)
    model.GUB = pyomo.ConstraintList()
    for i in model.points:
        for j in model.points:
            model.GUB.add(expr=model.x[i, j] <= model.y[i])
    # Add cardinality constraint on number of groups
    model.cardinality = pyomo.Constraint(expr=sum(model.y[i] for i in model.points) == model.k)
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