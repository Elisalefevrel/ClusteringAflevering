import pyomo.environ as pyomo       # Used for modelling the IP
import matplotlib.pyplot as plt     # Used to plot the instance
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

#Opgave 5: Opstil nu en model, der minimerer den største diameter over alle clustre.
#Implementer modellen i Pyomo og løs vha. en solver. Visualiser din løsning

def readData(clusterData: str) -> dict():
    data = rwJson.readJsonFileToDictionary(clusterData)
    data['nrPoints'] = len(data['State'])
    data['dist'] = makeLpNormDistanceMatrix(data, 2)
    return data

def buildModel(data: dict) -> pyomo.ConcreteModel():
    #Opretter modellen
    model = pyomo.ConcreteModel()
    #Tilføjer data
    model.State = data["State"]
    model.Murder = data["Murder"]
    model.Assault = data["Assault"]
    model.UrbanPop = data["UrbanPop"]
    model.Rape = data["Rape"]
    model.k = data['k']
    model.nrPoints = data['nrPoints']
    model.points = range(0, data['nrPoints'])
    model.dist = data['dist']
    model.groups = range(0,model.k)

    #Definerer variabler
    model.x=pyomo.Var(model.points,model.groups, within=pyomo.Binary)
    model.D=pyomo.Var(model.groups,within=pyomo.NonNegativeReals)
    model.Dmax=pyomo.Var(within=pyomo.NonNegativeReals)
    #Tilføjer objektfunktionen
    model.obj=pyomo.Objective(expr=model.Dmax, sense=pyomo.minimize)
    #Tilføjer definitionen af Dmax
    model.DmaxDef=pyomo.ConstraintList()
    for l in model.groups:
        model.DmaxDef.add(expr=model.Dmax >= model.D[l])
    #Tilføjer definitionen af D-variablerne
    model.DDef=pyomo.ConstraintList()
    for i in model.points:
        for j in model.points:
            for l in model.groups:
                model.DDef.add(expr=model.D[l] >= model.dist[i][j]*(model.x[i,l] + model.x[j,l] - 1.0))
    #Sørger for at alle punkter er i én gruppe
    model.AllInGroup=pyomo.ConstraintList()
    for i in model.points:
        model.AllInGroup.add(expr=sum(model.x[i,l] for l in model.groups) == 1)
    return model



def solveModel(model: pyomo.ConcreteModel()):
    #Sætter solveren
    solver = pyomo.SolverFactory('glpk')
    #Solver modellen
    solver.solve(model, tee=True)

def displaySolution(model: pyomo.ConcreteModel()):
    print('Optimal diameter is:', pyomo.value(model.obj))
    labels = [0] * model.nrPoints
    #Printer grupperne og deres tilhørende punkter
    for l in model.groups:
        print('Group', l, 'consists of:')
        for i in model.points:
            if pyomo.value(model.x[i, l]) == 1:
                print(model.State[i], ',', end='')
                labels[i] = l
        print('')

def main(clusterDataFile: str):
    data = readData(clusterDataFile)
    model = buildModel(data)
    solveModel(model)
    displaySolution(model)


if __name__ == '__main__':
    theDataFile = "USArrest"
    main(theDataFile)