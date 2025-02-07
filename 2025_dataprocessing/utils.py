import os
import random
import numpy as np


def allFilesExist(fileNames: list):

    for file in fileNames:
        if not os.path.exists(file):
            return False

    return True



def deltaNet(coord, delta):
    numLowerNodes = coord.shape[0]
    lead = []
    candidates = set(range(numLowerNodes))
    sClusters = []

    while len(candidates) != 0:
        p = random.choice(list(candidates))
        lead.append(p)

        clst = getCluster(p, coord, delta)
        sClusters.append(clst)

        candidates = candidates.difference(set(clst))

    sCoord = coord[np.array(lead)]

    return sCoord, sClusters


def getCluster(point,
                coords,
                delta):
    pCoord = coords[point]

    return np.where(np.linalg.norm(coords - pCoord, axis=-1) <= delta)[0]


def readBonds(filename):
    bond = 0
    first = 0
    edgelist = []
    with open(filename) as f:
        for line in f:
            rowlist = line.strip().split()
            if bond == 0:
                if len(rowlist) != 0 and rowlist[0] == 'Bonds':
                    bond = 1
            elif len(rowlist) == 0:  # bond==1
                if first == 0:
                    first = 1
                else:
                    break
            else:
                edgelist.append((eval(rowlist[2]), eval(rowlist[3])))
    return edgelist


def readCoords(fileName):
    atom = 0
    first = 0
    atoms = {}
    with open(fileName) as f:
        for line in f:
            rowlist = line.strip().split()
            if atom == 0:
                if len(rowlist) != 0 and rowlist[0] == 'Atoms':
                    atom = 1
            elif len(rowlist) == 0:  # atom==1
                if first == 0:
                    first = 1
                else:
                    break
            else:
                atoms[eval(rowlist[0])] = np.array((eval(rowlist[4]), eval(rowlist[5]), eval(rowlist[6])))
    return atoms

def readDihedrals(filename):
    dihedral = 0
    first = 0
    dihedrals = []
    with open(filename) as f:
        for line in f:
            rowlist = line.strip().split()
            if dihedral == 0:
                if len(rowlist) != 0 and rowlist[0] == 'Dihedrals':
                    dihedral = 1
            elif len(rowlist) == 0:  # atom==1
                if first == 0:
                    first = 1
                else:
                    break
            else:
                dihedrals.append([eval(rowlist[2]), eval(rowlist[5])])
    return dihedrals


def pd2counts(pd):
    cnt0 = 0
    cnt1 = 0
    cnt2 = 0
    for row in pd:
        u, v = row[1]
        if abs(1-u)<1e-2:
            if abs(2-v)<1e-2:
                cnt0 += 1
            else: 
                cnt1 += 1
        else:
            cnt2 += 1
    return [cnt0, cnt1, cnt2]


def spmt2edgelist(matrix):
    # matrix: sparse matrix

    numNodes = matrix.shape[0]

    srcNodes = [node for snode in range(numNodes) for node in matrix[snode].nonzero()[1]]
    dstNodes = [snode for snode in range(numNodes) for _ in matrix[snode].nonzero()[1]]

    return (srcNodes, dstNodes)
