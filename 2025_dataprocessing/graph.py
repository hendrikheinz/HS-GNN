import os
import torch
import dgl
import numpy as np
import networkx as nx
import scipy.sparse as sp
import gudhi as gd
from gtda.homology import EuclideanCechPersistence
from gtda.diagrams import PersistenceImage

from utils import *  # noqa: F403

BONDLENGTH = 1.42
DELTA_1 = 4 * BONDLENGTH
DELTA_2 = 8 * BONDLENGTH
MAX_CECH_1 = 8 * BONDLENGTH
MAX_CECH_2 = 16 * BONDLENGTH

class rawToDglGraph():

    def __init__(self, DELTA_1=DELTA_1, DELTA_2=DELTA_2, MAX_CECH_1=MAX_CECH_1, MAX_CECH_2=MAX_CECH_2):
        # DELTA_1, DELTA_1: delta-net clustering algorithm params
        # MAX_CECH_1, MAX_CECH_2: maximum edge length computing cech complex

        # Delta-Net
        self.DELTA_1 = DELTA_1
        self.DELTA_2 = DELTA_2

        # Persistent Summaries
        self.CP1 = EuclideanCechPersistence(homology_dimensions=[0, 1], max_edge_length=MAX_CECH_1)
        self.CP2 = EuclideanCechPersistence(homology_dimensions=[0, 1], max_edge_length=MAX_CECH_2)
        self.PI = PersistenceImage(n_bins=8)


    def getDglGraph(self, filePath, fileName, extraNodeFeatures=None):

        coordL1, eB, eD = self.extractFromRaw(filePath,fileName)

        _, sAdjL2, sMappingL2 = self.getHierarchy(filePath, level=2)
        srcAdj2, dstAdj2 = spmt2edgelist(sAdjL2)  # noqa: F405
        srcClst2, dstSuper2 = spmt2edgelist(sMappingL2)  # noqa: F405
        cechPimgL2, defectPhL2 = self.getPersistenceSummary(filePath, level=2, defect=True)
        phL2 = np.concatenate([cechPimgL2, defectPhL2], axis=-1)

        _, sAdjL3, sMappingL3 = self.getHierarchy(filePath, level=3)
        srcAdj3, dstAdj3 = spmt2edgelist(sAdjL3)  # noqa: F405
        srcClst3, dstSuper3 = spmt2edgelist(sMappingL3)  # noqa: F405
        phL3, _ = self.getPersistenceSummary(filePath, level=3, defect=False)

        graph_data = {
            ('A', 'B', 'A'): (np.append(eB[:, 0], eB[:, 1]), np.append(eB[:, 1], eB[:, 0])),
            # ('atom', 'edist', 'atom'): (np.append(eE[:, 0], eE[:, 1]), np.append(eE[:, 1], eE[:, 0])),
            ('A', 'D', 'A'): (np.append(eD[:, 0], eD[:, 1]), np.append(eD[:, 1], eD[:, 0])),
            # ('atom', 'rdist', 'atom'): (np.append(eR[:, 0], eR[:, 1]), np.append(eR[:, 1], eR[:, 0])),
            ('A', 'G1', 'C2'): (srcClst2, dstSuper2),
            ('C2', 'I2', 'C2'): (srcAdj2, dstAdj2),
            ('C2', 'G2', 'C3'): (srcClst3, dstSuper3),
            ('C3', 'I3', 'C3'): (srcAdj3, dstAdj3)
        }

        graph = dgl.heterograph(graph_data)

        nodeCoords = torch.FloatTensor(coordL1)
        nodeDegrees = graph.in_degrees(etype="B").float().view(-1,1)
        nodeFeats = torch.cat([nodeCoords, nodeDegrees], dim=-1)

        if extraNodeFeatures is not None:
            extraNodeFeatures = extraNodeFeatures.unsqueeze(0).repeat(nodeFeats.shape[0], 1)
            nodeFeats = torch.cat([nodeFeats, extraNodeFeatures], dim=-1)
            # a tip: if additional features (like bundle-level global features, e.g. tube diameter)
            # are desired to be part of input HS-GNN,
            # it is a good idea in practice to take them as (atomic level) node features along with [coordinates, degree]

        graph.nodes['A'].data['feats'] = torch.FloatTensor(nodeFeats)

        graph.nodes['C2'].data['pimg'] = torch.FloatTensor(phL2)
        # actually, pimg in level 2 contains not only persistent features (image) for shape information,
        # it also contains statistical persistent features capturing local defect

        graph.nodes['C3'].data['pimg'] = torch.FloatTensor(phL3)

        return graph

    def extractFromRaw(self, filePath, fileName):

        coordFile = os.path.join(filePath, 'coord.npy')
        bondFile = os.path.join(filePath, 'eB.npy')
        dihedralFile = os.path.join(filePath, 'eD.npy')

        if allFilesExist([coordFile, bondFile, dihedralFile]):  # noqa: F405

            coord = np.load(coordFile)
            eB = np.load(bondFile)
            eD = np.load(dihedralFile)

        else:

            rawfile = os.path.join(filePath, fileName)

            # please personalize read functions for different raw configuration / simulation data
            coord = readCoords(rawfile)  # noqa: F405
            eB = readBonds(rawfile)  # noqa: F405
            eD = readDihedrals(rawfile)  # noqa: F405

            g = nx.from_edgelist(eB)

            if len(coord) != len(g):
                print('Isolated atoms ' + rawfile) # todo: remove?

            for node in g.nodes():
                g.nodes()[node]['coord'] = coord[node]

            g = nx.convert_node_labels_to_integers(g, first_label=0, label_attribute='old')
            idmapping = {node[1]['old']: node[0] for node in g.nodes(data=True)}

            coord = np.array([g.nodes()[node]['coord'] for node in range(len(g))])
            eB = np.array(g.edges())
            eD = np.array([(idmapping[e[0]], idmapping[e[1]]) for e in eD])

            np.save(coordFile, coord)
            np.save(bondFile, eB)
            np.save(dihedralFile, eD)

        return coord, eB, eD

    def getHierarchy(self, filePath, level=2):

        sCoordFile = os.path.join(filePath, 'coord_L' + str(level) + '.npy')  # coordinate of super nodes
        sAdjFile = os.path.join(filePath, 'adj_L' + str(level) + '.npz')  # adj of coarser graph
        sMappingFile = os.path.join(filePath, 'clst_L' + str(level) + '.npz')  # mapping from super nodes to lower nodes

        if allFilesExist([sCoordFile, sAdjFile, sMappingFile]):  # noqa: F405
            # exists coarser graph

            sCoord = np.load(sCoordFile)
            sAdj = sp.load_npz(sAdjFile)
            sMapping = sp.load_npz(sMappingFile)

        else:
            # run delta-net algorithm to get the coarser graph
            suffix = '.npy' if level == 2 else '_L' + str(level - 1) + '.npy'
            coord = np.load(os.path.join(filePath, 'coord' + suffix))

            delta = self.DELTA_1 if level == 2 else self.DELTA_2
            sCoord, sCluster = deltaNet(coord, delta)  # noqa: F405

            numSNodes = len(sCluster)

            graph = nx.Graph()

            graph.add_nodes_from(range(numSNodes))

            for i in range(numSNodes):
                srcSet = set(sCluster[i])
                for j in range(i, numSNodes):
                    if len(srcSet.intersection(set(sCluster[j]))) > 0:
                        graph.add_edge(i, j)

            sAdj = sp.csr_matrix(nx.adjacency_matrix(graph, nodelist=list(range(numSNodes))))

            srcClst = [node for snode in range(numSNodes) for node in sCluster[snode]]
            dstClst = [snode for snode in range(numSNodes) for _ in sCluster[snode]]

            sMapping = sp.csr_matrix((np.ones(len(srcClst), dtype=int), (dstClst, srcClst)),
                                     shape=(numSNodes, len(coord)))

            np.save(sCoordFile, sCoord)
            sp.save_npz(sAdjFile, sAdj)
            sp.save_npz(sMappingFile, sMapping)

        return sCoord, sAdj, sMapping

    def getPersistenceSummary(self, filePath, level=2, defect=True):

        cechPimgFile = os.path.join(filePath, 'cechph_L' + str(level) + '.npy')
        defectPhFile = os.path.join(filePath, 'defph_L' + str(level) + '.npy')

        sMappingFile = os.path.join(filePath, 'clst_L' + str(level) + '.npz')

        if os.path.exists(cechPimgFile):

            cechPimg = np.load(cechPimgFile)

            if os.path.exists(defectPhFile):
                defectPh = np.load(defectPhFile)
            else:
                defectPh = None

        else:

            # compute persistent summaries from cech complex, characterizing shape of CNTs
            CechPH = self.CP1 if level == 2 else self.CP2

            sMapping = sp.load_npz(sMappingFile)
            numSNodes = sMapping.shape[0]
            coordFile = os.path.join(filePath, 'coord.npy') if level == 2 else os.path.join(filePath, 'coord_L' + str(
                level - 1) + '.npy')
            coord = np.load(coordFile)

            cechData = []  # cech complex: 3D coordinates of nodes in a lower cluster
            for sNode in range(numSNodes):
                lNodes = sMapping[sNode].nonzero()[1]
                cechData.append(coord[lNodes])

            pds = CechPH.fit_transform(cechData)
            cechPimg = self.PI.fit_transform(pds).reshape([-1, 2*8*8])

            np.save(cechPimgFile, cechPimg)
            np.save(os.path.join(filePath, 'pd_L' + str(level) + '.npy'), pds)

            if defect and level == 2:
                # usually if level >= 3, we do not compute ph of graph motif, since it in higher hierarchical level does not encode structure defect

                edges = np.load(os.path.join(filePath, 'eB.npy'))
                sMapping = sp.load_npz(sMappingFile)

                g = nx.from_edgelist(edges)

                defectPhFeats = []

                for sNode in sMapping:
                    atoms = sNode.nonzero()[1]
                    sg = g.subgraph(atoms)
                    st = gd.SimplexTree()
                    for i in atoms:
                        st.insert([i], filtration=-1e10)
                    for i, j in sg.edges():
                        st.insert([i, j], filtration=-1e10)
                    for i in atoms:
                        if i not in sg: continue  # noqa: E701
                        st.assign_filtration([i], sg.degree(i))
                    try:
                        if not st.make_filtration_non_decreasing():
                            raise (AssertionError)
                    except(AssertionError):
                        print("Error persistent filtration not decreasing -- defect, " + filePath)
                    st.extend_filtration()
                    ord0, rel1, ext0, ext1 = st.extended_persistence()
                    phord0 = np.array(pd2counts(ord0))  # noqa: F405
                    phext0 = np.array(pd2counts(ext0))  # noqa: F405
                    phext1 = np.array([len(ext1)])
                    phfeat = np.concatenate([phord0, phext0, phext1])
                    defectPhFeats.append(phfeat)

                defectPh = np.stack(defectPhFeats, axis=0)

                np.save(defectPhFile, defectPh)

            else:
                defectPh = None

        return cechPimg, defectPh

    def normalizeCoord(self, graph):
        # Normalizing atom coordinates is significant for downstream representation learning
        # We first recenter the coordinates (move the center of a structure into origin),
        # then employ min-max normalization to limit boundary of atomic coord the into a cube,
        # with range for any axis / x-y-z dimension: [-1,1]

        rawCoord = graph.nodes['A'].data['feats'][:, :3]

        maxBondary = torch.max(rawCoord, dim=0)[0]
        minBondary = torch.min(rawCoord, dim=0)[0]

        center = (maxBondary + minBondary) / 2.

        recenterCoord = rawCoord - center

        maxLength = torch.max(abs(recenterCoord), dim=0)[0]

        normCoord = recenterCoord / maxLength

        graph.nodes['A'].data['feats'][:, :3] = normCoord

        return graph


    def getGraphDataset(self, dirPath, fileNames, savePath, normalization=True, globalFeatures=None, labels=None, extraNodeFeatures=None):
        graphs = []

        for i, fileName in enumerate(fileNames):           
            # Unblock for junctions
            # try:
            #     parent = fileName.split('.lammps')[0]
            # except ValueError:
            #     parent = fileName
            # parent += '_newtopology'
            # filePath = os.path.join(dirPath, parent)
            # fileName = fileName.split('.lammps')
            # fileName[0] += '_newtopology'
            # fileName = fileName[0]+".lammps"
            
            # '.data' block for primary dataset
            try:
                parent = fileName.split('.data')[0]
            except ValueError:
                parent = fileName
            filePath = os.path.join(dirPath,parent)
            
            if extraNodeFeatures is not None:
                graph = self.getDglGraph(filePath,fileName,torch.FloatTensor(extraNodeFeatures[i]))
            else:
                graph = self.getDglGraph(filePath)
            if normalization: graph = self.normalizeCoord(graph)  # noqa: E701
            graphs.append(graph)

        labs = {}

        if labels is not None:
            labs["labels"] = torch.FloatTensor(labels)

        if globalFeatures is not None:
            labs['globalFeatures'] = torch.FloatTensor(globalFeatures)

        dgl.save_graphs(os.path.join(savePath, 'CNT_Dataset.dgl'), graphs, labels=labs)

        return graphs

