from typing import ForwardRef
import math
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv as csv
import re as re
import os, shutil
import time
import scipy

path = "./plots"
reportFileName = "SocialNetworks_ReportData.txt"
reportFile = open(reportFileName, "w")

def flatten(seq, container=None):
    if container is None:
        container = []
    for s in seq:
        try:
            iter(s)  # check if it's iterable
        except TypeError:
            container.append(s)
        else:
            flatten(s, container)
    return container

def generateCommaSeparatedFile(source, destination, run = False):
    if(run):
        f = open(destination, "w")
        f.writelines("source_id,target_id,timestamp")
        f.write("\n")
        with open(source,'rt') as file:
            data = csv.reader(file)
            for row in data:
                new_row_with_comma = [x.replace(" ", ",") for x in row]
                f.writelines(new_row_with_comma)
                f.write("\n")
        print("Ready!!!")

def createAndEmptyPlotsFolder(path):
    try: 
        os.mkdir(path) 
    except OSError as error: 
        print("Folder plots exists!")
        print("Deleting plots if exist!")
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

def readTxtFileAndProduceNodes(testing = True):
    if(testing): 
        nodes = pd.read_csv('./sampleTestData_source2.txt')
    else:
        nodes = pd.read_csv('sx-stackoverflowNewList.txt') # All of the data
    return nodes

def calculateMinMaxDTValues(nodes):
    tmin = nodes['timestamp'].min()
    tmax = nodes['timestamp'].max()
    return [tmin, tmax, tmax - tmin]

def calculateDt(userInput, DT):
    dt = round(DT / int(userInput))
    return dt

def calculateTj(value, minValuesOfTimestamp, dt):
    return minValuesOfTimestamp + (value * dt)

def produceT(userInput, minValuesOfTimestamp, dt):
    T = []
    for x in range(0, int(userInput)):
        tj_1 = calculateTj(x, minValuesOfTimestamp, dt) 
        tj = calculateTj(x+1, minValuesOfTimestamp, dt) 
        T.append([tj_1, tj])
        reportFile.write(f"tj-1={tj_1}\n")
        reportFile.write(f"tj={tj}\n")
    return T

def produceVE(T, nodes):
    V = []
    E = []
    for timestamps in T:
        if(len(timestamps) > 0):
            V.append([])
            E.append([])
            for row in nodes.values:
                # row[2] is the value of the timestamp
                if row[2] >= timestamps[0] and row[2] < timestamps[1]:
                    V[len(V) - 1].append([row[0], row[1]])
                    reportFile.write(f"V[{len(V)-1}] = row[0] = {row[0]}, row[1] = {row[1]}\n")
                    E[len(E) - 1].append(row[2])
                    reportFile.write(f"E[{len(E) - 1}] = {row[2]}\n")
    return V,E

def plotSubGraphs(vertices, interactive = False):
    plt.interactive = interactive
    noOfSubGraphs = len(V)
    # print("No of Graphs: ", noOfSubGraphs)
    reportFile.write(f"No of Graphs: {noOfSubGraphs}\n")
    counter = 0
    max_iterations = 100000
    for subVertices in vertices: # for subgraph in range(0, noOfSubGraphs - 1)
        # print("\n********* Graph ", counter, " *********\n")
        reportFile.write(f"\n********* Graph {counter} *********\n")
        if(len(subVertices) > 0):
            graph = nx.Graph()
            diGraph = nx.DiGraph()
            for nodes in subVertices:
                graph.add_node(nodes[0])
                graph.add_node(nodes[1])
                graph.add_edge(nodes[0], nodes[1])

                diGraph.add_node(nodes[0])
                diGraph.add_node(nodes[1])
                diGraph.add_edge(nodes[0], nodes[1])

            nx.draw_networkx(graph, with_labels=True)
            plt.title("Sub Graph - " + str(counter))
            plt.savefig(path + "/SubGraph-" + str(counter))
            if(interactive): plt.show()

            # print("Degree Centrality : ")
            degreeCentrality = nx.degree_centrality(graph)
            reportFile.write(f"\nDegree Centrality : {degreeCentrality}\n")
            # print(degreeCentrality)
            centralityMeasures = nx.Graph()
            centralityMeasures.add_nodes_from(degreeCentrality.keys())
            for k, v in degreeCentrality.items():
                centralityMeasures.add_edges_from(([(k,v)]))
            nx.draw_networkx(centralityMeasures, with_labels=True)
            plt.title("Degree Centrality, Sub Graph - " + str(counter))
            plt.savefig(path + "/Degree-Centrality_Sub Graph-" + str(counter))
            if(interactive): plt.show()

            # print("In Degree Centrality : ")
            inDegreeCentrality = nx.in_degree_centrality(diGraph)
            # print(inDegreeCentrality)
            reportFile.write(f"\nIn Degree Centrality : {inDegreeCentrality}\n")
            centralityMeasures = nx.Graph()
            centralityMeasures.add_nodes_from(inDegreeCentrality.keys())
            for k, v in inDegreeCentrality.items():
                centralityMeasures.add_edges_from(([(k,v)]))
            nx.draw_networkx(centralityMeasures, with_labels=True)
            plt.title("In Degree Centrality, Sub Graph - " + str(counter))
            plt.savefig(path + "/In-Degree-Centrality_SubGraph-" + str(counter))
            if(interactive): plt.show()

            # print("Out Degree Centrality : ")
            outDegreeCentrality = nx.out_degree_centrality(diGraph)
            # print(outDegreeCentrality)
            reportFile.write(f"\nOut Degree Centrality : {outDegreeCentrality}\n")
            centralityMeasures = nx.Graph()
            centralityMeasures.add_nodes_from(outDegreeCentrality.keys())
            for k, v in outDegreeCentrality.items():
                centralityMeasures.add_edges_from(([(k,v)]))
            nx.draw_networkx(centralityMeasures, with_labels=True)
            plt.title("Out Degree Centrality, Sub Graph - " + str(counter))
            plt.savefig(path + "/Out-Degree-Centrality_SubGraph-" + str(counter))
            if(interactive): plt.show()

            # print("Closeness Centrality : ")
            closenessCentrality = nx.closeness_centrality(graph, u=None, distance=None, wf_improved=True)
            # print(closenessCentrality)
            reportFile.write(f"\nCloseness Centrality : {closenessCentrality}\n")
            centralityMeasures = nx.Graph()
            centralityMeasures.add_nodes_from(closenessCentrality.keys())
            for k, v in closenessCentrality.items():
                centralityMeasures.add_edges_from(([(k,v)]))
            nx.draw_networkx(centralityMeasures, with_labels=True)
            plt.title("Closeness-Centrality_Sub Graph-" + str(counter))
            plt.savefig(path + "/Closeness-Centrality_SubGraph-" + str(counter))
            if(interactive): plt.show() 
            
            # print("Betweenness Centrality : ")
            betweennessCentrality = nx.betweenness_centrality(graph, k=None, normalized=True, weight=None, endpoints=False, seed=None)
            # print(betweennessCentrality)
            reportFile.write(f"\nBetweenness Centrality : {betweennessCentrality}\n")
            centralityMeasures = nx.Graph()
            centralityMeasures.add_nodes_from(betweennessCentrality.keys())
            for k, v in betweennessCentrality.items():
                centralityMeasures.add_edges_from(([(k,v)]))
            nx.draw_networkx(centralityMeasures, with_labels=True)
            plt.title("Betweenness Centrality, Sub Graph - " + str(counter))
            plt.savefig(path + "/Betweenness-Centrality_SubGraph-" + str(counter))
            if(interactive): plt.show()

            # print("Eigenvector Centrality : ")
            try:
                eigenvectorCentrality = nx.eigenvector_centrality(graph, max_iter=max_iterations, tol=1e-06, nstart=None, weight=None)
            except:
                reportFile.write(f"\nEigenvector Centrality : Couldn't calculate\n")
            else:
                reportFile.write(f"\nEigenvector Centrality : {eigenvectorCentrality}\n")
                # print(eigenvectorCentrality)
                centralityMeasures = nx.Graph()
                centralityMeasures.add_nodes_from(eigenvectorCentrality.keys())
                for k, v in eigenvectorCentrality.items():
                    centralityMeasures.add_edges_from(([(k,v)]))
                nx.draw_networkx(centralityMeasures, with_labels=True)
                plt.title("Eigenvector Centrality, Sub Graph - " + str(counter))
                plt.savefig(path + "/Eigenvector-Centrality_SubGraph-" + str(counter))
                if(interactive): plt.show()

            # print("Katz Centrality : ")
            phi = (1 + math.sqrt(5)) / 2.0  # largest eigenvalue of adj matrix
            try:
                katzCentrality = nx.katz_centrality(graph, 1 / phi - 0.01, 1.0, max_iterations, tol=1e-06)
            except:
                reportFile.write(f"\nKatz Centrality : Couldn't Calculate\n")
            else:
                reportFile.write(f"\nKatz Centrality : {katzCentrality}\n")
                centralityMeasures = nx.path_graph(len(V))
                centralityMeasures.add_nodes_from(katzCentrality.keys())
                for k, v in katzCentrality.items():
                    centralityMeasures.add_edges_from(([(k,v)]))
                nx.draw_networkx(centralityMeasures, with_labels=True)
                plt.title("Katz Centrality, Sub Graph - " + str(counter))
                plt.savefig(path + "/Katz-Centrality_SubGraph-" + str(counter))
                if(interactive): plt.show()
        counter += 1

def calculateVEStar(V,E):
    # calculate the similar V* when we inspect Tj, Tj+1 where the time spaces are
    # [tj-1,tj] and [tj,tj+1] <---- V*[tj-1,tj+1] = V[tj-1,tj] intersection V[tj,tj+1]
    # calculate V[tj-1,tj]
    # calculate V[tj,tj+1]
    # find the common ones
    VStar        = {}
    EStar        = {}
    counter      = 0
    fullNodes    = flatten(V)
    fullEdges    = flatten(E)
    noOfVertices = len(fullNodes)
    noOfEdges    = len(fullEdges)
    # print("\nNo Of Vertices: ", noOfVertices, 
    #       "\nNo Of Edges   : ", noOfEdges)
    for vertex in range(0, noOfVertices - 4, 4):
        if(fullNodes[vertex + 1] == fullNodes[vertex + 2]):
            VStar[counter]   = [[fullNodes[vertex], fullNodes[vertex + 1]],[fullNodes[vertex + 1], fullNodes[vertex + 2]]]
            EStar[counter]   = [[fullEdges[int(counter * 2)], fullEdges[int(counter * 2) + 1]]]
        counter += 1
    return VStar, EStar

def calculateSgd(vertices):
    graph = nx.Graph()
    for vertex in vertices.values():
        for v in vertex:
            graph.add_node(v[0])
            graph.add_node(v[1])
            graph.add_edge(v[0], v[1])
    shortestPathsLength = nx.all_pairs_shortest_path_length(graph)
    return shortestPathsLength

def calculateScn(vertices):
    graph = nx.complete_graph(vertices)
    Scn = nx.common_neighbor_centrality(graph)
    return Scn

def calculateSjc(vertices):
    graph = nx.complete_graph(vertices)
    Sjn = nx.jaccard_coefficient(graph)
    return Sjn

def calculateSa(vertices):
    graph = nx.complete_graph(vertices)
    Sa = nx.adamic_adar_index(graph)
    return Sa

def calculateSpa(vertices):
    graph = nx.complete_graph(vertices)
    Spa = nx.preferential_attachment (graph)
    return Spa

userInput  = input("Type a number of time divisions(> 1), e.g. 2: ")
pGDPerCent = input("Type pGD %, e.g. 5: ")
pCNPerCent = input("Type pCN %, e.g. 5: ")
pJCPerCent = input("Type pJC %, e.g. 5: ")
pAPerCent  = input("Type pA  %, e.g. 5: ")
pPAPerCent = input("Type pPA %, e.g. 5: ")

startTime  = time.time()
createAndEmptyPlotsFolder(path)
generateCommaSeparatedFile("sx-stackoverflow.txt", "sx-stackoverflowNewList.txt", False)

nodes = readTxtFileAndProduceNodes(True)
tminMaxDT = calculateMinMaxDTValues(nodes)
minValuesOfTimestamp = tminMaxDT[0]
maxValuesOfTimestamp = tminMaxDT[1]
DT = tminMaxDT[2]
print(tminMaxDT)
dt = calculateDt(userInput,DT)

reportFile.write("tmin=" + str(minValuesOfTimestamp) + "\n")
reportFile.write("tmax=" + str(maxValuesOfTimestamp) + "\n")
reportFile.write("DT=" + str(DT) + "\n")
reportFile.write("dt=" + str(dt) + "\n")

T = produceT(userInput,minValuesOfTimestamp,dt)
V,E = produceVE(T, nodes)
nodes = 0
print("nodes is zeroed")

VStar, EStar = calculateVEStar(V,E)
plotSubGraphs(V, True)
V = 0
E = 0

# shortestPathsLength = calculateSgd(VStar)
# commonNeighbours = calculateScn(VStar)
# jaccardCoefficient = calculateSjc(VStar)
# adamicAdar = calculateSa(VStar)
# preferentialAttachment = calculateSpa(VStar)

# maxPgd = max(shortestPathsLength.__dict__.values())
# maxPcn = max(commonNeighbours.__dict__.values())
# maxPjc = max(jaccardCoefficient.__dict__.values())
# maxPa  = max(adamicAdar.__dict__.values())
# maxPpa = max(preferentialAttachment.__dict__.values())

# print("V*:  ", VStar, "\nE*: ", EStar)
# reportFile.write(f"\nV*:  {VStar}, \nE*: {EStar}\n")

# print("\nPrint All the Shortest Paths\n")
# reportFile.write("\nPrint All the Shortest Paths\n")
# for x in shortestPathsLength:
#     print(x)
#     reportFile.write(x, "\n")

# print("\nCommon Neighbours\n")
# reportFile.write("\nCommon Neighbours\n")
# for u, v, p in commonNeighbours:
#     print(f"({u}, {v}) -> {p}")
#     reportFile.write(f"({u}, {v}) -> {p}\n")

# print("\nJaccard Coefficient\n")
# reportFile.write("\nJaccard Coefficient\n")
# for u, v, p in jaccardCoefficient:
#     print(f"({u}, {v}) -> {p: .8f}")
#     reportFile.write(f"({u}, {v}) -> {p: .8f}\n")

# print("\nAdamic / Adar Index\n")
# reportFile.write("\nAdamic / Adar Index\n")
# for u, v, p in adamicAdar:
#     print(f"({u}, {v}) -> {p: .8f}")
#     reportFile.write(f"({u}, {v}) -> {p: .8f}\n")

# print("\nPreferential Attachment\n")
# reportFile.write("\nPreferential Attachment\n")
# for u, v, p in preferentialAttachment:
#     print(f"({u}, {v}) -> {p: .8f}")
#     reportFile.write(f"({u}, {v}) -> {p: .8f}\n")

# print("maxPgd:  ", maxPgd, "\nmaxPcn: ", maxPcn, "\nmaxPjc: ", maxPjc, "\nmaxPa: ", maxPa, "\nmaxPpa: ", maxPpa)
# reportFile.write(f"maxPgd: {maxPgd}\nmaxPcn: {maxPcn}\nmaxPjc: {maxPjc}\nmaxPa: {maxPa}\nmaxPpa: {maxPpa}")

endTime = time.time()
print("\nTime Elapsed: ", endTime - startTime, "\n")
reportFile.write(f"\nTime Elapsed: {endTime - startTime}\n")
reportFile.close()