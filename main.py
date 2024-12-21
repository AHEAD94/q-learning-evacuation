#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:13:32 2020

@author: Dae-ha Yoo

© 2020 Dae-ha Yoo <ulysses94@naver.com>
"""

#######################################################################################
# [This program is for finding evacuation path on architectural drawing using Q-leaning]
# Please follow these steps.
# 1. Run and select starting point. (click red dot)
# 2. Press any key.
# 3. Select ending point.
# 4. Press any key and ckeck the graph on openCV window.
# 5. Press any key again.
# 6. After learning phase, you can find the evacuation path.
#######################################################################################

import math, random
import cv2

import pylab as pl
import networkx as nx
import numpy as np


class Node:
    def __init__(self, number=None, x=None, y=None):
        self.number = number
        self.x = None
        self.y = None
        if x is not None:
            self.x = x
        else:
            self.x = int(random.random() * 200)
        if y is not None:
            self.y = y
        else:
            self.y = int(random.random() * 200)

    def getNumber(self):
        return self.number

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distanceTo(self, node):
        xDistance = abs(self.getX() - node.getX())
        yDistance = abs(self.getY() - node.getY())
        distance = math.sqrt((xDistance * xDistance) + (yDistance * yDistance))
        return distance

    def __repr__(self):
        return "(" + str(self.getX()) + ", " + str(self.getY()) + ")"


class TourManager:
    srcNode = []
    transNodes = []
    dstNode = []

    def addNode(self, node):
        self.transNodes.append(node)

    def getNode(self, index):
        return self.transNodes[index]

    def numberOfNodes(self):
        return len(self.transNodes)

    def selectSrcNode(self, node):
        self.srcNode.append(node)

    def selectDstNode(self, node):
        self.dstNode.append(node)

    def getSrcNode(self, index):
        return self.srcNode[index]

    def getDstNode(self, index):
        return self.dstNode[index]


def arrangeDots(height, width, interX, interY):
    x = interX
    y = interY
    nodeNum = 0

    for i in range(0, int(height / y)):
        for j in range(0, int(width / x)):
            tourmanager.addNode(Node(number=nodeNum, x=15 + j * x, y=15 + i * y))
            nodeNum += 1
            cv2.circle(
                map_original,
                center=(15 + j * x, 15 + i * y),
                radius=1, color=(0, 0, 255), thickness=-1,
                lineType=cv2.LINE_AA
            )

    for i in range(0, tourmanager.numberOfNodes()):
        cv2.putText(
            map_original,
            text=str(tourmanager.getNode(i).getNumber()),
            org=(tourmanager.getNode(i).getX() + 5, tourmanager.getNode(i).getY() + 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.3,
            color=(0, 0, 255)
        )

    print("\n[Nx X Ny]:", int(width / x), int(height / y))
    cv2.imshow('map', map_original)


def mouseClickEvent_selectStarting(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        targetNode = Node(x=x, y=y)

        for i in range(0, tourmanager.numberOfNodes()):
            if tourmanager.getNode(i).distanceTo(targetNode) < 10:
                tourmanager.selectSrcNode(tourmanager.getNode(i))
                print("\nThe starting point is selected.")
                print(tourmanager.getSrcNode(0))


def mouseClickEvent_selectEnding(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        targetNode = Node(x=x, y=y)

        for i in range(0, tourmanager.numberOfNodes()):
            if tourmanager.getNode(i).distanceTo(targetNode) < 10:
                tourmanager.selectDstNode(tourmanager.getNode(i))
                print("\nThe Ending point is selected.")
                print(tourmanager.getDstNode(0))


def mouseClickEvent_getColor(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        B = map_original[y, x, 0]
        G = map_original[y, x, 1]
        R = map_original[y, x, 2]
        print("\nR:", R)
        print("G:", G)
        print("B:", B)


def mouseClickEvent_selectNode(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        targetNode = Node(x=x, y=y)

        for i in range(0, tourmanager.numberOfNodes()):
            if tourmanager.getNode(i).distanceTo(targetNode) < 10:
                print("node index:", i)


random.seed(1)
np.random.seed(2)


# Path finding with Q-Learning
class QSPF:
    def __init__(self, graph):
        self.grap = graph
        self.adj_matrix = nx.adjacency_matrix(graph).todense()
        self.num_nodes = len(self.adj_matrix)
        self.adj_matrix = nx.adjacency_matrix(graph, nodelist=range(self.num_nodes)).toarray()

    def epsilon_greedy(self, s_curr, q, epsilon):  # exploraiton vs exploitation
        potential_next_states = np.where(np.array(self.adj_matrix[s_curr]) > 0)[0]
        if random.random() > epsilon:
            q_of_next_states = q[s_curr][potential_next_states]
            s_next = potential_next_states[np.argmax(q_of_next_states)]
        else:
            s_next = random.choice(potential_next_states)
        return s_next

    def Q_learning(self, start, goal, num_epoch=200, gamma=0.5, epsilon=0.05, alpha=0.1):
        start = start
        goal = goal
        num_epoch = num_epoch
        gamma = gamma
        epsilon = epsilon
        alpha = alpha

        paths_len = []
        Q_table = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(1, num_epoch + 1):
            current = start
            path = [current]
            path_len = 0

            while (True):
                next_state = self.epsilon_greedy(current, Q_table, epsilon=epsilon)
                next_next = self.epsilon_greedy(next_state, Q_table, epsilon=-0.2)

                reward = -self.adj_matrix[current][next_state]
                print("reward: ", reward)
                delta = reward + gamma * Q_table[next_state, next_next] - Q_table[current, next_state]

                Q_table[current, next_state] = Q_table[current, next_state] + alpha * delta

                current = next_state
                path_len += -reward
                path.append(current)
                print("current: ", current)
                print("next_state: ", next_state)

                if current == goal:
                    break
            paths_len.append(path_len)
            print("----------------")

        return path


if __name__ == '__main__':
    n_nodes = 0

    # Load the map
    map_original = cv2.imread('test_small.png')
    # map_original = cv2.imread('test_big.png')
    height, width, channel = map_original.shape
    print("\n[image]\nheight:", height, "width:", width)
    cv2.imshow('map', map_original)

    # Set tour and nodes
    interX = 60  # default: 30
    interY = 60
    Nx = int(width / interX)
    Ny = int(height / interY)
    tourmanager = TourManager()
    arrangeDots(height, width, interX, interY)

    # Set n_nodes
    n_nodes = tourmanager.numberOfNodes()
    print("nodes:", n_nodes)

    # Set the source point
    srcIndex = -1
    cv2.setMouseCallback('map', mouseClickEvent_selectStarting)
    cv2.waitKey(0)
    for i in range(0, tourmanager.numberOfNodes()):
        if tourmanager.getSrcNode(0).getX == tourmanager.getNode(i).getX and tourmanager.getSrcNode(
                0).getY == tourmanager.getNode(i).getY:
            srcIndex = i
            print("Selected source node index is", i)

    # Set the destination point
    dstIndex = -1
    cv2.setMouseCallback('map', mouseClickEvent_selectEnding)
    cv2.waitKey(0)
    for i in range(0, tourmanager.numberOfNodes()):
        if tourmanager.getDstNode(0).getX == tourmanager.getNode(i).getX and tourmanager.getDstNode(
                0).getY == tourmanager.getNode(i).getY:
            dstIndex = i
            print("Selected destination node index is", i)

    # Initiate graph
    graph = [[0 for col in range(n_nodes)] for row in range(n_nodes)]
    graph = np.zeros((n_nodes, n_nodes))

    # Draw graph
    lineThickness = 1
    isWall = 0
    for i in range(0, tourmanager.numberOfNodes()):

        # Check wall + upper node
        if i - Nx >= 0:
            for j in range(0, interY):
                B = map_original[tourmanager.getNode(i).getY() - j, tourmanager.getNode(i).getX(), 0]
                G = map_original[tourmanager.getNode(i).getY() - j, tourmanager.getNode(i).getX(), 1]
                R = map_original[tourmanager.getNode(i).getY() - j, tourmanager.getNode(i).getX(), 2]

                if G > R and G > B and G - R > 100 and G - B > 100:
                    isWall = 1
                    break

            if isWall == 0:
                cv2.line(
                    map_original,
                    pt1=(tourmanager.getNode(i).getX(), tourmanager.getNode(i).getY()),
                    pt2=(tourmanager.getNode(i - Nx).getX(), tourmanager.getNode(i - Nx).getY()),
                    color=(255, 0, 0),
                    thickness=lineThickness,
                    lineType=cv2.LINE_AA
                )
                cv2.imshow('map', map_original)
                graph[i][i - Nx] = 1
            else:
                isWall = 0

        # Check wall + lower node
        if i + Nx < tourmanager.numberOfNodes():
            for j in range(0, interY):
                B = map_original[tourmanager.getNode(i).getY() + j, tourmanager.getNode(i).getX(), 0]
                G = map_original[tourmanager.getNode(i).getY() + j, tourmanager.getNode(i).getX(), 1]
                R = map_original[tourmanager.getNode(i).getY() + j, tourmanager.getNode(i).getX(), 2]

                if G > R and G > B and G - R > 100 and G - B > 100:
                    isWall = 1
                    break

            if isWall == 0:
                cv2.line(
                    map_original,
                    pt1=(tourmanager.getNode(i).getX(), tourmanager.getNode(i).getY()),
                    pt2=(tourmanager.getNode(i + Nx).getX(), tourmanager.getNode(i + Nx).getY()),
                    color=(255, 0, 0),
                    thickness=lineThickness,
                    lineType=cv2.LINE_AA
                )
                cv2.imshow('map', map_original)
                graph[i][i + Nx] = 1
            else:
                isWall = 0

        # Check wall + right node
        if i + 1 < tourmanager.numberOfNodes() and i % Nx != Nx - 1:
            for j in range(0, interX):
                B = map_original[tourmanager.getNode(i).getY(), tourmanager.getNode(i).getX() + j, 0]
                G = map_original[tourmanager.getNode(i).getY(), tourmanager.getNode(i).getX() + j, 1]
                R = map_original[tourmanager.getNode(i).getY(), tourmanager.getNode(i).getX() + j, 2]

                if G > R and G > B and G - R > 100 and G - B > 100:
                    isWall = 1
                    break

            if isWall == 0:
                cv2.line(
                    map_original,
                    pt1=(tourmanager.getNode(i).getX(), tourmanager.getNode(i).getY()),
                    pt2=(tourmanager.getNode(i + 1).getX(), tourmanager.getNode(i + 1).getY()),
                    color=(255, 0, 0),
                    thickness=lineThickness,
                    lineType=cv2.LINE_AA
                )
                cv2.imshow('map', map_original)
                graph[i][i + 1] = 1
            else:
                isWall = 0

        # Check wall + left node
        if i - 1 >= 0 and i % Nx != 0:
            for j in range(0, interX):
                B = map_original[tourmanager.getNode(i).getY(), tourmanager.getNode(i).getX() - j, 0]
                G = map_original[tourmanager.getNode(i).getY(), tourmanager.getNode(i).getX() - j, 1]
                R = map_original[tourmanager.getNode(i).getY(), tourmanager.getNode(i).getX() - j, 2]

                if G > R and G > B and G - R > 100 and G - B > 100:
                    isWall = 1
                    break

            if isWall == 0:
                cv2.line(
                    map_original,
                    pt1=(tourmanager.getNode(i).getX(), tourmanager.getNode(i).getY()),
                    pt2=(tourmanager.getNode(i - 1).getX(), tourmanager.getNode(i - 1).getY()),
                    color=(255, 0, 0),
                    thickness=lineThickness,
                    lineType=cv2.LINE_AA
                )
                cv2.imshow('map', map_original)
                graph[i][i - 1] = 1
            else:
                isWall = 0

    print("\n[result matrix]\n", graph)

    # Get the color
    # cv2.setMouseCallback('map', mouseClickEvent_getColor)

    # Get the point
    cv2.setMouseCallback('map', mouseClickEvent_selectNode)

    cv2.waitKey(0)

    # Learning part
    G = nx.DiGraph(graph)
    rl = QSPF(G)
    res = rl.Q_learning(start=tourmanager.getSrcNode(0).getNumber(), goal=tourmanager.getDstNode(0).getNumber())
    for i in range(len(res) - 1):
        cv2.line(
            map_original,
            pt1=(tourmanager.getNode(res[i]).getX(), tourmanager.getNode(res[i]).getY()),
            pt2=(tourmanager.getNode(res[i + 1]).getX(), tourmanager.getNode(res[i + 1]).getY()),
            color=(0, 0, 255),
            thickness=lineThickness + 1,
            lineType=cv2.LINE_AA
        )
    cv2.imshow('map', map_original)
    print("path ", res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

