from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from wekaI import Weka
from builtins import range
from builtins import object
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
from random import randint


class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        self.weka= Weka()
        self.weka.start_jvm()
        
        
    
    #x=[6,"East",-1, 1,1,0,1]
    #print(self.weka.predict("j48.model", x, "./training_tutorial1_classification_filter1.arff"))
    
    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP
    
    

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)
    
    
    
    def printLineData(self, gameState):
        
        #TUTORIAL1_CLASSIFICATION_FILTER1
        """
    
        state = str(gameState.getPacmanPosition()[1]) + ',' #py
        state +=  str(gameState.data.agentStates[0].getDirection()) + ',' #PDir
        state += str(gameState.getScore()) + ','
        if "North" in gameState.getLegalPacmanActions():
            state+=str(1) + "," #north in legal actions
        else:
            state+=str(0)+ ","
        
        if "East" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #east in legal actions
        else:
            state+=str(0)+ ","
        
        state += str(gameState.getNumFood()) + ',' #num of food
        if "North" in gameState.getLegalPacmanActions() and "South" in gameState.getLegalPacmanActions() :
            state+=str(2) + "," #north in legal actions
        elif "North" in gameState.getLegalPacmanActions():
            state+=str(1)+ ","
        elif "South" in gameState.getLegalPacmanActions():
            state+=str(1)+ ","
        else:
            state+=str(0)+ ","
        
        return state
    """
    #NEW INSTANCES
        state = str(gameState.getPacmanPosition()[0]) + ','+ str(gameState.getPacmanPosition()[1]) + ',' #pacman position
        state +=  str(gameState.data.agentStates[0].getDirection()) + ',' #pacman direction
        for i in range(0, len(gameState.data.ghostDistances)):
            if gameState.data.ghostDistances[i] == None:
                state += str(-1) + ','
            else:
                state += str(gameState.data.ghostDistances[i]) + ',' #distance to the ghosts 
    
        if "North" in gameState.getLegalPacmanActions():
            state+=str(1) + "," #north in legal actions
        else:
            state+=str(0)+ ","
        if "South" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #south in legal actions
        else:
            state+=str(0)+ ","
            
        if "West" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #west in legal actions
        else:
            state+=str(0)+ ","
            
        if "East" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #east in legal actions
        else:
            state+=str(0)+ ","
        
    
        return state
    

    """
    
    def printLineData(self, gameState):
        state = str(gameState.getPacmanPosition()[0]) + ','+ str(gameState.getPacmanPosition()[1]) + ',' #pacman position
        state +=  str(gameState.data.agentStates[0].getDirection()) + ',' #pacman direction
        state += str(gameState.getNumAgents()-1) + ',' #number of ghosts
        state += str(sum(gameState.getLivingGhosts())) + ',' #num living ghosts
        for i in range(0, len(gameState.data.ghostDistances)):
            if gameState.data.ghostDistances[i] == None:
                state += str(-1) + ','
            else:
                state += str(gameState.data.ghostDistances[i]) + ',' #distance to the ghosts 
        state += str(gameState.getScore()) + ','
        if "North" in gameState.getLegalPacmanActions():
            state+=str(1) + "," #north in legal actions
        else:
            state+=str(0)+ ","
        if "South" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #south in legal actions
        else:
            state+=str(0)+ ","
            
        if "West" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #west in legal actions
        else:
            state+=str(0)+ ","
            
        if "East" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #east in legal actions
        else:
            state+=str(0)+ ","
        
        state += str(gameState.getNumFood()) + ',' #num of food
        if gameState.getDistanceNearestFood() == None:
            state += str(-1)
        else:
            state += str(gameState.getDistanceNearestFood()) #distance to the nearest food
 
        return state
    
    """
    
    #python busters.py -p BustersKeyboardAgent -l openHunt -g RandomGhost

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState): #en vez de random, elegir la minima distancia a un objetivo
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        #####To find the mazeDistance between any two positions, use:
        #####self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class BasicAgentAA(BustersAgent):
    
    

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        self.list=[]
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())############
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())##############
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())#######
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)#----------------
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts()) #########
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore()) 
        
        
class BasicAgentAA(BustersAgent):
    
    

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        self.list=[]
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())############
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())##############
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())#######
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)#----------------
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts()) #########
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore()) 
        
        
    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        #self.printInfo(gameState)
        print(self.printLineData(gameState))
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        
        """
        move_random = random.randint(0,3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
       
       
       
        """
        pos1=gameState.getPacmanPosition()
        minghost=4567876567
        posghost=0
        
        
        
        for i in range(0,len(gameState.data.ghostDistances)):
            pos2=gameState.getGhostPositions()[i]
            if self.distancer.getDistance(pos1,pos2)<minghost  and gameState.getLivingGhosts()[i+1]==True: 
                minghost=self.distancer.getDistance(pos1,pos2)
                posghost=pos2
                
        
        
        x1,y1=pos1
        x2,y2=posghost
        
        if x1<x2 or x1==x2:
            if y1<y2:
                if Directions.NORTH in legal and str(gameState.data.agentStates[0].getDirection()) !="South": 
                    move = Directions.NORTH
                    
                             
                else:
                    if Directions.EAST in legal and str(gameState.data.agentStates[0].getDirection()) !="West":
                        move = Directions.EAST
                        
                    
                    
                    elif Directions.WEST in legal and str(gameState.data.agentStates[0].getDirection()) !="East":
                        move = Directions.WEST
                        
                    
                    elif  Directions.SOUTH in legal and str(gameState.data.agentStates[0].getDirection()) !="North":
                        move = Directions.SOUTH
                    
                    
                    
                    else:
                         if Directions.NORTH in legal:
                             move = Directions.NORTH
                         else:
                            if Directions.EAST in legal :
                                move = Directions.EAST
                               
                            
                            elif Directions.WEST in legal :
                                move = Directions.WEST
                               
                            
                            else:
                                move = Directions.SOUTH
                             
                        
                    
                        
                    
                    
                         
            if y1==y2:
                if Directions.EAST in legal  and str(gameState.data.agentStates[0].getDirection())!="West":
                    move = Directions.EAST
                   
                else:
                    if Directions.NORTH in legal and str(gameState.data.agentStates[0].getDirection())!="South" :
                        move = Directions.NORTH
                        
                       
                    elif Directions.SOUTH in legal and str(gameState.data.agentStates[0].getDirection())!="North":
                        move = Directions.SOUTH
                       
                    
                    elif Directions.WEST in legal and str(gameState.data.agentStates[0].getDirection())!="East":
                        move = Directions.WEST
                    
                    else:
                        
                        
                        if Directions.EAST in legal:
                             move = Directions.EAST
                        else:
                            if Directions.NORTH in legal :
                                move = Directions.NORTH
                                
                        
                            elif Directions.SOUTH in legal :
                                move = Directions.SOUTH
                                
                            
                            else:
                                move = Directions.WEST
            if y1>y2:
                if Directions.SOUTH in legal and str(gameState.data.agentStates[0].getDirection())!="North":
                        move = Directions.SOUTH
                       
                else:
                    
                    if Directions.EAST in legal and str(gameState.data.agentStates[0].getDirection())!="West":
                        move = Directions.EAST
                       
                        
                    elif Directions.WEST in legal and str(gameState.data.agentStates[0].getDirection())!="East":
                        move = Directions.WEST
                      
                        
                    elif  Directions.NORTH in legal and str(gameState.data.agentStates[0].getDirection())!="South" :
                        move = Directions.NORTH
                   
                    else:
                        if Directions.SOUTH in legal:
                             move = Directions.SOUTH
                        else:
                            if Directions.EAST in legal:
                                move = Directions.EAST
                          
                            elif Directions.WEST in legal:
                                move = Directions.WEST
                       
                        
                            else:
                                move = Directions.NORTH
                            
                                 
                            
                        
                    
                            
                 
                    
                        
        
        if x1>x2:
            if y1<y2:
                if Directions.NORTH in legal and str(gameState.data.agentStates[0].getDirection())!="South":
                    move = Directions.NORTH
                   
                    
                else:
                    if Directions.WEST in legal and str(gameState.data.agentStates[0].getDirection())!="East":
                        move = Directions.WEST
                      
                    elif Directions.EAST in legal and str(gameState.data.agentStates[0].getDirection())!="West":
                        move = Directions.EAST
                      
                    elif Directions.SOUTH in legal and str(gameState.data.agentStates[0].getDirection())!="North":
                        move = Directions.SOUTH
                    
                    else:
                        if Directions.NORTH in legal :
                            move = Directions.NORTH
                           
                                
                        else:
                            if Directions.WEST in legal :
                                move = Directions.WEST
                                
                            elif Directions.EAST in legal :
                                move = Directions.EAST
                               
                            elif Directions.SOUTH in legal :
                                move = Directions.SOUTH
                               
         
                
            if y1==y2:
                if Directions.WEST in legal and str(gameState.data.agentStates[0].getDirection())!="East":
                    move = Directions.WEST
                  
                else:
                    if Directions.NORTH in legal and str(gameState.data.agentStates[0].getDirection())!="South":
                        move = Directions.NORTH
                        
                    elif Directions.SOUTH in legal and str(gameState.data.agentStates[0].getDirection())!="North":
                        move = Directions.SOUTH
                     
                    elif Directions.EAST in legal and str(gameState.data.agentStates[0].getDirection())!="West":
                        move = Directions.EAST
                        
                    else:
                        if Directions.WEST in legal :
                            move = Directions.WEST
                           
                        else:
                            if Directions.NORTH in legal :
                                move = Directions.NORTH
                                 
                            elif Directions.SOUTH in legal :
                                move = Directions.SOUTH
                               
                            else:
                                move = Directions.EAST
                        
                        
                        
            if y1>y2:
                if Directions.SOUTH in legal and str(gameState.data.agentStates[0].getDirection())!="North":
                        move = Directions.SOUTH
                        
                else:
                    if Directions.WEST in legal and str(gameState.data.agentStates[0].getDirection())!="East":
                        move = Directions.WEST
                      
                        
                    elif Directions.EAST in legal and str(gameState.data.agentStates[0].getDirection())!="West":
                        move = Directions.EAST
                       
                    
                    elif  Directions.NORTH in legal and str(gameState.data.agentStates[0].getDirection())!="South" :
                        move = Directions.NORTH
                        
                    
                    else:
                        if Directions.SOUTH in legal :
                            move = Directions.SOUTH
                                
                               
                        else:
                            if Directions.WEST in legal :
                                move = Directions.WEST
                    
                            elif Directions.EAST in legal :
                                move = Directions.EAST
                                
                    
                            else:
                                move = Directions.NORTH
                                
                        
                                   
        
        return move 

   
    def printLineData(self, gameState):
      
    #NEW INSTANCES
        state = str(gameState.getPacmanPosition()[0]) + ','+ str(gameState.getPacmanPosition()[1]) + ',' #pacman position
        state +=  str(gameState.data.agentStates[0].getDirection()) + ',' #pacman direction
        for i in range(0, len(gameState.data.ghostDistances)):
            if gameState.data.ghostDistances[i] == None:
                state += str(-1) + ','
            else:
                state += str(gameState.data.ghostDistances[i]) + ',' #distance to the ghosts 
    
        if "North" in gameState.getLegalPacmanActions():
            state+=str(1) + "," #north in legal actions
        else:
            state+=str(0)+ ","
        if "South" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #south in legal actions
        else:
            state+=str(0)+ ","
            
        if "West" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #west in legal actions
        else:
            state+=str(0)+ ","
            
        if "East" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #east in legal actions
        else:
            state+=str(0)+ ","
        
    
        return state
    
class weka(BustersAgent):
    def printLineData(self, gameState):
        
        #TUTORIAL1_CLASSIFICATION_FILTER1
        """
    
        state = str(gameState.getPacmanPosition()[1]) + ',' #py
        state +=  str(gameState.data.agentStates[0].getDirection()) + ',' #PDir
        state += str(gameState.getScore()) + ','
        if "North" in gameState.getLegalPacmanActions():
            state+=str(1) + "," #north in legal actions
        else:
            state+=str(0)+ ","
        
        if "East" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #east in legal actions
        else:
            state+=str(0)+ ","
        
        state += str(gameState.getNumFood()) + ',' #num of food
        if "North" in gameState.getLegalPacmanActions() and "South" in gameState.getLegalPacmanActions() :
            state+=str(2) + "," #north in legal actions
        elif "North" in gameState.getLegalPacmanActions():
            state+=str(1)+ ","
        elif "South" in gameState.getLegalPacmanActions():
            state+=str(1)+ ","
        else:
            state+=str(0)+ ","
        
        return state
    """
    #NEW INSTANCES
        state = str(gameState.getPacmanPosition()[0]) + ','+ str(gameState.getPacmanPosition()[1]) + ',' #pacman position
        state +=  str(gameState.data.agentStates[0].getDirection()) + ',' #pacman direction
        for i in range(0, len(gameState.data.ghostDistances)):
            if gameState.data.ghostDistances[i] == None:
                state += str(-1) + ','
            else:
                state += str(gameState.data.ghostDistances[i]) + ',' #distance to the ghosts 
    
        if "North" in gameState.getLegalPacmanActions():
            state+=str(1) + "," #north in legal actions
        else:
            state+=str(0)+ ","
        if "South" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #south in legal actions
        else:
            state+=str(0)+ ","
            
        if "West" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #west in legal actions
        else:
            state+=str(0)+ ","
            
        if "East" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #east in legal actions
        else:
            state+=str(0)+ ","
        
    
        return state
    """
    def printLineData(self, gameState):
        state = str(gameState.getPacmanPosition()[0]) + ','+ str(gameState.getPacmanPosition()[1]) + ',' #pacman position
        state +=  str(gameState.data.agentStates[0].getDirection()) + ',' #pacman direction
        state += str(gameState.getNumAgents()-1) + ',' #number of ghosts
        state += str(sum(gameState.getLivingGhosts())) + ',' #num living ghosts
        for i in range(0, len(gameState.data.ghostDistances)):
            if gameState.data.ghostDistances[i] == None:
                state += str(-1) + ','
            else:
                state += str(gameState.data.ghostDistances[i]) + ',' #distance to the ghosts 
        state += str(gameState.getScore()) + ','
        if "North" in gameState.getLegalPacmanActions():
            state+=str(1) + "," #north in legal actions
        else:
            state+=str(0)+ ","
        if "South" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #south in legal actions
        else:
            state+=str(0)+ ","
            
        if "West" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #west in legal actions
        else:
            state+=str(0)+ ","
            
        if "East" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #east in legal actions
        else:
            state+=str(0)+ ","
        
        state += str(gameState.getNumFood()) + ',' #num of food
        if gameState.getDistanceNearestFood() == None:
            state += str(-1)
        else:
            state += str(gameState.getDistanceNearestFood()) #distance to the nearest food
 
        return state
    """
   
    
    def chooseAction(self, gameState): #en vez de random, elegir la minima distancia a un objetivo
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        x=self.printLineData(gameState).split(",")
        x.pop()
        print(x)
        #a=self.weka.predict("j48.model", x, "./training_tutorial1_classification_filter1.arff")
        #a=self.weka.predict("ibk1.model", x, "./training_tutorial1_classification_filter1.arff")
        #a=self.weka.predict("j48_2.model", x, "./NEWinstances_tutorial1.arff")
        #a=self.weka.predict("randomforest_tutorial1.model", x, "./NEWinstances_tutorial1.arff")
        #a=self.weka.predict("ibk_tutorial1.model", x, "./NEWinstances_tutorial1.arff")
        #a=self.weka.predict("j48_keyboard.model", x, "./NEWinstances_keyboard.arff")
        #a=self.weka.predict("randomforest_keyboard.model", x, "./NEWinstances_keyboard.arff")
        a=self.weka.predict("ibk_keyboard.model", x, "./NEWinstances_keyboard.arff")
        
        if a not in gameState.getLegalPacmanActions():
            
            move_random = random.randint(0, 3)
            if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
            if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
            if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
            if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
            a=move
        
        return a

    
    

    
   
#cd /home/ml-uc3m/Downloads
#cd pacman
#python3 busters.py -p weka -l NEWmaze -g RandomGhost

    
    
