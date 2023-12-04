# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util
import json

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint



#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='AgentLearnWeights', second='AgentLearnWeights', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class AgentLearnWeights(CaptureAgent):
    def __init__(self, index, time_for_computing=.1,alpha=0.15,gamma=0.9,epsilon=0.6):
        super().__init__(index, time_for_computing)
        self.start = None
        self.mode='attack'
        self.TotalFood=0
        self.Qvals = util.Counter()  # Q-table to store state-action values, initialized with zeros
        self.alpha = alpha  # learning rate
        self.discount = gamma  # discount factor
        self.epsilon = epsilon  # exploration-exploitation trade-off
        self.last_action=None
        self.savedSuccessor=0
        self.start=0
        self.exploration=0
        self.exploitation=0
        self.flag=False
        self.steps=0    
        self.flagLimit=0
        self.attackWeights=util.Counter()
        self.defenceWeights=util.Counter()
        self.enemyStart=0
        self.indexForLoadandSave=0

    

    
       

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        for enemy_index in self.get_opponents(game_state):
            self.enemyStart = game_state.get_agent_position(enemy_index)
        self.enemyStart
        CaptureAgent.register_initial_state(self, game_state)
        self.TotalFood = len(self.get_food(game_state).as_list())
        # print(self.index)
        if(self.index==0):
            self.indexForLoadandSave=1
        elif(self.index==1):
            self.indexForLoadandSave=1
        elif(self.index==2):
            self.indexForLoadandSave=3
        elif(self.index==3):
            self.indexForLoadandSave=3
        attack_file_path = "AttackWeights" + str(self.indexForLoadandSave) + ".json"
        #self.attackWeights=self.loadWeights(attack_file_path)
        defence_file_path = "DefenceWeights" + str(self.indexForLoadandSave) + ".json"
        self.defenceWeights=self.loadWeights(defence_file_path)
        self.attackWeights=self.loadWeights(attack_file_path)
        self.flagLimit=25 #random.randint(1,100)
        # print(self.start)
        # print(self.attackWeights)
        # print("\n")
        # self.mode='defence'
        # self.saveWeights(self.randomInitWeights(),defence_file_path)
        # self.mode='attack'
        # self.saveWeights(self.randomInitWeights(),attack_file_path)
        

        
     # Make sure to import the json module

    def saveWeights(self, Weights, filename): #only full path works, ask why tmrw
        try:
            # Save the Qvalues dictionary to a JSON file
            print("Inside try block")
            with open(filename, 'w') as file:
                json.dump(Weights, file)
            print("File saved successfully")
            file.close()
        except Exception as e:
            # Handle the exception and print an error message
            print(f"An error occurred: {str(e)}")
        finally:
            print("Done with the operation (whether it succeeded or failed)")


    def loadWeights(self, filename):
        try:
            # Load the Qvalues dictionary from the JSON file
            with open(filename, 'r') as file:
                qvalues = json.load(file)
            print("File loaded successfully")
            return qvalues
        except Exception as e:
            # Handle the exception and print an error message
            print(f"An error occurred while loading the file: {str(e)}")
            return None
        finally:
            print("Done with the operation (whether it succeeded or failed)")

    def CalculateReward(self, state, nextState):
        """
        
        """
        reward = 0.0  
        if(self.mode=="attack"):
                # Penalty for being in the starting position
            if nextState.get_agent_position(self.index) == self.start:
                reward -= 50.0  # Heavily penalize being in the starting position (dying)
            # else: #This could mess with enemy separation, mayeb remove
            #     distance_to_start=self.get_maze_distance(nextState.get_agent_position(self.index), self.start)
            #     reward-=1/distance_to_start

            #penalty for being too close to start
            
            
            # Reward for getting close to food
            current_food_list = self.get_food(state).as_list()
            next_food_list = self.get_food(nextState).as_list()
            current_min_distance = min([self.get_maze_distance(nextState.get_agent_position(self.index), food) for food in current_food_list])
            next_min_distance = min([self.get_maze_distance(nextState.get_agent_position(self.index), food) for food in next_food_list])
            if current_min_distance > next_min_distance:
                reward += 5.0+ 2/next_min_distance  # Reward for getting closer to food
            else: #agen tmoved away from food
                reward-=3.0
                

            # Penalty for getting close to enemies (not Pacman)
            for enemy_index in self.get_opponents(state):
                enemy_position = state.get_agent_position(enemy_index)
                if enemy_position:
                    distance_to_enemy = self.get_maze_distance(nextState.get_agent_position(self.index), enemy_position)
                    if not nextState.get_agent_state(enemy_index).is_pacman:
                        reward -= 2.0 / (distance_to_enemy + 1.0)  # Penalize proximity to non-Pacman enemies

            #Reward for eating food
            food_list = self.get_food(nextState).as_list()
            # reward+= 8.0/len(food_list) or 8.0 #decide whether how much food is left matters
            reward+=8.0
                    # reward+= 2.0/self.TotalFood-len(food_list)

            if(len(self.get_capsules(state))>len(self.get_capsules(nextState))):#ate a capsule
                reward+=0.5 #samll rward since in of its own its not that useful
            
            #Eatinf scared enemies
            for enemy_index in self.get_opponents(state):
                enemy_position = state.get_agent_position(enemy_index)
                #enemy_position_future=nextState.get_agent_position(enemy_index)
                if(enemy_position):
                    distance_to_enemy_future = self.get_maze_distance(nextState.get_agent_position(self.index), enemy_position)
                    distance_to_enemy_now = self.get_maze_distance(state.get_agent_position(self.index), enemy_position)
                    #6 to ensure definteley eaten
                    if(state.get_agent_state(enemy_index).scared_timer>0 and distance_to_enemy_now<1.1 and distance_to_enemy_future>6 ):
                    # if(state.get_agent_state(enemy_index).scared_timer>0 and distance_to_enemy_now<1.1 and enemy_position_future==self.enemyStart):#this crashes bc positon is none so do distances above
                        reward+=15
            
            #bring food home
            if(self.get_score(nextState)>self.get_score(state)):
                reward+=18 +self.get_score(nextState)

            #penalsie getting to close to pacman, maybe unnecessayr with dying penalty
            # for enemy_index in self.get_opponents(state):
            #     enemy_position = state.get_agent_position(enemy_index)
            #     if(enemy_position):
            #         distance_to_enemy = self.get_maze_distance(nextState.get_agent_position(self.index), enemy_position)
            #         if(not enemy_index.is_pacman and state.get_agent_state(enemy_index).scared_timer<=0 ):
            #             reward-=1 / distance_to_enemy


            
            
            
            
            
            
        elif(self.mode=="defence"):    
             # Reward for ctahcingenemy
            for enemy_index in self.get_opponents(state):
                enemy_position = state.get_agent_position(enemy_index)
                if enemy_position:
                    distance_to_enemy = self.get_maze_distance(nextState.get_agent_position(self.index), enemy_position)
                    if(state.get_agent_state(enemy_index).is_pacman and state.get_agent_state(self.index).scared_timer<=0):
                        distance_to_enemy = self.get_maze_distance(nextState.get_agent_position(self.index), enemy_position)
                        if not nextState.get_agent_state(enemy_index).is_pacman:
                            reward += 25.0 / (distance_to_enemy + 1.0)  # encourage proximity to Pacman enemies
                    elif(state.get_agent_state(enemy_index).is_pacman and state.get_agent_state(self.index).scared_timer>0):
                        reward-=state.get_agent_state(self.index).scared_timer/(distance_to_enemy + 1.0)

            
            for enemy_index in self.get_opponents(state):
                enemy_position = state.get_agent_position(enemy_index)
                #enemy_position_future=nextState.get_agent_position(enemy_index)
                if(enemy_position):
                    distance_to_enemy_future = self.get_maze_distance(nextState.get_agent_position(self.index), enemy_position)
                    distance_to_enemy_now = self.get_maze_distance(state.get_agent_position(self.index), enemy_position)
                    #6 to ensure definteley eaten
                    if(state.get_agent_state(enemy_index).is_pacman and distance_to_enemy_now<1.1 and distance_to_enemy_future>6 ):
                    # if(state.get_agent_state(enemy_index).scared_timer>0 and distance_to_enemy_now<1.1 and enemy_position_future==self.enemyStart):#this crashes bc positon is none so do distances above
                        reward+=45

            
                
                
        return reward
    
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
       
        actions = game_state.get_legal_actions(self.index)
        if self.mode == "attack" and "Stop" in actions:
            actions.remove("Stop")
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())
        
        
        
        if(util.flipCoin(self.epsilon)):
          
              toReturn= random.choice(actions)
              self.exploration+=1
        
        else:#no need to check for legal acitons here since compuet actiona alreayd does so
            
            self.exploitation+=1
            toReturn= random.choice(best_actions)
            
            
            
        nextState=self.get_successor(game_state,toReturn)
        reward=self.CalculateReward(game_state,nextState)
        self.update(game_state,toReturn,nextState,reward)
        return toReturn
        

        

        
    
    def final(self, game_state):
        # file_path = "C:\\Users\\ausia\\Desktop\\Pacman\\LAB4\\pacman-agent\\pacman-contest\\agents\\team_name_1\\Qvals" + str(self.index) + ".json"
        attack_file_path = "AttackWeights" + str(self.indexForLoadandSave) + ".json"
        
        defence_file_path = "DefenceWeights" + str(self.indexForLoadandSave) + ".json"
        print("\n")
        print(self.attackWeights)
        
        self.saveWeights(self.attackWeights,attack_file_path)
        self.saveWeights(self.defenceWeights,defence_file_path)
        
        print(f"Exploration Percentage: {100*(self.exploration/(self.exploitation+self.exploration))}%")
        print(f"Exploitation Percentage: {100*(self.exploitation/(self.exploitation+self.exploration))}%")   


    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
       

        
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        if(len(food_list)<self.TotalFood-2):
            self.mode='defence'
        else:

            if(self.invaderCount(game_state)>0):

                self.mode='defence'
            else:
                self.mode='attack'

        
    
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights
        
    def invaderCount(self,successor):
        
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        return len(invaders)

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        allFood=self.get_food(game_state)
        food_carried=game_state.get_agent_state(self.index).num_carrying
        if(self.mode=='attack'):

            
            successor = self.get_successor(game_state, action)
            food_list = self.get_food(successor).as_list()
            capsule_list=self.get_capsules(successor)
            # features['successor_score'] = -len(food_list)  # self.getScore(successor)

            # Compute distance to the nearest food

            if len(food_list) > 0:  # This should always be True,  but better safe than sorry
                my_pos = successor.get_agent_state(self.index).get_position()
                min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['distance_to_food'] = min_distance/(self.distanceNormaliser(game_state))



            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        
            dists_to_defenders = [self.get_maze_distance(my_pos, a.get_position()) for a in defenders]
            if(len(dists_to_defenders)==0):
                 features['proximity_to_enemy']=0
            else:
                min_distance_to_defender = min(dists_to_defenders)
                features['proximity_to_enemy'] = (10/min_distance_to_defender)/(self.distanceNormaliser(game_state))

            # enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            # friends = [successor.get_agent_state(i) for i in self.get_team(successor)]
            # attacking = [a for a in friends if a.is_pacman and a.get_position() is not None]

            if len(capsule_list) > 0:  # This should always be True,  but better safe than sorry
                my_pos = successor.get_agent_state(self.index).get_position()
                min_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in capsule_list])
                features['distance_to_capsule'] = min_distance/(self.distanceNormaliser(game_state))
            
            if(self.flag):
                if(food_carried>0): # as such, mor eimportnqace given to havign food but not too far from home
                    try:
                        features['food_carrying']=food_carried/ (self.distanceToHome(game_state,0)/self.distanceNormaliser(game_state))# need to find way to fix this
                    except:
                        print("Recursion too heavy")
                        
                
            else:
                if(food_carried>0): # as such, mor eimportnqace given to havign food but not too far from home
                    try:
                        features['food_carrying']=food_carried/ (self.get_maze_distance(my_pos,self.start)/self.distanceNormaliser(game_state))#self.distanceToHome(game_state,0) need to find way to fix this
                    except:
                        print(my_pos)
                        print(self.start)

                    

            #next are the º hot encoding ish
            if(self.get_score(successor)>self.get_score(game_state)):
                features['food_brought_home']=1.0
            else:
                features['food_brought_home']=0

            if game_state.get_agent_state(self.get_opponents(game_state)[0]).scared_timer>0:
                features['enemy_scared_1']=1.0
                features['distance_to_scared_1']=self.get_maze_distance(my_pos, game_state.get_agent_state(self.get_opponents(game_state)[0]).get_position())/(self.distanceNormaliser(game_state))
                if features['distance_to_scared_1']*(self.distanceNormaliser(game_state))<1.1:
                    features['eat_scared_1']=1.0
            else:
                if(game_state.get_agent_state(self.get_opponents(game_state)[0]).get_position() is not None):
                    features['distance_to_enemy_1']=self.get_maze_distance(my_pos, game_state.get_agent_state(self.get_opponents(game_state)[0]).get_position())/(self.distanceNormaliser(game_state))

            if game_state.get_agent_state(self.get_opponents(game_state)[1]).scared_timer>0:
                features['enemy_scared_2']=1.0
                features['distance_to_scared_2']=self.get_maze_distance(my_pos, game_state.get_agent_state(self.get_opponents(game_state)[1]).get_position())/(self.distanceNormaliser(game_state))
                if features['distance_to_scared_2']*(self.distanceNormaliser(game_state))<1.1:
                    features['eat_scared_2']=1.0
            else:
                if(game_state.get_agent_state(self.get_opponents(game_state)[1]).get_position() is not None):
                    features['distance_to_enemy_2']=self.get_maze_distance(my_pos, game_state.get_agent_state(self.get_opponents(game_state)[1]).get_position())/(self.distanceNormaliser(game_state))

            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance_enemy=10000
            for enemy in enemies:
                if enemy.get_position() is not None:
                    enemyDistance=self.get_maze_distance(my_pos, enemy.get_position()) 
                    if(enemyDistance<min_distance_enemy):
                        min_distance_enemy = enemyDistance
                    if min_distance_enemy > 5 and features['distance_to_food']*(self.distanceNormaliser(game_state)) < 1.1:
                        features['eat_food'] = 1.0

           

        
        elif(self.mode=='defence'):

            
            successor = self.get_successor(game_state, action)

            my_state = successor.get_agent_state(self.index)
            my_pos = my_state.get_position()

            # Computes whether we're on defenc
            features['on_defence'] = 1
            if my_state.is_pacman: features['on_defence'] = 0

            # Computes distance to invaders we can see
            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            features['num_invaders'] = len(invaders)
            if len(invaders) > 0:
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                features['invader_distance'] = min(dists)/(self.distanceNormaliser(game_state))
                my_pos = successor.get_agent_state(self.index).get_position()
                min_distance_invader = min([self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders])
                if(min_distance_invader<1.1 and game_state.get_agent_state(self.index).scared_timer<=0):
                    features['eat_invader']=1.0




            if action == Directions.STOP: features['stop'] = 1
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev: features['reverse'] = 1
       

        
        return self.normalize_features(features)
        
    def distanceNormaliser(self,game_state):
        return game_state.data.layout.width*game_state.data.layout.height
    import random
    def normalize_features(self,features):
        for feature in features:
            features[feature] /= 10
        return features
    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        if self.mode == 'attack':
            # Adding random weights to each feature
            return self.attackWeights

        elif self.mode == 'defence':
        # Adding random weights to features for defence mode
            return self.defenceWeights
        
    def randomInitWeights(self):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        if self.mode == 'attack':
            # Adding random weights to each feature
            return {
                
                'distance_to_food': random.uniform(1, 0.1),
                'proximity_to_enemy': random.uniform(-1, -0.1),
                'distance_to_capsule': random.uniform(0.5, 0.1),
                'food_carrying': random.uniform(0.1, 1),
                'food_brought_home': random.uniform(0, 1),
                'enemy_scared_1': random.uniform(0, 1),
                'distance_to_scared_1': random.uniform(1, 0.1),
                'enemy_scared_2': random.uniform(0, 1),
                'distance_to_scared_2': random.uniform(1, 0.1),
                'distance_to_enemy_1': random.uniform(-1, -0.1),
                'distance_to_enemy_2': random.uniform(-1, -0.1),
                'eat_food': random.uniform(0, 1),
                'eat_scared_1': random.uniform(0, 1),
                'eat_scared_2': random.uniform(0, 1),
            }

        elif self.mode == 'defence':
        # Adding random weights to features for defence mode
            return {
                'on_defence': random.uniform(0, 1),
                'num_invaders': random.uniform(-1, -0.1),
                'invader_distance': random.uniform(1, 0.1),
                'eat_invader': random.uniform(0, 1),
                'stop': random.uniform(0, 1),
                'reverse': random.uniform(0, 1)
            }
    
    def update(self, game_state, action, nextState, reward):
            """
            Q lienbar approx
            """
            
            # util.raiseNotDefined()
            #followed formula
            # tmp=((1-self.alpha)*self.getQValue(state,action))+ self.alpha*(reward+(self.discount*self.computeValueFromQValues(nextState)))
            # self.setQvalue(state,action,tmp)
            #delta=Temprola diffenece
            delta=reward+self.discount*(self.computeValueFromQValues(nextState)-self.getQValue(game_state,action))
            features = self.get_features(game_state, action)
            # print(self.attackWeights)
            # print("\n")
            if(self.mode=="attack"):
                for feature in features:
                    self.attackWeights[feature] += (self.alpha * delta * features[feature])
                    # print((self.alpha * delta * features[feature]))
            elif(self.mode=="defence"):
                for feature in features:
                    self.defenceWeights[feature] += (self.alpha * delta * features[feature])
            # print(self.defenceWeights)
    
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        
        
        return self.evaluate(state,action) #i think this goes here Q(s,a) computation goes in  update
    

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        ##CAREFUL thi should return  q val, the one returning action si the enxt fucntion
        # print(self.getLegalActions(state))
        if(len(state.get_legal_actions(self.index))==0):  # doing is none didnt work
                return 0.0
        else: #taken from valueiteraiton agtent
            maxQ=-100000000
            # Qvals=util.Counter()
            bestAction=''
            for action in state.get_legal_actions(self.index):
                Qval=self.getQValue(state,action)
                if(Qval>maxQ):
                    maxQ=Qval
                    bestAction=action 

            # return self.Qvals[state,bestAction]
            return self.getQValue(state,bestAction)

    def computeActionFromQValues(self, state):

        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        #smae as above but rteturn besta ction this time
        if(len(state.get_legal_actions(self.index))==0): 
                return None
        else:
            maxQ=-100000000
            # Qvals=util.Counter()
            bestAction=''
            for action in state.get_legal_actions(self.index):
                Qval=self.getQValue(state,action)
                if(Qval>maxQ):
                    maxQ=Qval
                    bestAction=action 
                elif(Qval==maxQ):
                    tieBreak=(bestAction,action)
                    bestAction=random.choice(tieBreak)

            return bestAction

    def distanceToHome(self,game_state,distance=0): #do this sinc ethe maze diatNC ETO HALFWYA DIDNT WORK
        actions = game_state.get_legal_actions(self.index)
        if("Stop" in actions):actions.remove("Stop")
        if(distance>17): #25 was too deep
            return 150#recurison depth limit
        if(not game_state.get_agent_state(self.index).is_pacman): 
            return distance
        
        # for action in actions:
        #     successor=self.get_successor(game_state,action)
        #     distance+=1
        #     self.distanceToHome(successor,distance)

        distances = []  # Store distances from recursive calls

        for action in actions:
            successor = self.get_successor(game_state, action)
            distances.append(self.distanceToHome(successor, distance + 1))

        # Filter out None values should fix issue
        distances = [d for d in distances if d is not None]

        # Check if distances list is not empty before returning
        return min(distances) if distances else 500


    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor


