"""
rein_learn:
reinforcement learning agent based for optimization problem
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback
from mallink.opt.monte_markov import MonteMarkov
import random

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class reinLearn(MonteMarkov):
    """reinforcement move in Monte Carlo"""

    def __init__(self,spotL,pathL=None,opsL=None,pairL=None,conf=None):
        """initialize the quantities for reinforcement learning"""
        MonteMarkov.__init__(self,spotL,pathL,opsL,pairL,conf)
        self.epsilon = 1.
        self.alpha = 0.1
        self.gamma = 0.6
        self.state = 0
        self.link = conf['link']
        self.train_frame = 100000
        self.buffer_size = 50000
        self.batchSize = 64
        self.q_table = []
        self.senseL = []
        self.rewardL = []
        self.moveL = []
        self.current_sense = self.sense()
        self.current_qval  = []
        self.current_reward = 0.
        self.loss_log = []
        self.nSense = len(self.sense())
        self.model = self.neural_net(conf)

    def neural_net(self,conf):
        """initiate the neural network"""
        nSense = self.nSense
        net_layer = conf['net_layer']
        link = conf['link']
        model = Sequential()
        model.add(Dense(net_layer[0],init='lecun_uniform',input_shape=(nSense,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(net_layer[1],init='lecun_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(link,init='lecun_uniform'))
        model.add(Activation('linear'))
        rms = RMSprop()
        model.compile(loss='mse',optimizer=rms)
        if conf['load_model'] != '':
            model.load_weights(conf['load_model'])
        return model

    def sense(self,move=None):
        """fill the sensor array"""
        if move == None:
            load=0;length=0;weight=np.zeros(self.link);potential=np.zeros(self.link)
        else:
            agent = move['agent'][0]
            state = move['state'][0]
            action = move['action'][0]
            weight = self.markovC.loc[:,state]
            weight = weight.sort_values().tail(self.link)
            potential = self.spotL.loc[weight.index,"potential"]
            load = self.pathL.loc[agent,"load"]
            length = self.pathL.loc[agent,"distance"]            
        return [load,length] + list(weight) + list(potential)

    def tryChange(self,m=None):
        """perform a move and try to accept it"""
        if m == None:
            m = np.random.choice(self.moveS,p=self.moveP)
        if   m == "move": move = self.tryMove()
        elif m == "distance": move = self.tryDistance()
        elif m == "markov": move = self.tryMarkov()
        elif m == "extrude": move = self.tryExtrude()
        elif m == "flat": move = self.tryMarkovFlat()
        elif m == "ai": move = self.tryReinforce()
        if self.isLog: print('%d) try change, move %d' % (self.step,m))
        for v,p,s in zip(move['agent'],move['state'],move['action']):
            if not self.checkAllowed(v,p,s):
                return False, {}
        return True, move

    def updateHistory(self,move):
        """update sense history"""
        s, a, act = move['state'], move['agent'], move['action']
        act = self.actionD[act[0]]
        sense = self.sense(move)
        reward = self.En
        # self.senseL.append(sense)
        # self.rewardL.append(self.En)
        # self.moveL.append(act)
        qval = [act] + [reward] + self.current_sense + sense
        self.q_table.append(qval)
        if len(self.q_table) > self.buffer_size:
            self.q_table.pop(0)

        if self.step > 100:
            self.fitState()
            if self.epsilon > 0.1:
                self.epsilon -= 1.0/self.train_frame
        self.current_reward = self.En
        self.current_sense = sense
        self.current_qval = qval

    def prepareBatch(self,minibatch):
        """prepare a training batch from history"""
        nSense = self.nSense
        q_table = np.array(minibatch)
        actionL = q_table[:,0].astype(int)
        rewardL = q_table[:,1]
        X_train = stateO  = q_table[:,2:nSense+2]
        stateN  = q_table[:,nSense+2:2*nSense+2]
        old_qval = self.model.predict(stateO, batch_size=self.batchSize)
        new_qval = self.model.predict(stateN, batch_size=self.batchSize)
        maxQs = np.max(new_qval, axis=1)
        y_train = old_qval
        y = np.choose(actionL, stateO.T)
        for i,j in enumerate(actionL):
            y_train[i,j] = rewardL[i] + self.gamma*maxQs[i]
        return X_train, y_train
        
    def fitState(self):
        """predict next state"""
        minibatch = random.sample(self.q_table,self.batchSize)
        X_train, y_train = self.prepareBatch(minibatch)
        history = LossHistory()
        self.model.fit(
            X_train, y_train, batch_size=self.batchSize,
            epochs=1, verbose=0, callbacks=[history]
        )
        self.loss_log.append(history.losses)

    def isMetropolis(self,dEn,weight=1.):
        """turn off metropolis acceptance"""
        return True

    def tryReinforce(self):
        """ reinforcement lerning status change """
        mode = "reinforce"
        if np.random.uniform(0, 1) < self.epsilon:
            agent = np.random.choice(self.agentL)
            state = np.random.choice(self.stateL)
            action = np.random.choice(self.actionL)
            mode = "explore"
        else:
            old_qval = np.array(self.current_qval)
            qval = self.model.predict(old_qval,batch_size=1)
            action  = qval.argmax()
            state = np.random.choice(self.stateL)
            # state = np.argmax(self.markovC[self.state])
            mode = "exploit"
        weight = 1.
        return {"weight":weight,"agent":[agent],"state":[state],"action":[action],"move":"reinforce"}
    
    def discReward(self):
        """"""
        timeL = pd.DataFrame(self.timeL)
        def discount(k):
            return 1/max(1,k)
        G = sum([discount(i)*x['current'] for i,x in timeL.iterrows()])

    def stateValue(self):
        """ """
        self.calcEnergy()

    def actionValue(self):
        """ """

    

        
