import tensorflow as tf
import numpy as np


class model(tf.keras.Model):
    def __init__(self, stateSize, actionSize, optimizer=None, loss = None):
        super(model,self).__init__()

        self.dense1 = tf.keras.layers.Dense(16,input_shape=(stateSize,),activation='relu')

        self.dense2 = tf.keras.layers.Dense(32,activation="relu")

        self.dense3 = tf.keras.layers.Dense(actionSize,activation='linear')

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001) if optimizer is None else optimizer

        self.loss = tf.keras.losses.MeanSquaredError() if loss is None else loss


    def call(self,x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

    @tf.function
    def trainStep(self,x,y):
        with tf.GradientTape() as tape:

            pred = self.call(x)
            currLoss = self.loss(y,pred)
        gradients = tape.gradient(currLoss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))






class DQN:


    def __init__(self, model, gamma=0.99, lr=0.001, actionSize = 0, batchSize = 32, maxMemorySize = 1000):
        self.model = model
        self.actionSize = actionSize
        self.gamma = gamma
        self.lr = lr
        self.expProb = 1.0
        self.decay = 0.05
        self.memory = list()
        self.batchSize = batchSize
        self.maxMemorySize = maxMemorySize


    def update_Explore_Prob(self):
        self.expProb = self.expProb * np.exp(-self.decay)


    def predict_Move(self,state):

        if self.expProb >= np.random.rand():
            return np.random.choice(range(self.actionSize))

        return np.argmax(self.model(state))


    def store_Data(self,prevState, action,reward,nextState, isGameOver):
        self.memory.append({
            "prev": prevState,
            "action": action,
            "reward": reward,
            "next": nextState,
            "over": isGameOver
        })
        if len(self.memory) > self.maxMemorySize:
            self.memory.pop(0)



    def train(self):
        np.random.shuffle(self.memory)
        sample = memory[0:self.batchSize]
        for experience in sample:
            currentQ = model(tf.convert_to_tensor(experience["prev"])).numpy()
            target = experience["reward"]
            if not experience["done"]:
                currentQ = currentQ + self.gamma*model(tf.convert_to_tensor(experience["next"]))
            currentQ[0][experience["action"]] = target
            # model.fit(experience["curr"], currentQ, epochs=1, verbose=1) if sequential else
            model.trainStep(experience["curr"], currentQ)

action = 0
state = 0
mod = model(stateSize=state,actionSize=action)
agent1 = DQN(mod)
games_to_simulate = 100

for i in range(games_to_simulate):
    while not gameOver:
        # get game State
        state = None
        # Predict on state
        move = agent1.predict_Move(state)
        # get new State
        NewState = None
        # Calculate reward
        reward = None
        # Check if game is over
        gameOver = None

        agent1.store_Data(state, move, reward, NewState, gameOver)
        if len(agent1.memory) >= agent1.batchSize:
            agent1.train()
    if len(agent1.memory) >= agent1.batchSize:
        agent1.train()
    agent1.update_Explore_Prob()
    # reset game
