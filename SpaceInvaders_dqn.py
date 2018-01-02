__author__ = 'baixiao'
__author__ = 'baixiao'
import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from skimage.transform import resize
from skimage.color import rgb2gray


# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

class DQN():
  # DQN Agent
    def __init__(self, env, load_tags):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = 84*84#env.observation_space.shape
        self.action_dim = env.action_space.n
        print 'state:%d action:%d' % (self.state_dim, self.action_dim)

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

        if len(load_tags) != 0:
            tf.saved_model.loader.load(self.session, load_tags, "./export")

    def create_Q_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder(tf.float32, [None, 84, 84])
        self.state_vector = tf.reshape(self.state_input, [-1, 84*84])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_vector, W1) + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer, W2) + b2

    def create_training_method(self):
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim]) # one hot presentation
        self.y_input = tf.placeholder(tf.float32, [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
            })

    def egreedy_action(self, observation):
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EPISODE
        Q_value = self.Q_value.eval(feed_dict={self.state_input: [observation]})[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    def action(self, observation):
        return np.argmax(self.Q_value.eval(feed_dict={self.state_input: [observation]})[0])

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'SpaceInvadersDeterministic-v4'
EPISODE = 2000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    # test_sample(env)
    agent = DQN(env, ["tag"])
    show(env, agent)
    return

    for episode in xrange(EPISODE):
        # initialize task
        total_reward = 0
        state = env.reset()
        observation = preprocess_image(state)
        # Train
        for step in xrange(STEP):
            action = agent.egreedy_action(observation) # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_observation = preprocess_image(next_state)
            agent.perceive(observation, action, reward, next_observation, done)
            observation = next_observation
            if done:
                break
        print 'episode:%d total_reward:%d' % (episode, total_reward)

        # Test every 100 episodes
        if episode % 100 == 0 and episode > 0:
            total_reward = 0
            for i in xrange(TEST):
                state = env.reset()
                observation = preprocess_image(state)
                for j in xrange(STEP):
                    env.render()
                    action = agent.action(observation) # direct action for test
                    state, reward, done, _ = env.step(action)
                    # print 'episode:%d step:%d reward:%d' % (episode, i, reward)
                    observation = preprocess_image(state)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print 'episode:', episode, 'Evaluation Average Reward:', ave_reward
            if ave_reward >= 200:
                break

    show(env, agent)
    # save the model
    save_model(agent)


def test_sample(env):
    # Reset it, returns the starting frame
    state = env.reset()
    # Render
    env.render()

    while True:
        # Perform a random action, returns the new frame, reward and whether the game is over
        state, reward, done, _ = env.step(env.action_space.sample())
        print state, reward, done
        # Render
        env.render()
        if done:
            break


def show(env, agent):
    i = 0
    total_reward = 0
    state = env.reset()
    observation = preprocess_image(state)
    while True:
        # print i
        env.render()
        action = agent.action(observation)  # direct action for test
        state, reward, done, _ = env.step(action)
        # print 'state:', state, 'reward:', reward
        observation = preprocess_image(state)
        total_reward += reward
        i += 1
        if done:
            break
    print 'show result reward', total_reward


# output 84*84
def preprocess_image(observation):  #takes about 20% of the running time!!!
    """ Grayscale, downscale and crop image for less data wrangling """
    #consider transfering this to TF
    out = rgb2gray(observation)    #takes about 5% of the running time!!!               #2s
    out = resize(out, (110, 84))    #takes about 9% of running time!!!
    return out[13:110 - 13, :]


def save_model(agent):
    # Create a builder to export the model
    builder = tf.saved_model.builder.SavedModelBuilder("./export")
    # Tag the model in order to be capable of restoring it specifying the tag set
    builder.add_meta_graph_and_variables(agent.session, ["tag"])
    builder.save()



if __name__ == '__main__':
    main()