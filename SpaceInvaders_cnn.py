__author__ = 'baixiao'
import argparse
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
FINAL_EPSILON = 0.1 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

class DQN():
  # DQN Agent
    def __init__(self, env, model_dir):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = 84*84#env.observation_space.shape
        self.action_dim = env.action_space.n
        print 'state:%d action:%d' % (self.state_dim, self.action_dim)
        self.random_count = 0

        # Init session
        self.session = tf.InteractiveSession()

        if model_dir != "":
            # tf.saved_model.loader.load(self.session, tags, model_dir)
            signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            input_key = 'x_input'
            output_key = 'y_output'

            export_path = model_dir
            meta_graph_def = tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], export_path)
            signature = meta_graph_def.signature_def

            x_tensor_name = signature[signature_key].inputs[input_key].name
            y_tensor_name = signature[signature_key].outputs[output_key].name
            t_tensor_name = signature[signature_key].outputs["TEST_STEP"].name

            self.state_input = self.session.graph.get_tensor_by_name(x_tensor_name)
            self.Q_value = self.session.graph.get_tensor_by_name(y_tensor_name)
            self.T = self.session.graph.get_tensor_by_name(t_tensor_name)
            print "load", self.session.run(self.T)
        else:
            self.create_Q_network()
            self.create_training_method()
            self.session.run(tf.global_variables_initializer())

    def save_model(self, model_dir, tags):
        # Create a builder to export the model
        builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
        # Tag the model in order to be capable of restoring it specifying the tag set

        tensor_info_x = tf.saved_model.utils.build_tensor_info(self.state_input)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(self.Q_value)
        tensor_info_t = tf.saved_model.utils.build_tensor_info(self.T)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'x_input': tensor_info_x},
                outputs={'y_output': tensor_info_y, "TEST_STEP": tensor_info_t},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            self.session, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature},
            )
        builder.save()

    def create_Q_network(self):
        # network weights
        self.T = tf.Variable(2.0, name="bias")
        W1 = tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="w_conv1")  #output will be of size (1, 21, 21, 16) for stride 4
        b1 = tf.Variable(tf.random_normal([1, 21, 21, 16], 0.00, 0.01), name="b_conv1")
        W2 = tf.Variable(tf.random_normal([4, 4, 16, 32], 0.00, 0.01), name="w_conv2")  #output will be of size (1, 11, 11, 32) for stride 2
        b2 = tf.Variable(tf.random_normal([1, 11, 11, 32], 0.00, 0.01), name="b_conv2")

        W3 = tf.Variable(tf.random_normal([11 * 11 * 32, 256], 0.00, 0.01), name="w_affine1")
        b3 = tf.Variable(tf.random_normal([256], 0.00, 0.01), name="b_affine1")
        W4 = tf.Variable(tf.random_normal([256, 6], 0.00, 0.01), name="w_affine2")
        b4 = tf.Variable(tf.random_normal([6], 0.00, 0.01), name="b_affine2")

        # input layer
        self.state_input = tf.placeholder(tf.float32, [None, 84, 84, 1])
        print "1. Direct Input: \t\t" + str(self.state_input.get_shape())
        # self.state_inputs = tf.reshape(self.state_input, [None, 84, 84, 1])
        # print "1.1 Reshape Input \t\t\t" + str(self.state_inputs.get_shape())

        Y1 = tf.nn.relu(tf.nn.conv2d(self.state_input, W1, strides=[1, 4, 4, 1], padding='SAME') + b1)
        print "2. Conv1 \t\t\t" + str(Y1.get_shape())

        Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 2, 2, 1], padding='SAME') + b2)
        print "3. Conv2 \t\t\t" + str(Y2.get_shape())

        Y2_ = tf.reshape(Y2, [-1, 11 * 11 * 32])
        print "4.1 Reshape before Affine1 \t" + str(Y2_.get_shape())
        Y3 = tf.nn.relu(tf.nn.relu(tf.matmul(Y2_, W3) + b3))
        print "4.2 Affine1 \t\t\t" + str(Y3.get_shape())

        # Q Value layer
        self.Q_value = tf.matmul(Y3, W4) + b4
        print "5. Affine2 \t\t\t" + str(self.Q_value.get_shape())

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
        Q_value = self.Q_value.eval(feed_dict={self.state_input: [observation]})[0]
        e = random.random()
        if e <= self.epsilon:
            # print 'random with %f %f' % (e, self.epsilon)
            self.random_count += 1
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    def set_epsilon(self):
        print 'random count %d with %f' % (self.random_count, self.epsilon)
        self.random_count = 0
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EPISODE

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
EPISODE = 10 # Episode limitation
STEP = 1000 # Step limitation in an episode
TEST_EPISODE = 10
TEST_STEP = 10 # The number of experiment TEST_STEP every 100 episode


def main(flags):
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)

    if flags.model_ops == "TEST_STEP":
        test_sample(env)
        return

    if flags.model_ops == "load" and flags.model_dir != "":
        agent = DQN(env, flags.model_dir)
        show(env, agent)
        return
    else:
        agent = DQN(env, "")

    for episode in xrange(EPISODE):
        # initialize task
        total_reward = 0
        state = env.reset()
        observation = preprocess_image(state)
        # Train
        steps = 0
        for step in xrange(STEP):
            steps = step
            action = agent.egreedy_action(observation) # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_observation = preprocess_image(next_state)
            agent.perceive(observation, action, reward, next_observation, done)
            observation = next_observation
            if done:
                break
        print 'episode:%d steps:%d total_reward:%d' % (episode, steps + 1, total_reward)
        agent.set_epsilon()

        # TEST_STEP every 100 episodes
        if episode % TEST_EPISODE == 0 and episode > 0:
            total_reward = 0
            for i in xrange(TEST_STEP):
                state = env.reset()
                env.render()
                observation = preprocess_image(state)
                for j in xrange(STEP):
                    env.render()
                    action = agent.action(observation) # direct action for TEST_STEP
                    next_state, reward, done, _ = env.step(action)
                    # print 'episode:%d step:%d reward:%d' % (episode, i, reward)
                    next_observation = preprocess_image(next_state)
                    observation = next_observation
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST_STEP
            print 'episode:', episode, 'Evaluation Average Reward:', ave_reward
            if ave_reward >= 200:
                break

    show(env, agent)

    if flags.model_ops == "save":
        # save the model
        agent.save_model(flags.model_dir, ["tag"])


def test_sample(env):
    # Reset it, returns the starting frame
    state = env.reset()
    # Render
    env.render()

    i = 0
    total_reward = 0
    while True:
        env.render()
        # Perform a random action, returns the new frame, reward and whether the game is over
        action = env.action_space.sample()
        print action
        state, reward, done, _ = env.step(action)
        total_reward += reward
        # Render
        i += 1
        if done:
            break
    print 'TEST_STEP result steps %d reward %d' %(i, total_reward)


def show(env, agent):
    # Reset it, returns the starting frame
    state = env.reset()
    # Render
    env.render()

    i = 0
    total_reward = 0
    observation = preprocess_image(state)
    while True:
        # print i
        env.render()
        action = agent.action(observation)  # direct action for TEST_STEP
        print action
        next_state, reward, done, _ = env.step(action)
        next_observation = preprocess_image(next_state)
        observation = next_observation
        total_reward += reward
        i += 1
        if done:
            break
    print 'show result steps %d reward %d' % (i, total_reward)


# input (210, 160, 3)
# output 84*84*1
def preprocess_image(observation):  #takes about 20% of the running time!!!
    """ Grayscale, downscale and crop image for less data wrangling """
    #consider transfering this to TF
    out = rgb2gray(observation)    #takes about 5% of the running time!!!               #2s
    out = resize(out, (110, 84))    #takes about 9% of running time!!!
    out = out[13:110 - 13, :]
    return np.array(out).reshape(84, 84, 1).astype(np.uint8)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default="./export",
        help='Directory to export model.'
    )
    parser.add_argument(
        '--model_ops',
        type=str,
        default="none",
        help='load or export model.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)