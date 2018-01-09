__author__ = 'baixiao'
import argparse
import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
INITAL_LR = 0.002
FINAL_LR = 0.0001
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
K = 4 # k frames per state

class DQN():
  # DQN Agent
    def __init__(self, env, model_dir):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.learning_rate = INITAL_LR
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
            t_tensor_name = signature[signature_key].outputs["TEST_ROUND"].name # Just test if load is ok
            d_tensor_name = signature[signature_key].inputs["dropout"].name

            self.state_input = self.session.graph.get_tensor_by_name(x_tensor_name)
            print "1. Direct Input: \t\t" + str(self.state_input.get_shape())
            self.Q_value = self.session.graph.get_tensor_by_name(y_tensor_name)
            print "5. Affine2 \t\t\t" + str(self.Q_value.get_shape())
            self.T = self.session.graph.get_tensor_by_name(t_tensor_name)
            print "load", self.session.run(self.T)
            self.pkeep = self.session.graph.get_tensor_by_name(d_tensor_name)
        else:
            self.create_Q_network()
            self.create_training_method()
            self.session.run(tf.global_variables_initializer())

    def save_model(self, model_dir):
        # Create a builder to export the model
        builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
        # Tag the model in order to be capable of restoring it specifying the tag set

        tensor_info_x = tf.saved_model.utils.build_tensor_info(self.state_input)
        tensor_info_d = tf.saved_model.utils.build_tensor_info(self.pkeep)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(self.Q_value)
        tensor_info_t = tf.saved_model.utils.build_tensor_info(self.T)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'x_input': tensor_info_x, "dropout": tensor_info_d},
                outputs={'y_output': tensor_info_y, "TEST_ROUND": tensor_info_t},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            self.session, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature},
            )
        builder.save()

    def create_Q_network(self):
        # network weights
        self.T = tf.Variable(2.0, name="bias")
        self.lr = tf.placeholder(tf.float32)
        self.pkeep = tf.placeholder(tf.float32)
        W1 = tf.Variable(tf.random_normal([8, 8, K, 16], 0.00, 0.01), name="w_conv1")  #output will be of size (1, 21, 21, 16) for stride 4
        b1 = tf.Variable(tf.random_normal([1, 21, 21, 16], 0.00, 0.01), name="b_conv1")
        W2 = tf.Variable(tf.random_normal([4, 4, 16, 32], 0.00, 0.01), name="w_conv2")  #output will be of size (1, 11, 11, 32) for stride 2
        b2 = tf.Variable(tf.random_normal([1, 11, 11, 32], 0.00, 0.01), name="b_conv2")

        W3 = tf.Variable(tf.random_normal([11 * 11 * 32, 256], 0.00, 0.01), name="w_affine1")
        b3 = tf.Variable(tf.random_normal([256], 0.00, 0.01), name="b_affine1")
        W4 = tf.Variable(tf.random_normal([256, 6], 0.00, 0.01), name="w_affine2")
        b4 = tf.Variable(tf.random_normal([6], 0.00, 0.01), name="b_affine2")

        # input layer
        self.state_input = tf.placeholder(tf.float32, [None, 84, 84, K]) # 4frames in one state
        print "1. Direct Input: \t\t" + str(self.state_input.get_shape())

        Y1 = tf.nn.relu(tf.nn.conv2d(self.state_input, W1, strides=[1, 4, 4, 1], padding='SAME') + b1)
        print "2. Conv1 \t\t\t" + str(Y1.get_shape())

        Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 2, 2, 1], padding='SAME') + b2)
        print "3. Conv2 \t\t\t" + str(Y2.get_shape())

        Y2_ = tf.reshape(Y2, [-1, 11 * 11 * 32])
        print "4.1 Reshape before Affine1 \t" + str(Y2_.get_shape())

        Y3 = tf.nn.relu(tf.nn.relu(tf.matmul(Y2_, W3) + b3))
        print "4.2 Affine1 \t\t\t" + str(Y3.get_shape())

        Y4 = tf.nn.dropout(Y3, self.pkeep)
        print "4.3 Dropout \t\t\t" + str(Y4.get_shape())

        # Q Value layer
        self.Q_value = tf.matmul(Y4, W4) + b4
        print "5. Affine2 \t\t\t" + str(self.Q_value.get_shape())

    def create_training_method(self):
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim]) # one hot presentation
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.y_input = tf.placeholder(tf.float32, [None])
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    def init_environment(self, env):
        # Output 4 random observations when episode begins
        next_obserbation_list = []
        for i in xrange(K):
            action = env.action_space.sample()
            new_state, reward, done, _ = env.step(action)
            new_observation = preprocess_image(new_state)
            next_obserbation_list.append(new_observation)

        return next_obserbation_list

    def run_environment(self, env, action):
        # Use this action to implement k times
        rewards = 0
        done = False
        next_obserbation_list = []

        for i in xrange(K):
            if RENDER:
                env.render()
            new_state, reward, done, _ = env.step(action)
            new_observation = preprocess_image(new_state)
            next_obserbation_list.append(new_observation)
            rewards += reward

        return rewards, next_obserbation_list, done

    def perceive(self, observation_list, action, rewards, next_obserbation_list, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((observation_list, one_hot_action, rewards, next_obserbation_list, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        observation_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_observation_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        next_input_batch = [np.array(input).reshape(84, 84, 4) for input in next_observation_batch]
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_input_batch, self.pkeep: 0.75})
        for i in range(0, BATCH_SIZE):
            done = done_batch[i]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        input_batch = [np.array(input).reshape(84, 84, 4) for input in observation_batch]
        self.optimizer.run(feed_dict={self.y_input: y_batch, self.action_input: action_batch, self.state_input: input_batch, self.lr: self.learning_rate, self.pkeep: 0.75})

    def egreedy_action(self, observation_list):
        input = np.array(observation_list).reshape(84, 84, 4)
        Q_value = self.Q_value.eval(feed_dict={self.state_input: [input], self.pkeep: 1.0})[0]
        e = random.random()
        if e <= self.epsilon:
            # print 'random with %f %f' % (e, self.epsilon)
            self.random_count += 1
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    def set_epsilon(self):
        print 'random count %d with epsilon:%f, lr:%f' % (self.random_count, self.epsilon, self.learning_rate)
        self.random_count = 0
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EPISODE
        self.learning_rate -=(INITAL_LR - FINAL_LR)/EPISODE

    def tack_action(self, observation_list):
        input = np.array(observation_list).reshape(84, 84, 4)
        return np.argmax(self.Q_value.eval(feed_dict={self.state_input: [input], self.pkeep: 1.0})[0])

    def save_figs(self, model_dir, train_episodes,  train_steps, train_rewards, test_episodes, test_steps, test_rewards):
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plot1=plt.plot(train_episodes, train_steps, 'r')
        plot2=plt.plot(train_episodes, train_rewards, 'g')
        plt.subplot(2, 1, 2)
        plot3=plt.plot(test_episodes, test_steps, 'b')
        plot4=plt.plot(test_episodes, test_rewards, 'y')
        plt.savefig(model_dir+'/xxx.png')


# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'SpaceInvadersDeterministic-v4'
EPISODE = 100 # Episode limitation
STEP = 1000 # Step limitation in an episode
TEST_EPISODE = 100 # Test every TEST_EPISODE
TEST_ROUND = 10 # The number of experiment TEST_ROUND every 100 episode
RENDER = False

def main(flags):
    global RENDER
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)

    RENDER = True if flags.render == 1 else False
    if flags.model_ops == "test":
        test_sample(env)
        return

    if flags.model_ops == "load" and flags.model_dir != "":
        agent = DQN(env, flags.model_dir)
        show(env, agent)
        return
    else:
        agent = DQN(env, "")

    train_episodes = []
    train_steps = []
    train_rewards = []
    test_episodes = []
    test_steps = []
    test_rewards = []
    for episode in xrange(EPISODE):
        train_episodes.append(episode)
        # initialize task
        total_reward = 0
        env.reset()
        observation_list = agent.init_environment(env)
        # Train
        steps = 0
        for step in xrange(STEP):
            steps = step
            action = agent.egreedy_action(observation_list) # e-greedy action for train
            # Use this action for 4 times
            rewards, next_observation_list, done = agent.run_environment(env, action)
            total_reward += rewards
            agent.perceive(observation_list, action, rewards, next_observation_list, done)
            observation_list = next_observation_list
            if done:
                break
        print 'episode:%d steps:%d total_reward:%d' % (episode, steps + 1, total_reward)
        agent.set_epsilon()
        train_steps.append(steps+1)
        train_rewards.append(total_reward)

        # TEST_ROUND every 100 episodes
        if episode % TEST_EPISODE == 0 and episode > 0:
            test_episodes.append(episode)
            test_step = 0
            test_reward = 0
            for i in xrange(TEST_ROUND):
                state = env.reset()
                if RENDER:
                    env.render()
                observation_list = agent.init_environment(env)
                for j in xrange(STEP):
                    test_step = j
                    if RENDER:
                        env.render()
                    action = agent.tack_action(observation_list) # direct action for TEST_ROUND
                    rewards, next_observation_list, done = agent.run_environment(env, action)
                    test_reward += rewards
                    observation_list = next_observation_list
                    if done:
                        break
            ave_reward = test_reward/TEST_ROUND
            print 'Test episode:', episode, 'Evaluation Average Reward:', ave_reward
            test_steps.append(test_step+1)
            test_rewards.append(ave_reward)


    show(env, agent)

    if flags.model_ops == "save":
        # save the model
        agent.save_model(flags.model_dir)
        agent.save_figs(flags.model_dir, train_episodes,  train_steps, train_rewards, test_episodes, test_steps, test_rewards)


def test_sample(env):
    global RENDER
    # Reset it, returns the starting frame
    state = env.reset()
    # Render
    if RENDER:
        env.render()

    actions = []
    i = 0
    total_reward = 0
    while True:
        # Render
        if RENDER:
            env.render()
        # Perform a random action, returns the new frame, reward and whether the game is over
        action = env.action_space.sample()
        actions.append(action)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        i += 1
        if done:
            break
    print actions
    print 'TEST_ROUND result steps %d reward %d' %(i, total_reward)


def show(env, agent):
    # Reset it, returns the starting frame
    env.reset()
    # Render
    if RENDER:
        env.render()

    actions = []
    i = 0
    total_reward = 0
    observation_list = agent.init_environment(env)
    # print observation[0, :, :]
    # show_image(observation)
    while True:
        # print i
        action = agent.tack_action(observation_list)  # direct action for TEST_ROUND
        actions.append(action)
        rewards, next_observation_list, done = agent.run_environment(env, action)
        total_reward += rewards
        observation_list = next_observation_list
        i += 1
        if done:
            break
    print actions
    print 'show result steps %d reward %d' % (i, total_reward)


# input (210, 160, 3)
# output 84*84*1
def preprocess_image(observation):  #takes about 20% of the running time!!!
    """ Grayscale, downscale and crop image for less data wrangling """
    #consider transfering this to TF
    # plt.figure(figsize=(10, 10))
    # plt.subplot(5, 2, 1)
    # plt.imshow(observation)
    out = rgb2gray(observation)    #takes about 5% of the running time!!!               #2s
    # plt.subplot(5, 2, 2)
    # plt.imshow(out)
    out = resize(out, (110, 84))    #takes about 9% of running time!!!
    # plt.subplot(5, 2, 3)
    # plt.imshow(out)
    out = out[13:110 - 13, :]
    # plt.subplot(5, 2, 4)
    # plt.imshow(out)
    # out = out.reshape(84, 84, 1)
    # plt.subplot(5, 2, 5)
    # plt.imshow(out.reshape(84, 84))
    # plt.show()
    return out


def show_image(data):
    fig = plt.figure(1)
    # plt.axis('off')
    xx = data.reshape(84, 84)
    # print xx
    plt.imshow(xx)
    plt.show()



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
    parser.add_argument(
        '--render',
        type=int,
        default=0,
        help='Render gym environment.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)