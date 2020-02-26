import numpy as np
import tensorflow as tf
import random
import copy as cp
from collections import deque
from environment_DQN import *
from show_result import *

env=Env()
dis=0.8 ; replay_memory=50000
input_size=MapWidth*MapHeight ; output_size=len(env.action_space)

l_rate = 0.005 ; Activation_Function = 'Leak_relu'

def state_to_one_hot(state):
    n=MapWidth*state[0]+state[1]
    state_one_hot=np.identity(MapHeight*MapWidth)[n:n+1]
    return state_one_hot

def one_hot_to_state(state_one_hot):
    idx=list(state_one_hot[0]).index(1)
    state=[idx//MapWidth, idx%MapWidth]
    return state

def action_selection_boundary(state):

    # map 모서리 처리
    if state in list(range(1, 13)):
        if state == 5 :
            return [1, 2]
        elif state == 7 :
            return [1, 3]
        else :
            return [1, 2, 3]
        
    elif state in list(range(14, 99, 14)):
        return [0, 1, 3]
    
    elif state in list(range(113, 125)):
        if state == 117 :
            return [0, 2]
        elif state == 119 :
            return [0, 3]
        else :
            return [0, 2, 3]
        
    elif state in list(range(27, 112, 14)):
        return [0, 1, 2]

    elif state == 0 :
        return [1, 3]
    elif state == 13 :
        return [1, 2]
    elif state == 112 :
        return [0, 3]

    # wall 주변 처리
    elif state in list(range(19, 104, 14)):
        if state == 75 :
            return [3]
        else :
            return [0, 1, 2]
    elif state in list(range(21, 106, 14)):
        return [0, 1, 3]

    # base_camp에서 처리
    elif state == 76 :
        return [3]

    # The rest
    else :
        return [0, 1, 2, 3]

class DQN:
    
    def __init__(self, session, input_size, output_size, name="main"):
        
        self.session=session
        self.input_size=input_size
        self.output_size=output_size
        self.net_name=name
        
        self._build_network()
            
    def _build_network(self, h0_size=256, h1_size=256, l_rate=0.005) : # learning_rate = 0.01 -> 0.005

        with tf.variable_scope(self.net_name):

            self._X=tf.placeholder(tf.float32, shape=[None, self.input_size], name='input_x')
            
            w1=tf.Variable(tf.random_uniform([self.input_size, h0_size], -1., 1.), name='w1')
            layer1=tf.nn.leaky_relu(tf.matmul(self._X, w1)) # tanh -> leaky_relu 

            w2=tf.Variable(tf.random_uniform([h0_size, h1_size], -1., 1.), name='w2')
            layer2=tf.nn.leaky_relu(tf.matmul(layer1, w2)) # tanh -> leaky_relu 

            w3=tf.Variable(tf.random_uniform([h1_size, self.output_size], -1., 1.), name='w3')

            self._Qpred=tf.matmul(layer2, w3)

        self._Y=tf.placeholder(tf.float32, shape=[None, self.output_size])

        self._loss=tf.reduce_mean(tf.square(self._Y - self._Qpred))
        
        self._train=tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
            
    def predict(self, state_one_hot):
        return self.session.run(self._Qpred, feed_dict={self._X : state_one_hot})
    
    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X : x_stack, self._Y : y_stack})
    
def replay_train(mainDQN, targetDQN, train_batch):
    
    x_stack=np.empty(0).reshape(0, input_size)
    y_stack=np.empty(0).reshape(0, output_size)
    
    for state, action, reward, next_state, done in train_batch:

        state_one_hot=state_to_one_hot(state)
        next_state_one_hot=state_to_one_hot(next_state)
        
        Q=mainDQN.predict(state_one_hot)
        
        if done :
            Q[0, action]=reward
        else :
            Q[0, action]=reward + dis * np.max(targetDQN.predict(next_state_one_hot))
            # reward + dis * targetDQN.predict(next_state_one_hot)[0, np.argmax(mainDQN.predict(next_state_one_hot))] 에서
            # reward + dis * np.max(targetDQN.predict(next_state_one_hot)) 으로 변경
        
        y_stack=np.vstack([y_stack, Q])
        x_stack=np.vstack([x_stack, state_one_hot])
        
    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(*, dest_scope_name='target', src_scope_name='main'):
    
    op_holder=[]
    
    src_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))
    
    return op_holder

def main():

    # accumulated average version
    max_episodes=10002 ; replay_buffer=deque() ; success_cnt_list = [] 
    
    with tf.Session() as sess :
        
        mainDQN=DQN(sess, input_size, output_size, name='main')
        targetDQN=DQN(sess, input_size, output_size, name='target')
        tf.global_variables_initializer().run()
        
        copy_ops=get_copy_var_ops(dest_scope_name='target', src_scope_name='main')
        sess.run(copy_ops)
        cnt=0 ; cnt_bs = 0
        
        for episode in range(max_episodes):
            
            e = 1. / ((episode / 300) + 1)
            
            done=False
            
            state=env.reset()
            env.render()
            
            while not done :

                env.render() # 화면 출력 
                state_one_hot=state_to_one_hot(state)
                state_number = MapWidth*state[0] + state[1]

                # Act_Move 함수를 호출하여 agent가 이동하기 전까지는 agent의 방향을 정하는 부분이다.
                if np.random.rand(1) < e :
                    action = np.random.choice(action_selection_boundary(state_number))
                    
                else :
                    temp = action_selection_boundary(state_number)
                    action_list = []
                    prediction = cp.deepcopy(mainDQN.predict(state_one_hot)[0])
                    
                    for i in range(4) :
                        if i in temp :
                            action_list.append(prediction[i])
                        else :
                            action_list.append(min(prediction)-1)

                    action = np.argmax(action_list)
                    
                    action_list.clear()
                
                next_state, reward, done, whether_success = env.Action_Function(action)

                if done and whether_success==1 : cnt+=1
                                                
                replay_buffer.append((state, action, reward, next_state, done))
                
                if len(replay_buffer) > replay_memory :
                    replay_buffer.popleft()
                
                state=next_state

            if episode > 0 and episode % 100 == 0 :
                print("Episode : {}".format(episode))
                success_cnt_list.append(round(cnt/episode, 3)*100)
                print("Success Rate : {}\n".format(str(cnt)))
                cnt=0
                
            if episode % 20 == 1 : # train per 20 episodes
                
                for _ in range(50):
                    
                    minibatch=random.sample(replay_buffer, 10)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                
                sess.run(copy_ops)
                
    result=line_plot(success_cnt_list, "DQN with MLP accum_average", l_rate, Activation_Function)
    result.showing()
    
    return

if __name__=="__main__":
    main()
