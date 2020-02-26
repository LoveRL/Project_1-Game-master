import random
import numpy as np
import copy as cp
from environment import *
from show_result import *

env = Env()
dis = 0.9
learning_rate = 0.01
max_episode = 10002

def coords_to_state(coords):
    return MapWidth * coords[0] + coords[1]

def make_not_none_idx_list(temp):
    not_none_idx_list = []

    for i in range(4) :
        if temp[i] != None :
            not_none_idx_list.append(i)

    return not_none_idx_list

def arg_max(state_action, not_idx_list):

    max_index_list = []
    max_value = state_action[not_idx_list[0]]
    for i in not_idx_list :
        if state_action[i] > max_value :
            max_index_list.clear()
            max_value = state_action[i]
            max_index_list.append(i)
        elif state_action[i] == max_value :
            max_index_list.append(i)
            
    return random.choice(max_index_list)

def max_q_value(temp):
    tmp = make_not_none_idx_list(temp)
    tmp2 = []
    for i in tmp :
        tmp2.append(temp[i])
    return np.max(tmp2)   

def setting_map(Map):

    # 모서리
    for i in range(1, 13):
        Map[i][0] = None
        Map[i+112][1] = None
    for i in range(14, 99, 14):
        Map[i][2] = None
        Map[i+13][3] = None

    # 꼭지점
    Map[0][0] = None ; Map[0][2] = None
    Map[13][0] = None ; Map[13][3] = None
    Map[112][1] = None ; Map[112][2] = None
    Map[125][1] = None ; Map[125][3] = None

    # wall 양옆
    for i in range(5, 118, 14):
        Map[i][3] = None
        Map[i+2][2] = None

        if i == 5 :
            Map[i][0] = None ; Map[i+2][0] = None
        elif i == 117 :
            Map[i][1] = None ; Map[i+2][1] = None

    # base_camp
    for i in range(3):
        Map[76][i] = None
        Map[75][i] = None
    Map[75][3] = 0

    return Map

def main():

    # accum_average version
    cnt=0 ; success_cnt_list=[]
    
    Q_table = [[0 for action in range(4)] for state in range(126)]

    Q_table = setting_map(Q_table)

    success_cnt_list = []
    
    for episode in range(max_episode):

        agent_curr_cell_coords = env.reset()
        done = False
        
        while not done :

            if episode > 9500 :
                env.render()
            
            e = 1 / ((episode + 1)/300)
            
            agent_state = coords_to_state(agent_curr_cell_coords)
            
            if np.random.rand(1) < e :
                action = random.choice(make_not_none_idx_list(Q_table[agent_state]))
            else :
                not_none_idx_list = make_not_none_idx_list(Q_table[agent_state])
                action = arg_max(Q_table[agent_state], not_none_idx_list)
   
            agent_next_cell_coords, reward, done, whether_success = env.Action_Function(action)
            agent_next_state = coords_to_state(agent_next_cell_coords)

            Q_now = Q_table[agent_state][action]
            Q_next = reward + dis * max_q_value(Q_table[agent_next_state])

            Q_table[agent_state][action] += learning_rate * (Q_next - Q_now)
            
            Q_table[agent_state][action] = round(Q_table[agent_state][action], 3)

            if done and whether_success == 1 : cnt += 1

            agent_curr_cell_coords = agent_next_cell_coords
            
        if episode > 0 and episode % 100 == 0 :
            print("Episode :", episode)
            #print("Success rate : {}\n".format(str(cnt)))
            success_cnt_list.append(round(cnt/episode, 3) * 100)
            
        if episode > 0 and episode % 1000 == 0 :
            f=open("Result_"+str(episode)+".txt", "w")
            for i in range(126):
                f.write(str((i))+str(" : ")+str(Q_table[i])+"\n")
            f.close()

    result = line_plot(success_cnt_list, "Q-Learning with Q-Table accum_average")
    result.showing()
    return

if __name__=="__main__":
    main()
