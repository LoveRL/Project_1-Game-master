import os
import time
import random
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
# python3에서는 pillow로 대체 되었다. pillow 설치하고 이렇게 사용해도 된다.

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage # 사진을 object로 받을 것이라는 의미
PixelPercell = 50 ; CellCenter = 25
MapWidth = 14 ; MapHeight = 9  

# 논리적 맵 좌표
# MineAndHint_Map[row][col] 으로 접근해야 함. 좌표는 (x, y)
MineAndHint_Map = [[0 for col in range(MapWidth)] for row in range(MapHeight)] # 실제 데이터 있는 맵
broken = [[0 for col in range(MapWidth)] for row in range(MapHeight)]
direction = [[0,1],[0,-1],[1,0],[-1,0]]

class Env(tk.Tk): # tkinter module 상속받기
    
    # 생성자
    def __init__(self):
        
        super(Env, self).__init__() # tkinter module의 생성자 실행
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.title('RL_DQN_Project 9*14 size_map')
        self.geometry('{0}x{1}'.format(MapWidth * PixelPercell, MapHeight * PixelPercell)) # setting window_size
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        return

    # canvas 그리는 method
    def _build_canvas(self):
        
        canvas = tk.Canvas(self, bg='white', height = MapHeight * PixelPercell, width = MapWidth * PixelPercell)

        # 가로 방향으로 움직이면서 세로 선 한 칸씩 그리기 : 0, 800, 100
        # 시점=(x0, y0), 종점=(x1, y1)
        # line 그릴 때 for loop에서 0을 반드시 추가...
        for c in range(0, MapWidth * PixelPercell, PixelPercell):  
            x0, y0, x1, y1 = c, 0, c, MapHeight * PixelPercell
            canvas.create_line(x0, y0, x1, y1)

        # 세로 방향으로 움직이면서 가로 선 한 칸씩 선 그리기 : 0, 1300, 100
        for r in range(0, MapHeight * PixelPercell, PixelPercell):
            x0, y0, x1, y1 = 0, r, MapWidth * PixelPercell, r           
            canvas.create_line(x0, y0, x1, y1)
        
        # agent의 시작위치 
        tempPixel = self.CellToPixel([0,0])
        self.agent = canvas.create_image(tempPixel[0], tempPixel[1], image=self.shapes[0], anchor=tk.CENTER)

        # mine 위치 설정
        for row in range(MapHeight):
            for col in range(MapWidth):
                if(MineAndHint_Map[row][col] == -1):
                    tempPixel = self.CellToPixel([row, col])
                    canvas.create_image(tempPixel[0], tempPixel[1], image=self.shapes[1])
                    
        # goal 위치 설정
        tempPixel = self.CellToPixel([MapHeight-1, MapWidth-1]) 
        self.goal = canvas.create_image(tempPixel[0], tempPixel[1], image=self.shapes[2], anchor=tk.CENTER, tag="Goal")

        # base camp 설정
        tempPixel = self.CellToPixel([5, 6])
        self.base_camp = canvas.create_image(tempPixel[0], tempPixel[1], image=self.shapes[3], anchor=tk.CENTER)

        # wall 설정
        for i in range(9):
            if i == 5 :
                continue
            else :
                tempPixel = self.CellToPixel([i, 6])
                self.wall = canvas.create_image(tempPixel[0], tempPixel[1], image=self.shapes[4], anchor=tk.CENTER)
        
        canvas.pack()
        
        return canvas

    # canvas에 넣을 img file object로 불러오는 method
    def load_images(self):
        agent = PhotoImage(Image.open(os.getcwd()+"\\img\\agent.png").resize((35, 35)))
        mine = PhotoImage(Image.open(os.getcwd()+"\\img\\mine.png").resize((35, 35)))
        goal = PhotoImage(Image.open(os.getcwd()+"\\img\\finish.png").resize((35, 35)))
        base_camp = PhotoImage(Image.open(os.getcwd()+"\\img\\base_camp.png").resize((35, 35)))
        wall = PhotoImage(Image.open(os.getcwd()+"\\img\\wall.png").resize((48, 48)))
        return agent, mine, goal, base_camp, wall

    def render(self):
        time.sleep(0.0001) # agent의 움직임 속도 조절 
        self.update()
        return

    def DFS(self, row, col):

        # DFS의 종료조건 == 블럭을 어디까지 깨지게 할 것인지 boundary 설정
        if(col < 0 or col >= MapWidth) :
            return
        elif(row < 0 or row >= MapHeight) :
            return
        elif [row, col] in [[i, 6] for i in range(9)] :
            return
        elif MineAndHint_Map[row][col] == -1 :
            return
        
        if broken[row][col] == 0 : # 한번도 깨지지 않은 위치라면
            broken[row][col] = 1 # 깨졌다고 표시한 후, 
            
            if(MineAndHint_Map[row][col] != -1) : # 맵 데이터가 mine이 아니라면
                data = MineAndHint_Map[row][col] # 맵 데이터 보여주기
                pixelcoords = self.CellToPixel([row, col])

                # 매 episode마다 생성된 text를 지워야하기 때문에 tag를 쓰고 손쉽게 지울 수 있도록 한다.
                self.canvas.create_text(pixelcoords[0], pixelcoords[1], font="Times 13 italic bold", text=str(data), tag='tagging_dfs')
        
            if(MineAndHint_Map[row][col] > 0) :
                return 0
            
            for i in range(4) : # 맵데이터가 비어있다면 
                direction = [[0,1],[0,-1],[1,0],[-1,0]]
                if 0 <= row+direction[i][0] <= 8 and 0 <= col+direction[i][1] <= 13 :
                    self.DFS(row+direction[i][0], col+direction[i][1]) # 현재 위치 기준 사방 각 블록 체크
                
        elif broken[row][col] == 1 :
            return 

    # 픽셀값을 셀 인덱스 값으로 변환해서 돌려줌.
    def PixelToCell(self, p_coords):
        cellX = int(p_coords[1] / PixelPercell)
        cellY = int(p_coords[0] / PixelPercell)
        return [cellX, cellY]

    # 셀 픽셀값으로 변환해서 돌려줌. 이때 계산 편의성 때문에 셀 중앙값으로 변환
    def CellToPixel(self, c_coords): 
        pixelX = int(c_coords[1] * PixelPercell) + CellCenter
        pixelY = int(c_coords[0] * PixelPercell) + CellCenter
        return [pixelX, pixelY]
    
    def reset(self):
        time.sleep(0.05)
        for i in range(MapHeight):
            for j in range(MapWidth):
                broken[i][j] = 0
        broken[0][0] = 1
        x, y = self.canvas.coords(self.agent)
        self.canvas.move(self.agent, CellCenter-x, CellCenter-y)
        self.canvas.delete("tagging_dfs") # grid_world에 남아있는 text지우기 위함.
        return self.PixelToCell(self.canvas.coords(self.agent))
    
    # Agent action and Reward function
    def Action_Function(self, action):

        # agent의 현재 위치
        self.agent_curr_pixel_coords = self.canvas.coords(self.agent)
        self.agent_curr_cell_coords = self.PixelToCell(self.agent_curr_pixel_coords)
        
        # agent에서 success counting check 위한 flag variable {-1 : fail, 0 : not-yet, 1 : success}
        self.whether_success=0

        # base_action는 agent가 move할 때 얼만큼 옮길 것인지를 표현하는 delta value.
        self.action_delta = np.array([0, 0])
        
        # Tkinter 좌표계는 -> : x = col, | : y = row 이고 표기는 (x, y)로 한다.
        # action selection algorithm : 만약 map의 boundary를 넘어서면 그 자리에 가만히 있는다. + Cycle 방지 위해 pre_action 추가        
        if action == 0 :  # 상
            if self.agent_curr_pixel_coords[1] > PixelPercell : # map의 경계선 확인
                self.action_delta[1] -= PixelPercell
        elif action == 1 :  # 하
            if self.agent_curr_pixel_coords[1] < (MapHeight - 1) * PixelPercell:
                self.action_delta[1] += PixelPercell
        elif action == 2 :  # 좌
            if self.agent_curr_pixel_coords[0] > PixelPercell:
                self.action_delta[0] -= PixelPercell
        elif action == 3 :  # 우
            if self.agent_curr_pixel_coords[0] < (MapWidth - 1) * PixelPercell:
                self.action_delta[0] += PixelPercell

        # 에이전트 이동 : canvas.move가 agent를 설정한 delta-value 만큼 이동하게 해주는 핵심적인 method이다.
        # 따로 저장하지 않아도 object에 변화된 값이 적용된다.
        self.canvas.move(self.agent, self.action_delta[0], self.action_delta[1])
        self.agent_moved_pixel_coords = self.canvas.coords(self.agent)
        self.agent_moved_cell_coords = self.PixelToCell(self.agent_moved_pixel_coords)
        self.next_row, self.next_col = self.agent_moved_cell_coords

        # 에이전트를 가장 상위로 배치 : 마치 한글에서 맨 앞으로 배치와 같다.
        self.canvas.tag_raise(self.agent)

        base_camp = [5, 6]

        # # of mine과 reward를 반비례 위해, agent가 이동 전 위치의 data 저장
        self.map_data = MineAndHint_Map[self.agent_curr_cell_coords[0]][self.agent_curr_cell_coords[1]] + 1

        self.done = False ; self.reward = 0

        # Reward Function in case of mine
        # agent가 이동하려는 곳에 mine이 있다면,
        if MineAndHint_Map[self.next_row][self.next_col] == -1 :
            self.reward = -100 / self.map_data ; self.done=True ; self.whether_success=-1
            
        # Reward function in case of Not mine
        else :
            
            # goal인 경우
            if self.agent_moved_cell_coords == [8, 13] :
                self.reward = 1000 ; self.done = True ; self.whether_success=1
            
            # 앞으로 전진한 경우
            elif (self.agent_moved_cell_coords[0] - self.agent_curr_cell_coords[0]) > 0 or (self.agent_moved_cell_coords[1] - self.agent_curr_cell_coords[1]) > 0 :

                self.reward = 100 ; self.whether_success = 0 ; self.DFS(self.next_row, self.next_col)

            # 그외 경우
            else :
                self.whether_success = 0
                self.DFS(self.next_row, self.next_col)

        return self.agent_moved_cell_coords, self.reward, self.done, self.whether_success

# 사방의 지뢰의 갯수를 세어서 리턴하자
def count(row, col):
    
    cnt_ = 0
    
    if (row in list(range(1, 8))) and (col in list(range(1, 13))) :
            
        if(MineAndHint_Map[row + 1][col] == -1):
            cnt_ += 1
        if(MineAndHint_Map[row - 1][col] == -1):
            cnt_ += 1
        if(MineAndHint_Map[row][col + 1] == -1):
            cnt_ += 1
        if(MineAndHint_Map[row][col - 1] == -1):
            cnt_ += 1

        if(MineAndHint_Map[row + 1][col + 1] == -1):
            cnt_ += 1
        if(MineAndHint_Map[row + 1][col - 1] == -1):
            cnt_ += 1
        if(MineAndHint_Map[row - 1][col + 1] == -1):
            cnt_ += 1
        if(MineAndHint_Map[row - 1][col - 1] == -1):
            cnt_ += 1
                
    return cnt_

# 폭탄 세팅
MineAndHint_Map[1][3]=-1 ; MineAndHint_Map[4][7]=-1 ; MineAndHint_Map[1][11]=-1
MineAndHint_Map[7][4]=-1 ; MineAndHint_Map[5][11]=-1 ; MineAndHint_Map[4][1]=-1
MineAndHint_Map[8][9]=-1 ; MineAndHint_Map[5][2]=-1 ; MineAndHint_Map[7][7]=-1
MineAndHint_Map[3][9]=-1 ; MineAndHint_Map[0][8]=-1 ; MineAndHint_Map[3][5]=-1
MineAndHint_Map[4][13]=-1 ; MineAndHint_Map[3][12]=-1 ; MineAndHint_Map[7][1]=-1
MineAndHint_Map[6][0]=-1

# 지뢰 주변에 번호 세팅
for row in range(MapHeight):
    for col in range(MapWidth):
        if(MineAndHint_Map[row][col] == 0):
            MineAndHint_Map[row][col] = count(row, col)
