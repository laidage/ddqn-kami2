"""
搭建游戏环境，并设定相应奖励
"""
import random
from collections import deque

class Env:
    def __init__(self):
        self.colors = 5
        self.rest_of_colors = []
        self.state = []

    def tran_state(self):
        # 将游戏状态转换为可供pytorch训练的格式
        state = []
        for color_index in range(5):
            if color_index >= self.colors:
                state.append([[-1 for _ in range(15)] for _ in range(20)])
                continue
            types = []
            for column_index in range(20):
                column = [0 for _ in range(15)]
                if column_index % 4 == 1 or column_index % 4 == 2: 
                    column[14] = -1
                types.append(column)
            state.append(types)
        for i in range(20):
            for j in range(len(self.state[i])):
                color = self.state[i][j]
                state[color][i][j] = 1
        return state

    def random(self):
        # 产生随机的游戏环境
        self.colors = random.randint(2, 5)
        self.rest_of_colors = [i for i in range(0, self.colors)]
        self.state = []

        for i in range(20):
            max_index = 14 if i % 4 == 1 or i % 4 == 2 else 15
            column = []
            for j in range(max_index):
                column.append(random.randint(0, self.colors-1))
            self.state.append(column)
        over, _ = self.game_over()
        if over:
            self.state[0][0] = (self.state[0][0] + 1) % self.colors
        
        return self.tran_state(), self.colors

    def load_config_file(self, file):
        # 导入状态文件，获取对应的游戏环境
        self.state = []
        with open(file, 'r') as f:
            lines = f.readlines()
            self.colors = int(lines[0])
            self.rest_of_colors = [i for i in range(0, self.colors)]
            color_rgbs = []
            for i in range(self.colors):
                color_rgbs.append(eval(lines[i+1]))
            for line in lines[self.colors+1:]:
                self.state.append(eval(line))
        return self.tran_state(), self.colors

    

    def game_over(self):
        # 判断游戏是否结束，并返回每一步的奖励，消除一个颜色奖励为3，游戏结束奖励为10，其余为-1
        reward = -1
        for color in self.rest_of_colors:
            if not any(color in column for column in self.state):
                self.rest_of_colors.remove(color)
                reward = 3
        if len(self.rest_of_colors) == 1:
            return True, 10
        return False, reward

    def deque_util(self, deq, x, y, color):
        deq.append((x, y))
        self.state[x][y] = color

    def valid(self, action):
        # 判断动作是否合法
        color = action // 300
        index = action - color * 300
        x = index // 15
        y = index % 15
        if (y == 14 and (x % 4 == 1 or x % 4 == 2)) or (color >= self.colors) or (self.state[x][y] == color):
            return False
        return True

    def step(self, action):
        # 在游戏中执行一步，获取执行后的游戏状态，奖励，以及游戏是否已经结束

        #将action值转换为具体的游戏行为，包括使用哪种颜色点击那个三角块
        color = action // 290
        index = action - color * 290
        four_columns = index // (29 * 2)
        index = index - four_columns * 29 * 2
        if index < 15:
            y = index
            x = four_columns * 4
        elif index < 15 + 14:
            y = index - 15
            x = four_columns * 4 + 1
        elif index < 15 + 14 + 14:
            y = index - 29
            x = four_columns * 4 + 2
        else:
            y = index - 43
            x = four_columns * 4 + 3
        
        rest_colors = self.rest_of_colors[:]
        rest_colors.remove(self.state[x][y])
        color = rest_colors[color % len(rest_colors)]
        before_color = self.state[x][y]
        deq = deque()
        deq.append((x, y))
        self.state[x][y] = color

        while len(deq) != 0:
            # 执行动作，修改游戏状态
            x, y = deq.pop()
            if x-1 >= 0 and len(self.state[x-1]) > y and self.state[x - 1][y] == before_color:
                self.deque_util(deq, x-1, y, color)
            if x+1 < 20 and len(self.state[x+1]) > y and self.state[x + 1][y] == before_color:
                self.deque_util(deq, x+1, y, color)
            if x % 4 == 0:
                if x + 1 < 20 and y - 1 >= 0 and self.state[x+1][y-1] == before_color:
                    self.deque_util(deq, x+1, y-1, color)
            elif x % 4 == 1:
                if x - 1 >= 0 and len(self.state[x-1]) > y + 1 and self.state[x-1][y+1] == before_color:
                    self.deque_util(deq, x-1, y+1, color)
            elif x % 4 == 2:
                if x + 1 < 20 and len(self.state[x+1]) > y + 1 and self.state[x+1][y+1] == before_color:
                    self.deque_util(deq, x+1, y+1, color)
            else:
                if x - 1 >=0 and y - 1 >= 0 and self.state[x-1][y-1] == before_color:
                    self.deque_util(deq, x-1, y-1, color)
        
        done, reward = self.game_over()
        return self.tran_state(), reward, done
