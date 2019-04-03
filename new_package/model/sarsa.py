from base_model import Base

class SarsaTable(Base):
    self.juncs = state.copy()
    def __init__(self,learning_rate = 0.01, reward_decay=0.9,e_greedy = 0.9):
        super(SarsaTable, self).__init__(self, learning_rate , reward_decay,e_greedy)
            
    def learn(self,s,a,a_):
        q_predict = self.q_table[a] #a is the index of action
        r = reward(s)
        q_target = r + self.gamma*self.q_table[a_] 
        
        self.q_table[a] += self.lr * (q_target - q_predict) #update q_table


class long(Base):
    pass
