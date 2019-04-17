import numpy as np
import pandas as pd

Data_File_Path='E:\\project\\car\\201904\\dataoutput' #请在这里输入存放day1/2/19/25四个文件夹的路径
ChosenDate='day1' #请在day1/2/19/25中选择
'''
其中这四天的情况分别为
day1:晴天+工作日
day2:雨天+工作日
day19:雨天+周末
day25:晴天+周末
'''
speed_matrix_data=pd.read_csv(Data_File_Path+'\\'+ChosenDate+'\\'+ChosenDate+'_updated.csv')
poisson_matrix_data=pd.read_csv(Data_File_Path+'\\'+ChosenDate+'\\'+ChosenDate+'_Count_TR.csv')
poisson_matrix_data['Code2']=poisson_matrix_data.apply(lambda x:x['RoadCode'][-2:],axis=1)
#以下四个是四个路口的转移矩阵，使用dayx_Cross1\2\3\4.csv
transition_matrix_data=[]
for i in range(1,5):
    transition_matrix_data.append(np.array(pd.read_csv(Data_File_Path+'\\'+ChosenDate+'\\'+ChosenDate+'_Cross'+str(i)+'.csv',index_col=0)))


'''
注意在调用时，return的matrix中，如speed_maxtrix(cross_id,epoch)[i,j]，代表从i方向通过cross_id到j方向的车在epoch时间中的平均速度
其中，i,j的取值为0,1,2,3，分别代表西，北，东，南
'''
    

def speed_matrix(cross_id, epoch):
    outMat=np.zeros((4,4))
    order=['baxc','xdyz','yefg','hzji']
    cross=ord(cross_id)-65 #ABCD通过ASCII读取转为0123
    for i in range(0,4):
        for j in range(0,4):
            key=order[cross][i]+str(cross+1)+order[cross][j] #转弯方式对应的code
            speed=sum(speed_matrix_data[(speed_matrix_data['Code']==key) & (speed_matrix_data['Time']==int(epoch/90))]['AvgSpeed']) #对应的时间段
            outMat[i,j]=speed
                              
    '''
    :param:
    cross_id:   "A"/"B"/"C"/"D" #id似乎是python的保留变量名，我改成cross_id了
    epoch: 以10s为单位,
    :return:
    4x4 的速度矩阵
    '''
    return outMat


def transition_matrix(cross_id, epoch):
    cross=ord(cross_id)-65 #ABCD通过ASCII读取转为0123
    return transition_matrix_data[cross]


def poisson_matrix(cross_id, epoch):
    outArray=np.zeros(4)
    cross=ord(cross_id)-65 #ABCD通过ASCII读取转为0123
    for i in range(0,4):
        key=order[cross][i]+str(cross+1)
        count=sum(poisson_matrix_data[(poisson_matrix_data['Code2']==key) & (poisson_matrix_data['Time']==int(epoch/90))]['Count'])
        outArray[i]=count/90
    return outArray

'''这里依次返回西北东南四个方向前往路口的期望车辆数（10秒内）'''
