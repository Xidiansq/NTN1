# -*-coding:utf-8-*-
import Parameters
import numpy as np
import math as m
def get_paper_reward_info(extra,MAX_DOWN_Rate):
    """
    Function:
    Input:
    Output:
    """
    #print(extra)
    #input()
    Sate_step_Qos, Sate_step_new, Sate_step_wait, Sate_step_downtx, Sate_step_downthroughput,  = 0, 0, 0, 0, 0
    Bs_step_Qos, Bs_step_new, Bs_step_wait, Bs_step_downtx, Bs_step_downthroughput = 0, 0, 0, 0, 0
    sate_extra = extra['Sate_User']
    bs_extra = extra['Bs_User']
    if sate_extra==[]: return 1
    Maxdis_Bs = max(max(item["Dis_Bs"] for item in sate_extra), max(item["Dis_Bs"] for item in bs_extra))
    Qos_sa=[]
    Qos_bs=[]
    for Sate_user in sate_extra:
        Sate_step_new += Sate_user["NewData"]        #该时刻新到的数据
        Sate_step_wait += Sate_user["Total_WaitData"]#所有等待的数据
        Sate_step_downtx +=Sate_user["Down_TxData"]  #该时刻传输的数据
        Sate_step_downthroughput += Sate_user["Down_Throughput"] #该时刻的下行吞吐量
        Qos_sa.append(reward_sa_Qos(Sate_user["Angle"],          #卫星仰角
                                    Sate_user["Dis_Bs"],         #到基站的距离
                                    Sate_user["Last_WaitData"], #所有等待的数据
                                    Sate_user["Down_TxData"],
                                    Maxdis_Bs,
                                    MAX_DOWN_Rate[int(Sate_user["UserID"])])    #该时刻传输的数据
                                    )
        #Sate_step_Qos += Sate_user["Dis_Bs"]/Maxdis_Bs * Sate_user["Down_TxData"] / Sate_user["Total_WaitData"] if Sate_user["Total_WaitData"] != 0 else 1

    # for Bs_User in bs_extra:
    #     Bs_step_new += Bs_User["NewData"]        #该时刻新到的数据
    #     Bs_step_wait += Bs_User["Total_WaitData"]#所有等待的数据
    #     Bs_step_downtx +=Bs_User["Down_TxData"]  #该时刻传输的数据
    #     Bs_step_downthroughput += Bs_User["Down_Throughput"] #该时刻的下行吞吐量
    #     Qos_bs.append(reward_bs_Qos(Bs_User["Total_WaitData"],
    #                                 Bs_User["Down_TxData"])
    #                                 if Bs_User["Total_WaitData"] != 0 else 1)

        

        #先用一个常数代替
        # Bs_step_Qos += Bs_User["Down_TxData"] / Bs_User["Total_WaitData"]
        # Bs_step_Qos = 0.3
    Sate_step_ave_downtx = Sate_step_downtx / len(sate_extra) if len(sate_extra) > 0 else 0
    Sate_step_ave_downthroughput = Sate_step_downthroughput / len(sate_extra) if len(sate_extra) > 0 else 0

    Bs_step_ave_downtx = Bs_step_downtx / len(bs_extra) if len(bs_extra) > 0 else 0
    Bs_step_ave_downthroughput = Bs_step_downthroughput / len(bs_extra) if len(bs_extra) > 0 else 0

    # r1 = reward_Qos(Sate_step_Qos, Bs_step_Qos)
    r1 =  1 * (sum(Qos_sa)/len(Qos_sa))  #+0.5 * (sum(Qos_bs)/len(Qos_bs)) +
    print("---------------------------------")
    print("reward",r1)
    print("---------------------------------")
    return r1


def reward_sa_Qos(angle_Sa2User, distance_bs,Request,Capacity,Maxdis_Bs,MAX_DOWN_Rate):
    """
    Function: 计算卫星用户的Qos
    Input:
        angle_Sa2User:卫星与用户的夹角
        distance_bs:用户与基站的距离
        Request:用户传输需求量
        Capacity:信道容量
    Output:卫星用户的Qos
    """
    Q_n=channel_quality_factor(angle_Sa2User)
    beta_n=service_priority(distance_bs,Maxdis_Bs)
    eta_n=capacity_request_ratio(Request,Capacity,MAX_DOWN_Rate)
    print("---------------------------------")
    print("Q_n",Q_n)
    print("beta_n",beta_n)
    print("eta_n",eta_n)
    #print("reward",beta_n*eta_n 
    return eta_n 
def reward_bs_Qos(Request,Capacity):
    """
    Function:计算基站用户的Qos
    Input:
        Request:用户传输需求量
        Capacity:信道容量
    Output:基站用户的Qos
    """
    return capacity_request_ratio(Request,Capacity)


def channel_quality_factor(angle_Sa2User):
    """
    计算卫星信道质量
    """
    return  np.log(angle_Sa2User/Parameters.Elevation_Angle)
def service_priority(distance_bs,Maxdis_Bs):
    """
    计算卫星用户的优先级
    """
    return loss_path(distance_bs)/loss_path(Maxdis_Bs)
    
def loss_path(distance):
    """
    计算路径损失(dBi)
    """
    return 32.44+20*np.log10(distance/1000)+20*np.log10(Parameters.frequency/1e6)

def capacity_request_ratio(Request,Capacity,MAX_DOWN_Rate):
    """
    计算要求和容量的比值
    """
    if Request==0: return 0
    
    
    return Capacity/min(Request,MAX_DOWN_Rate)

if __name__ == '__main__':
    print(channel_quality_factor(80))