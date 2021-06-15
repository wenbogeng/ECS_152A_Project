# @author Wenbo Geng
import numpy as np
import random
import simpy
import math
import matplotlib.pyplot as plt
import sys
import getopt

class G:
    RANDOM_SEED = 33
    SIM_TIME = 100000
    SLOT_TIME = 1
    N = 30  # number of hosts

    # N = int(sys.argv[1])
    # ARRIVAL_RATES = sys.argv[3]
    # four algorithms
    # RETRANMISSION_POLICIES = sys.argv[2]
    # lambda in the range of [0.001, 0.03]
    ARRIVAL_RATES = [0.001, 0.004, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.02, 0.024]
    # high lambda
    # ARRIVAL_RATES = [0.003, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
        
    # four algorithms
    RETRANMISSION_POLICIES = ["pp", "op", "beb", "lb"]  # 四种算法


class Server_Process(object):
    def __init__(self, env, dictionary_of_nodes, hostcount, arrival_rate, slot_time):
        # number of hosts
        self.hostcount = hostcount
        # store the node informations
        self.dictionary_of_nodes = dictionary_of_nodes
        # running environment
        self.env = env
        # Lambda
        self.arrival_rate = arrival_rate
        self.current_slot = 0
        self.slot_time = slot_time
        # successfully stored packages
        self.slot_succeeded = 0
        # collision packages
        self.slot_collided = 0

    def run(self, env, retran_policy):
        # creates the nodes for hosts
        for x in range(self.hostcount):
            self.dictionary_of_nodes[x] = Node_Process(env, x + 1, self.arrival_rate, self)
            # create host array
            self.env.process(self.dictionary_of_nodes[x].run(self.env, self))

        while True:
            yield self.env.timeout(self.slot_time)
            # store the host index and transmit
            self.node_index = []
            for x in range(self.hostcount):
                # check the packets is transmitted in the host
                if (self.dictionary_of_nodes[x].queue_length == 0):
                    continue
                if (self.dictionary_of_nodes[x].queue_length >= 1 and self.dictionary_of_nodes[
                    x].slot_number_transmission == self.current_slot):
                    # store the index
                    self.node_index.append(x)
            # check the collisions
            if len(self.node_index) == 1:
                self.slot_succeeded += 1
                self.dictionary_of_nodes[self.node_index[0]].access(self.env)
            if len(self.node_index) > 1:
                self.slot_collided += 1
                for x in range(self.hostcount):
                    if (self.dictionary_of_nodes[x].slot_number_transmission == self.current_slot):
                        self.dictionary_of_nodes[x].delay(retran_policy, self)
            self.current_slot += 1

    # return the calculated throughput round in 2 decimals
    def throughput(self):
        return round(self.slot_succeeded * 1.0 / self.current_slot, 2)


# Host Node informations
class Node_Process(object):
    def __init__(self, env, id, arrival_rate, server_process):
        self.env = env
        self.id = id
        self.arrival_rate = arrival_rate
        self.queue_length = 0
        self.slot_number_transmission = 0
        # number of retransmissions
        self.number_reattempts = 0
        self.server_process = server_process
        # number of packages successful transmitted
        self.transmitpackets = 0

     # transmited the package
    def access(self, env):
        self.queue_length -= 1
        self.slot_number_transmission += 1
        self.number_reattempts = 0
        self.transmitpackets += 1
    # recevied the packages
    def run(self, env, server_process):
        while True:
            yield env.timeout(random.expovariate(self.arrival_rate))
            if (self.queue_length == 0):
                self.slot_number_transmission = server_process.current_slot + 1
            self.queue_length += 1
    # delay for the four different algorithms and waiting for the next available time slot
    def delay(self, retran_policy, server_process):
        # P-Persistent
        if retran_policy == "pp":
            self.slot_number_transmission = self.slot_number_transmission + np.random.geometric(0.5)  # p = 0.5
            self.number_reattempts += 1
        # Non -Persistent
        if retran_policy == "op":
            self.slot_number_transmission = self.slot_number_transmission + np.random.geometric(1 / G.N)  # p = 1/N
            self.number_reattempts += 1
        # Binary Exponential backoff
        if retran_policy == "beb":
            K = min(self.number_reattempts, 10)
            R = round(random.uniform(0, 2 ** K))
            self.slot_number_transmission = (self.slot_number_transmission + R + 1)
            self.number_reattempts += 1
        # Linear backoff
        if retran_policy == "lb":
            K = min(self.number_reattempts, 1024)
            R = random.randint(0, K)
            self.slot_number_transmission = self.slot_number_transmission + R + 1
            self.number_reattempts += 1


def main():
    print("Simiulation Analysis of Random Access Protocols")
    random.seed(G.RANDOM_SEED)

    print("Simulation Time: ", G.SIM_TIME)
    for retran_policy in G.RETRANMISSION_POLICIES:
        # store the throughput
        throughput = []
        # for each arrivate rate
        for arrival_rate in G.ARRIVAL_RATES:
            env = simpy.Environment()
            # store the nodes
            dictionary_of_nodes = {}
            serverProcess = Server_Process(env, dictionary_of_nodes, G.N, arrival_rate, G.SLOT_TIME)
            env.process(serverProcess.run(env, retran_policy))
            env.run(until=G.SIM_TIME)
            throughput.append(serverProcess.throughput())
            print(
                "Number of Nodes: " + str(G.N) + ", Retransmission policy: " + retran_policy + ", Arrival Rate: " + str(
                    arrival_rate) + ", Throughput: " + str(serverProcess.throughput()))
        # plot the waveforms
        plt.plot([x * G.N for x in G.ARRIVAL_RATES], throughput, label=retran_policy)
    plt.xlabel('Arrivate Rate N * Lambda')
    plt.ylabel('Throughput')
    plt.legend(loc='best')
    plt.title("Throughtput Analysis For Different Backoff Algorithms")
    plt.show()

def test_ethernet(N, retran_policy, arrival_rate):
   N = int(N)
   arrival_rate = float(arrival_rate)
   random.seed(G.RANDOM_SEED)

   throughput = []
   env = simpy.Environment()
   dictionary_of_nodes = {}
   serverProcess = Server_Process(env, dictionary_of_nodes, G.N, arrival_rate, G.SLOT_TIME)
   env.process(serverProcess.run(env, retran_policy))
   env.run(until=G.SIM_TIME)
   throughput.append(serverProcess.throughput())
   print("Nodes:" + str(N) + " Re-transmission Algorithm:" + retran_policy + " Arrival rate(lambda):" + str(arrival_rate) + " Throughput:" + str(serverProcess.throughput()))


if __name__ == '__main__':
    num,retran_policy,arrival_rate = sys.argv[1:4]
    test_ethernet(num, retran_policy, arrival_rate)
