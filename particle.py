import random
from functools import cmp_to_key

import numpy as np
import math
from kmeans import Kmeans
from index import reset_centroids, assign_cluster, update_centroids, cal_fitness, assign_cluster1, update_centroids1, \
    cal_similar


class Particle:
    def __init__(self,
                 n_cluster: int,
                 data: np.ndarray,
                 labels:np.ndarray,
                 use_kmeans: bool
                 ):
        self.n_cluster=n_cluster
        self.data=data
        self.dim=self.data.shape[1]
        self.labels = labels

        if use_kmeans:
            kmeans = Kmeans(n_cluster=n_cluster, init_pp=False,max_iter=100)
            kmeans.fit(data,kmeans._init_centroid(self.data))
            self.centroids=kmeans.centroids
            print("use-kmeans",use_kmeans)
        else:
            self.centroids = self.init_centroids2(data,n_cluster)

        self.cluster = assign_cluster(self.centroids,data)
        self.centroids = update_centroids(self.n_cluster, self.cluster, data)
        self.fitness = cal_fitness(self.centroids, self.cluster, self.data)

        self.best_cluster=self.cluster.copy()
        self.best_centroids = self.centroids.copy()
        self.best_fitness=self.fitness
        self.velocity = np.zeros_like(self.centroids)

    def init_centroids(self,data,n_cluster):
        # 通过随机指定样本初始化粒子
        idx = np.random.choice(range(len(data)), size=(n_cluster), replace=False)
        centroids = data[idx].copy()
        # 通过随机指定样本的类别，然后求聚类中心初始化粒子
        # cluster = np.random.choice(list(range(n_cluster)), len(data))
        # centroids = update_centrois(data, cluster,n_cluster)
        return centroids

    def init_centroids1(self,data,n_cluster):
        max_val=np.max(data,axis=0)
        min_val=np.min(data,axis=0)
        centroids = np.array([[random.uniform(min_val[i], max_val[i]) for i in range(data.shape[1])] for j in range(n_cluster)])
        return centroids
    def init_centroids4(self,data,n_cluster,n,distanceMeasure=0):
        centroids=[]
        centroids.append(np.random.sample(data,1))
        for _ in range(n_cluster):
            swarm=np.random.sample(data,n)
            dist=[]
            for sample in swarm:
                min_val=np.inf
                for centroid in centroids:
                    d=cal_similar.get(distanceMeasure)(centroid,sample)
                    if d<min_val:
                        min_val=d
                dist.append(min_val)
            max_dist=-np.inf
            id=0
            for i in range(len(swarm)):
                if dist[i]>max_dist:
                    max_dist=dist[i]
                    id=i
            centroids.append(swarm[id])
        return np.array(centroids)

    def init_centroids2(self,data,n_cluster,n=3):
        dim=len(data[0])
        centroids=[]
        for i in range(n_cluster):
            swarm = random.sample(list(data), n)
            centroid=[[swarm[k][j]for k in range(n)]for j in range(dim)]
            median_centroid=np.median(centroid,axis=1)
            centroids.append(list(median_centroid))
        # print(centroids)
        return np.array(centroids)

    def init_centroids3(self,data,n_cluster,clusters):
        centroids=[]
        for i in range(n_cluster):
            idx=np.where(clusters==i)

            # print(idx[0])
            # k=random.sample(idx[0],1)
            # print(k)
            # centroids.append(data[k])
            centroids.append(list(random.choice(data[idx])))
        # print(centroids)
        return np.array(centroids)

    def k_means_localSearch(self,data,centroids,n_cluster):
        for i in range(1e10):
            new_centroids = centroids.copy()
            new_cluster = assign_cluster(new_centroids, data)
            centroids = update_centroids(n_cluster, new_cluster, data)
            diff = np.abs(centroids - new_centroids).sum()
            if np.abs(centroids - new_centroids).sum() < 1e-8:
                break
        fitness = cal_fitness(new_centroids, new_cluster, self.data)
        return fitness,new_centroids,new_cluster
    def re_adjust_centroids(self,centroids):
        new_centroids=sorted(centroids, key=lambda x: [x[i] for i in range(centroids.shape[1])])
        return np.array(new_centroids)

    def calc_fitness(self, data,assign_version=0):
        #划分样本并计算评价函数，个体聚类中心也随之更新
        new_centroids = self.centroids.copy()
        for _ in range(1):
            if assign_version==0:
                new_cluster = assign_cluster(new_centroids,data)
            else:
                new_cluster = assign_cluster1(new_centroids, data)
            new_centroids=update_centroids(self.n_cluster, new_cluster, data)
            # if np.abs(self.centroids - new_centroids).sum() < 1e-8:
            #     break
        self.fitness = cal_fitness(new_centroids, new_cluster, self.data)
        self.centroids = new_centroids
        self.cluster = new_cluster

    def cmp_centroids(self,centroids,assign_version=0):
        if assign_version == 0:
            new_cluster = assign_cluster(centroids, self.data)
        else:
            new_cluster = assign_cluster1(centroids, self.data)
        new_centroids = update_centroids(self.n_cluster, new_cluster, self.data)
        new_fitness=cal_fitness(new_centroids,new_cluster,self.data)
        if new_fitness > self.fitness:
            self.cluster = new_cluster.copy()
            self.centroids =new_centroids.copy()  #centroids
            self.fitness = new_fitness

    def pso_update(self, gbest_centroids, use_ACI,  w, c1, c2):
        v_old = w * self.velocity.copy()
        # if use_ACI:
            # self.best_centroids=reset_centroids(self.best_centroids,self.centroids)
            # gbest_centroids=reset_centroids(gbest_centroids,self.centroids)

        cognitive_component=np.zeros_like(self.centroids)  #差分向量（pbest-x）
        social_component=np.zeros_like(self.centroids)     #交叉向量（gbest-x）
        '''
        改变r1,r2的取值范围
        r1=random.uniform(0,2)
        r2=random.uniform(0,0.5)
        '''
        r1 = random.random()
        r2 = random.random()
        for i in range(self.n_cluster):
            for j in range(self.dim):
                cognitive_component[i][j]=c1 * r1 *(self.best_centroids[i][j] - self.centroids[i][j])
                social_component[i][j]=c2 * r2 * (gbest_centroids[i][j] - self.centroids[i][j])

        #cognitive_component=reset_centroids(cognitive_component,social_component)
        self.velocity=v_old+cognitive_component+social_component
        #P1:标准PSO,每次迭代都更新个体
        self.centroids += self.velocity
        self.calc_fitness(self.data)

    def pso_cmp_update(self, gbest_centroids, use_ACI, w, c1, c2):
        v_old = w * self.velocity.copy()
        # if use_ACI:
        # self.best_centroids=reset_centroids(self.best_centroids,self.centroids)
        # gbest_centroids=reset_centroids(gbest_centroids,self.centroids)
        cognitive_component = np.zeros_like(self.centroids)  # 差分向量（pbest-x）
        social_component = np.zeros_like(self.centroids)  # 交叉向量（gbest-x）
        for i in range(self.n_cluster):
            r1 = random.random()
            r2 = random.random()
            for j in range(self.dim):
                cognitive_component[i][j] = c1 * r1 * (self.best_centroids[i][j] - self.centroids[i][j])
                social_component[i][j] = c2 * r2 * (gbest_centroids[i][j] - self.centroids[i][j])

        # cognitive_component=reset_centroids(cognitive_component,social_component)
        self.velocity = v_old + cognitive_component + social_component
        #P2: 当个体位置更新后计算的适应度值更好时才更新个体
        new_centroids=self.velocity +self.centroids
        self.cmp_centroids(new_centroids)

    def clpso_update(self, learn_centroids,use_ACI,w,c):
        v_old = w * self.velocity
        if use_ACI:
            self.centroids=reset_centroids(self.centroids,learn_centroids)
        for i in range(self.n_cluster):
            r1 = random.random()
            for j in range(self.dim):
                self.velocity[i][j]=v_old[i][j]+c * r1 *(learn_centroids[i][j] - self.centroids[i][j])
        # P1：直接更新个体并计算个体适应度值
        self.centroids=self.velocity+self.centroids
        self.calc_fitness(self.data)

        #P2: 当个体位置更新后计算的适应度值更好时才更新个体
        # new_centroids= v_old+self.centroids
        # self._cmp_centroids(new_centroids)


    # DE/rand/1/bin
    def de_rand_update(self, sample_position,use_ACI,CR,F):
        if use_ACI:
            sample_position[1].centroids=reset_centroids(sample_position[1].centroids,sample_position[2].centroids)
            sample_position[0].centroids=reset_centroids(sample_position[0].centroids,sample_position[2].centroids)

        diff_centroids=F*(sample_position[1].centroids-sample_position[2].centroids)
        mute_centroids=sample_position[0].centroids+diff_centroids
        j_rand = random.sample(range(self.n_cluster), 1)
        new_centroids=self.centroids.copy()
        for i in range(self.n_cluster):
            if random.random() <= CR or i is j_rand:
                new_centroids[i]=mute_centroids[i]
        self.cmp_centroids(new_centroids)

    # DE/best/bin1
    def de_best_update(self, sample_position,use_ACI,CR,F):
        if use_ACI:
            self.best_centroids = reset_centroids(self.best_centroids, sample_position[0].centroids)
            sample_position[1].centroids = reset_centroids(sample_position[1].centroids, sample_position[0].centroids)

        mute_centroids = self.best_centroids + F * (sample_position[0].centroids - sample_position[1].centroids)
        j_rand = random.sample(range(self.n_cluster), 1)
        new_centroids=self.centroids.copy()
        for i in range(self.n_cluster):
            if random.random() <= CR or i is j_rand:
                new_centroids[i]=mute_centroids[i]
        self.cmp_centroids(new_centroids)

    def de_pso_update(self, sample_position ,gbest_position,use_ACI, w, c1, c2, CR,F):
        #取适应度更好时在更新
        self.pso_cmp_update(gbest_position, use_ACI=use_ACI, w = w, c1 =c1 ,c2 = c2)
        self.de_rand_update(sample_position, use_ACI = use_ACI, CR = CR, F = F)

    def qpso_update(self, gbest_position: np.ndarray,mean_best_position:np.ndarray,use_ACI,alpha):
        attarctor1=[]
        attarctor2=[]
        for i in range(self.n_cluster):
            for j in range(len(self.centroids.shape[0])):
                r1 = np.random.random()
                attarctor1.append(r1 * self.best_centroids[i][j])
                attarctor2.append((1 - r1) * gbest_position[i][j])
        attarctor1=np.array(attarctor1)
        attarctor2=np.array(attarctor2)
        if use_ACI:
            attarctor1=reset_centroids(attarctor1,attarctor2)
        local_attractor =attarctor1+attarctor2
        if use_ACI:
            self.centroids=reset_centroids(self.centroids,mean_best_position)
        delta_potentia=[]
        cross=mean_best_position - self.centroids
        for i in range(self.n_cluster):
            r2 = np.random.random()
            r2 = 1 if r2 < 0.5 else -1
            delta_potentia.append(r2 * alpha *np.abs(cross[i])*math.log(1.0 / np.random.random()))
        delta_potentia=np.array(delta_potentia)

        self.centroids = local_attractor +delta_potentia
        self.calc_fitness(self.data)
        #改进的QPSO
        # r3=random.uniform(0,2)
        # r4=random.uniform(0,0.5)
        # self.centroids=r3*(local_attractor+alpha*delta_potential*r4)

def cmp_personal(x,y):
    for i in range(len(x)):
        if x[i]>y[i]:
            return 1
        elif x[i]<y[i]:
            return -1
        else:
            pass

if __name__ == "__main__":
    nums=np.array([[1,2,3],[3,8,7],[3,7,1]])
    nums=sorted(nums, key=lambda x: [x[i] for i in range(nums.shape[1])])
    # nums=sorted(nums, key=lambda x: (x[0],x[1]))
    # sorted(nums, key=cmp_to_key(cmp_personal),reverse=1)
    print(np.array(nums))

    mid_sample = np.median(nums[0])
    print(mid_sample)