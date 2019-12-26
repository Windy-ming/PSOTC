import random
import numpy as np
from sklearn.metrics import *
from particle import Particle
from index import reset_centroids, cal_cluster_result, update_centroids, assign_cluster, CH_index, cal_disXX, max_index


class Clustering:
    def __init__(self,
                 n_cluster: int,
                 n_particles: int,
                 data: np.ndarray,
                 labels: np.ndarray,
                 use_kmeans: bool,
                 max_iter: int,
                 w,
                 print_debug: int = 10):
        self.n_cluster = n_cluster
        self.n_particles = n_particles
        self.labels=labels
        self.data = data
        self.max_iter = max_iter
        self.particles = []
        self.mean_particle_position=None
        self.use_kmeans = use_kmeans
        self.print_debug = print_debug
        self.gbest_cluster=None
        self.gbest_fitness = -np.inf
        self.gbest_centroids = None
        self.assign_version = 0  # 0表示采用常规划分样本方法，1表示采用NMP划分方法
        self._init_particles()
        #w采用递减的方式
        self.w=w
        self.w_step=0.5/max_iter
        self.stop_iter_num=20
        self.tolerance=1e-4

    def _init_particles(self):
        for i in range(self.n_particles):
            #是否对采用k-means的聚类结果初始化第一个粒子
            if i < 1 and self.use_kmeans:
                particle = Particle(self.n_cluster, self.data,self.labels, self.use_kmeans)
                #print("K-Means的聚类结果",particle.fitness)
            else:
                particle = Particle(self.n_cluster, self.data,self.labels,self.use_kmeans)
            self.particles.append(particle)
        self._update_gbest()

    def _update_gbest(self):
        for particle in self.particles:
            if particle.fitness > particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_centroids = particle.centroids.copy()
                particle.best_cluster = particle.cluster.copy()

            if particle.fitness > self.gbest_fitness:
                self.gbest_centroids = particle.centroids.copy()
                self.gbest_fitness = particle.fitness
                self.gbest_cluster = particle.cluster.copy()

    def _reset_particles(self):
        for particle in self.particles:
            particle.centroids = reset_centroids(particle.centroids,self.gbest_centroids)
            particle.best_centroids=reset_centroids(particle.best_centroids, self.gbest_centroids)

    def pso_run(self,w,c1,c2,use_ACI):
        iter_result = []
        print('Initial gbest fitness by PSO', self.gbest_fitness)
        count=0
        for i in range(self.max_iter):
            if use_ACI:
                self._reset_particles()
            for particle in self.particles:
                particle.pso_update(self.gbest_centroids,use_ACI=use_ACI,w=w,c1=c1,c2=c2)
            cur_fitness=self.gbest_fitness
            self._update_gbest()
            if np.linalg.norm(cur_fitness-self.gbest_fitness)>self.tolerance :
                count=count+1
            else:
                count=0
            if count>self.stop_iter_num:
                break
            iter_result.append(self.gbest_fitness)
            ari=adjusted_rand_score(self.labels, self.gbest_cluster)
            if i % self.print_debug == 0:
                print('Iter {:04d}/{:04d} current cluster fitness: {:.4f}, ARI:{:.4f}'
                      .format(i + 1, self.max_iter,self.gbest_fitness, ari))
        print('Finish {:04d}/{:04d} opt cluster fitness: {:.4f}, ARI:{:.4f}'
              .format(i + 1, self.max_iter,self.gbest_fitness, ari))

        return self.gbest_fitness, self.gbest_cluster, self.gbest_centroids, iter_result

    def clpso_run(self,w,c,use_ACI):
        iter_result = []
        print('Initial global best fitness by PSO', self.gbest_fitness)
        count=0
        for i in range(self.max_iter):
            if use_ACI:
                self._reset_particles()
            for j in range(self.n_particles):
                pc = 0.25 + 0.45 * (np.math.exp(10 * j / (self.n_particles - 1))-1)/(np.math.exp(10)-1)
                if random.random()<pc:
                    p=random.sample(range(self.n_particles), 2)
                    learn=p[0] if self.particles[p[0]].best_fitness>self.particles[p[1]].best_fitness else p[1]
                else:
                    learn=j
                # print(j,learn,pc)
                self.particles[j].clpso_update(self.particles[learn].centroids,use_ACI=use_ACI,w=w-self.w_step,c=c)
            cur_fitness=self.gbest_fitness
            self._update_gbest()
            if np.linalg.norm(cur_fitness-self.gbest_fitness)>self.tolerance :
                count=count+1
            else:
                count=0
            iter_result.append(self.gbest_fitness)
            ari=adjusted_rand_score(self.labels, self.gbest_cluster)

            if i % self.print_debug == 0:
                print('Iter {:04d}/{:04d} current cluster fitness: {:.4f}, ARI:{:.4f}'
                      .format(i + 1, self.max_iter,self.gbest_fitness, ari))
        print('Finish {:04d}/{:04d} opt cluster fitness: {:.4f}, ARI:{:.4f}'
              .format(i + 1, self.max_iter,self.gbest_fitness, ari))
        return self.gbest_fitness, self.gbest_cluster, self.gbest_centroids, iter_result

    def de_run(self,CR,F,use_ACI,method_type="de_rand"):
        iter_result=[]
        print('Initial global best fitness by ', self.gbest_fitness)
        count=0
        for i in range(self.max_iter):
            if use_ACI:
                for particle in self.particles:
                    particle.centroids=reset_centroids(particle.centroids, self.gbest_centroids)
            for j in range(self.n_particles):
                id_list=list(range(self.n_particles))
                id_list.remove(j)
                if method_type=="de_rand":
                    id_list=random.sample(id_list,3)
                    sample_position=[self.particles[k] for k in id_list]
                    self.particles[j].de_rand_update(sample_position,use_ACI=use_ACI,CR=CR,F=F)
                else:
                    id_list = random.sample(id_list, 2)
                    sample_position = [self.particles[i] for i in id_list]
                    self.particles[j].de_best_update(sample_position, use_ACI=use_ACI, CR=CR, F=F)

            cur_fitness=self.gbest_fitness
            self._update_gbest()
            if np.linalg.norm(cur_fitness-self.gbest_fitness)>self.tolerance :
                count=count+1
            else:
                count=0
            if count>self.stop_iter_num:
                break
            iter_result.append(self.gbest_fitness)
            ari=adjusted_rand_score(self.labels, self.gbest_cluster)
            if i % self.print_debug == 0:
                print('Iter {:04d}/{:04d} current cluster fitness: {:.4f}, ARI:{:.4f}'
                      .format(i + 1, self.max_iter,self.gbest_fitness,ari))
        print('Finish {:04d}/{:04d} opt cluster fitness: {:.4f}, ARI:{:.4f}'
              .format(i + 1, self.max_iter,self.gbest_fitness,ari))
        return self.gbest_fitness, self.gbest_cluster, self.gbest_centroids, iter_result

    def de_pso_run(self,use_ACI,w,c1,c2,CR,F):
        iter_result=[]
        ari=adjusted_rand_score(self.labels, self.gbest_cluster)
        count=0
        print('Initial global best fitness by DE-PSO:', self.gbest_fitness,ari)
        for i in range(self.max_iter):
            if use_ACI:
                self._reset_particles()
            for j in range(self.n_particles):
                #DE/rand/2
                id_list = list(range(self.n_particles))
                id_list.remove(j)
                id_list = random.sample(id_list, 3)
                sample_position = [self.particles[k] for k in id_list]
                self.particles[j].de_pso_update(sample_position,self.gbest_centroids,use_ACI,w=w,c1=c1,c2=c2,CR=CR,F=F)

            cur_fitness = self.gbest_fitness
            self._update_gbest()
            if np.linalg.norm(cur_fitness-self.gbest_fitness)>self.tolerance :
                count=count+1
            else:
                count=0
            if count > self.stop_iter_num:
                break
            ari = adjusted_rand_score(self.labels, self.gbest_cluster)

            if i % self.print_debug == 0:
                print('Iter {:04d}/{:04d} current cluster fitness: {:.4f}, ARI:{:.4f}'
                      .format(i + 1, self.max_iter, self.gbest_fitness, ari))
        print('Finish {:04d}/{:04d} opt cluster fitness: {:.4f}, ARI:{:.4f}'
              .format(i + 1, self.max_iter, self.gbest_fitness, ari))
        return self.gbest_fitness, self.gbest_cluster, self.gbest_centroids, iter_result

    def pso_run_Subpopulation(self,w,c1,c2,pro,use_ACI):
        iter_result = []
        k=int(pro*self.n_particles)
        print('Initial gbest fitness by PSO', self.gbest_fitness)
        count=0
        for i in range(self.max_iter):
            if use_ACI:
                self._reset_particles()
            fitness_list=[self.particles[t].fitness for t in range(self.n_particles)]
            #若不排序
            gbest_centroids1=self.particles[max_index(fitness_list,0,k)].best_centroids.copy()
            gbest_centroids2=self.particles[max_index(fitness_list,k,self.n_particles)].best_centroids.copy()

            # idx = np.argsort(fitness_list)
            # self.particles = [self.particles[t] for t in idx]
            # gbest_centroids1=self.particles[0].best_centroids.copy()
            # gbest_centroids2=self.particles[k].best_centroids.copy()

            for j in range(self.n_particles):
                gbest_centroids = gbest_centroids1 if j < k else gbest_centroids2
                self.particles[j].pso_update(gbest_centroids,use_ACI=use_ACI,w=w,c1=c1,c2=c2)
            cur_fitness=self.gbest_fitness
            self._update_gbest()
            if np.linalg.norm(cur_fitness-self.gbest_fitness)>self.tolerance :
                count=count+1
            else:
                count=0
            if count>self.stop_iter_num:
                break
            iter_result.append(self.gbest_fitness)
            ari=adjusted_rand_score(self.labels, self.gbest_cluster)
            if i % self.print_debug == 0:
                print('Iter {:04d}/{:04d} current cluster fitness: {:.4f}, ARI:{:.4f}'
                      .format(i + 1, self.max_iter,self.gbest_fitness, ari))
        print('Finish {:04d}/{:04d} opt cluster fitness: {:.4f}, ARI:{:.4f}'
              .format(i + 1, self.max_iter,self.gbest_fitness, ari))
        return self.gbest_fitness, self.gbest_cluster, self.gbest_centroids, iter_result

    def de_run_Subpopulation(self,CR,F,use_ACI,pro,method_type="de_best"):
        iter_result=[]
        k=int(pro*self.n_particles)
        print('Initial global best fitness by ',method_type, self.gbest_fitness)
        count=0
        for i in range(self.max_iter):
            if use_ACI:
                for particle in self.particles:
                    particle.centroids=reset_centroids(particle.centroids, self.gbest_centroids)
            fitness_list=[self.particles[t].fitness for t in range(self.n_particles)]
            idx = np.argsort(fitness_list)
            self.particles=[self.particles[t] for t in idx]
            swarm1=self.particles[:k]
            swarm2=self.particles[k:]
            for j in range(k):
                if j<k:
                    id_list = list(range(k))
                    id_list.remove(j)
                    id_list = random.sample(id_list, 2)
                else:
                    id_list = list(range(k,self.n_particles))
                    id_list.remove(j)
                    id_list = random.sample(id_list, 2)
                sample_position = [self.particles[t] for t in id_list]
                self.particles[j].de_best_update(sample_position, use_ACI=use_ACI, CR=CR, F=F)

            cur_fitness=self.gbest_fitness
            self._update_gbest()
            if np.linalg.norm(cur_fitness-self.gbest_fitness)>self.tolerance :
                count=count+1
            else:
                count=0
            if count>self.stop_iter_num:
                break
            iter_result.append(self.gbest_fitness)
            ari=adjusted_rand_score(self.labels, self.gbest_cluster)
            if i % self.print_debug == 0:
                print('Iter {:04d}/{:04d} current cluster fitness: {:.4f}, ARI:{:.4f}'
                      .format(i + 1, self.max_iter,self.gbest_fitness,ari))
        print('Finish {:04d}/{:04d} opt cluster fitness: {:.4f}, ARI:{:.4f}'
              .format(i + 1, self.max_iter,self.gbest_fitness,ari))
        return self.gbest_fitness, self.gbest_cluster, self.gbest_centroids, iter_result

    def pso_de_run_Subpopulation(self,w,c1,c2,CR,F,use_ACI,pro,method_type="de_best"):
        iter_result=[]
        k=int(pro*self.n_particles)
        print('Initial global best fitness by ',method_type, self.gbest_fitness)
        count=0
        for i in range(self.max_iter):
            if use_ACI:
                for particle in self.particles:
                    particle.centroids=reset_centroids(particle.centroids, self.gbest_centroids)
            fitness_list=[self.particles[t].fitness for t in range(self.n_particles)]
            idx = np.argsort(fitness_list)
            self.particles=[self.particles[t] for t in idx]
            swarm1=self.particles[:k]
            swarm2=self.particles[k:]

            for particle in range(self.n_particles[:k]):
                self.gbest_centroids=self.particles[k-1].best_centroids
                particle.pso_update(self.gbest_centroids,use_ACI=use_ACI,w=w,c1=c1,c2=c2)

            for particle in range(self.n_particles[k:]):
                sample_position = random.sample(swarm2, 2)
                particle.de_best_update(sample_position, use_ACI=use_ACI, CR=CR, F=F)

            cur_fitness=self.gbest_fitness
            self._update_gbest()
            if np.linalg.norm(cur_fitness-self.gbest_fitness)>self.tolerance :
                count=count+1
            else:
                count=0
            if count>self.stop_iter_num:
                break
            iter_result.append(self.gbest_fitness)
            ari=adjusted_rand_score(self.labels, self.gbest_cluster)
            if i % self.print_debug == 0:
                print('Iter {:04d}/{:04d} current cluster fitness: {:.4f}, ARI:{:.4f}:'
                      .format(i + 1, self.max_iter,self.gbest_fitness,ari))
        print('Finish {:04d}/{:04d} opt cluster fitness: {:.4f}, ARI:{:.4f}'
              .format(i + 1, self.max_iter,self.gbest_fitness,ari))
        return self.gbest_fitness, self.gbest_cluster, self.gbest_centroids, iter_result

    def pso_de_run_Subpopulation1(self,w,c1,c2,CR,F,use_ACI,pro,method_type="de_best"):
        iter_result=[]
        k=int(pro*self.n_particles)
        fg=10
        print('Initial global best fitness by ',method_type, self.gbest_fitness)
        count=0
        for i in range(self.max_iter):
            if use_ACI:
                for particle in self.particles:
                    particle.centroids=reset_centroids(particle.centroids, self.gbest_centroids)
            if i%fg<fg/2:
                for particle in range(self.n_particles[:k]):
                    self.gbest_centroids = self.particles[k - 1].best_centroids
                    particle.pso_update(self.gbest_centroids, use_ACI=use_ACI, w=w, c1=c1, c2=c2)
            else:
                sample_position = random.sample(self.particles, 2)
                particle.de_best_update(sample_position, use_ACI=use_ACI, CR=CR, F=F)

            cur_fitness=self.gbest_fitness
            self._update_gbest()
            if np.linalg.norm(cur_fitness-self.gbest_fitness)>self.tolerance :
                count=count+1
            else:
                count=0
            if count>self.stop_iter_num:
                break
            iter_result.append(self.gbest_fitness)
            ari=adjusted_rand_score(self.labels, self.gbest_cluster)
            if i % self.print_debug == 0:
                print('Iter {:04d}/{:04d} current cluster fitness: {:.4f}, ARI:{:.4f}:'
                      .format(i + 1, self.max_iter,self.gbest_fitness,ari))
        print('Finish {:04d}/{:04d} opt cluster fitness: {:.4f}, ARI:{:.4f}'
              .format(i + 1, self.max_iter,self.gbest_fitness,ari))
        return self.gbest_fitness, self.gbest_cluster, self.gbest_centroids, iter_result

    def pso_de_run_Subpopulation2(self,w,c1,c2,CR,F,use_ACI,pro,method_type="de_best"):
        iter_result=[]
        k=int(pro*self.n_particles)
        print('Initial global best fitness by ',method_type, self.gbest_fitness)
        for i in range(self.max_iter):
            if use_ACI:
                for particle in self.particles:
                    particle.centroids=reset_centroids(particle.centroids, self.gbest_centroids)
            fitness_list=[self.particles[t].fitness for t in range(self.n_particles)]
            idx = np.argsort(fitness_list)
            self.particles=[self.particles[t] for t in idx]
            swarm1=self.particles[:k]
            swarm2=self.particles[k:]

            for particle in range(self.n_particles[:k]):
                self.gbest_centroids=self.particles[k-1].best_centroids
                particle.pso_update(self.gbest_centroids,use_ACI=use_ACI,w=w,c1=c1,c2=c2)

            for particle in range(self.n_particles[k:]):
                sample_position = random.sample(swarm2, 2)
                particle.de_best_update(sample_position, use_ACI=use_ACI, CR=CR, F=F)

            self._update_gbest()
            iter_result.append(self.gbest_fitness)
            ari=adjusted_rand_score(self.labels, self.gbest_cluster)
            if i % self.print_debug == 0:
                print('Iter {:04d}/{:04d} current cluster fitness: {:.4f}, ARI:{:.4f}:'
                      .format(i + 1, self.max_iter,self.gbest_fitness,ari))
        print('Finish {:04d}/{:04d} opt cluster fitness: {:.4f}, ARI:{:.4f}'
              .format(i + 1, self.max_iter,self.gbest_fitness,ari))
        return self.gbest_fitness, self.gbest_cluster, self.gbest_centroids, iter_result

    def qpso_run(self,alpha,use_ACI):
        print('Initial global best fitness by QPSO', self.gbest_fitness)
        iter_cluster_result=[]
        iter_result=[]
        for i in range(self.max_iter):
            # self._reset_particles()
            self.mean_particle_position = np.mean(np.array([particle.best_position for particle in self.particles]), axis=0)
            # self.mean_particle_position=reset_centroids(self.mean_particle_position,self.gbest_centroids)
            for particle in self.particles:
                particle.QPSO_update(self.gbest_centroids, self.mean_particle_position,use_ACI=use_ACI,alpha=alpha)
                #print(i, particle.best_fitness, self.gbest_fitness)
            self._update_gbest()
            iter_result.append(self.gbest_fitness)
            opt_centroids = update_centroids( self.n_cluster,self.labels,self.data)
            centroids = reset_centroids(self.gbest_centroids, opt_centroids)
            clusters = assign_cluster(centroids, self.data)
            cluster_result=cal_cluster_result(self.gbest_fitness, clusters,self.labels)
            iter_cluster_result.append(cluster_result)

            # if i % self.print_debug == 0:
            # print('Iter {:04d}/{:04d} current cluster result {.5f}:'.format(i + 1, self.max_iter,cluster_result))
            # print(cluster_result)
        # print('Finish {:04d}/{:04d} opt cluster result {.5f}:'.format(i + 1, self.max_iter,cluster_result))
        iter_cluster_result = np.array(iter_cluster_result)
        # np.savetxt("QPSO_iter_result.csv", iter_cluster_result, delimiter="\t", fmt="%.3f")
        return self.gbest_fitness, self.gbest_cluster, self.gbest_centroids,iter_result, np.max(iter_cluster_result[1:],axis=0)

    def de_qpso_run(self,alpha,CR,F,use_ACI):
        iter_result=[]
        print('Initial global best fitness by DE-QPSO: ', self.gbest_fitness)
        for i in range(self.max_iter):
            self._reset_particles()
            self.mean_particle_position = np.mean(np.array([particle.best_position for particle in self.particles]), axis=0)
            self.mean_particle_position = reset_centroids(self.mean_particle_position, self.gbest_centroids)
            for particle in self.particles:
                sample_position=random.sample(self.particles,3)
                particle.de_qpso_update(sample_position,self.gbest_centroids,self.mean_particle_position,alpha=alpha,CR=CR,F=F)
            self._update_gbest()
            iter_result.append(self.gbest_fitness)
            #print(i,self.gbest_fitness)
            #if i % self.print_debug == 0:
                #print('Iteration {:04d}/{:04d} current gbest fitness {:.18f},gbest sse {:.18f}'.format(
                    #i + 1, self.max_iter, self.gbest_fitness,self.gbest_sse))
            #print(i,self.gbest_fitness)
        print('Finish with gbest fitness {:.18f},gbest sse {:.18f}'.format(self.gbest_fitness,self.gbest_sse))
        return self.gbest_sse,self.gbest_fitness,self.gbest_cluster,self.gbest_centroids,iter_result
if __name__ == "__main__":
    a=np.random.choice(list(range(100)),5)
    print(a)
    nums = np.array([4, 1, 5, 2, 9, 6, 8, 7])
    idx=nums[nums > 7]
    res=np.mean(nums[np.array([1,2])])
    print(res)
    print(idx.any())
    print(random.choice(nums))
    print(nums[4])
    c=np.argsort(nums)
    print(c)
    sorted_nums = sorted(enumerate(nums), key=lambda x: x[1])
    idx = [i[0] for i in sorted_nums]
    nums = [i[1] for i in sorted_nums]
    c=np.where(np.array(nums)>9)
    print(type(c))
    print(nums[nums>7])
    pass