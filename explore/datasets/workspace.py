from data_tools import h5_helper, mujoco_io
import robotic as ry
import sim_wrappers as sim
import numpy as np
from time import sleep
from scipy.spatial import KDTree
#from diversemanip.mlp_basics import *
import omegaconf
from datetime import datetime
#from .mujoco_goal_gym import MujocoGoalGym, VectorGym
from .ManipulationNLP import ManipulationNLP

class H5MatchFile:
    def __init__(self, h5_filename):
        h5 = h5_helper.H5Reader(h5_filename)
        self.__dict__ = h5.read_all() #MAGIC!

class ConfigsFile:
    manifest: dict
    supports: list
    config: dict
    q: np.array
    contacts: np.array
    forces: np.array

    def __init__(self, h5_filename):
        h5 = h5_helper.H5Reader(h5_filename)
        self.__dict__ = h5.read_all() #MAGIC!
        if 'potential_supports' in self.manifest:
            self.supports = self.manifest['potential_supports']
        else:
            self.supports = self.manifest['potential supports'] #OLD convension...

class SplinesFile(H5MatchFile):
    manifest: dict
    params: list
    tasks: np.array
    splines: np.array
    costs: np.array

    def select_sucessful(self, threshold):
        good = np.argwhere(self.costs<threshold).reshape(-1)
        self.splines = self.splines[good]
        self.tasks = self.tasks[good]
        self.costs = self.costs[good]

class EpisodesFile(H5MatchFile):
    manifest: dict
    config: dict
    tau_step: float
    task: np.array
    start: np.array
    goal: np.array
    final_cost: np.array
    state: np.array
    action: np.array

class TrajectoryData:
    goal: np.array
    state: np.array
    observation: np.array
    action: np.array
    time_to_go: np.array

class Workspace:
    tau_sim = .001

    def __init__(self, cfg = None):
        if cfg is None:
            self.cfg = omegaconf.OmegaConf.load('config.yaml')
        else:
            self.cfg = cfg
        self.cfg.tag = f'{datetime.now().strftime("%b%d_%H-%M-%S")}'
        omegaconf.OmegaConf.save(config=self.cfg, f=f'data/tmp/cfg_{self.cfg.tag}.yaml')

        print('-- creating Workspace for scenario', self.cfg.scenario)
        self.cfg.files=self.cfg.files[self.cfg.scenario]
        # print(self.cfg.files)

        self.C = ry.Config()
        self.C.addFile(self.cfg.files.scene_yaml)
        self.F = self.C.getFrames()
        self.q_dim = self.C.getJointDimension()
        self.q_home = self.C.getJointState()
        # self.Chelper = self.C.addFrame('helper')
        base = self.C.getFrame('base', warnIfNotExist=False)
        if base is None:
            base = self.C.getFrame('l_base', warnIfNotExist=False)
        if base is None:
            base = self.C.getFrame('panda_base')
        # self.Chelper.setParent(base)
        print('-- loaded config q-dim:', self.C.getJointDimension(), 'with joints', self.C.getJointNames())

    def load_configs(self, verbose=1):
        self.configs = ConfigsFile(self.cfg.files.configs)
        self.supportObjs = {}
        for f_name in self.configs.supports:
            self.supportObjs[f_name] = self.C.getFrame(f_name).ID
        if self.configs.q.shape[1]+4==self.q_dim: #quaternions missing
            self.configs.q = np.hstack((self.configs.q, np.tile(np.array([1.,0.,0.,0.]), (self.configs.q.shape[0],1))))
        assert self.configs.q.shape[1]==self.q_dim
        if verbose>0:
            print('-- loaded configs:', self.configs.q.shape[0])

    def load_splines(self, threshold=1e-3, fixed_goal=None, verbose=1):
        self.traj = SplinesFile(self.cfg.files.splines)
        if verbose>0:
            print('-- loaded trajectories:', self.traj.splines.shape[0])
        self.traj.select_sucessful(threshold)
        if verbose>0:
            print('-- selected trajectories:', self.traj.splines.shape[0])

    def load_episodes(self, verbose=1):
        self.episodes = EpisodesFile(self.cfg.files.episodes)
        self.episodes.tau_step = self.episodes.manifest['tau_step']
        if self.episodes.start.shape[1]+4==self.q_dim: #quaternions missing
            self.episodes.start = np.hstack((self.episodes.start, np.tile(np.array([1.,0.,0.,0.]), (self.episodes.start.shape[0],1))))
            self.episodes.goal = np.hstack((self.episodes.goal, np.tile(np.array([1.,0.,0.,0.]), (self.episodes.goal.shape[0],1))))
        assert self.episodes.manifest['num_episodes']==self.episodes.start.shape[0]
        assert self.episodes.tau_step==self.cfg.Gym.tau_step
        assert self.episodes.start.shape[1]==self.q_dim
        assert self.episodes.goal.shape[1]==self.q_dim
        if verbose>0:
            print('-- loaded episodes:', self.episodes.start.shape[0])

    def load_TrajectoryData(self, build_observations=True, verbose=1):
        self.load_episodes()
        S = self.episodes.state.shape

        self.traj_data = TrajectoryData()
        self.traj_data.state = self.episodes.state
        self.traj_data.action = self.episodes.action
        self.traj_data.goal = np.empty((S[0], S[1], S[2]))
        self.traj_data.time_to_go = np.empty((S[0], S[1], 1))

        for i in range(S[0]):
            if (i%100)==0:
                print('\r   processing: ', i, '/', S[0], end='')
            Ti = S[1]
            goal = self.traj_data.state[i,-1]
            self.traj_data.goal[i,:] = goal
            for t in range(Ti):
                self.traj_data.time_to_go[i,t,0] = (Ti-1-t)*self.episodes.tau_step
        
        if build_observations:
            # save the current gym goal
            if hasattr(self.gym, 'goal_feat'):
                org_goal_feat = self.gym.goal_feat.copy()
            else:
                org_goal_feat = None
                # just to have some goal set:
                x = self.gym.sim.to_state(self.traj_data.state[0,0])
                _, self.gym.goal_feat = self.gym.observation_fct(x, without_goal=True)

            # resize the observation array
            x = self.gym.sim.to_state(self.traj_data.state[0,0])
            obs, _ = self.gym.observation_fct(x)
            self.traj_data.observation = np.empty((S[0], S[1], obs.size))

            for i in range(S[0]):
                if (i%100)==0:
                    print('\r   processing: ', i, '/', S[0], end='')
                Ti = S[1]
                # set gym goal to traj goal
                goal = self.traj_data.state[i,-1]
                x = self.gym.sim.to_state(goal)
                _, self.gym.goal_feat = self.gym.observation_fct(x, without_goal=True)
                for t in range(Ti):
                    # get the state observation (which may be relative to the goal)
                    state = self.traj_data.state[i,t]
                    x = self.gym.sim.to_state(state)
                    obs, _ = self.gym.observation_fct(x)
                    self.traj_data.observation[i,t] = obs

            # restore the current gym goal
            self.gym.goal_feat = org_goal_feat

        print(f'\n-- loaded trajectories: {self.traj_data.state.shape[0]}')

    def create_NLP(self, goal_state, num_scenes=1):
        goal_feat_map = lambda qpos, qvel: self.get_config_embedding(qpos, qvel, self.cfg.NLP.metric)

        nlp = ManipulationNLP(self.sim, goal_state=goal_state,
                                num_scenes=num_scenes, total_time=self.cfg.NLP.total_time, ctrl_pts=self.cfg.NLP.ctrl_pts, goal_feat_map=goal_feat_map)

        return nlp

    def get_xml(self, C):
        M = mujoco_io.MujocoWriter(C, friction=self.cfg.Gym.friction)
        xml = M.str().decode('ascii')
        with open('z.xml', 'w') as fil:
            fil.write(xml)
        return xml

    def create_Sim(self, engine='mujoco'):
        if engine=='mujoco':
            xml = self.get_xml(self.C)
            self.sim = sim.MujocoSim(xml, self.C, tau_sim=self.tau_sim, warp_worlds=self.cfg.Gym.warp_worlds, use_mj_viewer=False)
        elif engine=='physx':
            self.sim = ry.Simulation(self.C, engine=ry.SimulationEngine.physx, verbose=0)
        else:
            raise Exception(f'engine "{engine}" not defined')
        return self.sim

    def create_Gym(self, num_scenes=1):
        _sim = self.create_Sim()

        # goal_feat_map = lambda obs: self.get_config_embedding(obs, None, self.cfg.RL_goal_feat_map)
        goal_feat_map = lambda qpos, qvel: qpos[:, -7:-4].copy() #W.get_config_embedding(qpos, qvel, W.cfg.RL_goal_feat_map)

        self.gym = sim.MujocoGym(_sim, cfg=self.cfg.Gym, goal_feat_map=goal_feat_map, num_scenes=num_scenes, terminal_bounds=_sim.C.getJointLimits())
        self.gyms = [self.gym]
        return self.gym

    def create_GoalGym(self):
        self.goal_gym = MujocoGoalGym(self.gym)

    def create_VectorGym(self, num_envs):
        make_env = lambda: self.create_Gym()
        # [make_env] * num_envs
        self.gym = VectorGym([make_env for _ in range(num_envs)]) # gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
        self.gyms = self.gym.envs

    def get_support_freq(self, configs):
        frames = self.C.getFrames()
        freq = {}
        for i in configs:
            s = self.configs.contacts[i].reshape(-1,2)[:,1]
            s = [frames[id].name for id in s]
            s = '-'.join(s)
            # c1 = tuple(c1)
            if s in freq:
                freq[s] += 1
            else:
                freq[s] = 1
        return freq

    def get_random_task(self, goal_not_on_table=True, balance_goals_only=False, goal_not_balance=False, fixed_goal=None):
        if not hasattr(self, 'configs'):
            self.load_configs()

        if goal_not_on_table:
            table_id = self.supportObjs['table']
        while(True):
            task = np.random.randint(self.configs.q.shape[0]-1, size=2)
            if fixed_goal is not None:
                task[1] = fixed_goal
            if task[0]>=task[1]:
                task[0] += 1
            if goal_not_on_table:
                if table_id in self.configs.contacts[task[1]]:
                    continue
            if balance_goals_only:
                if not np.array_equal(self.configs.contacts[task[1]], np.array([9, 4], dtype='int32')):
                    continue
            if goal_not_balance:
                if np.array_equal(self.configs.contacts[task[1]], np.array([9, 4], dtype='int32')):
                    continue
            break
        return task.tolist()

    def get_interp_task(self, alpha, goal_not_on_table=True, balance_goals_only=False, goal_not_balance=False, fixed_goal=None):
        if not hasattr(self, 'configs'):
            self.load_configs()

        if not hasattr(self.configs, 'tree'):
            self.create_config_KDTree()
        
        tree: KDTree = self.configs.tree

        task = self.get_random_task(goal_not_on_table, balance_goals_only, goal_not_balance, fixed_goal)

        if alpha<=1.:
            t_interp = np.random.uniform(1.-alpha, 1.)
        else:
            t_interp = np.random.uniform(1.-alpha, 2.-alpha)

        if t_interp<=0.:
            return task #same as t_interp=0
        
        z0 = self.configs.z[task[0]]
        z1 = self.configs.z[task[1]]
        z = (1.-t_interp) * z0 + t_interp * z1
        d, i = tree.query(z, k=2)
        # print(f'eps task: org: {task} org-dist: {np.linalg.norm(z0-z1)} alpha: {alpha}, near: {i} distance: {d}')
        # j = np.random.choice(i)
        # while(j == task[1]):
        #     j = np.random.choice(i)
        j = i[0]
        if j == task[1]:
            j = i[1]
        task[0] = int(j)
        return task

    def getStartsGoals_fromRandomTasks(self, n=10_000, alpha=None):
        starts_q, goals_q = [],[]

        for i in range(n):
            if alpha is None:
                task = self.get_random_task(goal_not_on_table=self.cfg.goal_not_on_table, goal_not_balance=self.cfg.goal_not_balance, balance_goals_only=self.cfg.balance_goals_only)
            else:
                task = self.get_interp_task(alpha, goal_not_on_table=self.cfg.goal_not_on_table, goal_not_balance=self.cfg.goal_not_balance, balance_goals_only=self.cfg.balance_goals_only)
            starts_q.append(self.configs.q[task[0]])
            goals_q.append(self.configs.q[task[1]])
                           

            if (i%100)==0:
                print('\r   processing: ', i, '/', n, end='')

        print(f'\n-- created rand starts+goals: {len(starts_q)} (alpha: {alpha})')

        return np.stack(starts_q), np.stack(goals_q)

    def getStartsGoals_fromTrajData(self, alpha):
        if not hasattr(self, 'traj_data'):
            self.load_TrajectoryData(build_observations=False)
            
        S = self.traj_data.state.shape
        T = S[1]
        if alpha<=1.:
            T0 = int((1.-alpha)*T)
        else:
            T0 = 0
        starts_q, starts_v, goals_q, goals_v = [],[],[],[]

        for i in range(S[0]):
            for t in range(T0, S[1]):
                x = self.sim.to_state(self.traj_data.goal[i,t])
                goals_q.append(x.qpos.copy())
                goals_v.append(x.qvel.copy())
                x = self.sim.to_state(self.traj_data.state[i,t])
                starts_q.append(x.qpos.copy())
                starts_v.append(x.qvel.copy())

        # goals = self.traj_data.goal[:,T0:,:]
        # starts = self.traj_data.state[:,T0:,:]

        # # WATCH - static (i.e. zero velocity) resets:
        # S = starts.shape
        # for i in range(S[0]):
        #     if (i%100)==0:
        #         print('\r   processing: ', i, '/', S[0], end='')
        #     for t in range(S[1]):
        #         s = self.sim.to_state(starts[i,t])
        #         s.time *= 0.
        #         # s.qvel *= 0.
        #         s.act *= 0.
        #         starts[i,t] = s.as_vector()

        # starts = starts.reshape(-1,S[2])
        # goals = goals.reshape(-1,S[2])

        print(f'\n-- created traj starts+goals: {len(starts_q)} (alpha: {alpha})')

        return np.stack(starts_q), np.stack(goals_q) #starts, goals

    def getStartsGoals(self, n, alpha, mode):
        assert alpha>0.
        if mode=='direct':
            return self.getStartsGoals_fromRandomTasks(n, alpha=None)
        elif mode=='interp':
            return self.getStartsGoals_fromRandomTasks(n, alpha=alpha)
        elif mode=='traj':
            return self.getStartsGoals_fromTrajData(alpha=alpha)
        elif mode=='trajInterp':
            S1,G1 = self.getStartsGoals_fromRandomTasks(n, alpha=alpha)
            S2,G2 = self.getStartsGoals_fromTrajData(alpha=alpha)
            return np.vstack((S1,S2)), np.vstack((G1,G2))
        else:
            raise Exception(f'mode {mode} not defines')

    def set_sim_state_to_config_id(self, i):
        return self.set_sim_state_to_config(self.configs.q[i])

    def set_sim_state_to_config(self, q):
        q[-4] = 1.
        q[-3:] = .0
        qdim = self.sim.qpos_dim
        if qdim>q.size and (qdim % q.size==0):
            qq = np.tile(q, (qdim//q.size, 1)).reshape(-1)
            if self.sim.qpos_offset is not None:
                qq += self.sim.qpos_offset
            self.sim.set_state(qq)
        else:
            self.sim.set_state(q)
        x = self.sim.getState()
        return x

    def rollout_spline(self, ctrl_pts, total_time, tau_step):
        '''equivalent to ManipulationNLP.evaluate - but chopped in action intervals and returning action sequence'''
        q0 = self.sim.data.qpos[self.sim.ctrl_indices]
        qDim = q0.size

        splineX = ctrl_pts.reshape(-1, qDim).copy()
        splineX += q0 # make control points relative to q0!!!
        splineT = np.linspace(0, total_time, splineX.shape[0]+1)
        splineT = splineT[1:]
        
        self.sim.resetSplineRef(ctrl_time=0.)
        self.sim.updateSplineRef(splineX, splineT, append=False)

        actions = []

        while self.sim.ctrl_time<total_time:
            # current_ref = self.sim.ctrlRef_spline.eval3(self.sim.ctrl_time)
            current_ref1 = self.sim.ctrlRef_spline.eval3(self.sim.ctrl_time+.001)
            current_vel = self.sim.data.qvel[self.sim.ctrl_indices]
            #self.a  = (delta-tau*v0)/(tau*tau) #at time t=tau, f(t) = x0 + .5*tau*v0 + .5*delta
            tau = 2.*tau_step
            action = (tau*tau)*current_ref1[2]+tau*current_vel
            actions.append(action)

            self.sim.step(tau_step=tau_step)

        return np.stack(actions)

    def eval_action_sequence(self, task, actions, tau_step, polyActions, verbose=2):
        x0 = self.set_sim_state_to_config_id(task[0])
        if verbose>0:
            self.sim.C.view(verbose>1, 'start')
            self.sim.view_speed=1.
        else:
            self.sim.view_speed=-1.

        if polyActions:
            self.sim.resetPolyRef(ctrl_time=0.)
        else:
            self.sim.resetSplineRef(ctrl_time=0.)
        T = actions.shape[0]
        assert actions.ndim==2

        for t in range(T):
            if polyActions:
                current_ref = self.sim.ctrlRef_poly.eval(self.sim.ctrl_time)
                current_vel = self.sim.ctrlRef_poly.eval_vel(self.sim.ctrl_time) #self.qvel[:, self.ctrl_indices]
                self.sim.ctrlRef_poly = sim.SecondOrderCtrlRef(self.sim.ctrl_time, current_ref, current_vel, actions[t], 2.*tau_step)
            else:
                ctrl_ref = actions[t].reshape(1, self.sim.ctrl_dim).copy() #assuming absolute actions?
                # ctrl_ref += self.qpos[:, self.ctrl_indices]
            # ctrl_ref += self.sim.spline_ref.eval3(self.sim.ctrl_time)[0] # NEW! relative
                self.sim.updateSplineRef(ctrl_ref, np.array([2.*tau_step]), append=False)

            self.sim.step(tau_step=tau_step)

        xT = self.sim.getState()
        zT = self.get_config_embedding(xT.qpos, xT.qvel, self.cfg.NLP.metric)
        x1 = self.set_sim_state_to_config_id(task[1])
        z1 = self.get_config_embedding(x1.qpos, x1.qvel, self.cfg.NLP.metric)
        phi = z1-zT

        if verbose>0:
            print('phi:', phi)
            print('phi^2:', np.sum(np.square(phi)))
            self.sim.C.view(verbose>1, 'end manip')
            self.sim.setState(x1)
            self.sim.C.view(verbose>1, 'goal')

        return phi

    def eval_NLP_spline(self, task, spline, verbose=2):
        x1 = self.set_sim_state_to_config_id(task[1])
        x0 = self.set_sim_state_to_config_id(task[0])
        if verbose>0:
            self.sim.C.view(verbose>1, 'start')

        nlp = self.create_NLP(goal_state=x1)

        if verbose>0:
            self.sim.view_speed=1.

        phi, _ = nlp.evaluate(spline)
        key = 0

        if verbose>0:
            print('phi:', phi)
            print('phi^2:', np.sum(np.square(phi)))
            self.sim.C.view(verbose>1, 'end manip')
            self.sim.setState(x1)
            key = self.sim.C.view(verbose>1, 'goal')

        return phi, key

    def get_config_embedding(self, qpos, qvel, cfg): #scale_obj=2., scale_q=1., scale_vel=0., scale_d=0., d_sigma=.05): #c, scale_c=.0, 
        if qpos.size>self.q_dim and (qpos.size % self.q_dim==0): #batch evaluation
            feat = []
            n = qpos.size // self.q_dim
            _qpos = qpos.reshape(n, -1)
            _qvel = qvel.reshape(n, -1)
            for i in range(n):
                feat.append(self.get_config_embedding(_qpos[i], _qvel[i], cfg))
            return np.stack(feat)

        assert qpos.size==self.q_dim

        # object position
        z = cfg.scale_obj * qpos[-7:-4]

        # robot angles (=finger position)
        if cfg.scale_q>0.:
            z_q = qpos[:-3]
            z = np.concat((z, cfg.scale_q*z_q))

        # object velocity
        if cfg.scale_vel>0. and qvel is not None:
            qvel = qvel[:self.q_dim]
            z_vel = qvel[-3:]
            # if scale_q>0.:
            #     z_vel = np.concat((z_vel, scale_q * qvel[:-3]))
            z = np.concat((z, cfg.scale_vel*z_vel))

        # concat mode binary indicator vector
        # z_c = np.any(c[:,None] == self.I, axis=0).astype(np.float64)

        if cfg.scale_d>0.:
            # distance 'sigmoid'
            self.C.setJointState(qpos)
            D = np.array([self.C.eval(ry.FS.negDistance, [f_name, 'obj'])[0][0] for f_name in self.configs.supports])
            z_d = np.clip(1.+D/cfg.d_sigma,0., 1.) #negative distance clipped to [-1,0]
            # z_d = np.exp(D/d_sigma) # e^{-d/sig} = contact indicator in [0,1]
            z = np.concatenate((z, cfg.scale_d*z_d))

        return z

    def retired__get_other_config_embedding(self, q, d_sigma=.05, scale_obj=3., scale_q=.5, scale_d=1.): #c, scale_c=.0, 
        C = self.C
        C.setJointState(q)
        z_obj = C.getFrame('obj').getPosition()
        z_finger = C.getFrame('fing').getPosition()

        z_dists = []
        for f_name in self.configs.supports:
            y_dist, _ = C.eval(ry.FS.pairCollision_negScalar, [f_name, 'obj'])
            y_norm, _ = C.eval(ry.FS.pairCollision_normal, [f_name, 'obj'])
            z_dists.append(np.concatenate((y_dist, y_norm)))

        z_dists = np.stack(z_dists)
        z = np.concat((z_obj, z_finger, z_dists.reshape(-1)))
        return z

    def embed_i(self, i):
        q = self.configs.q[i]
        # c = self.configs.contacts[i].reshape(-1,2)[:,1]
        z = self.get_config_embedding(q, np.zeros(q.size), self.cfg.NLP.metric)
        return z

    def embed_dist(self, task):
        return np.linalg.norm(self.embed_i(task[0]) - self.embed_i(task[1]), ord=2)

    def embed_task(self, task):
        return np.concatenate((self.embed_i(task[0]), self.embed_i(task[1])))

    def display_task(self, task, pause=True):
        # self.Chelper.setLines([], colors=[0,0,0])
        self.C.setJointState(self.configs.q[task[0]])
        self.C.view(pause, f'task {task}\nSTART')
        sleep(.05)
        self.C.setJointState(self.configs.q[task[1]])
        key = self.C.view(pause, f'task {task}\nGOAL')
        sleep(.05)
        return key

    def display_config_and_spline(self, q, x, msg):
        x0 = q[:3].reshape(1,3) #start position of finger: is missing in the ctrl spline (smooth overwrite would do this internally)
        x = np.vstack((x0, x.reshape(-1,3)))
        S = ry.BSpline()
        S.set(2, x, np.linspace(0.,1.,x.shape[0]))

        line = S.eval(np.linspace(0., 1., 50))
        # self.Chelper.setLines(line, colors=[0,0,0])

        self.C.setJointState(q)
        self.C.view(True, msg)

    def display_task_and_spline(self, task, x):
        self.display_config_and_spline(self.configs.q[task[0]], x, f'task {task}\nSTART')
        self.C.setJointState(self.configs.q[task[1]])
        self.C.view(True, f'task {task}\nGOAL')

    def create_config_KDTree(self):
        n = self.configs.q.shape[0]
        z = self.get_config_embedding(self.configs.q[0], None, self.cfg.metric)

        Z = np.empty((n, z.size))
        for i in range(n):
            Z[i] = self.get_config_embedding(self.configs.q[i], None, self.cfg.metric)
            if (i%1000)==0:
                print('\r   computing features: ', i, '/', n, end='')

        self.configs.z = Z
        self.configs.tree = KDTree(data=Z)
        print('\n   KD tree built')

        return self.configs.tree

    def build_task_KDTree(self):
        Z = []
        for task in self.traj.tasks:
            Z.append(self.embed_task(task))

        X = np.stack(Z)
        self.tree = KDTree(data=X)

    def get_nearest_solution(self, task, translate_to_start=True):
        z = self.embed_task(task)
        d, i_near = self.tree.query(z)
        if translate_to_start:
            task_near = self.traj.tasks[i_near]
            q0_near = self.configs.q[task_near[0]]
            q0 = self.configs.q[task[0]]
            delta = (q0-q0_near)[:3]
            x = self.traj.splines[i_near] + delta
        else:
            x = self.traj.splines[i_near]

        return x, i_near

    def load_spline_predictor(self, file='data/tmp/model.torch'):
        task = self.get_random_task()
        z0 = self.get_other_config_embedding(self.configs.q[task[0]])
        z1 = self.get_other_config_embedding(self.configs.q[task[1]])
        z = np.concatenate((z0, z1))

        self.pi_spline = FullyConnectedModel([z.size, 256, 32, 12])
        trainer = Trainer(self.pi_spline, learning_rate=3e-4, weight_decay=1e-3, huberLoss_beta=None)
        trainer.load(file)

    def get_spline_prediction(self, task):
        z0 = self.get_other_config_embedding(self.configs.q[task[0]])
        z1 = self.get_other_config_embedding(self.configs.q[task[1]])
        z = np.concatenate((z0, z1))

        with torch.no_grad():
            x = self.pi_spline.predict(z)

        return x
