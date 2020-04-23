import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import pandas as pd

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def map_action(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def map_epsilon(epsilon):
    return map_action(epsilon, -1, 1, 0.1, 1.5)

def map_kappa(kappa):
    return map_action(kappa, -1, 1, 0.1, 10)

def map_u(u):
    return map_action(u, -1, 1, -1, 3)

def map_w(w):
    return map_action(w, -1, 1, -4, 4)

def show_paths(data):
    fig = plt.figure(figsize=plt.figaspect(1/3))
    for i in range(3):
        ax_2d = fig.add_subplot(1, 3, i+1)
        ax_2d.plot(*[data['ground_truth'][i]['path'][...,j] for j in range(2)], label="Ground truth")
        ax_2d.plot(*[data['fb_baseline'][i]['path'][...,j] for j in range(2)], label="Feedback normalization baseline")
        ax_2d.plot(*[data['complex'][i]['path'][...,j] for j in range(2)], label="End-to-end solution")
        ax_2d.plot(*[data['fb'][i]['path'][...,j] for j in range(2)], label="Feedback normalization")

        ax_2d.set_aspect("equal")
        ax_2d.grid()
        if i == 1:
            ax_2d.legend()
    fig.tight_layout()
    plt.savefig("paths.pdf")
    plt.show()

def show_complex_path_speed(data):
    fig = plt.figure(figsize=[3.5,5])#plt.figaspect(1/1))
    if True:#for i in range(2, 3):
        i = 2
        ax_2d = fig.add_subplot(1, 1, 1)
        path_cutoff = 7000
        ax_2d.plot(*[data['ground_truth'][i]['path'][...,j][:900] for j in range(2)], color='b')

        mvg_avg = 501
        speed_avg = moving_average(data['complex'][i]['actions'][...,0], mvg_avg)[:path_cutoff]
        speed_avg -= speed_avg.min()
        speed_avg /= speed_avg.max()
        rot_avg = moving_average(data['complex'][i]['actions'][...,1], mvg_avg)[:path_cutoff]
        rot_neg = rot_avg < 0
        rot_avg[rot_neg] /= rot_avg[rot_neg].min()
        rot_avg[rot_neg] = -rot_avg[rot_neg]
        rot_avg[~rot_neg] /= rot_avg[~rot_neg].max()
        ax_2d.scatter(
            *[data['complex'][i]['path'][...,j][int(mvg_avg/2):-int(mvg_avg/2)][:path_cutoff] for j in range(2)],
            s=speed_avg*20,
            c=rot_avg,
            cmap='RdYlGn'
        )
        ax_2d.set_aspect("equal")
        ax_2d.grid()
        #if i == 0:
        #ax_2d.legend()
    fig.tight_layout()
    plt.savefig("complex_path_actions.pdf")
    plt.show()

def compute_solutions_path_speed(data):
    for exp in ['fb_baseline', 'complex', 'fb']:
        lengths = np.array([len(trial['path']) for trial in data[exp]])/240
        print(exp, np.mean(lengths), np.std(lengths))

def show_fb_path_speed(data):
    fig = plt.figure(figsize=[3.5,5])#plt.figaspect(1/1))
    if True:#for i in range(2, 3):
        i = 2
        ax_2d = fig.add_subplot(1, 1, 1)
        path_cutoff = 12500
        ax_2d.plot(*[data['ground_truth'][i]['path'][...,j][:900] for j in range(2)], color='b')

        mvg_avg = 501
        speed_avg = moving_average(data['fb'][i]['actions'][...,1], mvg_avg)[:path_cutoff]
        speed_avg -= speed_avg.min()
        speed_avg /= speed_avg.max()
        kappa_avg = moving_average(data['fb'][i]['actions'][...,0], mvg_avg)[:path_cutoff]
        ax_2d.scatter(
            *[data['fb'][i]['path'][...,j][int(mvg_avg/2):-int(mvg_avg/2)][:path_cutoff] for j in range(2)],
            s=speed_avg*20,
            c=kappa_avg,
            cmap='Greens'
        )
        ax_2d.set_aspect("equal")
        ax_2d.grid()
    fig.tight_layout()
    plt.savefig("fb_path_actions.pdf")
    plt.show()

def eval_action_observation(data):
    def process(experiment_id, action_id, poly_f=lambda x: x):
        polys = []
        actions = []
        for trial in data[experiment_id]:
            poly = []
            for i in range(len(trial['observations'])):
                obs = trial['observations'][i].reshape(10, 3)[:5]
                z = np.polyfit(*[obs[...,j] for j in range(2)], 1)
                poly.append(poly_f(z[0])) # abs if speed is considered
                if False:#abs(z[0]) > 10:
                    plt.scatter(*[obs[...,j] for j in range(2)])
                    p = np.poly1d(z)
                    xp = np.linspace(-2, 6, 100)
                    plt.plot(xp, p(xp))
                    plt.xlim(0, 2)
                    plt.ylim(-1.5, 1.5)
                    plt.show()
            mvg_avg = 501
            avg_poly = moving_average(poly, mvg_avg)
            #plt.plot(avg_poly)
            avg_a = moving_average(trial['actions'][...,action_id], mvg_avg)
            polys.append(avg_poly)
            actions.append(avg_a)
        return np.concatenate(actions), np.concatenate(polys)
    
    fig = plt.figure(figsize=plt.figaspect(1/1))
    for ax_i, i in enumerate([60,100,250,600]):
        ax = fig.add_subplot(2, 2, ax_i+1)
        
        obs = data['complex'][0]['observations'][i].reshape(10, 3)
        z = np.polyfit(*[obs[...,j][:5] for j in range(2)], 1)
        ax.scatter(*[obs[...,j] for j in range(2)])
        p = np.poly1d(z)
        xp = np.linspace(-2, 6, 100)

        ax.plot(xp, p(xp), label=f"Slope m={z[0]:0.2f}")
        ax.grid()
        ax.legend()
        ax.set_xlim(0, 2)
        ax.set_ylim(-1.5, 1.5)
    plt.savefig("act_obs_slope_examples.pdf")
    plt.show()

    # complex action indexes: [u, w, h]
    actions, polys = process('complex', 0, abs) # forward speed
    plt.hist2d(actions, polys, bins=30)
    locs, labels = plt.xticks()
    plt.xticks(locs[1:-1], [f'{map_u(l):.2f}' for l in locs])
    plt.xlabel("Forward speed [m/s]")
    plt.ylabel("Path slope")
    plt.savefig("act_obs_corr_complex_u.pdf")
    plt.show()
    
    actions, polys = process('complex', 1) # rotation speed
    plt.hist2d(actions, polys, bins=35)
    locs, labels = plt.xticks()
    plt.xticks(locs[1:-1], [f'{map_w(l):.2f}' for l in locs])
    plt.xlabel("Angular yaw speed [rad/s]")
    plt.ylabel("Path slope")
    plt.savefig("act_obs_corr_complex_w.pdf")
    plt.show()
    
    # fb action indexes: [epsilon, kappa]
    
    actions, polys = process('fb', 0) # epsilon
    plt.hist2d(actions, polys, bins=35)
    locs, labels = plt.xticks()
    plt.xticks(locs[1:-1], [f'{map_epsilon(l):.2f}' for l in locs])
    plt.xlabel("Epsilon")
    plt.ylabel("Path slope")
    plt.savefig("act_obs_corr_fb_epsilon.pdf")
    plt.show()
    
    actions, polys = process('fb', 1, abs) # kappa (forward speed amplifier)
    plt.hist2d(actions, polys, bins=35)
    locs, labels = plt.xticks()
    plt.xticks(locs[1:-1], [f'{map_kappa(l):.2f}' for l in locs])
    plt.xlabel("Kappa")
    plt.ylabel("Path slope")
    plt.savefig("act_obs_corr_fb_kappa.pdf")
    plt.show()

def training_curves(path, end_index, result_filename):
    results = pd.concat([pd.read_csv(open(f, "rb"), skiprows=1) for f in glob.glob(path+"/*.monitor.csv")]).sort_values(by=['t'])
    #results = pd.read_csv(open(path+"/0.monitor.csv", "rb"), skiprows=1)
    t = results['t'][:end_index].to_numpy()
    fig, ax1 = plt.subplots(figsize=[8,4])
    color = 'tab:red'
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('Return', color=color)
    ax1.plot(t, results['r'][:end_index].to_numpy(), c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid()
    
    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.plot(t, results['l'][:end_index].to_numpy(), c=color)
    ax2.set_ylabel('Episode length', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    #ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
    
    plt.savefig(result_filename)
    plt.show()
    
    
if __name__ == '__main__':
    data = pickle.load(open("eval_results.pkl", "rb"))
    #eval_action_observation(data)
    #show_paths(data)
    show_complex_path_speed(data)
    #show_fb_path_speed(data)
    #compute_solutions_path_speed(data)
    #training_curves("./results/DronePathComplex-v0_52", 200, "./img/train_complex.pdf")
    #training_curves("./results/DronePathComplexFB-v0_2", 100, "./img/train_fb.pdf")
    #eval_action_observation(data)
    #eval_act_obs_path_slope(data)
    #eval_action_observation(data)
    #eval_action(data)
