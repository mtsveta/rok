import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Do not show any warnings
mpl.set_loglevel("critical")
mpl.rc_file("plotting/matplotlibrc")

#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = 'TeX Gyre Adventor'
#mpl.rcParams['font.style'] = 'normal'

def plot_on_demand_learning_countings_mpl(time_steps, learnings, folder_results):

    plt.xlabel('Time Step')
    plt.ylabel('On-demand learnings')
    plt.xlim(left=0, right=len(learnings))
    plt.ylim(bottom=0, top=np.max(learnings)+1)
    plt.ticklabel_format(style='plain', axis='x')
    plt.plot(time_steps, learnings, color='C0', linewidth=1.5)
    plt.tight_layout()
    plt.savefig(folder_results + '/on-demand-learning-countings.png')
    plt.close()

def plot_computing_costs_mpl(time_steps, times, step, folder_results):

    (timings_equilibrium_conv, timings_equilibrium_smart, timings_transport) = times

    plt.xlabel('Time Step')
    plt.ylabel('Computing Cost [s]')
    plt.xlim(left=0, right=len(time_steps))
    plt.yscale('log')
    plt.ticklabel_format(style='plain', axis='x')
    plt.plot(time_steps[0:len(time_steps):step], np.array(timings_equilibrium_conv[0:len(time_steps):step]), label="Chemical Equilibrium (Conventional)", color='C0', linewidth=1.5)
    plt.plot(time_steps[0:len(time_steps):step], np.array(timings_equilibrium_smart[0:len(time_steps):step]), label="Chemical Equilibrium (Smart)", color='C1', linewidth=1.5, alpha=1.0)
    plt.plot(time_steps[0:len(time_steps):step], np.array(timings_transport[0:len(time_steps):step]), label="Transport", color='C2', linewidth=1.5, alpha=1.0)
    leg = plt.legend(loc='right', bbox_to_anchor=(0.5, 0.3, 0.5, 0.5))
    for line in leg.get_lines(): line.set_linewidth(2.0)
    plt.tight_layout()
    plt.savefig(folder_results + '/computing-costs-nolegend-with-smart-ideal.png')
    plt.close()

def plot_speedups_mpl(time_steps, times, step, folder_results):

    (timings_equilibrium_conv, timings_equilibrium_smart) = times
    speedup = np.array(timings_equilibrium_conv) / np.array(timings_equilibrium_smart)

    plt.xlim(left=0, right=len(time_steps))
    plt.xlabel('Time Step')
    plt.ylabel('Speedup [-]')
    plt.ticklabel_format(style='plain', axis='x')
    plt.plot(time_steps[0:len(time_steps):step], speedup[0:len(time_steps):step], label="Conventional vs. Smart ", color='C0', linewidth=1.5)
    leg = plt.legend(loc='lower right')
    for line in leg.get_lines(): line.set_linewidth(2.0)
    plt.tight_layout()
    plt.savefig(folder_results + '/speedups.png')
    plt.close()
