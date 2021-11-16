import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#contains plotting functions that we use often

font = {'family': 'serif',
    'color':  'black',
    'weight': 'bold',
    'size': 35,
    'horizontalalignment' : 'right',
    'verticalalignment' : 'top'
    }

#Basic Mixed States Processing functions
####

def three_states_to_simplex(all_states):
    '''
    takes in an array of mixed states (from 3 state machine) and returns two ordered lists of x and y coordinates for the 2-simplex
    '''
    x = []
    y = []
    # run through keys and extract MSs
    for ms in range(len(all_states)):
        state = all_states[ms]
        a = state[0]; b = state[1]; c = state[2]
        # WLOG take two coordinates
        x.append(b+0.5*c)
        y.append(c*np.sqrt(3)/2.0)
    return x, y

def two_states_to_simplex(all_states):
    '''
    takes in an array of mixed states (from 2 state machine) and returns two ordered lists of x and y coordinates for the 1-simplex
    '''
    x = []
    for ms in xrange(len(all_states)):
        x.append(all_states[ms][0])
    return x

#Basic mixed states plotting functions
#####

def three_state_msp_scatter(x, y,filename='none'):
    '''
    takes in list of ordered x and y values and returns a scatter plot with mixed states in the 2-simplex
    '''
    plt.figure(figsize=(10,10))
    plt.clf()
    SimplexVertices = np.array([ [0,0],[1,0],[0.5,np.sqrt(3)/2.0],[0,0] ])
    plt.plot(SimplexVertices[:,0],SimplexVertices[:,1],color='k')
    plt.scatter(x, y, marker='.',alpha=0.1, s=2)

    plt.xticks([]); plt.yticks([])
    plt.xlim([0,1]); plt.ylim([0,1])
    plt.axis('off')
    plt.axis('equal')

    plt.text(0,-0.025,r'$(1,0,0)$',fontdict=font,ha='center',va='top')
    plt.text(1,-0.025,r'$(0,1,0)$',fontdict=font,ha='center',va='top')
    plt.text(0.5,np.sqrt(3)/2.0,r'$(0,0,1)$',fontdict=font,ha='center',va='bottom')
    if filename != 'none':
         #plt.savefig(filename+'.pdf',format='pdf',bbox_inches='tight')
        plt.savefig(filename+'.png',format='png',bbox_inches='tight')
    plt.show()
    return


#TODO: Add 2D plotting functionality: msp points in simplex, histogram
#TODO: Add 3D animation code