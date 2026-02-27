import argparse
import distutils.util
import numpy as np

parser = argparse.ArgumentParser()
arg = parser.add_argument

arg('--epochs', type=int, default=100)
#arg('--n_gpus', type=int, default=2)
arg('--batch_size', type=int, default=64)
arg('--filters', type=int, default=32)
#arg('--lr', type=float, default=0.001)
#arg('--shufs', type=distutils.util.strtobool, default='true')
arg('--results_path', type=str, default='Results/')
arg('--alpha', type=float, default=8.0,  help='L_HD weight for CustomLoss (train.py)')
arg('--beta',  type=float, default=5.0,  help='L_IN weight for CustomLoss (train.py)')
arg('--gamma', type=float, default=10.0, help='L_torus weight for CustomLoss (train.py)')
arg('--delta', type=float, default=4.0,  help='L_atm weight for CustomLoss (train.py)')
arg('--weights_path', type=str, default='dl_fluids.h5', help='Path to saved model weights (.h5)')
arg('--data_path',    type=str, default='x.npy',        help='Path to input data file (.npy)')
#arg('--reload_data', type=distutils.util.strtobool, default='false')
#arg('--use_bn', type=distutils.util.strtobool, default='False')
#arg('--high_density', type=float, default='1')
#arg('--accretion_disk', type=float, default='1')
#arg('--torus', type=float, default='1')
#arg('--diffusion', type=float, default='1')


args = parser.parse_args()

def write_results(folder_name, total_time):
    results_values = open(folder_name + "result_values.txt", "w")
    results_values.write(str(args.epochs) + "\n")
    #results_values.write(str(args.n_gpus) + "\n")
    results_values.write(str(args.batch_size) + "\n")
    #results_values.write(str(args.reload_data) + "\n")
    #results_values.write(str(args.use_bn) + "\n")
    #results_values.write(str(args.shufs) + "\n")
    #results_values.write(str(scores) + "\n")
    results_values.write(str(total_time) + "\n")
