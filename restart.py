
import os, sys
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-source', type=str)
    args = parser.parse_args()

    with open('./results/{}/cmd.txt'.format(args.source)) as cmd_file:
        cmd = cmd_file.read()
    cmd_split = cmd.split(' -')
    for i in range(len(cmd_split)):
        arg = cmd_split[i]
        if 'pretrain' in arg:
            cmd_split[i] = 'pretrain {}'.format(args.source)
        elif 'launch_tmux' in arg:
            cmd_split[i] = 'launch_tmux yes'
    cmd = ' -'.join(cmd_split)

    os.system(cmd)
