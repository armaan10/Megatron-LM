import os
import subprocess
import time 
import signal
import argparse
import shlex

#train_script 
#recovery 
#monitroing 

def parse_train_args (cmd):
    tokens = shlex.split(cmd)
    train_file = tokens[0]
    args = tokens[1:]
    parsed = {}
    i = 0 
    while i<len(args):
        if args[i].startswith('--'):
            key = args[i]
            if i+1 < len(args) and not args[i+1].startswith('--'):
                parsed[key] = args[i+1]
                i+=2
            else:
                parsed[key] = None 
                i+=1
        else:
            i+=1
    return train_file, parsed
def add_main_args():
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',allow_abbrev=False)
    group  = parser.add_argument_group("Inputs")
    group.add_argument('--train-cmd', type = str,default=None,  required=True, help='Commmand to run training along with its arguements')
    group.add_argument('--recovery-path', default=None, required=True, help='Path to recovery script')
    group.add_argument('--monitor-path', default=None, required=True, help='Path to GPU monitoring script')
    group.add_argument('--use-avail-gpus-on-reboot',action='store_true' ,default=None, help='On failure restart training using available gpus. If None program will wait till all gpus are online again')
    return parser
def start_proc(cmd):
    return subprocess.Popen(cmd, shell= True, preexec_fn=os.setsid)
def is_proc_alive(proc):
    return proc.poll() is None
def main():
    parser = add_main_args()
    args = parser.parse_args()
    train_file, train_args = parse_train_args(args.train_cmd)
    log_path = "./logs/train_run.log"

    #send stdout logs to file or sumn
    train_proc = start_proc(f"python {args.train_cmd} 2>&1 | tee {log_path}")
    monitor_proc = start_proc(f"python {args.monitor_path}")

    try: 
        while True:
            print("Launched")
            if not is_proc_alive(train_proc):
                train_started = False
                with open(log_path, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        if "training ..." in line:
                            train_started = True
                if train_started:
                    #do recovery
                    print("Booting up recovery")
                    #reset pp and tp and build train cmd
                    
                    train_proc = start_proc()
                    pass
                else:
                    print("Training Processes died before training started. Likely not a GPU issue. Check logs for details.")
                           
    #need exception different path for non gpu errors
    except KeyboardInterrupt:
        print("Shutting down")
        os.killpg(os.getpgid(train_proc.pid), signal.SIGTERM)
        os.killpg(os.getpgid(monitor_proc.pid), signal.SIGTERM)

if __name__ == "__main__":
    main()
