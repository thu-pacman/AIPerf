# Copyright (c) Peng Cheng Laboratory
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import re
import json
import time
import datetime
import math
import profiler
import numpy as np

def find_all_trials(nnidir, expid, trial_id_list):
    experiment_data = {}

    for trial_id in trial_id_list:
        hp_list=[]
        filepath = os.path.join(nnidir, expid)
        filepath = os.path.join(filepath, 'trials')
        filepath = os.path.join(filepath, trial_id)
        filepath = os.path.join(filepath, 'trial.log')
        if not os.path.exists(filepath):
            continue
        f = open(filepath)
        f_list = f.readlines()
        f.close()
        count = 0
        for index in range(len(f_list)):
            if "Epoch" in f_list[index]:
                count = index
                continue
            elif ' val_acc' in f_list[index]:
                if "".join(re.findall('Epoch (.*?)/',f_list[count],re.S)) == '1':
                    hp_list.append([])
                hp_list[-1].extend([[int("".join(re.findall('Epoch (.*?)/',f_list[count],re.S))), conversion_time("".join(re.findall('\[(.*?)\]',f_list[index],re.S))),float("".join(re.findall(' val_acc.*?: (.*?)(?:\s|$)',f_list[index],re.S)))]])
                continue
            elif not f_list[index].strip():
                continue
        if hp_list:
            print(trial_id, hp_list)
            experiment_data[trial_id] = hp_list
    return experiment_data

def find_max_acc(stop_time, experiment_data):
    #print(trial)
    max_acc = 0
    lastest_time = 0
    result_dict = {}
    for trial in experiment_data:
        hy_count=0
        tmp_dict = {}
        for hp_data in experiment_data[trial]:
            hp_latest_time = 0
            hy_max_epoch = 0
            if hp_data:
                for epoch_data in hp_data:
                    if epoch_data[1] < stop_time:
                        #????????????????????????????????????????????????
                        if epoch_data[2] > max_acc:
                            max_acc = epoch_data[2]
                        #?????????????????????????????????????????????????????????
                        if epoch_data[1] > lastest_time:
                            lastest_time = epoch_data[1]
                        #???????????????????????????????????????epoch
                        if epoch_data[1] > hp_latest_time:
                            hp_latest_time = epoch_data[1]
                            if epoch_data[0] == 0:
                                continue
                            hy_max_epoch = epoch_data[0]
                if not hy_max_epoch == 0:
                    tmp_dict[hy_count] = hy_max_epoch
            hy_count+=1
        if tmp_dict:
            result_dict[trial] = tmp_dict
    return lastest_time, max_acc, result_dict


def conversion_time(string):
    '''
    ????????????,?????????trial??????trial.log???????????????????????????????????????????????????AM???PM????????????
    string???????????????: "06/02/2020, 09:12:50 PM"
    '''
    if string.split()[1].split(':')[0] =='12':
        #trial.log??????????????????12??????????????????????????????????????????????????????????????????????????????????????????????????????
        if string.split()[2] == 'AM':
            date = string.split()[0]
            chj_time = ':'.join(['00',string.split()[1].split(':')[1], string.split()[1].split(':')[2]])
            time_system = string.split()[2]
            string = " ".join([date , chj_time , time_system])
        elif string.split()[2] == 'PM':
            string = string.split()[0] + ' ' + string.split()[1] + ' AM'

    #???????????????string???????????????????????????????????????
    mytime = string.split()[0][:-1]+' '+string.split()[1]
    mytime = time.mktime(time.strptime(mytime,"%m/%d/%Y %H:%M:%S"))

    #???????????????PM???????????????12??????, ?????????????????????
    if string.split()[2] == 'PM':
        datetime_struct = datetime.datetime.fromtimestamp(mytime)
        datetime_obj = (datetime_struct + datetime.timedelta( hours=12 ))
        mytime = datetime_obj.timestamp()     
    return mytime

def find_startime(trial_id_list, t, experiment_path):
    start_time_list = []
    for i in range(len(trial_id_list)):
        trial_id = trial_id_list[i]
  
        # ???????????????trial 0??????????????????????????????
        if os.path.isfile(experiment_path + "/hyperparameter_epoch/" + trial_id + '/0.json'):
            with open(experiment_path + "/hyperparameter_epoch/" + trial_id + '/0.json') as hyperparameter_json:
                hyperparameter = json.load(hyperparameter_json)
                start_time = time.mktime(time.strptime(hyperparameter['start_date'], "%m/%d/%Y, %H:%M:%S"))
                start_time_list.append(start_time)
    start_time = min(start_time_list)
    # ????????????????????????????????????????????????????????????????????????????????????
    datetime_struct = datetime.datetime.fromtimestamp(start_time)
    datetime_obj = (datetime_struct + datetime.timedelta(hours=t))
    stop_time = datetime_obj.timestamp()
    return start_time,stop_time

def process_log(trial_id_list, experiment_data, dur, experiment_path):
    results = {}
    results['real_time'] = ['0.00']
    results['GFLOPS'] = ['0.0']
    results['Error'] = ['100.00']
    results['Score'] = ['0.0']
    results['real_time'] = []
    results['GFLOPS'] = []
    results['Error'] = []
    results['Score'] = []
    flops_info = profiler.profiler(experiment_path)
    for index in np.arange(0.2, dur+0.1, 0.1):
        start_time,stop_time = find_startime(trial_id_list, index, experiment_path)
        # ???????????????????????????
        lastest_time, max_acc, result_dict = find_max_acc(stop_time, experiment_data)
        # print(result_dict)
        # ????????????
        run_sec = lastest_time - start_time
        # print(datetime.datetime.fromtimestamp(start_time),'\t',datetime.datetime.fromtimestamp(stop_time),'\t',stop_time-start_time)

        total_FLOPs=0
        faild_trial=[]
        for i in range(len(flops_info['trialid'])):
            trial_id = flops_info['trialid'][i]
            if trial_id in result_dict:
                hp_num = int(flops_info['hpoid'][i])
                eval_ops = float(flops_info['eval_per_image'][i])
                trian_ops = float(flops_info['train_per_image'][i])
                #??????????????????????????????????????????????????????
                if hp_num in result_dict[trial_id].keys():
                    #???????????????????????????epoch
                    epoch = result_dict[trial_id][hp_num]
                    total_FLOPs += (eval_ops * 50000 + trian_ops * 1280000) * epoch
                    #print(trial_id,hp_num,epoch)
            else:
                faild_trial.append(trial_id)
        fraction = float(float(total_FLOPs) * float(abs(math.log(1-max_acc,math.e)))) / float(run_sec)
        fraction = fraction / (10**9)

        if run_sec < 0:
            continue

        results['real_time'].append('{:.2f}'.format(run_sec / 3600.))
        results['GFLOPS'].append('{:.1f}'.format(float(total_FLOPs) / float(run_sec) / (10**9)))
        results['Error'].append('{:.2f}'.format(100 - max_acc * 100))
        results['Score'].append('{:.1f}'.format(fraction))
    return results

def cal_report_results(expid):
    id_dict = {}
    nnidir = os.path.join(os.environ["HOME"], "nni/experiments/")
    mountdir = os.path.join(os.environ["HOME"], "mountdir/nni/experiments/")
    experiment_path = os.path.join(mountdir, expid)
    #??????sequence_id???trial_id?????????sequence_id??????????????????
    for trials in os.listdir(os.path.join(nnidir, expid,'trials')):
        with open(os.path.join(nnidir, expid,'trials',trials,'parameter.cfg'), "r") as file_read:
            json_read = json.load(file_read)
            parameter_id = json_read['parameter_id']
            id_dict[parameter_id] = trials
    #?????? sequence_id ?????????????????? id_dict = {sequence_id : trial_id}
    id_dict = sorted(zip(id_dict.keys(),id_dict.values()))
    id_dict = dict(id_dict)
    print("id dict:",id_dict)
    trial_id_list = list(id_dict.values())
 
    start_time_list = []
    for i in range(len(trial_id_list)):
        trial_id = trial_id_list[i]

        # ???????????????trial 0??????????????????????????????
        if os.path.isfile(experiment_path + "/hyperparameter_epoch/" + trial_id + '/0.json'):
            with open(experiment_path + "/hyperparameter_epoch/" + trial_id + '/0.json') as hyperparameter_json:
                hyperparameter = json.load(hyperparameter_json)
                start_time = time.mktime(time.strptime(hyperparameter['start_date'], "%m/%d/%Y, %H:%M:%S"))
                start_time_list.append(start_time)
    start_time = min(start_time_list)
    print("start time:",start_time)
    experiment_data = find_all_trials(nnidir, expid, trial_id_list)
    stop_time=0
    for index in range(len(trial_id_list)-1,-1,-1):
        if trial_id_list[index] in experiment_data:
            stop_time = max(stop_time,experiment_data[trial_id_list[index]][-1][-1][1])
            break
    print("stop time:",stop_time)
    dur = (stop_time - start_time) / 3600.
    results = process_log(trial_id_list, experiment_data, dur, experiment_path)
    return results, trial_id_list, experiment_data
