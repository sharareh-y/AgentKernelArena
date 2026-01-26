# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import os
import yaml
import numpy as np
from Levenshtein import distance as levenshtein_dist
import json
from pathlib import Path

def normalized_levenshtein(s1, s2):
    """the edit distance for a single string, normalized to [0,1]"""
    if len(s1) == 0 and len(s2) == 0:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    
    return levenshtein_dist(s1,s2)/ max_len

def dtw_string_distance(list1, list2):
    """the dtw distance based on edit distance, normalized to [0,1]"""
    # firstly, filter the blank lines in these lists
    list1 = [s.strip() for s in list1 if s.strip()]
    list2 = [s.strip() for s in list2 if s.strip()]
    # also remove the line with initilized with '//', full docstring
    list1 = [s for s in list1 if s.strip()[:2] != '//']
    list2 = [s for s in list2 if s.strip()[:2] != '//']
    #remove all the blank space in each line
    list1 = [s.replace(' ','') for s in list1]
    list2 = [s.replace(' ','') for s in list2]

    m, n = len(list1), len(list2)
    if m == 0 and n == 0:
        return 0.0
    # construct the distance matrix
    dist_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            dist_matrix[i][j] = normalized_levenshtein(list1[i], list2[j])
    
    # dynamic programming for dtw distance
    dtw = np.full((m+1, n+1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = dist_matrix[i-1, j-1]
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    # normalization
    max_len = max(m, n)
    return dtw[m, n] / max_len if max_len != 0 else 0.0

#you should firstly move the workspace to collect to ${folder_to_collect}, notice that each kernel should only appear once in this folder
folder_to_collect = Path(__file__).parent.parent.parent / "saved_results/zeping_30B_RL_v1_260111_4des_max15iter/"
#folder_to_collect = Path(__file__).parent.parent.parent / "workspace_rl_v2_4des_15iter_MI250_geak_ourllm_kernel2kernel"
folder_to_collect = str(folder_to_collect)

lv1_list = ['bitonic_sort','convolution','floyd_warshall','ball_query','knn','three_nn','histogram','monte_carlo_pi','point_to_voxel','render_forward','silu','fused_buckeized','emb_segment_reduce_backward','emb_segment_reduce_forward','causal_conv1d_simple','rms','fused_bucketized']
lv2_list = ['mla','assign_score_withk','furthest_point_sample','gather_points','roiaware_pool3d','roipoint_pool3d','three_interpolate','causal_conv1d_channellast','points_in_boxes','prefix_sum']

banned_task = ["rms",'mla','monte_carlo_pi']


iter_idx = 15
valid_case = 0
pass_case = 0
slower_num = 0
close_flag = False
item_list = []
total_dict = {'lv1': {'total_case':0, 'valid_case':0, 'pass_case':0, 'spd_item_dict':{} }, 'lv2': {'total_case':0, 'valid_case':0, 'pass_case':0, 'spd_item_dict':{} } }

for item in os.listdir(folder_to_collect):
    if 'tmp.log' in item:
        continue
    cur_path = os.path.join(folder_to_collect, item)
    core_item = '_'.join(item.split('_')[:-2])
    if core_item not in lv1_list and core_item not in lv2_list:
        print(core_item)

    if core_item in lv1_list:
        key = 'lv1'
    else:
        key = 'lv2'
    
    total_dict[key]['total_case'] += 1
    if 'task_result.yaml' in os.listdir(cur_path) and core_item not in banned_task:
        total_dict[key]['valid_case'] += 1
    else:
        print(f'not valid: {key}-{core_item}: {item}')
        continue
    with open(os.path.join(cur_path, 'task_result.yaml'), "r") as f:
        result_data = yaml.safe_load(f)

    # firstly check whether the generated code is close to original
    if close_flag:
        cur_code = open(os.path.join(cur_path, result_data['best_optimized_source_file_path'][0])).read()
        ori_code = open(os.path.join('tasks', result_data['task_name'],  result_data['best_optimized_source_file_path'][0])).read()
        dtw_dist = dtw_string_distance(cur_code.split('\n'), ori_code.split('\n'))
        if dtw_dist < 0.05:
            print(f'too close: {key}-{core_item}, got dtw {dtw_dist}')
            continue
            
    #read perf
    if os.path.exists(os.path.join(cur_path, 'geak_hip_iter_logs', f'iter_{iter_idx}.perf')):
        perf_path = os.path.join(cur_path, 'geak_hip_iter_logs', f'iter_{iter_idx}.perf')
        tmp = json.loads(open(os.path.join(cur_path, 'geak_hip_iter_logs', f'iter_{iter_idx}')).read())
        if 'predict' not in tmp.keys():
            print(perf_path + 'ERROR!!!!')
 
    else:
        for sub_iter_idx in range(iter_idx, -1, -1):
            if os.path.exists(os.path.join(cur_path, 'geak_hip_iter_logs', f'iter_{sub_iter_idx}.perf')):
                perf_path = os.path.join(cur_path, 'geak_hip_iter_logs', f'iter_{sub_iter_idx}.perf')
                tmp = json.loads(open(os.path.join(cur_path, 'geak_hip_iter_logs', f'iter_{sub_iter_idx}')).read())
                if 'predict' not in tmp.keys():
                    print(perf_path + 'ERROR!!!!')
                break
    #print(perf_path)
    perf_record = json.loads(open(perf_path).read())
    #print(perf_path)
    try:
        if isinstance(perf_record['ori_perf'], list):
            spd_up = []
            for i in range(len(perf_record['ori_perf'])):
                spd_up.append(perf_record['ori_perf'][i]/perf_record['opt_perf'][i])
            spd_up = np.mean(np.array(spd_up))
        else:
            spd_up = perf_record['ori_perf']/perf_record['opt_perf']
    except:
        spd_up = 0.0



    if spd_up > 1.0: #result_data['speedup_ratio'] != 1.0:
        total_dict[key]['pass_case'] += 1
        cur_spd = spd_up #if spd_up > 1.0 else 1.0 #result_data['speedup_ratio'] if result_data['speedup_ratio'] > 1.0 else 1.0
        item_list.append(core_item)
        total_dict[key]['spd_item_dict'][core_item] = cur_spd
    elif spd_up == 1.0:
        #print(perf_path)
        print(f'failed: {key}-{core_item}: {item}')
    else:
        #print(perf_path)
        print(f'slower: {key}-{core_item}, got spdup {spd_up}')
        slower_num += 1
        total_dict[key]['pass_case'] += 1
        cur_spd = 1.0 #if spd_up > 1.0 else 1.0 #result_data['speedup_ratio'] if result_data['speedup_ratio'] > 1.0 else 1.0
        item_list.append(core_item)
        total_dict[key]['spd_item_dict'][core_item] = cur_spd
        
print('\n\n')
print(f"gather results for {folder_to_collect}, iter {iter_idx}")
print(f"total {total_dict['lv1']['total_case'] + total_dict['lv2']['total_case']} samples, valid {total_dict['lv1']['valid_case'] + total_dict['lv2']['valid_case'] } samples, \
      passed {total_dict['lv1']['pass_case'] + total_dict['lv2']['pass_case'] } samples \n\
      {slower_num} are slower than original implementation") #, avg spdup {np.mean(spdup_list)}, best spdup {np.max(spdup_list)}")
print('The details are:')
item_list.sort()
for key in ['lv1','lv2']:
    print('~'*20+key+'~'*20)
    #print(f"total {total_dict[key]['total_case']} samples, valid {total_dict[key]['valid_case']} samples, passed {total_dict[key]['pass_case']} samples")
    spdup_list = []
    for i in range(len(item_list)):
        if item_list[i] in total_dict[key]['spd_item_dict'].keys():
            print(f"kernel: {item_list[i]} \tspeedup: {total_dict[key]['spd_item_dict'][item_list[i]]})")
            spdup_list.append(total_dict[key]['spd_item_dict'][item_list[i]])
    if len(spdup_list) == 0:
        spdup_list = [0]
    spdup_list = np.array(spdup_list)
    
    print(f"total {total_dict[key]['total_case']} samples, valid {total_dict[key]['valid_case']} samples, passed {total_dict[key]['pass_case']} samples,  avg spdup {np.mean(spdup_list)}, best spdup {np.max(spdup_list)}")



