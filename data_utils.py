import numpy as np
import pickle
import re

def load_data(models, Ns, ns, temps, task='addition'):

    # load data
    results = {}
    for model, N in zip(models, Ns):
        results[N] = {}
        for temp in temps:
            results[N][temp] = {}
            for n in ns:
                
                print(f"Loading N={N/1e9:.1f}, temp={temp:.1f}, n={n}", end='\r')

                try:
                    if task == 'addition':
                        with open(f"./{model}/{temp:.1f}/result_size={n:.0f}_value=0-100_mod=inf-temp={temp:.1f}-flash=True_sglang_batched_fp32.pkl", 'rb') as f:
                            res = pickle.load(f)
                    elif task =='brackets':
                        with open(f'./{model}/{temp:.1f}/result_size={n:.0f}_brackets-temp={temp:.1f}-flash=True_sglang_batched_fp32.pkl', 'rb') as f:
                            res = pickle.load(f)
                    elif task == 'parity':
                        with open(f'./{model}/{temp:.1f}/result_size={n:.0f}_parity-temp={temp:.1f}-flash=True_sglang_batched_fp32.pkl', 'rb') as f:
                            res = pickle.load(f)
                    elif task == 'index':
                        with open(f'./{model}/{temp:.1f}/result_size={n:.0f}_index-temp={temp:.1f}-flash=True_sglang_batched_fp32.pkl', 'rb') as f:
                            res = pickle.load(f)
                    else:
                        raise ValueError(f"Unknown task: {task}")
                    
                    results[N][temp][n] = res
                
                except:
                    print(f"Missing File: N={N/1e9:.1f}, temp={temp:.1f}, n={n}")
    
    return results

def _get_correct_addition(results, Ns, ns, temps, num_runs):

    # regex pattern to extract numbers in \boxed{ } or \boxed{\text{ }} format. Allows for "," in the number.
    pattern = r"\\boxed\{\s*(?:\\text\{\s*)?((?:[0-9]{1,3}(?:,[0-9]{3})+)|[0-9]+)\s*\}?\s*\}"

    correct = {}
    for N in Ns:
        
        correct[N] = {}
        
        for temp in temps:
            correct[N][temp] = np.zeros((ns.shape[0], num_runs))
            
            for i, n in enumerate(ns):
                for j in range(num_runs):

                    correct_answer = int(results[N][temp][n]['results'][j]['correct_answer'])
                    answer_text = results[N][temp][n]['results'][j]['output']

                    if r"</think>" in answer_text:
                        endthink_idx = answer_text.find(r"</think>")
                    else:
                        endthink_idx = None

                    if (endthink_idx is not None):
                        matches = re.search(pattern, answer_text[endthink_idx:])
                        answer = int(matches.group(1).replace(',','')) if matches else None
                    else:
                        answer = None
                
                    if (answer is not None) and (answer == correct_answer):
                        correct[N][temp][i, j] = 1.0
    
    return correct

def _get_correct_brackets(results, Ns, ns, temps, num_runs):

    pattern = r"boxed\{\s*(?:\\text\{\s*)?(true|false|True|False)\s*\}?\s*\}"

    correct = {}
    for N in Ns:
        correct[N] = {}
        for temp in temps:
            correct[N][temp] = np.zeros((ns.shape[0], num_runs))
            for i, n in enumerate(ns):
                for j in range(num_runs):

                    correct_answer = bool(results[N][temp][n]['results'][j]['correct_answer'])
                    answer_text = results[N][temp][n]['results'][j]['output']
                    
                    if r"</think>" in answer_text:
                        endthink_idx = answer_text.find(r"</think>")
                    else:
                        endthink_idx = None

                    if endthink_idx is not None:
                        matches = re.search(pattern, answer_text[endthink_idx:], flags=re.IGNORECASE)

                        if matches is None:
                            answer = None
                        elif matches.group(1) in ['false', 'False']:
                            answer = False
                        elif matches.group(1) in ['true', 'True']:
                            answer = True
                        else:
                            answer = None
                            
                        if (answer is not None) and (answer == correct_answer):
                            correct[N][temp][i, j] = 1.0
                        elif (correct_answer == True) and ("true" in answer_text[-10:] or "True" in answer_text[-10:]):
                            correct[N][temp][i,j] = 1.0
                        elif (correct_answer == False) and ("false" in answer_text[-10:] or "False" in answer_text[-10:]):
                            correct[N][temp][i,j] = 1.0
            
    return correct

def _get_correct_parity(results, Ns, ns, temps, num_runs):

    pattern = r"\\boxed\{\s*(?:\\text\{\s*)?(even|odd)\s*\}?\s*\}"

    correct = {}

    for N in Ns:
        correct[N] = {}
        for temp in temps:
            correct[N][temp] = np.zeros((ns.shape[0], num_runs))
            for i, n in enumerate(ns):
                for j in range(num_runs):

                    correct_answer = results[N][temp][n]['results'][j]['correct_answer']
                    answer_text = results[N][temp][n]['results'][j]['output']

                    if r"</think>" in answer_text:
                        endthink_idx = answer_text.find(r"</think>")
                    else:
                        endthink_idx = None

                    if endthink_idx is not None:

                        matches = re.search(pattern, answer_text[endthink_idx:], flags=re.IGNORECASE)
                        
                        if matches is None:
                            answer = None
                        elif matches.group(1) in ['even', 'Even']:
                            answer = 'even' 
                        elif matches.group(1) in ['odd', 'Odd']:
                            answer = 'odd'
                        else:
                            answer = None
                    
                        if (answer is not None) and (answer == correct_answer):
                            correct[N][temp][i, j] = 1.0
    
    return correct


def _get_correct_index(results, Ns, ns, temps, num_runs):

    pattern = r"\\boxed\{\s*(?:\\text\{\s*)?((?:[0-9]{1,3}(?:,[0-9]{3})+)|[0-9]+)\s*\}?\s*\}"

    correct = {}
    for N in Ns:
        correct[N] = {}
        for temp in temps:
            correct[N][temp] = np.zeros((ns.shape[0], num_runs))
            for i, n in enumerate(ns):
                for j in range(num_runs):

                    correct_answer = int(results[N][temp][n]['results'][j]['correct_answer'])
                    answer_text = results[N][temp][n]['results'][j]['output']

                    if r"</think>" in answer_text:
                        endthink_idx = answer_text.find(r"</think>")
                    else:
                        endthink_idx = None

                    if (endthink_idx is not None):
                        matches = re.search(pattern, answer_text[endthink_idx:])
                        answer = int(matches.group(1).replace(',','')) if matches else None
                    else:
                        answer = None

                    if (answer is not None) and (answer == correct_answer):
                        correct[N][temp][i, j] = 1.0

    return correct                         

def get_correct(results, Ns, ns, temps, num_runs, task='addition'):

    if task == 'addition':
        correct = _get_correct_addition(results, Ns, ns, temps, num_runs)
    elif task == 'brackets':
        correct = _get_correct_brackets(results, Ns, ns, temps, num_runs)
    elif task == 'parity':
        correct = _get_correct_parity(results, Ns, ns, temps, num_runs)
    elif task == 'index':
        correct = _get_correct_index(results, Ns, ns, temps, num_runs)
    else:
        raise ValueError(f"Unknown task: {task}")

    return correct

def count_correct(correct_indicator, Ns, ns, temps):

    correct_count = {}
    for temp in temps:
        correct_count[temp] = np.zeros((Ns.shape[0], ns.shape[0]))
        for i, N in enumerate(Ns):
            correct_count[temp][i, :] = np.nansum(correct_indicator[N][temp], axis=1)


    return correct_count


def get_length(results, Ns, ns, temps, num_runs):

    length = {}
    for N in Ns:
        length[N] = {}
        for temp in temps:
            length[N][temp] = np.zeros((ns.shape[0], num_runs))
            for i, n in enumerate(ns):
                for j in range(num_runs):
                    length[N][temp][i, j] = results[N][temp][n]['results'][j]['num_tokens']

    return length

def group_length_correct(length, correct_indicator, Ns, ns, temp):

    Y = []
    group_ns_idx = []
    group_Ns_idx = []
    L_mean = []
    L_std = []

    for i, N in enumerate(Ns):
        for j, _ in enumerate(ns):

            num_correct = np.nansum(correct_indicator[N][temp][j, :])
            L = length[N][temp][j, :]
                
            if np.isfinite(num_correct) and num_correct > 0:

                Y.append(num_correct)
                
                group_Ns_idx.append(i)
                group_ns_idx.append(j)

                L_mean.append(np.mean(L[correct_indicator[N][temp][j, :]==1.0]))
                L_std.append(np.std(L[correct_indicator[N][temp][j, :]==1.0]))

    return np.asarray(L_mean), np.asarray(L_std), np.asarray(Y), np.asarray(group_Ns_idx), np.asarray(group_ns_idx)