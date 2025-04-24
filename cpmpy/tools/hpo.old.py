import subprocess
import sys
import os
import time
from random import shuffle
import pandas as pd
from skopt import Optimizer
from skopt.utils import point_asdict, dimensions_aslist
import numpy as np
from ..solvers.utils import SolverLookup, param_combinations
from ..solvers.solver_interface import ExitStatus
import pandas as pd
from pychoco import solver as chocosolver
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

class Probe:
    @staticmethod
    def geometric_sequence(i):
        return 1.2 * i

    @staticmethod
    def luby_sequence(i, current_timeout, timeout_list):
        timeout_list.append(current_timeout)
        sequence = [timeout_list[0]]
        while len(sequence) <= i:
            sequence += sequence + [2 * sequence[-1]]
        return sequence[i]

    def __init__(self, solvername, model, time_limit, max_tries, all_config, default_config, fix_params, **kwargs):
        self.solvername = solvername
        self.time_limit = time_limit
        self.model = model
        self.problem_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        self.script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.relative_path = os.path.join(self.script_dir, os.path.basename(sys.argv[0]))
        if len(sys.argv) > 2:
            self.HPO = sys.argv[1]
            self.solver_name = sys.argv[2]
        elif len(sys.argv) == 2:
            self.HPO = sys.argv[1]
            self.solver_name = self.solvername or "ortools"
        else:
            self.HPO = "Hamming"
            self.solver_name = self.solvername or "ortools"
        if not all_config or not default_config or all_config == {} or default_config == {} or solvername is None or solvername=="":
            all_config , default_config = self.set_hp(self.solver_name)
        self.mode = "minimize" if self.model.objective_is_min else "maximize"
        self.global_timeout = time_limit
        self.max_tries = max_tries
        self.all_config = all_config
        self.default_config = default_config
        self.param_order = list(self.all_config.keys())
        self.best_config = self._params_to_np([self.default_config])
        self.best_config_str = self.dict_to_string(self.default_config)
        self.probe_timeout = None
        self.solving_time = None
        self.round_timeout = None
        self.best_obj = 1e10 if self.mode == "minimize" else -1e10
        self.timeout_list = []
        self.none_change_flag = False
        self.solution_list = []
        self.unchanged_count = 0
        self.fix_params = fix_params
        self.best_params = None
        self.best_runtime = None
        round_counter = total_time_used = 0
        self.default_flag = True
        self.additional_params = kwargs
        print("self.additional_params", self.additional_params, self.HPO, self.solver_name)
        self.init_round_type = kwargs.get('init_round_type', None)
        self.stop_type = kwargs.get('stop_type', None)
        self.tuning_timeout_type = kwargs.get('tuning_timeout_type', None)
        self.time_evol = kwargs.get('time_evol', None)
        self.results_file = f"hpo_result_{self.solver_name}.csv"
        if self.tuning_timeout_type == "Static":
            self.probe_timeout, self.none_change_flag = Probe.Tuning_global_timeout(self, self.global_timeout, self.tuning_timeout_type, self.solution_list, round_counter, total_time_used)
        if self.HPO != "freesearch":
            self.round_timeout = Probe.initialize_round_timeout(self, self.solver_name, self.model, self.init_round_type)
        self.stop = Probe.stop_condition(self, self.stop_type)
        if self.HPO == "Hamming":
            Probe.Hamming_Distance(self)
        elif self.HPO == "Bayesian":
            Probe.Bayesian_Optimization(self)
        elif self.HPO == "Grid":
            Probe.Grid_Search(self)
        elif self.HPO == "freesearch":
            Probe.free_search(self)
        Probe.save_result(self)

    def set_hp(self, solver_name):
        if solver_name == "ortools":
            tunables = {
                # 'optimize_with_core': [False, True],
                # 'search_branching': [0, 1, 2, 3, 4, 5, 6],
                # 'boolean_encoding_level': [0, 1, 2, 3],
                'linearization_level': [0, 1, 2],
                'core_minimization_level': [0, 1, 2],  # new in OR-tools>= v9.8
                'cp_model_probing_level': [0, 1, 2, 3],
                # 'cp_model_presolve': [False, True],
                # 'clause_cleanup_ordering': [0, 1],
                # 'binary_minimization_algorithm': [0, 1, 2, 3, 4],
                # 'minimization_algorithm': [0, 1, 2, 3],
                # 'use_phase_saving': [False, True]
                }

            defaults = {
                # 'optimize_with_core': False,
                # 'search_branching': 0,
                # 'boolean_encoding_level': 1,
                'linearization_level': 1,
                'core_minimization_level': 2,# new in OR-tools>=v9.8
                'cp_model_probing_level': 2,
                # 'cp_model_presolve': True,
                # 'clause_cleanup_ordering': 0,
                # 'binary_minimization_algorithm': 1,
                # 'minimization_algorithm': 2,
                # 'use_phase_saving': True
                }
        elif solver_name == "choco":
            tunables = {
                "solution_limit": [None, 0, 100, 500, 1000],
                "node_limit": [None, 1000, 5000, 10000],
                "fail_limit": [None, 100, 500, 1000],
                "restart_limit": [None],
                "backtrack_limit": [None]
            }
            defaults = {
                "solution_limit": None,
                "node_limit": None,
                "fail_limit": None,
                "restart_limit": None,
                "backtrack_limit": None
            }
            for key in tunables:
                tunables[key] = [None if v is None else int(v) for v in tunables[key]]
            for key in defaults:
                defaults[key] = None if defaults[key] is None else int(defaults[key])

            defaults = {key: defaults[key] for key in defaults}
            tunables = {key: tunables[key] for key in tunables}
        elif solver_name == "ACE":
            tunables = {
                'varh': ['input', 'dom', 'rand'],
                'valh': ['min', 'max', 'rand'],
            }
            defaults = {
                'varh': 'input',
                'valh': 'min',
            }
        return tunables, defaults

    def initialize_round_timeout(self, solver_name, model, round_type):
        if self.HPO == "Hamming":
            round_type = "Dynamic"
        if round_type == "Dynamic":
            solver = SolverLookup.get(solver_name, model)
            solver.solve(time_limit=self.probe_timeout ,**self.default_config)
            print("First_round_obj",solver.objective_value())
            print("First_round_runtime",solver.status().runtime)
            self.base_runtime = solver.status().runtime
            self.round_timeout = self.base_runtime
            self.probe_timeout = round(self.probe_timeout-solver.status().runtime, 3)
        elif round_type == "Static":
            self.round_timeout = 5
        else:
            self.round_timeout = self.time_limit
        return self.round_timeout

    def stop_condition(self, stop_type):
        if stop_type == "First_Solution":
            self.stop = "First_Solution"
        elif stop_type == "Timeout":
            self.stop = "Timeout"
        return self.stop

    def round_timeout_evolution(self, time_evol, current_timeout):
        if time_evol == "Static":
            round_timeout = current_timeout
        elif time_evol == "Dynamic_Geometric":
            round_timeout = Probe.geometric_sequence(current_timeout)
        elif time_evol == "Dynamic_Luby":
            index = len(self.timeout_list)
            round_timeout = self.luby_sequence(index, current_timeout, self.timeout_list)
        return round_timeout

    def Tuning_global_timeout(self, global_timeout, tuning_timeout_type, solution_list, round_counter, total_time_used):
        if tuning_timeout_type == "Static":
            self.probe_timeout = global_timeout * 0.2
            self.solving_time = global_timeout - self.probe_timeout
        elif tuning_timeout_type == "Dynamic":
            if round_counter < self.max_tries and total_time_used < self.probe_timeout:
                if len(self.solution_list) > 1:
                    if self.solution_list[-1].get('objective') == self.solution_list[-2].get('objective'):
                        self.unchanged_count += 1
                    else:
                        self.unchanged_count = 0
                if self.unchanged_count >= 8:
                    self.none_change_flag = True
                    self.solving_time = global_timeout - total_time_used
            else:
                self.none_change_flag = True

        return self.probe_timeout, self.none_change_flag

    def memory(self):
        print("MEMORY")

    def _get_score(self, combos):
        """
            Return the hamming distance for each remaining configuration to the current best config.
            Lower score means better configuration, so exploit the current best configuration by only allowing small changes.
        """
        return np.count_nonzero(combos != self.best_config, axis=1)

    def _params_to_np(self,combos):
        arr = [[params[key] for key in self.param_order] for params in combos]
        return np.array(arr)

    def _np_to_params(self,arr):
        return {key: val for key, val in zip(self.param_order, arr)}

    def dict_to_string(self, input_dict):
        return ' '.join([f"{key}={value}" for key, value in input_dict.items()])

    def Hamming_Distance(self):
        if self.time_limit is not None:
            start_time = time.time()
        combos = list(param_combinations(self.all_config))
        combos_np = self._params_to_np(combos)
        self.best_runtime = self.round_timeout
        # Ensure random start
        np.random.shuffle(combos_np)
        total_time_used = current_timeout = 0
        i = 0
        if self.max_tries is None:
            self.max_tries = len(combos_np)
        while (i < self.max_tries if hasattr(self,
                                             'max_tries') else True) and total_time_used + current_timeout < self.probe_timeout:
            # Make new solver total_time_used += current_timeout
            solver = SolverLookup.get(self.solver_name, self.model)
            # Apply scoring to all combos
            scores = self._get_score(combos_np)
            if scores.size == 0:
                print("Warning: scores array is empty.")
                return
            max_idx = np.where(scores == scores.min())[0][0]
            # Get index of optimal combo
            params_np = combos_np[max_idx]
            # Remove optimal combo from combos
            combos_np = np.delete(combos_np, max_idx, axis=0)
            # Convert numpy array back to dictionary
            params_dict = self._np_to_params(params_np)
            # set fixed params
            params_dict.update(self.fix_params)
            timeout = self.best_runtime
            # set timeout depending on time budget
            if self.time_limit is not None:
                timeout = min(timeout, self.time_limit - (time.time() - start_time))
            # run solver
            if self.solver_name == "ortools":
                solver.solve(**params_dict, time_limit=timeout)
            elif self.solver_name == "choco":
                if timeout < 0.5 :
                    timeout = 0.5
                best_params = {key: value for key, value in params_dict.items() if value is not None}
                print("best_params_first_step:",best_params)
                best_params = {key: int(value) for key, value in best_params.items()}
                print("best_params_second_step:",best_params)
                solver.solve(time_limit=timeout, **{k: int(v) for k, v in best_params.items()})
            elif self.solver_name == "ACE":
                solver.solve(**params_dict, time_limit=timeout)
            if solver.status().exitstatus == ExitStatus.OPTIMAL and solver.status().runtime < self.best_runtime:
                self.best_runtime = solver.status().runtime
                # update surrogate
                self.best_config = params_np
                self.best_obj = solver.objective_value()
            if solver.objective_value() is not None and  solver.objective_value() < self.best_obj:
                self.best_obj = solver.objective_value()
            print("obj",solver.objective_value())
            print("runtime",solver.status().runtime)
            current_timeout = solver.status().runtime
            if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                break
            i += 1
            if current_timeout < 0.5:
                total_time_used += 0.5
            else:
                total_time_used += current_timeout
        if hasattr(self, 'max_tries'):
            self.solving_time = self.global_timeout - total_time_used
        print("total_time_used",total_time_used)
        print("self.solving_time",self.solving_time)
        print("remaining time:",self.global_timeout - total_time_used)
        self.best_params, self.best_runtime, self.best_obj = Probe.solving(self)
        self.best_params = self._np_to_params(self.best_config)
        self.best_params.update(self.fix_params)
        print(self.best_params, self.best_runtime, self.best_obj)
        return self.best_params, self.best_runtime

    def Grid_Search(self):
        if self.time_limit is not None:
            start_time = time.time()
        self.best_runtime = self.round_timeout

        # Get all possible hyperparameter configurations
        combos = list(param_combinations(self.all_config))
        shuffle(combos)  # test in random order

        if self.max_tries is not None:
            combos = combos[:self.max_tries]

        for params_dict in combos:
            # Make new solver
            solver = SolverLookup.get(self.solver_name, self.model)
            # set fixed params
            params_dict.update(self.fix_params)
            timeout = self.best_runtime
            # set timeout depending on time budget
            if self.time_limit is not None:
                timeout = min(timeout, self.time_limit - (time.time() - start_time))
            # run solver
            solver.solve(**params_dict, time_limit=timeout)
            if solver.status().exitstatus == ExitStatus.OPTIMAL and solver.status().runtime < self.best_runtime:
                self.best_runtime = solver.status().runtime
                # update surrogate
                self.best_params = params_dict

            if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                break
        print(self.best_params , self.best_runtime)
        return self.best_params , self.best_runtime

    def Bayesian_Optimization(self):
        current_timeout = self.best_runtime = self.round_timeout
        opt = Optimizer(dimensions=dimensions_aslist(self.all_config), base_estimator="GP", acq_func="EI")
        first_non_none_objective = False
        self.best_obj = obj = 1e10 if self.mode == "minimize" else -1e10
        round_counter = total_time_used = solve_call_counter = seen_counter = 0
        self.solution_list.append({'params': self.best_params})
        while (self.tuning_timeout_type == "Static" and total_time_used + current_timeout < self.probe_timeout and current_timeout != 0)or(self.tuning_timeout_type == "Dynamic" and round_counter<self.max_tries):
            solver = SolverLookup.get(self.solver_name, self.model)
            params = opt.ask()
            parameters = point_asdict(self.all_config, params) if total_time_used != 0 else self.default_config
            seen = False
            for solution in self.solution_list:
                if solution.get('params') == parameters:
                    seen = True
                    obj = solution.get('objective')
                    runtime = solution.get('runtime')
                    status = solution.get('status')
                    break
            if seen:
                print("Parameters seen before. Using stored results.")
                seen_counter += 1
                if obj is None or np.isnan(obj) or np.isinf(obj):
                    if self.mode == 'maximize':
                        obj = -1e10
                    elif self.mode == 'minimize':
                        obj = 1e10
                total_time_used += 1
                # if seen_counter == 8:
                #     break
            else:
                if self.stop == "Timeout":
                    if self.solver_name == "ortools":
                        solver.solve(time_limit=current_timeout, **parameters)
                    elif self.solver_name == "choco":
                        parameters = {key: value for key, value in parameters.items() if value is not None}
                        parameters = {key: int(value) for key, value in parameters.items()}
                        solver.solve(time_limit=current_timeout, **{k: int(v) for k, v in parameters.items()})
                    elif self.solver_name == "ACE":
                        solver.solve(time_limit=current_timeout, **parameters)
                elif self.stop == "First_Solution":
                    if self.solver_name == "ortools":
                        solver.solve(**parameters)
                    elif self.solver_name == "choco":
                        parameters = {key: value for key, value in parameters.items() if value is not None}
                        parameters = {key: int(value) for key, value in parameters.items()}
                        solver.solve(**{k: int(v) for k, v in parameters.items()})
                    elif self.solver_name == "ACE":
                        solver.solve(**parameters)
                if solver.objective_value() is not None:
                    solve_call_counter += 1
                    self.first_non_none_objective = True
                if self.mode == "minimize":
                    if (solver.objective_value() is not None and solver.objective_value() < self.best_obj) or (solver.objective_value() == self.best_obj and (self.best_runtime is None or solver.status().runtime < self.best_runtime)):
                        self.best_obj = solver.objective_value()
                        self.best_params = parameters
                        self.best_runtime = round(solver.status().runtime,3)
                else:
                    if (solver.objective_value() is not None and solver.objective_value() > self.best_obj) or (solver.objective_value() == self.best_obj and (
                            self.best_runtime is None or solver.status().runtime < self.best_runtime)):
                        self.best_obj = solver.objective_value()
                        self.best_params = parameters
                        self.best_runtime = round(solver.status().runtime,3)
                if current_timeout < 0.5 :
                    total_time_used += 0.5
                else:
                    total_time_used += current_timeout

                obj = solver.objective_value() if solve_call_counter > 0 else None
                if obj is None or not np.isfinite(obj):
                    obj = self.best_obj if self.best_obj is not None and np.isfinite(self.best_obj) else 1e10
                    obj = round(float(obj), 3)
                    if not first_non_none_objective:
                        current_timeout = Probe.round_timeout_evolution(self, self.time_evol, current_timeout)
                        current_timeout = round(current_timeout, 2)
                        if self.tuning_timeout_type == "Static" and current_timeout > self.probe_timeout - total_time_used:
                            current_timeout = self.probe_timeout - total_time_used
                else:
                    obj = round(float(obj), 3)
                    first_non_none_objective = True
                self.solution_list.append({
                    'params': dict(parameters),
                    'objective': obj,
                    'runtime': solver.status().runtime,
                    'status': solver.status().exitstatus
                })
                obj = -obj if self.mode == "maximize" else obj
            if self.tuning_timeout_type == "Dynamic":
                self.probe_timeout, self.none_change_flag = Probe.Tuning_global_timeout(self, self.global_timeout,
                                                                                   self.tuning_timeout_type,
                                                                                   self.solution_list,
                                                                                   round_counter,
                                                                                   total_time_used)
                if self.none_change_flag:
                    break
            print("obj", solver.objective_value())
            print("runtime", solver.status().runtime)
            if self.tuning_timeout_type == "Static" and total_time_used >= self.probe_timeout:
                print("Timeout reached. Exiting.")
                break
            print("total_time_used",total_time_used)
            opt.tell(params, obj)
            round_counter += 1
        solve_call_counter += 1
        self.best_params , self.best_runtime , self.best_obj = Probe.solving(self)
        print(self.best_params , self.best_runtime , self.best_obj)
        return self.best_params, self.best_runtime

    def free_search(self):
        solver = SolverLookup.get(self.solver_name, self.model)
        solver.solve(time_limit=self.global_timeout)
        self.best_obj = solver.objective_value()
        self.best_runtime = solver.status().runtime
        print(self.best_params, self.best_runtime, self.best_obj)
        return self.best_params, self.best_runtime

    def get_best_params_and_runtime(self):
        return self.best_params, self.best_runtime

    def save_result(self):
        self.results_file = f"hpo_result_{self.solver_name}.csv"

        if os.path.exists(self.results_file):
            df = pd.read_csv(self.results_file)
        else:
            if self.solver_name == "choco":
                df = pd.DataFrame(columns=[
                    "problem", "Global_timeout", "Mode",
                    "objective_Hamming", "run_time_Hamming", "best_configuration_Hamming",
                    "objective_BO", "run_time_BO", "best_configuration_BO",
                    "objective_Grid", "run_time_Grid", "best_configuration_Grid",
                    "objective_free", "run_time_free", "best_configuration_free",
                    "Best_HPO_Method"
                ])
            else:
                df = pd.DataFrame(columns=[
                    "problem", "Global_timeout", "Mode",
                    "objective_Hamming", "run_time_Hamming", "best_configuration_Hamming",
                    "objective_BO", "run_time_BO", "best_configuration_BO",
                    "objective_Grid", "run_time_Grid", "best_configuration_Grid",
                    "Best_HPO_Method"
                ])

        if self.problem_name in df["problem"].values:
            row_index = df[df["problem"] == self.problem_name].index[0]
        else:
            if self.solver_name == "choco":
                new_row = {
                    "problem": self.problem_name,
                    "Global_timeout": self.global_timeout,
                    "Mode": self.mode,
                    "objective_Hamming": None, "run_time_Hamming": None, "best_configuration_Hamming": None,
                    "objective_BO": None, "run_time_BO": None, "best_configuration_BO": None,
                    "objective_Grid": None, "run_time_Grid": None, "best_configuration_Grid": None,
                    "objective_free": None, "run_time_free": None, "best_configuration_free": None,
                    "Best_HPO_Method": None
                }
            else:
                new_row = {
                    "problem": self.problem_name,
                    "Global_timeout": self.global_timeout,
                    "Mode": self.mode,
                    "objective_Hamming": None, "run_time_Hamming": None, "best_configuration_Hamming": None,
                    "objective_BO": None, "run_time_BO": None, "best_configuration_BO": None,
                    "objective_Grid": None, "run_time_Grid": None, "best_configuration_Grid": None,
                    "Best_HPO_Method": None
                }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            row_index = df.index[-1]

        # Update results
        if self.HPO == "Hamming":
            df.at[row_index, "objective_Hamming"] = self.best_obj
            df.at[row_index, "run_time_Hamming"] = str(self.best_runtime)
            df.at[row_index, "best_configuration_Hamming"] = str(self.best_params)

        elif self.HPO == "Bayesian":
            df.at[row_index, "objective_BO"] = self.best_obj
            df.at[row_index, "run_time_BO"] = str(self.best_runtime)
            df.at[row_index, "best_configuration_BO"] = str(self.best_params)

        elif self.HPO == "Grid":
            df.at[row_index, "objective_Grid"] = self.best_obj
            df.at[row_index, "run_time_Grid"] = str(self.best_runtime)
            df.at[row_index, "best_configuration_Grid"] = str(self.best_params)

        elif self.solver_name == "choco" and self.HPO == "freesearch":
            df.at[row_index, "objective_free"] = self.best_obj
            df.at[row_index, "run_time_free"] = str(self.best_runtime)
            df.at[row_index, "best_configuration_free"] = str(self.best_params)

        # Compare methods and update best one
        self.compare_hpo_methods(df)

        df.to_csv(self.results_file, index=False)
        print(f"Results updated for {self.problem_name} in {self.results_file}")

    def compare_hpo_methods(self, df):
        for i, row in df.iterrows():
            mode = str(row["Mode"]).strip().lower()

            if self.solver_name == "choco":
                if mode == "maximize":
                    hamming_obj = row["objective_Hamming"] if pd.notna(row["objective_Hamming"]) else -1e10
                    bo_obj = row["objective_BO"] if pd.notna(row["objective_BO"]) else -1e10
                    grid_obj = row["objective_Grid"] if pd.notna(row["objective_Grid"]) else -1e10
                    free_obj = row["objective_free"] if pd.notna(row["objective_free"]) else -1e10
                elif mode == "minimize":
                    hamming_obj = row["objective_Hamming"] if pd.notna(row["objective_Hamming"]) else 1e10
                    bo_obj = row["objective_BO"] if pd.notna(row["objective_BO"]) else 1e10
                    grid_obj = row["objective_Grid"] if pd.notna(row["objective_Grid"]) else 1e10
                    free_obj = row["objective_free"] if pd.notna(row["objective_free"]) else 1e10
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            else:
                if mode == "maximize":
                    hamming_obj = row["objective_Hamming"] if pd.notna(row["objective_Hamming"]) else -1e10
                    bo_obj = row["objective_BO"] if pd.notna(row["objective_BO"]) else -1e10
                    grid_obj = row["objective_Grid"] if pd.notna(row["objective_Grid"]) else -1e10
                elif mode == "minimize":
                    hamming_obj = row["objective_Hamming"] if pd.notna(row["objective_Hamming"]) else 1e10
                    bo_obj = row["objective_BO"] if pd.notna(row["objective_BO"]) else 1e10
                    grid_obj = row["objective_Grid"] if pd.notna(row["objective_Grid"]) else 1e10
                else:
                    raise ValueError(f"Unknown mode: {mode}")

            if self.solver_name == "choco":
                print(f"Row {i} - Hamming: {hamming_obj}, BO: {bo_obj}, Grid: {grid_obj}, free: {free_obj}")
            else:
                print(f"Row {i} - Hamming: {hamming_obj}, BO: {bo_obj}, Grid: {grid_obj}")
            print(f"Mode: {mode}")

            # Convert runtime values to float (set inf if missing)
            hamming_time = float(row["run_time_Hamming"]) if pd.notna(row["run_time_Hamming"]) else float("inf")
            bo_time = float(row["run_time_BO"]) if pd.notna(row["run_time_BO"]) else float("inf")
            grid_time = float(row["run_time_Grid"]) if pd.notna(row["run_time_Grid"]) else float("inf")
            if self.solver_name == "choco":
                free_time = float(row["run_time_free"]) if pd.notna(row["run_time_free"]) else float("inf")

            # Determine best HPO method
            if self.solver_name == "choco":
                best_obj = max(hamming_obj, bo_obj, grid_obj, free_obj) if mode == "maximize" else min(hamming_obj, bo_obj, grid_obj, free_obj)
            else:
                best_obj = max(hamming_obj, bo_obj, grid_obj) if mode == "maximize" else min(hamming_obj, bo_obj, grid_obj)
            print(f"Best objective found: {best_obj}")

            # Select the best method (if there's a tie, pick the one with the shortest runtime)
            best_methods = []
            if abs(hamming_obj - best_obj) < 1e-6:
                best_methods.append(("Hamming", hamming_time))
            if abs(bo_obj - best_obj) < 1e-6:
                best_methods.append(("Bayesian", bo_time))
            if abs(grid_obj - best_obj) < 1e-6:
                best_methods.append(("Grid", grid_time))
            if self.solver_name == "choco":
                if abs(free_obj - best_obj) < 1e-6:
                    best_methods.append(("free", free_time))

            print(f"Candidate best methods before sorting: {best_methods}")
            best_methods.sort(key=lambda x: x[1])  # Sort by runtime (ascending)
            df.at[i, "Best_HPO_Method"] = best_methods[0][0]  # Pick the fastest one
            print(f"Selected Best HPO Method: {df.at[i, 'Best_HPO_Method']}")

        df.to_csv(self.results_file, index=False)

    def config_(self):
        default_params = {
            "init_round_type": "Static",
            "stop_type": "Timeout",
            "tuning_timeout_type": "Static",
            "time_evol": "Dynamic_Geometric",
        }
        user_params = {
            "init_round_type": "Dynamic",  # "Dynamic", "Static" , "None"
            "stop_type": "Timeout",  # "First_Solution" , "Timeout"
            "tuning_timeout_type": "Static",  # "Static" , "Dynamic", "None"
            "time_evol": "Static"  # "Static", "Dynamic_Geometric" , "Dynamic_Luby"
        }
        params = {**default_params, **user_params}

    def solving(self):
        if self.best_params is None:
            self.best_params = self.default_config
        solver = SolverLookup.get(self.solver_name, self.model)
        if self.solver_name == "ortools":
            print("IM runnig solving phase")
            solver.solve(**self.best_params, time_limit=self.solving_time)
        elif self.solver_name == "choco":
            self.best_params = {key: value for key, value in self.best_params.items() if value is not None}
            self.best_params = {key: int(value) for key, value in self.best_params.items()}
            solver.solve(time_limit=self.solving_time, **{k: int(v) for k, v in self.best_params.items()})
        elif self.solver_name == "ACE":
            solver.solve(**self.best_params, time_limit=self.solving_time)
        if self.mode == "minimize":
            if (solver.objective_value() is not None and solver.objective_value() < self.best_obj) or (
                    solver.objective_value() == self.best_obj and (
                    self.best_runtime is None or solver.status().runtime < self.best_runtime)):
                self.best_obj = solver.objective_value()
                self.best_runtime = round(solver.status().runtime, 3)
        else:
            if (solver.objective_value() is not None and solver.objective_value() > self.best_obj) or (
                    solver.objective_value() == self.best_obj and (
                    self.best_runtime is None or solver.status().runtime < self.best_runtime)):
                self.best_obj = solver.objective_value()
                self.best_runtime = round(solver.status().runtime, 3)

        return self.best_params, self.best_runtime, self.best_obj


