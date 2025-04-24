#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Interface to the ACE native solver via XCSP3.

This module allows using the ACE solver (native Java binary) with CPMpy, by launching it as an external process.
It supports time-limited solving, status extraction, and minimal output parsing (objective value, satisfiability).

.. note::
    This interface assumes the `pycsp3` package is installed and accessible.
    The solver is executed through a Java jar wrapper (`ace.jar`) exposed via `pycsp3`.

Always use :func:`cp.SolverLookup.get("ace") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

============
Installation
============

Requires the `pycsp3` Python package:

.. code-block:: console

    $ pip install pycsp3

See https://github.com/xcsp3team/pycsp3 for additional details.

================
List of classes
================

.. autosummary::
    :nosignatures:

    CPM_ace
    ACESolver
    JavaJar
    ProcessNativeSolver
"""

import subprocess
import sys
from abc import ABC, abstractmethod
import psutil

import cpmpy
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from timeit import default_timer as timer

ANSWER_PREFIX = "s" + chr(32)
OBJECTIVE_PREFIX = "o" + chr(32)
SOLUTION_PREFIX = "v" + chr(32)

class ProcessNativeSolver(ABC):
    """
    Abstract base class for launching a solver as a native process.

    Provides utility methods for argument management, process control,
    and output collection.
    """

    def __init__(self):
        """Initialize an empty command and process reference."""
        self._cmd = []
        self._process = None

    @abstractmethod
    def build_command(self):
        """Construct the full command to be executed as a list of strings."""
        pass

    def add_argument(self, arg):
        """Append an argument to the command list.

        Args:
            arg (str): Argument to add.
        """
        self._cmd.append(arg)

    def start(self):
        """Start the solver subprocess using the built command."""
        cmd = self.build_command()
        self._process = psutil.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

    def is_running(self) -> bool:
        """Check if the subprocess is currently running.

        Returns:
            bool: True if running, False otherwise.
        """
        return self._process and self._process.is_running()

    def terminate(self):
        """Gracefully terminate the subprocess."""
        if self._process:
            self._process.terminate()

    def kill(self):
        """Forcefully kill the subprocess."""
        if self._process and self._process.returncode is None:
            self._process.kill()

    def wait(self):
        """Wait for the subprocess to complete.

        Returns:
            int: Return code of the process.
        """
        if self._process:
            return self._process.wait()

    def get_stdout(self):
        """Collect and return standard output as a list of stripped lines.

        Returns:
            list[str]: Lines from stdout.
        """
        if self._process and self._process.stdout:
            return [line.strip() for line in self._process.stdout.readlines()]
        return []

    def get_stderr(self):
        """Collect and return standard error as a list of stripped lines.

        Returns:
            list[str]: Lines from stderr.
        """
        if self._process and self._process.stderr:
            return [line.strip() for line in self._process.stderr.readlines()]
        return []

class JavaJar(ProcessNativeSolver):
    """
    Specialization of ProcessNativeSolver for launching Java .jar solvers.

    Args:
        jar_path (str): Path to the .jar file.
    """
    def __init__(self, jar_path):
        super().__init__()
        self.jar_path = jar_path

    def build_command(self):
        """Return the full Java -jar command with all arguments.

        Returns:
            list[str]: Command to execute.
        """
        return ["java", "-jar", self.jar_path] + self._cmd


class ACESolver(JavaJar):
    """
    Concrete solver interface for ACE launched as a Java jar.

    Parses the output to extract satisfiability status and objective value.
    """
    def __init__(self):
        from pycsp3.solvers.ace.ace import ACE_CP
        super().__init__(ACE_CP)
        self._exit_status = ExitStatus.UNKNOWN
        self.add_argument("-npc")
        self.add_argument("-sh=false")

    def set_instance(self, instance):
        """Prepend the instance file path to the command.

        Args:
            instance (str): Path to the XCSP3 instance.
        """
        self._cmd = [instance] + self._cmd

    def set_timelimit_seconds(self, seconds):
        """Add a time limit to the command line.

        Args:
            seconds (int): Time limit in seconds.
        """
        self.add_argument(f"-t={seconds}s")

    def _process_stdout(self):
        """Parse the solver stdout to determine status and extract objective.

        Returns:
            tuple: (ExitStatus, int or None)
        """
        stdout = self.get_stdout()
        answer = None
        obj = None

        for line in stdout:
            if line.startswith(ANSWER_PREFIX):
                answer = line[len(SOLUTION_PREFIX):].strip()
            if line.startswith(OBJECTIVE_PREFIX):
                try:
                    objective = line[len(OBJECTIVE_PREFIX):].split()[0]
                    obj = int(objective)
                except ValueError as e:
                    print(f"Impossible to transform '{line[len(OBJECTIVE_PREFIX):].split()[0]}' to a valid objective value",file=sys.stderr)
                    raise e

        if answer == "SATISFIABLE":
            self._exit_status = ExitStatus.FEASIBLE
        elif answer == "OPTIMUM FOUND":
            self._exit_status = ExitStatus.OPTIMAL
        elif answer == "UNSATISFIABLE":
            self._exit_status = ExitStatus.UNSATISFIABLE
        else:
            self._exit_status = ExitStatus.UNKNOWN

        return self._exit_status, obj, 0

    def solve(self):
        """Execute the solver process and parse its output.

        Returns:
            tuple: (ExitStatus, int or None)
        """
        self.start()
        try:
            self.wait()
            if self._process.returncode != 0:
                return ExitStatus.UNKNOWN, None, self._process.returncode
            return self._process_stdout()
        except Exception as e:
            self.kill()
            raise e

    def set_limit_solution(self, solution_limit):
        self.add_argument(f"-s={solution_limit}")


class CPM_ace(SolverInterface):
    """
    CPMpy interface for the ACE solver (native executable).

    Args:
        cpm_model (Model): Optional CPMpy model.
        subsolver (str): Ignored.
        xpath (str): XCSP3 instance file path.
    """

    @staticmethod
    def supported():
        """Check if the solver is available and pycsp3 is installed.

        Returns:
            bool: True if supported, else False
        """
        try:
            import pycsp3
            return True
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e

    def __init__(self, cpm_model=None, subsolver=None, xpath=None):
        if not self.supported():
            raise Exception("CPM_ace: Install the python package 'pycsp3' to use this solver interface.")

        assert subsolver is None
        assert xpath is not None
        self._xcsp_solver = ACESolver()
        self._xcsp_model = xpath
        self._xcsp_solver.set_instance(xpath)
        super().__init__(name="ace", cpm_model=cpm_model)
        self.cpm_status = SolverStatus(self.name)
        self._cpm_model:cpmpy.Model|None = cpm_model
        self._objective_min = False

    def solve(self, time_limit=None, solution_limit=None):
        """Solve the problem with optional time limit.

        Args:
            time_limit (int, optional): Timeout in seconds.

        Returns:
            bool: True if a solution is found, False otherwise.
        """
        if time_limit is not None:
            self._xcsp_solver.set_timelimit_seconds(time_limit)
        if solution_limit is not None:
            self._xcsp_solver.set_limit_solution(solution_limit)
        start = timer()
        self.cpm_status.exitstatus, self.objective_value_, return_code = self._xcsp_solver.solve()
        if return_code != 0:
            print(f"Solver failed with return code : {return_code}",file=sys.stderr)
            return False
        end = timer()
        self.cpm_status.runtime = end - start
        has_sol = self._solve_return(self.cpm_status)
        return has_sol

    def objective(self, expr, minimize):
        self._objective_min = minimize

    def add(self, cpm_expr):
        return self

    def has_objective(self):
        return self._cpm_model is not None and self._cpm_model.has_objective()
