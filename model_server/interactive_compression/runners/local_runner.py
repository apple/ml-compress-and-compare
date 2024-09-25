"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import random, string
import os
import json
import multiprocessing as mp
import shutil
from functools import partial
from json.decoder import JSONDecodeError


def _random_task_id(k=16):
    """Generates a random alphanumeric task ID string"""
    x = "".join(random.choices(string.ascii_letters + string.digits, k=k))
    return x


def json_objects_equal(obj1, obj2):
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        return set(obj1.keys()) == set(obj2.keys()) and all(
            json_objects_equal(obj1[k], obj2[k]) for k in obj1
        )
    elif isinstance(obj1, list) and isinstance(obj2, list):
        return len(obj1) == len(obj2) and all(
            json_objects_equal(obj1[i], obj2[i]) for i in range(len(obj1))
        )
    return obj1 == obj2


def exception_handling_wrapper(target_fn, args, output_dir):
    try:
        target_fn(args, output_dir)
    except Exception as e:
        with open(os.path.join(output_dir, "output.json"), "w") as file:
            json.dump({"status": "error", "message": f"{type(e)}: {e}"}, file)
        raise e


def inspector_task_runner(cls, *args, **kwargs):
    inspector = cls()
    exception_handling_wrapper(inspector, *args, **kwargs)


class LocalRunner:
    """
    A task runner that runs tasks as subprocess calls. You initialize the LocalRunner with
    a function that takes an argument dictionary and an output directory. The function should
    write a JSON file to output.json in the output directory which should
    contain the outputs or intermediate progress in one of the following formats:

    { "status": "running", "progress": 0.5 }
    { "status": "complete": "result": ... }
    { "status": "error", "message": ... }

    You should assume that the function will run in its own process, so it will not
    have access to any locally-scoped variables.
    """

    def __init__(
        self, inspector_class, task_directory, clear_files=True, max_workers=5
    ):
        super().__init__()
        self.target_fn = partial(inspector_task_runner, inspector_class)
        self.task_directory = task_directory
        self.scheduler = None
        self.task_cache = []  # List of tuples (args, task_id)
        self.callback_functions = {}
        if clear_files and os.path.exists(self.task_directory):
            shutil.rmtree(self.task_directory)
        if not os.path.exists(self.task_directory):
            os.mkdir(self.task_directory)
        self.max_workers = max_workers
        self.running_processes = {}  # task_id: Process
        self.queued_processes = []  # Tuples (task_id, Process)

    def set_scheduler(self, scheduler):
        """
        Adds an APScheduler object to the task runner. The scheduler instance is used
        to schedule a periodic cron job to poll for task results.
        """
        self.scheduler = scheduler
        if scheduler is not None:
            self.poll_task = "poll_worker"
            self.scheduler.add_job(
                id=self.poll_task,
                func=self.poll_task_results,
                trigger="interval",
                seconds=5,
            )

    def get_task_result(self, task_id):
        if not any(id == task_id for _, id in self.task_cache):
            return {
                "status": "error",
                "task_id": task_id,
                "message": f"Task with {task_id} could not be found",
            }

        path = os.path.join(self.task_directory, f"{task_id}", "output.json")
        if os.path.exists(path):
            with open(path, "r") as file:
                return {**json.load(file), "task_id": task_id}

        return {"status": "waiting", "task_id": task_id}

    def run_task(self, args, callback_fn, target_fn=None, dry_run=False):
        """
        If dry_run is True, the task will not be started, but just returned
        if a matching existing task was found. If not found, None will be returned.
        """

        existing_task = next(
            (
                task_id
                for arg_set, task_id in self.task_cache
                if json_objects_equal(arg_set, args)
            ),
            None,
        )
        if existing_task is not None:
            if not dry_run:
                self.callback_functions.setdefault(existing_task, []).append(
                    callback_fn
                )
            result = self.get_task_result(existing_task)
            if (
                result
                and "status" in result
                and result["status"] not in ("stopped", "error")
            ):
                return result

        if dry_run:
            return None

        task_id = _random_task_id()
        while os.path.exists(os.path.join(self.task_directory, f"{task_id}")):
            print("Regenerating task ID because of duplicate!")
            task_id = _random_task_id()
        self.task_cache.append((args, task_id))
        self.callback_functions.setdefault(task_id, []).append(callback_fn)

        output_dir = os.path.join(self.task_directory, f"{task_id}")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # We use multiprocessing instead of APScheduler to handle actual task
        # runs to give us more control of starting and stopping jobs (APScheduler
        # jobs cannot be terminated).
        p = mp.Process(
            target=partial(
                exception_handling_wrapper,
                target_fn,
            )
            if target_fn is not None
            else self.target_fn,
            args=(args, output_dir),
        )
        if len(self.running_processes) >= self.max_workers:
            print("Queueing process", task_id)
            self.queued_processes.append((task_id, p))
            return {"status": "waiting", "task_id": task_id}
        else:
            print("Starting process immediately", task_id, self.running_processes)
            p.start()
            self.running_processes[task_id] = p
            return {"status": "running", "task_id": task_id}

    def run_tasks(self, arg_sets, callback_fn, target_fn=None, dry_run=False):
        """Runs multiple tasks in parallel"""

        if not len(arg_sets):
            if callback_fn:
                callback_fn([], [])
            return

        results = {}
        task_id_list = []

        def callback(task_id, progress):
            nonlocal results
            results[task_id] = progress
            callback_fn(task_id_list, [results[id] for id in task_id_list])

        start_results = []
        for args in arg_sets:
            progress = self.run_task(
                args, callback, target_fn=target_fn, dry_run=dry_run
            )
            start_results.append(progress)
            if not dry_run:
                task_id_list.append(progress["task_id"])
                try:
                    callback(progress["task_id"], progress)
                except Exception as e:
                    callback(
                        progress["task_id"], {"status": "error", "message": str(e)}
                    )
                    raise e
        return start_results

    def stop_task(self, task_id, call_callbacks=False):
        current_result = self.get_task_result(task_id)
        if current_result["status"] in ("error", "complete"):
            return

        print("Stopping", task_id, current_result)
        current_result = {"status": "stopped", "task_id": task_id}
        with open(
            os.path.join(self.task_directory, f"{task_id}", "output.json"), "w"
        ) as file:
            json.dump(current_result, file)

        if task_id in self.running_processes:
            self.running_processes[task_id].terminate()
            del self.running_processes[task_id]
        else:
            queued_process_idx = next(
                (i for i, (t, p) in enumerate(self.queued_processes) if t == task_id),
                None,
            )
            if queued_process_idx is not None:
                del self.queued_processes[queued_process_idx]
            else:
                print(
                    f"Unable to terminate task {task_id}, not found in running or queued processes"
                )
        # self.scheduler.remove_job(id=task_id)
        if call_callbacks:
            for fn in self.callback_functions[task_id]:
                try:
                    fn(task_id, current_result)
                except Exception as e:
                    fn(task_id, {"status": "error", "message": str(e)})
                    raise e
        del self.callback_functions[task_id]

    def poll_task_results(self):
        old_callbacks = {k: v for k, v in self.callback_functions.items()}
        for task_id, callback_fns in old_callbacks.items():
            try:
                current_results = self.get_task_result(task_id)
            except JSONDecodeError:
                print("Decode error (JSON file is likely being concurrently written)")
                continue

            for fn in callback_fns:
                try:
                    fn(task_id, current_results)
                except Exception as e:
                    fn(task_id, {"status": "error", "message": str(e)})
                    raise e

            if current_results["status"] in ("complete", "error", "stopped"):
                print("Removing task", task_id)
                del self.callback_functions[task_id]
                if task_id in self.running_processes:
                    print("Removing running process")
                    del self.running_processes[task_id]
        # Start processes that were waiting
        if self.queued_processes:
            while (
                len(self.running_processes) < self.max_workers and self.queued_processes
            ):
                id, process = self.queued_processes.pop(0)
                print("Starting process", id)
                process.start()
                self.running_processes[id] = process

    def __del__(self):
        print("Killing processes")
        if self.scheduler is not None:
            self.scheduler.remove_job(self.poll_task)
            # for task_id in self.callback_functions:
            #     self.scheduler.remove_job(task_id)

        for p in self.running_processes:
            p.terminate()
            p.close()
        self.running_processes = {}
        self.queued_processes = []
