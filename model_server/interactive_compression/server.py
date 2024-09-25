"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

from flask import Flask, render_template, request
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
from flask_apscheduler import APScheduler
from engineio.payload import Payload
import pyarrow as pa
from .sockets import SocketConnector
from .components import LayerDetailComponent, InstanceDetailComponent, ModelMapComponent
from .monitors import LocalModelMonitor
import atexit

Payload.max_decode_packets = 200

app = None
socketio = None
socket_connector = None
scheduler = None


class SchedulerConfig:
    SCHEDULER_EXECUTORS = {
        "default": {"type": "threadpool", "max_workers": 20},
        "poll": {"type": "threadpool", "max_workers": 20},
    }


def start_flask_server(model_info, task_runner=None, port=5001, debug=True):
    """
    Starts a server on localhost at the given port, which can provide information
    about the given models. This method will run continuously once called.

    :param model_info: A model monitor object, or a dictionary conforming to the model
        info structure. It should contain the following keys:
        *operations*: A list of JSON objects describing compression operations
            that are used by the models. Each compression operation should have
            a unique `name` and an object containing `parameters` to that
            operation.
        *metrics*: A list of JSON objects defining metrics by which each model
            is evaluated, such as accuracy, model size, etc. Each object should
            have a `name` and an optional `range` object with optional `min`
            and `max` fields. The optional `primary` field can be `true` to indicate
            that the metric should be shown on the sidebar and tooltips.
        *models*: A list of JSON-style objects for each model to be visualized.
            Each model can have the following properties:
            * `id` - string identifier (required)
            * `tag` -  optional human-readable `tag`
            * `base` - optional string reference to a model ID, denoting that
                the model is derived from the given base
            * `operation` - optional operation that was applied, formatted as an
                object with a `name` and `parameters` corresponding to the name
                and parameters defined in the *operations* top-level field
            * `metrics` - an optional object keyed by metric names and where
                the values are the model's value for that metric
    :param task_runner: An object that has a run_task method, which should take a
        dictionary of task parameters and either return the value of the task
        result or start a background task and return an identifier for the task. It
        should also define a check_task method that polls for results from the task,
        and returns the results if they are present or an object representing the
        task progress.
    :param port: Port on localhost at which to serve the model server.
    :param debug: Debug flag passed to socket.io.
    """
    if not debug:
        import eventlet

        eventlet.monkey_patch(thread=True, time=True)

    global socketio, app, socket_connector, scheduler
    assert (
        socketio is None and app is None
    ), "A flask server is already running - shut it down before starting a new one"

    app = Flask(__name__)
    app.config.from_object(SchedulerConfig())
    app.config["SECRET_KEY"] = "b2c23fc5d6f62449cebbf60d6ebd1ccec31ea6e9"
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="eventlet" if not debug else "threading",
    )

    def make_components():
        if isinstance(model_info, LocalModelMonitor):
            model_info.load_metadata()
            info = model_info.make_info_dictionary()
        else:
            info = model_info

        model_list = info["models"]
        return [
            {
                "namespace": "model_detail",
                "component": LayerDetailComponent(model_list, task_runner),
            },
            {"namespace": "model_map", "component": ModelMapComponent(info)},
            {
                "namespace": "instance_detail",
                "component": InstanceDetailComponent(model_list, task_runner),
            },
        ]

    socket_connector = SocketConnector(
        app,
        socketio,
        make_components,
    )

    if task_runner is not None:
        scheduler = APScheduler()
        scheduler.init_app(app)
        scheduler.start()

        task_runner.set_scheduler(scheduler)
        atexit.register(lambda: scheduler.shutdown())

    print(f"Start flask server at port {port}, debug={debug}")
    
    if debug:
        socketio.run(
            app,
            port=port,
            debug=debug,
            use_reloader=debug,
            use_debugger=debug,
            exclude_patterns=["*.cache*"],
        )
    else:
        socketio.run(
            app,
            host="0.0.0.0",
            port=port,
            debug=False,
            use_reloader=False,
        )


def stop_flask_server():
    global socketio, app, socket_connector, scheduler
    if scheduler is not None:
        scheduler.shutdown()
        scheduler = None
    socketio.stop()
    app.stop()
    app = None
    socketio = None
    socket_connector = None
