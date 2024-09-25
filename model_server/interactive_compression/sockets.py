"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

from flask import Flask, render_template, request
from flask_socketio import SocketIO, send, emit

import time
import datetime
import traitlets
import functools
import inspect
from collections import namedtuple


class _SocketRequest:
    def __init__(self, endpoint, func, asynchronous=False):
        self.endpoint = endpoint
        self.func = func
        self.endpoint_owner = None
        self.asynchronous = asynchronous

    def __call__(self, *args, **kwargs):
        return self.func(self.endpoint_owner, *args, **kwargs)


_SocketRequestResponse = namedtuple(
    "_SocketRequestResponse", ("data_or_generator", "batched")
)


class bound_method(object):
    """
    This decorator should be applied BEFORE socket_route for endpoints
    that are declared within a class and require the use of self.
    """

    def __init__(self, f):
        self.func = f

    def __call__(self, *args, **kwargs):
        ret = self.func(*args, **kwargs)
        return ret

    def __get__(self, instance, owner):
        self.func.endpoint_owner = instance
        return self.func


class SocketRequestable:
    """
    Instances of this class can have methods decorated with the
    @socket_route() decorator, which will be processed by the
    SocketConnector as request/response messages.
    """

    @property
    def request_endpoints(self):
        return [
            getattr(self, name)
            for name in dir(self)
            if name != "request_endpoints"
            and isinstance(getattr(self, name), _SocketRequest)
        ]


def socket_route(endpoint_name, batched=False):
    """
    Decorator that can be used to mark a method as handling a named
    request from the client, essentially reproducing a traditional
    request/response model within Socket.IO with optional batching. If
    batched is set to True, the wrapped function should return an
    iterable
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # If batched is True, func should return a generator. Otherwise,
            # it should return the data to provide in the response
            results = func(*args, **kwargs)
            if not batched:
                assert not inspect.isgenerator(
                    results
                ) and not inspect.isgeneratorfunction(
                    results
                ), "Results should not be a generator for socket route not marked as batched"
            return _SocketRequestResponse(results, batched)

        return _SocketRequest(endpoint_name, wrapper)

    return decorator


def socket_async_route(endpoint_name):
    """
    Decorator that can be used to mark a method as handling a named
    request from the client, similar to socket_route but with a
    callback function to send the response instead of a direct
    return value from the function. The decorated function's first
    argument should be a callback function that the function calls
    with the desired response as the first argument.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)

        return _SocketRequest(endpoint_name, wrapper, asynchronous=True)

    return decorator


class SocketConnector:
    user_data = {}

    def __init__(self, app, socketio, component_factory):
        """
        component_factory should be a function that takes no arguments and
        returns a list of dictionaries, where each dictionary
        can contain the following keys: namespace, component (an
        object inheriting from HasTraits and/or SocketRequestable), and
        subcomponents (a list of dictionaries in identical format).
        """
        self.component_factory = component_factory
        self.app = app
        self.socketio = socketio
        for comp in self.component_factory():
            self.connect_events(comp, only_connect_events=True)

    def connect(self):
        print("connected", request.sid)
        if request.sid not in SocketConnector.user_data:
            components = self.component_factory()
            # Set up user data
            SocketConnector.user_data[request.sid] = {
                "components": components,
                "dt": datetime.datetime.now(),
                "locks": {},
            }
        else:
            components = SocketConnector.user_data[request.sid]["components"]
        for comp in components:
            self.connect_events(comp)

    def connect_events(
        self, component_dict, base_namespace="", only_connect_events=False
    ):
        namespace = base_namespace + "/" + component_dict["namespace"]
        if only_connect_events:
            self.socketio.on_event("connect", self.connect, namespace=namespace)
            self.socketio.on_event("disconnect", self.disconnect, namespace=namespace)
        elif "component" in component_dict:
            comp = component_dict["component"]
            if not only_connect_events:
                # Tell the component that a socket session is started
                if hasattr(comp, "connect"):
                    comp.connect()
            if isinstance(comp, traitlets.HasTraits):
                for trait_name in comp.trait_names(sync=lambda x: x):
                    # Register callbacks for getting and setting from frontend
                    self.socketio.on_event(
                        "get:" + trait_name,
                        self._read_value_handler(comp, trait_name),
                        namespace=namespace,
                    )
                    self.socketio.on_event(
                        "set:" + trait_name,
                        self._write_value_handler(comp, trait_name),
                        namespace=namespace,
                    )

                    # Emit responses when backend state changes
                    comp.observe(
                        self._emit_value_handler(trait_name, request.sid, namespace),
                        trait_name,
                    )
            if isinstance(comp, SocketRequestable):
                for requester in comp.request_endpoints:
                    self.socketio.on_event(
                        "request:" + requester.endpoint,
                        self._request_handler(
                            requester.endpoint,
                            requester,
                            request.sid,
                            namespace=namespace,
                        ),
                        namespace=namespace,
                    )
        if "subcomponents" in component_dict:
            for subcomp in component_dict["subcomponents"]:
                self.connect_events(
                    subcomp,
                    base_namespace=namespace,
                    only_connect_events=only_connect_events,
                )

    def disconnect(self):
        print("disconnected", request.sid)
        if request.sid in SocketConnector.user_data:
            del SocketConnector.user_data[request.sid]

    def _read_value_handler(self, component, name):
        def handle_msg():
            if request.sid not in SocketConnector.user_data:
                print("Missing request SID:", request.sid)
                return None
            session_data = SocketConnector.user_data[request.sid]
            session_data["dt"] = datetime.datetime.now()
            return getattr(component, name)

        return handle_msg

    def _write_value_handler(self, component, name):
        def handle_msg(data):
            if request.sid not in SocketConnector.user_data:
                print("Missing request SID:", request.sid)
                return
            session_data = SocketConnector.user_data[request.sid]
            session_data["dt"] = datetime.datetime.now()
            # Set a lock on this value so that if the widget emits a change for it,
            # we do not redundantly send the client a change message
            session_data["locks"][name] = data
            try:
                setattr(component, name, data)
            except Exception as e:
                raise e
            finally:
                del session_data["locks"][name]

        return handle_msg

    def _emit_value_handler(self, name, sid, namespace="/"):
        def handle_msg(change):
            if sid not in SocketConnector.user_data:
                print("Missing request SID:", sid)
                return
            with self.app.app_context():
                session_data = SocketConnector.user_data[sid]
                if (
                    name not in session_data["locks"]
                    or session_data["locks"][name] != change.new
                ):
                    emit("change:" + name, change.new, room=sid, namespace=namespace)

        return handle_msg

    def _request_response_batcher(self, uid, event_name, sid, namespace, generator):
        if hasattr(generator, "__len__"):
            # All elements already exist - we can show determinate progress
            all_results = list(generator)
            for i, result in enumerate(all_results):
                with self.app.app_context():
                    emit(
                        "response:" + event_name,
                        {
                            "uid": uid,
                            "data": result,
                            "index": i,
                            "total": len(all_results),
                            "completed": i == len(all_results) - 1,
                        },
                        room=sid,
                        namespace=namespace,
                    )
                time.sleep(BATCHING_SLEEP_TIME)
        else:
            # All elements haven't been generated yet - don't show progress
            all_results = list(generator)
            last_result = None
            result_index = 0
            for result in all_results:
                if last_result is not None:
                    with self.app.app_context():
                        emit(
                            "response:" + event_name,
                            {
                                "uid": uid,
                                "data": last_result,
                                "index": result_index,
                                "total": 0,
                                "completed": False,
                            },
                            room=sid,
                            namespace=namespace,
                        )
                        result_index += 1
                    self.socketio.sleep(0)
                last_result = result
            if last_result is not None:
                with self.app.app_context():
                    emit(
                        "response:" + event_name,
                        {
                            "uid": uid,
                            "data": last_result,
                            "index": result_index,
                            "total": 0,
                            "completed": True,  # mark the process as done
                        },
                        room=sid,
                        namespace=namespace,
                    )

    def _request_handler(self, name, func, sid, namespace="/"):
        def handle_msg(data):
            uid = data[
                "uid"
            ]  # keep track of who requested this so we can return the id when sending results
            if hasattr(func, "asynchronous") and func.asynchronous:

                def callback(response):
                    self.socketio.sleep(0)
                    print("Sending asynchronous callback", uid, sid)
                    with self.app.app_context():
                        emit(
                            "response:" + name,
                            {"uid": uid, "data": response},
                            room=sid,
                            namespace=namespace,
                        )

                func(callback, data["args"])
                return {"asynchronous": True}
            else:
                result = func(data["args"])
                if result.batched:
                    self.socketio.start_background_task(
                        self._request_response_batcher,
                        uid,
                        name,
                        sid,
                        namespace,
                        result.data_or_generator,
                    )
                    return {"batched": True}
                else:
                    return {"batched": False, "result": result.data_or_generator}

        return handle_msg
