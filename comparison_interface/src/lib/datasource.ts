/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2024 Apple Inc. All Rights Reserved.
 */

import { io } from "socket.io-client";

const CHANGE_EVENT_PREFIX = "change:";

function dec2hex(dec: number): string {
  return dec.toString(16).padStart(2, "0");
}

function _generateRandomID(len: number): string {
  var arr = new Uint8Array((len || 40) / 2);
  window.crypto.getRandomValues(arr);
  return Array.from(arr, dec2hex).join("");
}

export interface DataSource {
  connect(): void;
  disconnect(): void;
  onAttach(func: (model: DataSource) => void): void;
  onDetach(func: (model: DataSource, error: Error | null) => void): void;
  get(): any;

  fetch<ValueType>(name: string): Promise<ValueType>;

  set(name: string, value: any): void;

  // Do nothing - this is for compatibility with jupyter widget model
  save_changes(): void;

  on<ValueType>(
    eventName: string,
    func: (model: DataSource, val: ValueType) => void
  ): void;

  request(
    eventName: string,
    args: any,
    progressCallback?:
      | ((data: any, index: number, total: number, completed: boolean) => void)
      | null
  ): Promise<any>;
}

export class SocketDataSource implements DataSource {
  url: string | null = null;
  urlFunction: (() => string) | null = null;
  socket_: any;
  connected: boolean = false;
  attachListeners_: { func: any; runOnce: boolean }[] = [];
  detachListeners_: any = [];
  requestListeners_: Map<string, any> = new Map();
  nameListeners_: Map<string, any> = new Map();
  namespace: string = "";
  maxReconnectionAttempts: number = 1;
  retryCount: number = 0;
  sentDetachMessage: boolean = false;

  constructor(
    url: string | (() => string) = "",
    namespace = "/",
    maxReconnectionAttempts = 1
  ) {
    if (typeof url === "string") this.url = url;
    else this.urlFunction = url;
    this.namespace = namespace;
    this.maxReconnectionAttempts = maxReconnectionAttempts;
  }

  connect(reset = false) {
    this.sentDetachMessage = false;
    if (!!this.urlFunction) this.url = this.urlFunction();
    if (this.socket_ != null && !reset) {
      this.socket_.connect();
    } else {
      let hadOldSocket = this.socket_ != null;
      if (this.socket_ != null && this.socket_.connected) {
        this.socket_.disconnect();
        this.socket_ = null;
      }
      this.socket_ = io(this.url + this.namespace);
      this.socket_.on("connect", () => this._attached());
      this.socket_.on("disconnect", () => this._detached());
      this.socket_.on("connect_error", (error: Error) => this._detached(error));
      if (hadOldSocket) {
        // Add all the handlers from the previous socket instance
        console.log("restoring name listeners");
        Array.from(this.nameListeners_.entries()).forEach(([name, func]) => {
          console.log(name);
          this.socket_.on(name, (val: any) => func(this, val));
        });
      }
    }
  }

  disconnect() {
    this.socket_.disconnect();
  }

  _attached() {
    console.log("attached", this.socket_.io.engine.id);
    this.connected = true;
    this.retryCount = 0;
    this.attachListeners_.forEach(({ func }) => {
      func(this);
    });
    this.attachListeners_ = this.attachListeners_.filter(
      ({ runOnce }) => !runOnce
    );
  }

  _detached(error: any = null) {
    console.log("disconnection error:", error, this.retryCount);
    if (this.retryCount < this.maxReconnectionAttempts) {
      console.log(
        `attempting reconnect, try ${this.retryCount + 1} of ${
          this.maxReconnectionAttempts
        }`
      );
      this.connect();
      this.retryCount++;
    } else if (!this.sentDetachMessage) {
      console.log("detached");
      this.detachListeners_.forEach((func: any) => {
        func(this, error);
      });
      this.connected = false;
      this.sentDetachMessage = true;
    }
  }

  onAttach(func: (model: DataSource) => void, runOnce: boolean = false) {
    this.attachListeners_.push({ func, runOnce });
    if (this.connected) {
      func(this);
    }
  }

  onDetach(func: (model: DataSource, error: Error | null) => void) {
    this.detachListeners_.push(func);
  }

  removeListeners() {
    this.attachListeners_ = [];
    this.detachListeners_ = [];
    this.nameListeners_.clear();
    this.requestListeners_.clear();
  }

  // Doesn't support get() - this is for compatibility with jupyter widget model
  get() {
    return null;
  }

  fetch<ValueType>(name: string): Promise<ValueType> {
    return new Promise((resolve) => {
      this.socket_.emit("get:" + name, (data: any) => {
        resolve(data);
      });
    });
  }

  set(name: string, value: any) {
    if (!this.socket_) return;
    this.socket_.emit("set:" + name, value);

    // Check if there are other handlers for this name. We can update them
    // without even going through the socket
    if (this.nameListeners_.has(name))
      this.nameListeners_.get(name).forEach((f: any) => f(this, value));
  }

  // Do nothing - this is for compatibility with jupyter widget model
  save_changes() {}

  on<ValueType>(
    eventName: string,
    func: (model: DataSource, val: ValueType) => void
  ) {
    if (!this.socket_ && !this.connected) {
      this.onAttach(() => this.on(eventName, func), true);
      return;
    }
    if (eventName.startsWith(CHANGE_EVENT_PREFIX)) {
      let propName = eventName.slice(CHANGE_EVENT_PREFIX.length);
      if (!this.nameListeners_.has(propName))
        this.nameListeners_.set(propName, []);
      this.nameListeners_.get(propName).push(func);
      console.log("listening for", eventName);
      this.socket_.on(eventName, (val: any) => func(this, val));
    } else {
      console.error(
        `Tried to register an unsupported event '${eventName}' with SocketModel. Only events starting with '${CHANGE_EVENT_PREFIX}' are supported.`
      );
    }
  }

  /**
   * Sends a request to the server and listens for a response.
   *
   * The server should behave as follows: when it receives a 'request:{name}'
   * message, it can either return a batched or unbatched response, or an error.
   * In the case of an error, it should acknowledge the request: event with an
   * object containing an 'error' property. If the response is unbatched, the
   * acknowledgement should contain the response data. If the response is batched,
   * the server should emit one or more 'response:{name}' events whose bodies contain
   * five properties: the UID passed in the original request (uid), the current batch
   * of data (data), the current batch index (index), the total number
   * of batches (total - zero if the batching process is indeterminate), and whether
   * this batch is the last one (completed). The last batch must send a value of
   * `true` for completed.
   *
   * @param eventName name of the request endpoint
   * @param args arguments to pass to the server
   * @param progressCallback if the response is a batched result, this function
   *  will be repeatedly called with four arguments: the contents of the most
   *  recently received batch, the index of this batch, the total number of batches
   *  that will be sent, and a flag indicating whether the response is completed. The
   *  returned Promise will still be resolved at the end of the final sent batch, but
   *  the promise resolution will not include data. NOTE: if the batching process on the
   *  server side cannot determine how many batches will be sent, the `total` value
   *  will be zero.
   * @returns a Promise that is resolved when the server emits a response event.
   */
  request(
    eventName: string,
    args: any = null,
    progressCallback:
      | ((data: any, index: number, total: number, completed: boolean) => void)
      | null = null
  ) {
    if (args instanceof Function) {
      progressCallback = args;
      args = null;
    }

    let uid = _generateRandomID(20);

    return new Promise((resolve, reject) => {
      const asyncListener = (batch: any) => {
        console.log("received asynchronous response for", eventName, batch);
        if (batch.uid != uid) return;
        this.socket_.off("response:" + eventName, asyncListener);
        if (batch.error) reject(batch.error);
        else if (batch.data && batch.data.error) reject(batch.data.error);
        else resolve(batch.data);
      };
      this.socket_.on("response:" + eventName, asyncListener);

      this.socket_.emit("request:" + eventName, { uid, args }, (data: any) => {
        if (data.error) {
          reject(data.error);
          this.socket_.off("response:" + eventName, asyncListener);
        } else if (data.asynchronous) {
          console.log("expecting asynchronous response for", eventName, uid);
          if (!!progressCallback)
            console.error(
              `The server did not send a batched response for event ${eventName}, but a progress callback was provided`
            );
          console.log(
            "callbacks:",
            this.socket_.listeners("response:" + eventName)
          );
        } else if (data.batched) {
          if (!progressCallback)
            console.error(
              `The server sent a batched response for event ${eventName}, but no progress callback was provided`
            );
          this.socket_.on("response:" + eventName, (batch: any) => {
            if (batch.uid != uid) return;
            progressCallback!(
              batch.data,
              batch.index,
              batch.total,
              batch.completed
            );
            if (batch.completed) resolve(null);
          });
          this.socket_.off("response:" + eventName, asyncListener);
        } else {
          if (!!progressCallback)
            console.error(
              `The server did not send a batched response for event ${eventName}, but a progress callback was provided`
            );
          resolve(data.result);
          this.socket_.off("response:" + eventName, asyncListener);
        }
      });
    });
  }
}

export function hasModelServerURL(): boolean {
  return window.localStorage.getItem("modelServerURL") != null;
}

export function getModelServerURL(): string {
  let storedURL = window.localStorage.getItem("modelServerURL");
  if (!storedURL) return "http://localhost:5001";
  return storedURL;
}

export function setModelServerURL(newURL: string) {
  window.localStorage.setItem("modelServerURL", newURL);
}
