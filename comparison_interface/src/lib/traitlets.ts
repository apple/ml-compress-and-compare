/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2024 Apple Inc. All Rights Reserved.
 */

import type { Writable } from "svelte/store";
import { writable } from "svelte/store";
import type { DataSource } from "./datasource";

export interface Traitlet<T> extends Writable<T> {}

export interface TraitletOptions<LocalType, RemoteType> {
  inConverter?: ((x: RemoteType) => LocalType) | null;
  outConverter?: ((x: LocalType) => RemoteType) | null;
  fetchInitial?: boolean;
}

export function traitlet<LocalType, RemoteType = LocalType>(
  model: any,
  name_: string,
  defaultVal: LocalType,
  options: TraitletOptions<LocalType, RemoteType> | null = null
): Traitlet<LocalType> {
  const name: string = name_;
  let receivedInitialValue = false;
  const curVal: Writable<LocalType> = writable(model.get(name) || defaultVal);
  const inConverter =
    (options || {}).inConverter ||
    ((x: RemoteType): LocalType => x as unknown as LocalType);
  const outConverter =
    (options || {}).outConverter ||
    ((x: LocalType): RemoteType => x as unknown as RemoteType);

  model.on(
    "change:" + name,
    (model: DataSource, val: RemoteType) => {
      console.log("change:", name, val);
      curVal.set(inConverter(val));
    },
    null
  );

  if (!!model.onAttach)
    model.onAttach(async () => {
      let v = inConverter(await model.fetch(name));
      if (
        !receivedInitialValue &&
        (options?.fetchInitial == undefined || options.fetchInitial == true)
      )
        curVal.set(v);
      receivedInitialValue = true;
    });

  return {
    set: (v: LocalType) => {
      curVal.set(v);
      model.set(name, outConverter(v));
      model.save_changes();
    },
    subscribe: curVal.subscribe,
    update: (func: any) => {
      curVal.update((v: any) => {
        let out = func(v);
        model.set(name, outConverter(out));
        model.save_changes();
        return out;
      });
    },
  };
}
