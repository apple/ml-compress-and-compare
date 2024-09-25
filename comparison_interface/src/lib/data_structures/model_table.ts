/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2024 Apple Inc. All Rights Reserved.
 */

import { distinctObjects } from "$lib/utils";

export enum OperationVariableType {
  parameter = 1,
  presence = 2,
  operation = 5,
  operationSequence = 6,
}
export class OperationVariable {
  type: OperationVariableType;
  parameterName?: string | null;
  operationName?: string | null;

  constructor(
    type: OperationVariableType,
    operationName: string | null | undefined = undefined,
    parameterName: string | null | undefined = undefined
  ) {
    this.type = type;
    this.parameterName = parameterName;
    this.operationName = operationName;
  }

  equals(other: OperationVariable): boolean {
    return (
      other.type == this.type &&
      other.operationName == this.operationName &&
      other.parameterName == this.parameterName
    );
  }

  toString(): string {
    if (this.type == OperationVariableType.parameter)
      return `${this.parameterName}`;
    else if (this.type == OperationVariableType.presence)
      return `${this.operationName}`;
    return "";
  }
}

export type OperationVariableValue = {
  variable: OperationVariable;
  value: any;
};

/**
 * Represents a tabular dataset where the rows are models, and the columns are
 * "variables" (changes to model structure enacted through compression operations).
 * ModelTable instances are designed to be immutable.
 */
export class ModelTable {
  models: { id: string; values: any[] }[];
  _modelIDMap: Map<string, number>;
  flags: Map<string, Map<string, any>>;
  variables: OperationVariable[];
  size: number;
  numVariables: number;
  orderings: Map<number, any[]>;

  constructor(
    models: { id: string; values: any[] }[] | null = null,
    flags: Map<string, Map<string, any>> | null = null,
    variables: OperationVariable[] | null = null,
    orderings: Map<number, any[]> | null = null
  ) {
    this.models = models || [];
    this.flags = flags || new Map();
    this.variables = variables || [];
    this._modelIDMap = new Map(this.models.map((m, i) => [m.id, i]));
    this.size = this.models.length;
    this.numVariables = this.variables.length;
    this.orderings = orderings || new Map();
  }

  has(modelID: string): boolean {
    return this._modelIDMap.has(modelID);
  }

  modelIDs(): string[] {
    return [...this._modelIDMap.keys()];
  }

  getModelValues(modelID: string): any[] {
    if (!this.has(modelID))
      console.error(
        `Can't get values for model ID ${modelID} because it is not present in the model table`
      );
    return [...this.models[this._modelIDMap.get(modelID)!].values];
  }

  getVariableIndexValues(index: number): any[] {
    if (index < 0) console.error(`Variable index ${index} out of bounds`);
    return this.models.map((modelInfo) => modelInfo.values[index]);
  }

  getFlag(modelID: string, flagName: string): any {
    if (!this.has(modelID))
      console.error(
        `Can't get flag for model ID ${modelID} because it is not present in the model table`
      );
    if (!this.flags.has(flagName)) return undefined;
    return this.flags.get(flagName)!.get(modelID);
  }

  distinctValues(variableIndex: number): any[] {
    return distinctObjects(this.getVariableIndexValues(variableIndex));
  }

  filterModels(
    predicate: (
      modelInfo: { id: string; values: any[] },
      flags: Map<string, any>
    ) => boolean
  ): ModelTable {
    let newModels = this.models.filter((modelInfo, i) => {
      let flags = new Map(
        [...this.flags.entries()].map((entry) => [
          entry[0],
          entry[1].get(modelInfo.id),
        ])
      );
      return predicate(modelInfo, flags);
    });
    let newModelIDs = new Set(newModels.map((m) => m.id));
    return new ModelTable(
      newModels,
      new Map(
        [...this.flags.entries()].map(([flagName, flags]) => [
          flagName,
          new Map(
            [...flags.entries()].filter((entry) => newModelIDs.has(entry[0]))
          ),
        ])
      ),
      this.variables,
      this.orderings
    );
  }

  filterVariables(
    predicate: (variable: OperationVariable, index: number) => boolean
  ): ModelTable {
    let variableMask = this.variables.map((variable, i) =>
      predicate(variable, i)
    );
    return new ModelTable(
      this.models.map(({ id, values }) => ({
        id,
        values: values.filter((v, i) => variableMask[i]),
      })),
      this.flags,
      this.variables.filter((v, i) => variableMask[i]),
      new Map(
        [...this.orderings.entries()].filter(
          ([position, _]) => variableMask[position]
        )
      )
    );
  }

  withFlags(flagName: string, value: ((id: string) => any) | any): ModelTable {
    let valueMap: Map<string, any>;
    if (typeof value === "function") {
      let valueFn = value as (id: string) => any;
      valueMap = new Map(
        this.models.map((modelInfo) => [modelInfo.id, valueFn(modelInfo.id)])
      );
    } else
      valueMap = new Map(this.models.map((modelInfo) => [modelInfo.id, value]));

    return new ModelTable(
      [...this.models],
      new Map([...this.flags.entries(), [flagName, valueMap]]),
      [...this.variables],
      new Map([...this.orderings.entries()])
    );
  }

  withModels(
    newModels: { id: string; values: any[] }[],
    flags: { [key: string]: any } | null = null
  ): ModelTable {
    newModels = newModels.filter((modelInfo) => {
      if (this.has(modelInfo.id)) {
        console.error(
          `Can't add existing model ${modelInfo.id} to model table`
        );
        return false;
      }
      if (modelInfo.values.length != this.variables.length) {
        console.error(
          `Model ${modelInfo.id} should have ${this.variables.length}, but it has ${modelInfo.values.length}`
        );
        return false;
      }
      return true;
    });

    let newFlags = new Map(this.flags.entries());
    if (flags != null) {
      // Merge flags together
      Object.keys(flags).forEach((flagName) => {
        let flagMap: Map<string, any> = newFlags.has(flagName)
          ? new Map(newFlags.get(flagName)!.entries())
          : new Map();
        if (typeof flags[flagName] === "function") {
          newModels.forEach((modelInfo) =>
            flagMap.set(modelInfo.id, flags[flagName](modelInfo.id))
          );
        } else {
          newModels.forEach((modelInfo) =>
            flagMap.set(modelInfo.id, flags[flagName])
          );
        }
        newFlags.set(flagName, flagMap);
      });
    }
    return new ModelTable(
      [...this.models, ...newModels],
      newFlags,
      [...this.variables],
      new Map([...this.orderings.entries()])
    );
  }

  withVariable(
    newVariable: OperationVariable,
    value: ((id: string) => any) | any,
    index: number | null = null,
    ordering: any[] | null = null
  ): ModelTable {
    let valueList: any[];
    if (typeof value === "function") {
      let valueFn = value as (id: string) => any;
      valueList = this.models.map((modelInfo) => valueFn(modelInfo.id));
    } else valueList = this.models.map((_) => value);

    if (valueList.length != this.models.length)
      console.error("length of values must match number of models in table");

    if (index == null) index = this.variables.length;

    let newOrdering = new Map([...this.orderings.entries()]);
    if (ordering != null) {
      newOrdering.set(index, ordering);
    }
    return new ModelTable(
      this.models.map((modelInfo, i) => ({
        id: modelInfo.id,
        values: [
          ...modelInfo.values.slice(0, index!),
          valueList[i],
          ...modelInfo.values.slice(index!),
        ],
      })),
      new Map(this.flags.entries()),
      [
        ...this.variables.slice(0, index!),
        newVariable,
        ...this.variables.slice(index!),
      ],
      newOrdering
    );
  }

  withVariables(
    variableValues: {
      variable: OperationVariable;
      value: ((id: string) => any) | any;
      ordering?: any[];
    }[]
  ): ModelTable {
    let result = this as ModelTable;
    variableValues.forEach(({ variable, value, ordering }) => {
      result = result.withVariable(
        variable,
        value,
        result.numVariables,
        ordering || null
      );
    });
    return result;
  }

  removingVariables(variableIndexes: number[]): ModelTable {
    return new ModelTable(
      this.models.map((modelInfo) => ({
        id: modelInfo.id,
        values: modelInfo.values.filter((v, i) => !variableIndexes.includes(i)),
      })),
      new Map(this.flags.entries()),
      this.variables.filter((v, i) => !variableIndexes.includes(i)),
      new Map(
        [...this.orderings.entries()].filter(
          ([i, _]) => !variableIndexes.includes(i)
        )
      )
    );
  }
}
