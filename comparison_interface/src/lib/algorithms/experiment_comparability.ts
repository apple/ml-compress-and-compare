/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2024 Apple Inc. All Rights Reserved.
 */

import {
  TreeNode,
  modelListToTrees,
  type TreeNodeCollection,
  BaseOperationName,
  BaseParentName,
} from "$lib/algorithms/map_layout";
import { distinctObjects, objectsEqual } from "$lib/utils";
import {
  operationToString,
  type CompressionOperation,
  type Model,
} from "$lib/data_structures/model_map";
import {
  OperationVariableType,
  OperationVariable,
  ModelTable,
} from "$lib/data_structures/model_table";

/**
 * Computes the list of variables for the given operations, all of which
 * are the same operation (name) but have different parameters.
 */
function _variablesForOperations(
  operations: (CompressionOperation | null)[]
): OperationVariable[] {
  let opName = operations[0] != null ? operations[0].name : null;
  if (!operations.every((op) => (op != null ? op!.name : null) == opName)) {
    console.error(
      "Cannot use _variablesForOperations if not all operations have the same name"
    );
    return [];
  }

  let parameterValues: Map<string, Set<any>> = new Map();
  operations.forEach((op) => {
    if (!op) return;
    Object.entries(op.parameters).forEach((param) => {
      if (!parameterValues.has(param[0]))
        parameterValues.set(param[0], new Set());
      parameterValues.get(param[0])!.add(param[1]);
    });
  });

  let variablesToAdd: OperationVariable[] = [];
  Array.from(parameterValues.entries()).forEach((paramSet) => {
    if (paramSet[1].size > 1) {
      variablesToAdd.push(
        new OperationVariable(
          OperationVariableType.parameter,
          opName,
          paramSet[0]
        )
      );
    }
  });

  return variablesToAdd;
}

function deduplicatedOperationName(
  graph: TreeNodeCollection,
  models: Map<string, Model>,
  node: TreeNode,
  opName: string
): string {
  // Count number of operations in the node's ancestors that match the opName
  let numMatchingOperations = 0;
  let curr: TreeNode | null | undefined = node;
  while (curr != null) {
    let model = models.get(curr.id);
    if (!!model && !!model.operation && model.operation.name == opName)
      numMatchingOperations += 1;
    curr = curr.parent != null ? graph.get(curr.parent) || null : null;
  }

  return numMatchingOperations <= 1
    ? opName
    : `${opName} ${numMatchingOperations}`;
}

function getOperationForModel(
  models: Map<string, Model>,
  id: string
): CompressionOperation | null {
  if (id == BaseParentName) return null;
  let op = models.get(id)!.operation;
  if (op != null) return op;
  return { name: BaseOperationName, parameters: { Base: id } };
}

function computePossibleDiffsForChildren(
  graph: TreeNodeCollection,
  models: Map<string, Model>,
  pointers: ModelTable,
  maxDistinctOperations: number = 3
): { newPointers: ModelTable }[] {
  let childNodesAndOperations = pointers
    .modelIDs()
    .filter((id) => !pointers.getFlag(id, "explored"))
    .map((id) => {
      let node = graph.get(id)!;
      let children = node.children;
      if (children.length == 0) return [];
      return children
        .filter(
          (childID) =>
            graph.has(childID) && models.has(childID) && !pointers.has(childID)
        )
        .map((childID) => [
          graph.get(childID)!,
          getOperationForModel(models, childID),
        ]);
    })
    .flat() as [TreeNode, CompressionOperation | null][];
  let childrenOperations = childNodesAndOperations.map(([node, op]) => op);

  console.log("pointers:", pointers);
  console.log("operations:", childrenOperations);

  if (childrenOperations.length == 0) return [];

  let candidates: {
    newPointers: ModelTable;
  }[] = [];

  // Simple cases: all identical, all same operation name but different parameters
  let opNames = new Set(
    childNodesAndOperations.map(([node, op]) =>
      op != null
        ? deduplicatedOperationName(graph, models, node, op.name)
        : null
    )
  );
  if (opNames.size == 1) {
    let opName = Array.from(opNames.values())[0]!;
    if (opName == null) return [];

    let variablesToAdd = _variablesForOperations(childrenOperations);

    // Move all pointers to children
    let newPointers = pointers
      .withVariables([
        {
          variable: new OperationVariable(
            OperationVariableType.presence,
            opName
          ),
          value: false,
        },
        ...variablesToAdd.map((variable) => ({
          variable,
          value: "None",
        })),
      ])
      .withFlags("explored", true)
      .withModels(
        pointers
          .filterModels((_, flags) => !flags.get("explored"))
          .modelIDs()
          .map((pointerID) => {
            let node = graph.get(pointerID)!;

            return node.children
              .filter(
                (id) => graph.has(id) && models.has(id) && !pointers.has(id)
              )
              .map((id) => ({
                id,
                values: [
                  ...pointers.getModelValues(pointerID),
                  true, // operation presence
                  ...variablesToAdd.map(
                    // operation parameters
                    (variable) =>
                      getOperationForModel(models, id)!.parameters[
                        variable.parameterName!
                      ]
                  ),
                ],
              }));
          })
          .flat()
      );

    if (newPointers.size != pointers.size) candidates.push({ newPointers });
  } else {
    // General cases: different operation names
    let distinctOperations = distinctObjects(childrenOperations);

    // Try adding all possible operations and advancing to all children
    let differentOpPointers = pointers
      .withVariable(
        new OperationVariable(OperationVariableType.operation),
        "None"
      )
      .withFlags("explored", true)
      .withModels(
        pointers
          .filterModels((_, flags) => !flags.get("explored"))
          .modelIDs()
          .map((pointerID) => {
            let node = graph.get(pointerID)!;

            return node.children
              .filter(
                (id) => graph.has(id) && models.has(id) && !pointers.has(id)
              )
              .map((id) => ({
                id,
                values: [
                  ...pointers.getModelValues(pointerID),
                  operationToString(getOperationForModel(models, id)!, false),
                ],
              }));
          })
          .flat()
      );

    if (differentOpPointers.size != pointers.size)
      candidates.push({ newPointers: differentOpPointers });

    // Also try adding each individual operation and advancing only the pointers for the
    // nodes that have that operation
    let distinctOperationNames = new Set(
      distinctOperations.map((op) => (op != null ? op.name : null))
    );

    if (distinctOperationNames.size <= maxDistinctOperations) {
      distinctOperationNames.forEach((opName) => {
        if (opName == null) return;

        let matchingOperations = childrenOperations.filter(
          (childOp) =>
            (childOp == null && opName == null) ||
            (childOp != null && opName != null && childOp.name == opName)
        );
        if (matchingOperations.length == 0)
          console.error(
            "Missing matching operations:",
            childrenOperations,
            opName
          );
        let variablesToAdd = _variablesForOperations(matchingOperations);
        let absentOpPointers = pointers.withVariable(
          new OperationVariable(
            OperationVariableType.presence,
            matchingOperations.length == 1
              ? !!matchingOperations[0]
                ? operationToString(matchingOperations[0], false)
                : "Operation"
              : opName
          ),
          false
        );
        variablesToAdd.forEach(
          (variable) =>
            (absentOpPointers = absentOpPointers.withVariable(variable, "none"))
        );
        absentOpPointers = absentOpPointers
          // do not set explored to true for all previous nodes - we are only advancing for some children
          .withModels(
            pointers
              .filterModels((_, flags) => !flags.get("explored"))
              .modelIDs()
              .map((pointerID) => {
                let node = graph.get(pointerID)!;

                return node.children
                  .filter(
                    (id) => graph.has(id) && models.has(id) && !pointers.has(id)
                  )
                  .filter((id) =>
                    matchingOperations.some((op) =>
                      objectsEqual(getOperationForModel(models, id), op)
                    )
                  )
                  .map((id) => ({
                    id,
                    values: [
                      ...pointers.getModelValues(pointerID),
                      true,
                      ...variablesToAdd.map(
                        // operation parameters
                        (variable) =>
                          getOperationForModel(models, id)!.parameters[
                            variable.parameterName!
                          ]
                      ),
                    ],
                  }));
              })
              .flat()
          );

        if (absentOpPointers.size != pointers.size)
          candidates.push({ newPointers: absentOpPointers });
      });
    }
  }

  return candidates;
}

/**
 * Returns the subgraph of `graph` that contains all ancestors of nodes in `subset`.
 * @param graph
 * @param subset
 */
export function getAncestorGraph(
  graph: TreeNodeCollection,
  subset: string[]
): TreeNodeCollection {
  let subgraph: TreeNodeCollection = new Map();
  subset.forEach((id) => {
    let curr: TreeNode | null = graph.get(id) || null;
    while (curr != null) {
      subgraph.set(curr.id, curr);
      if (curr.parent == null) break;
      curr = graph.get(curr.parent) || null;
    }
  });
  return subgraph;
}

function computeShortestDiffPath(
  graph: TreeNodeCollection,
  models: Map<string, Model>,
  pointers: ModelTable | null = null
): ModelTable {
  if (pointers == null) {
    // find root of the current models
    let roots = Array.from(graph.keys()).map((nodeID) => {
      let curr = graph.get(nodeID);
      while (
        curr!.parent != null &&
        graph.has(curr!.parent!) &&
        (curr!.parent == BaseParentName ||
          models.has(graph.get(curr!.parent!)!.id))
      ) {
        curr = graph.get(curr!.parent);
      }
      return curr!;
    });
    console.log("roots:", roots);
    if (!roots.every((r) => r.id == roots[0].id))
      console.error("Not all models in the list have the same root");
    pointers = new ModelTable([{ id: roots[0].id, values: [] }]);
  }

  // Compare the children of the current pointers
  let candidates = computePossibleDiffsForChildren(graph, models, pointers);
  let candidateResults = candidates.map((candidate) => {
    let shortestPath = computeShortestDiffPath(
      graph,
      models,
      candidate.newPointers
    );
    return {
      pointers: shortestPath,
      maxLength: shortestPath.variables.reduce((a, b) => a + b.type, 0),
    };
  });

  if (candidateResults.length == 0) return pointers;

  if (candidateResults.length > 1) console.log("candidates:", candidateResults);
  // Find the candidate with the shortest max number of variables
  let bestCandidate = candidateResults.reduce(
    (prev, curr) => (prev.maxLength < curr.maxLength ? prev : curr),
    { pointers: null, maxLength: 1e9 } as {
      pointers: null;
      maxLength: number;
    }
  );
  return bestCandidate.pointers;
}

/**
 * Checks if the given variable value sets (which are assumed to all be boolean)
 * are cumulative. Cumulative means that if a preceding value is false, all successive
 * values in that row are also false, and if a value is true, all values preceding it
 * are true. The below is an example of a cumulative set of variables:
 *
 * [[ false, false, false],
 *  [ true, true, false ],
 *  [ true, true, true ],
 *  [ true, false, false ]]
 *
 */
function _areBooleanVariablesCumulative(variableValues: any[][]) {
  let testValue = false; // check that values in column otherIdx = testValue when values in column startIdx = testValue
  for (let startIdx = 0; startIdx < variableValues.length; startIdx++) {
    for (
      let otherIdx = startIdx + 1;
      otherIdx < variableValues.length;
      otherIdx++
    ) {
      let filteredValues = variableValues[otherIdx].filter(
        (v, i) => variableValues[startIdx][i] == testValue
      );
      if (!filteredValues.every((v) => v == testValue)) {
        console.log(startIdx, otherIdx, filteredValues);
        return false;
      }
    }
  }
  return true;
}

/**
 * Generates a string describing the values of the model with the given ID at the
 * variables at `positions`.
 *
 * @param table table containing variable values to summarize
 * @param id ID string of the model to summarize
 * @param positions indexes of the variable columns to combine
 * @param cumulative If true, the set is assumed to contain only
 *  boolean values. Only the last variable value that is true will
 *  be included in the description
 *
 * @returns a string description of the value of the variables for this model
 */
function _concatenateValuesForPositions(
  table: ModelTable,
  id: string,
  positions: number[],
  cumulative: boolean
): string {
  let originalValues = table.getModelValues(id);
  let valSequence = positions.map((i) => originalValues[i]);
  let stringVals = valSequence.map((val, i) => {
    let variable = table.variables[positions[i]];
    if (variable.type == OperationVariableType.presence) {
      return val ? variable.toString() : "";
    } else if (val == null || val == "None") return "";
    else return `${variable.toString()} ${val}`;
  });
  if (cumulative) {
    let lastTrueValue = valSequence.findLastIndex((v) => v);
    if (lastTrueValue < 0) stringVals = [];
    else stringVals = [stringVals[lastTrueValue]];
  }
  let combined = stringVals.filter((s) => s.length > 0).join(" & ");
  if (combined.length == 0) return "None";
  return combined;
}

/**
 * Provides an unambiguous ordering for the given cumulative variable
 * defined by the given variable indexes. Values are ordered such that
 * increasing index corresponds to more operations performed (true values).
 *
 * @param table
 * @param positions
 * @returns an array of value strings that can serve as a domain for a
 *  visual encoding
 */
function cumulativeVariableValueOrder(
  table: ModelTable,
  positions: number[]
): string[] {
  let sortedArrayValues = table.models.slice().sort((model1, model2) => {
    let originalValuesA = positions.map((i) => model1.values[i]);
    let originalValuesB = positions.map((i) => model2.values[i]);
    return (
      originalValuesA.reduce((a, b) => a + b, 0) -
      originalValuesB.reduce((a, b) => a + b, 0)
    );
  });
  return distinctObjects(
    sortedArrayValues.map((modelInfo) =>
      _concatenateValuesForPositions(table, modelInfo.id, positions, true)
    )
  );
}

// helper to simplify and combine multiple variables in a given set of positions
function _simplifyVariablesAtPositions(
  diffTable: ModelTable,
  positionsToAggregate: number[],
  {
    requireConsecutive = true,
    requireCumulative = false,
  }: { requireConsecutive?: boolean; requireCumulative?: boolean } = {}
): { table: ModelTable; insertionIndex: number } | null {
  if (
    requireConsecutive &&
    positionsToAggregate.reduce((a, b) => Math.max(a, b), -1e9) -
      positionsToAggregate.reduce((a, b) => Math.min(a, b), 1e9) !=
      positionsToAggregate.length - 1
  )
    return null;
  let isCumulative =
    positionsToAggregate.every(
      (p) => diffTable.variables[p].type == OperationVariableType.presence
    ) &&
    _areBooleanVariablesCumulative(
      positionsToAggregate.map((p) => diffTable.getVariableIndexValues(p))
    );
  if (requireCumulative && !isCumulative) return null;

  let newVariable = new OperationVariable(
    OperationVariableType.operationSequence,
    positionsToAggregate
      .map((p) => diffTable.variables[p].toString())
      .join(", ")
  );

  console.log(positionsToAggregate, newVariable);

  let insertionIndex = positionsToAggregate.reduce(
    (a, b) => Math.min(a, b),
    1e9
  );
  let newTable = diffTable
    .removingVariables(positionsToAggregate)
    .withVariable(
      newVariable,
      (id: string) =>
        _concatenateValuesForPositions(
          diffTable,
          id,
          positionsToAggregate,
          isCumulative
        ),
      insertionIndex,
      isCumulative
        ? cumulativeVariableValueOrder(diffTable, positionsToAggregate)
        : null
    );
  console.log("simplifying:", newTable);
  return { table: newTable, insertionIndex };
}

/**
 * Simplifies a set of variables by combining variables that imply each other. For example,
 * if we see the following table:
 *
 * Model  Var A  Var B
 *  1      false false
 *  2      true  false
 *  3      true  true
 *
 * we can see that when Var B is true, Var A is also true. This allows us to simplify
 * the two variables to one variable with three values: None, Var A, and Var A & Var B.
 *
 * @param diffTable The model table to simplify
 * @param maxAdditionalDistinctValues The maximum number of additional values that can
 *  be allowed for a set of variables. For example, if two variables are being simplified
 *  and maxAdditionalDistinctValues is 1, the maximum number of distinct combined values
 *  for the two variables is the sum of the distinct count for each individual variable
 *  plus one.
 * @param requireConsecutive If true, then require that the positions to be combined into
 *  one be consecutive in the original model table.
 * @param requireCumulative If true, then require that the positions only use boolean variables
 *  (presence/absence of operation) and be cumulative such that a preceding variable being false
 *  implies all successive variables are false.
 * @returns a simplified model table
 */
export function simplifyConstantValuePositions(
  diffTable: ModelTable,
  maxAdditionalDistinctValues = 1,
  maxGroupedVariables = 1e9,
  {
    requireConsecutive = true,
    requireCumulative = false,
  }: { requireConsecutive?: boolean; requireCumulative?: boolean } = {}
): ModelTable {
  let constantValuePositions: Map<
    number,
    Map<any, { position: number; value: any }[]>
  > = new Map();
  let positionValues = diffTable.variables.map((v, i) =>
    diffTable.distinctValues(i)
  );

  positionValues.forEach((values, i) => {
    // Check if filtering to one specific value also removes
    if (values.length <= 1) return;
    if (
      requireCumulative &&
      diffTable.variables[i].type != OperationVariableType.presence
    )
      return;

    values.forEach((val) => {
      let filteredTable = diffTable.filterModels((modelInfo) =>
        objectsEqual(modelInfo.values[i], val)
      );
      filteredTable.variables.forEach((otherVariable, j) => {
        let valSet = filteredTable.distinctValues(j);
        if (i > j && positionValues[j].length > 1 && valSet.length == 1) {
          // The filter on column i has led to only one distinct value in column j
          if (!constantValuePositions.has(i))
            constantValuePositions.set(i, new Map());
          if (!constantValuePositions.get(i)?.has(val))
            constantValuePositions.get(i)?.set(val, []);
          constantValuePositions
            .get(i)
            ?.get(val)
            ?.push({ position: j, value: valSet[0] });
        }
      });
    });
  });

  let sortedPositionSets = Array.from(constantValuePositions.entries()).sort(
    (a, b) =>
      Array.from(b[1].values()).reduce((prev, curr) => prev + curr.length, 0) -
      Array.from(a[1].values()).reduce((prev, curr) => prev + curr.length, 0)
  );
  if (sortedPositionSets.length == 0) return diffTable;

  for (
    let positionSetIdx = 0;
    positionSetIdx < sortedPositionSets.length;
    positionSetIdx++
  ) {
    let [position, valuePositions] = sortedPositionSets[0];

    console.log(position, valuePositions);

    let possibleAggregatePositions = [
      position,
      ...Array.from(valuePositions.values())
        .map((valueSet) => valueSet.map((v) => v.position).flat())
        .flat(),
    ];
    possibleAggregatePositions = Array.from(
      new Set(possibleAggregatePositions)
    ).sort();

    for (
      let startIdx = 0;
      startIdx <
      Math.max(1, possibleAggregatePositions.length - maxGroupedVariables);
      startIdx++
    ) {
      let positionsToAggregate = possibleAggregatePositions.slice(
        startIdx,
        startIdx + maxGroupedVariables
      );

      // Don't simplify variables that are already simplified
      if (
        positionsToAggregate.some(
          (p) =>
            diffTable.variables[p].type ==
            OperationVariableType.operationSequence
        )
      )
        continue;

      let newTable = _simplifyVariablesAtPositions(
        diffTable,
        positionsToAggregate,
        { requireConsecutive, requireCumulative }
      );
      if (!newTable) continue;
      if (
        newTable.table.distinctValues(newTable.insertionIndex).length >
        positionsToAggregate.length + maxAdditionalDistinctValues
      ) {
        console.log(
          "table doesn't reduce dimensions enough:",
          newTable,
          newTable.table.distinctValues(newTable.insertionIndex),
          positionsToAggregate.length
        );
        continue;
      }

      return simplifyConstantValuePositions(
        newTable.table,
        maxAdditionalDistinctValues
      );
    }
  }
  return diffTable;
}

/**
 * Combines pairs of variables for whom the total number of unique values is at most
 * maxUniqueValues.
 *
 * @param diffTable
 * @param maxUniqueValues
 * @param requireConsecutive
 */
export function simplifySmallDomainVariables(
  diffTable: ModelTable,
  {
    maxUniqueValues = 6,
    requireConsecutive = true,
  }: { requireConsecutive?: boolean; maxUniqueValues?: number } = {}
): ModelTable {
  let variablePair: [number, number] | null = null;
  for (let i = 0; i < diffTable.numVariables; i++) {
    for (
      let j = i + 1;
      j <
      Math.min(
        requireConsecutive ? i + 2 : diffTable.numVariables,
        diffTable.numVariables
      );
      j++
    ) {
      let numDistinct = distinctObjects(
        diffTable.models.map((modelInfo) =>
          _concatenateValuesForPositions(diffTable, modelInfo.id, [i, j], false)
        )
      ).length;
      if (numDistinct <= maxUniqueValues) {
        variablePair = [i, j];
        break;
      }
    }
    if (variablePair != null) break;
  }

  console.log("variable pair:", variablePair);
  if (variablePair == null) return diffTable;
  if (
    variablePair.some(
      (p) =>
        diffTable.variables[p].type == OperationVariableType.operationSequence
    )
  )
    return diffTable;

  let newTable = _simplifyVariablesAtPositions(diffTable, variablePair, {
    requireConsecutive,
    requireCumulative: false,
  });
  console.log("new table:", newTable);
  if (!newTable) return diffTable;

  return simplifySmallDomainVariables(newTable.table, {
    maxUniqueValues,
    requireConsecutive,
  });
}

/**
 * Simplifies the model table by converting all variables to one single
 * variable containing the full operation sequence for each model. This can
 * be used to display comparisons with a small number of experiments but a
 * large number of variables.
 *
 * @param diffTable the model table to flatten
 */
export function flattenExperimentTable(diffTable: ModelTable): ModelTable {
  let newVariable = new OperationVariable(
    OperationVariableType.operationSequence
  );
  let newModels = diffTable.models.map(({ id, values }) => {
    let operationStrings = values
      .map((v, i) => {
        let variableString = diffTable.variables[i].toString();
        return [variableString, v];
      })
      .filter(
        (v) => !!v[1] && !["none", "null"].includes(`${v[1]}`.toLowerCase())
      )
      .map((v) => {
        if (v[1] == true) return `${v[0]}`;
        else if (v[0].length > 0) return `${v[0]} ${v[1]}`;
        return `${v[1]}`;
      });
    return {
      id,
      values: [
        operationStrings.length > 0 ? operationStrings.join(" & ") : "None",
      ],
    };
  });
  console.log("new models:", newModels);
  return new ModelTable(newModels, null, [newVariable]);
}

/**
 * Computes a table that indicates how each of the models in the given
 * subset varies with respect to the operations performed on it.
 *
 * @param models A list of Model objects representing the full tree of
 *  experiments
 * @param subsetIDs The subset of model IDs to generate the table for
 * @returns A ModelTable instance where each row is a model, and the columns
 *  are variables which take on at least two distinct values in the given
 *  model subset.
 */
export function computeModelExperimentTable(
  models: Model[],
  subsetIDs: string[]
): ModelTable {
  // we use a model tree representation that includes a "base parent", which is the
  // parent of all base models. Then the base model will itself be an operation that
  // we can compare on
  let graph = getAncestorGraph(modelListToTrees(models, true), subsetIDs);
  let modelIDMap = new Map(models.map((m) => [m.id, m]));
  let diffTable = computeShortestDiffPath(graph, modelIDMap).filterModels(
    (modelInfo) => subsetIDs.includes(modelInfo.id)
  );
  console.log("raw table:", diffTable);

  let positionValues = diffTable.variables.map((v, i) =>
    diffTable.distinctValues(i)
  );
  return diffTable.filterVariables((_, i) => positionValues[i].length > 1);
}
