/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2024 Apple Inc. All Rights Reserved.
 */

/**
 * Layout of the model map using basic graph algorithms. The model layout
 * algorithm is a topological sort, which depends on the model list being
 * a collection of trees (e.g. DAGs in which each node has only one parent).
 */

import type {
  CompressionOperation,
  Model,
} from "$lib/data_structures/model_map";

/**
 * A representation of a graph node/vertex with one parent and
 * zero or more children. This is used for basic graph manipulation
 * such as topological sorting.
 */
import type { Position } from "$lib/utils";

export class TreeNode {
  id: string;
  parent: string | null;
  parentEdgeType: string | null;
  children: string[] = [];
  visible: boolean = true;

  constructor(
    id: string,
    parent: string | null = null,
    parentEdgeType: any = null
  ) {
    this.id = id;
    this.parent = parent;
    this.parentEdgeType = parentEdgeType;
  }

  hasParent(): boolean {
    return this.parent != null;
  }
}

export type TreeNodeCollection = Map<string, TreeNode>;

type Spacing = { horizontal: number; vertical: number };

export enum TreeType {
  BY_STEP = 0,
  BY_EDGE_TYPE = 1,
}

export interface TreeColumn {
  displayName(): string;
}

type NodePlacement<TreeColumnType extends TreeColumn> = Map<
  string,
  TreeColumnType
>;

class EdgeTypeColumn implements TreeColumn {
  edgeType: string | null;
  index: number;

  constructor(edgeType: string | null, index = 0) {
    this.edgeType = edgeType;
    this.index = index;
  }

  next(): EdgeTypeColumn {
    return new EdgeTypeColumn(this.edgeType, this.index + 1);
  }

  matchesEdgeType(edgeType: string | null): boolean {
    return edgeType == this.edgeType;
  }

  displayName(): string {
    return this.edgeType || "Base";
  }
}

class StepColumn implements TreeColumn {
  stepNumber: number;

  constructor(stepNumber: number) {
    this.stepNumber = stepNumber;
  }

  displayName(): string {
    if (this.stepNumber == 0) return "Base";
    return `Step ${this.stepNumber}`;
  }
}

export const BaseParentName = "###BASE_PARENT###";
export const BaseOperationName = "Base Model";

export function modelListToTrees(
  models: Model[],
  addBaseParent: boolean = false
): TreeNodeCollection {
  if (new Set(models.map((m) => m.id)).size < models.length)
    console.error(
      "Model list has duplicate IDs:",
      models.map((m) => m.id)
    );
  let nodes = new Map(
    models.map((m) => {
      if (m.base != null && m.operation == null)
        console.warn(`Model '${m.id}' has a base but no operation.`);
      return [
        m.id,
        new TreeNode(
          m.id,
          m.base != null ? m.base : addBaseParent ? BaseParentName : null,
          !!m.operation
            ? m.operation.name
            : addBaseParent
            ? BaseOperationName
            : null
        ),
      ];
    })
  );
  if (addBaseParent) {
    nodes.set(BaseParentName, new TreeNode(BaseParentName, null, null));
  }
  // Add each node to its parent's children
  Array.from(nodes.values()).forEach((n) => {
    if (!!n.parent) {
      if (!nodes.has(n.parent))
        console.error(`Node '${n.id}' is missing its parent, '${n.parent}'`);
      nodes.get(n.parent)!.children.push(n.id);
    }
  });
  return nodes;
}

/**
 * Filters the given trees so that any child of a node for whom
 * visiblePredicate returns false will be filtered out.
 *
 * @param graph A collection of tree nodes representing a set of
 *  trees
 * @param visiblePredicate A function that returns false if the
 *  node and its children should be hidden
 */
export function filterTrees(
  graph: TreeNodeCollection,
  visiblePredicate: (nodeID: string, node: TreeNode) => boolean
): TreeNodeCollection {
  let visitedNodes: Set<string> = new Set();
  let finalGraph: TreeNodeCollection = new Map();
  let roots = Array.from(graph.keys()).filter(
    (nodeID) => !graph.get(nodeID)?.hasParent()
  );
  roots.forEach((startNode) => {
    if (visitedNodes.has(startNode)) return;
    if (!visiblePredicate(startNode, graph.get(startNode)!)) return;

    let stack: TreeNode[] = [graph.get(startNode)!];
    while (stack.length > 0) {
      let currentNode = stack.pop()!;
      if (visitedNodes.has(currentNode.id))
        console.error(
          "Already visited a node that was popped from the search stack - is there a cycle?"
        );
      visitedNodes.add(currentNode.id);

      let visibleChildren = currentNode.children.filter((child) =>
        visiblePredicate(child, graph.get(child)!)
      );
      let newNode = new TreeNode(
        currentNode.id,
        currentNode.parent,
        currentNode.parentEdgeType
      );
      newNode.children = visibleChildren;
      finalGraph.set(currentNode.id, newNode);
      visibleChildren.forEach((child) => stack.push(graph.get(child)!));
    }
  });

  return finalGraph;
}

function _findConnectedComponent(
  graph: TreeNodeCollection,
  startNodeID: string
): string[] {
  let component = [startNodeID];
  graph.get(startNodeID)!.children.forEach((child) => {
    component = [...component, ..._findConnectedComponent(graph, child)];
  });
  return component;
}

/**
 * Separates a collection of trees into individual trees.
 *
 * @param graph A collection of possibly disconnected trees.
 * @returns A list of trees where each element contains all nodes
 *  from `graph` that have a connection to each other.
 */
function findConnectedComponents(
  graph: TreeNodeCollection
): TreeNodeCollection[] {
  let visited = new Set<string>();
  let results = new Array<TreeNodeCollection>();
  while (visited.size < graph.size) {
    let startID = Array.from(graph.keys()).filter((n) => !visited.has(n))[0];
    let componentIDs = _findConnectedComponent(graph, startID);
    componentIDs.forEach((c) => visited.add(c));
    results.push(new Map(componentIDs.map((id) => [id, graph.get(id)!])));
  }
  return results;
}

export function pathFromRoot(graph: TreeNodeCollection, id: string): string[] {
  let node = graph.get(id);
  let path: string[] = [];
  while (!!node) {
    path = [node.id, ...path];
    let parent = node.parent;
    if (!!parent && graph.has(parent)) {
      node = graph.get(parent);
    } else break;
  }
  return path;
}

function listPaths(graph: TreeNodeCollection): string[][] {
  // Find all LEAF nodes, and walk back from them
  let result = Array.from(graph.values())
    .filter((n) => n.children.length == 0)
    .map((node) => {
      let path: string[] = [];
      let curr: TreeNode | undefined = node;
      while (!!curr) {
        path.push(curr.id);
        if (!curr.parent) break;
        curr = graph.get(curr.parent!);
      }
      path.reverse();
      return path;
    });
  // Sort in decreasing order of length, as well as lexicographic order of operations that differ
  result.sort((a, b) => {
    if (a.length != b.length) return b.length - a.length;

    for (let i = 0; i < a.length; i++) {
      let node1 = graph.get(a[i]);
      let node2 = graph.get(b[i]);
      if (!node1 && !node2) continue;
      else if (!node1 || !node2) return 0;
      if (node1.parentEdgeType != node2.parentEdgeType)
        return (node1.parentEdgeType || "") < (node2.parentEdgeType || "")
          ? -1
          : 1;
    }

    return 0;
  });
  return result;
}

function calculateEdgeTypeIndexBound(
  columns: EdgeTypeColumn[],
  nodePlacement: NodePlacement<EdgeTypeColumn>,
  path: string[],
  nodeIndex: number,
  assignedNodeIndexes: number[]
): { lowerBound: number; upperBound: number } {
  let node = path[nodeIndex];
  let minPathIndex = assignedNodeIndexes
    .filter((ni) => ni < nodeIndex)
    .reduce((prev, curr) => Math.max(prev, curr), 0);
  let maxPathIndex = assignedNodeIndexes
    .filter((ni) => ni > nodeIndex)
    .reduce((prev, curr) => Math.min(prev, curr), path.length - 1);
  let minEdgeDataIndex = columns.indexOf(
    nodePlacement.get(path[minPathIndex])!
  );
  if (minEdgeDataIndex < 0) minEdgeDataIndex = 0;
  else minEdgeDataIndex += 1;
  let maxEdgeDataIndex = columns.indexOf(
    nodePlacement.get(path[maxPathIndex])!
  );
  if (maxEdgeDataIndex < 0) maxEdgeDataIndex = columns.length;

  return { lowerBound: minEdgeDataIndex, upperBound: maxEdgeDataIndex };
}

/**
 * Iteratively adds a path to the diagram by aligning its edge
 * data with those in the previous edge data sequence. The path
 * is added following a three step process:
 *
 * 1. Fix all nodes whose positions are already part of the
 *    diagram (e.g. they are shared with a different path).
 * 2. Assign positions for nodes whose edge data exists in the
 *    edge data sequence, but the nodes themselves have not been
 *    placed yet. The node will be assigned to the first
 *    edge data position that matches the node's edge data that falls
 *    between the position of the most recent already-assigned
 *    preceding node and the next already-assigned subsequent node.
 * 3. Insert edge data and assign positions for nodes for whom the
 *    edge data does not yet exist in the sequence. The edge data
 *    will be placed immediately after the position of the most recent
 *    already-assigned preceding node.
 *
 * @param graph Tree structure containing all nodes
 * @param path The specific path to add, described using each node's
 *  ID string
 * @param columns The sequence of edge data that has been
 *  generated so far. Each edge data is a tuple of the edge data itself
 *  and the index of the appearance of that edge data in the sequence.
 *  For example: [('a', 0), ('b', 0), ('c', 0), ('a', 1)].
 * @param nodePlacement A Map relating node IDs to edge data
 *  and index tuples. This indicates the positions of all nodes that have
 *  already been placed.
 * @returns The updated columns and nodePlacement, returned
 *  as a copy of the inputs with additional nodes and edge data items added.
 */
function addEdgeDataAlignedPath(
  graph: TreeNodeCollection,
  path: string[],
  columns: EdgeTypeColumn[] = [],
  nodePlacement: NodePlacement<EdgeTypeColumn> = new Map()
): {
  columns: EdgeTypeColumn[];
  nodePlacement: NodePlacement<EdgeTypeColumn>;
} {
  columns = Array.from(columns);
  nodePlacement = new Map(nodePlacement.entries());

  let log = false;

  if (log) console.log(path);

  // First see if any of the nodes along the path are already
  // assigned to a column.
  let assignedNodeIndexes = path
    .map((n, i) => i)
    .filter((i) => nodePlacement.has(path[i]));

  if (log) console.log("assigned initially:", assignedNodeIndexes);

  // Now place any nodes whose edge data (operations) are already
  // present in the edge data sequence
  path.forEach((node, i) => {
    if (nodePlacement.has(node)) return;
    let edgeType = graph.get(node)!.parentEdgeType || null;
    let { lowerBound, upperBound } = calculateEdgeTypeIndexBound(
      columns,
      nodePlacement,
      path,
      i,
      assignedNodeIndexes
    );
    if (log)
      console.log(
        "bounds on placement of " + node + ": ",
        lowerBound,
        upperBound
      );
    let existingIndex = columns
      .slice(lowerBound, upperBound)
      .findIndex((item) => item.matchesEdgeType(edgeType));
    if (existingIndex < 0) return; // operation isn't yet placed - we will get to these later
    existingIndex += lowerBound;

    nodePlacement.set(node, columns[existingIndex]);
    if (log)
      console.log(
        "placing",
        node,
        edgeType,
        existingIndex,
        columns[existingIndex]
      );
    assignedNodeIndexes.push(i);
  });

  if (log) console.log("intermediate node placement:", nodePlacement, columns);

  // Finally, place nodes whose edge data is NOT yet present
  path.forEach((node, i) => {
    if (nodePlacement.has(node)) return;
    let edgeType = graph.get(node)!.parentEdgeType || null;
    let { lowerBound, upperBound } = calculateEdgeTypeIndexBound(
      columns,
      nodePlacement,
      path,
      i,
      assignedNodeIndexes
    );

    let existingIndex = columns
      .slice(lowerBound, upperBound)
      .findIndex((item) => item.matchesEdgeType(edgeType));

    if (existingIndex >= 0)
      console.error(
        `Edge data already exists: '${edgeType}' was assigned at index ${existingIndex} (value ${columns[existingIndex].edgeType}, ${columns[existingIndex].index})`
      );
    existingIndex += lowerBound;

    let indexOfNewEdgeData = columns.filter((ed) =>
      ed.matchesEdgeType(edgeType)
    ).length;
    let insertionIndex = columns.length > 0 ? lowerBound : 0;

    if (log) console.log("adding new step", node, edgeType, insertionIndex);
    columns = [
      ...columns.slice(0, insertionIndex),
      new EdgeTypeColumn(edgeType, indexOfNewEdgeData),
      ...columns.slice(insertionIndex),
    ];
    nodePlacement.set(node, columns[insertionIndex]);
    assignedNodeIndexes.push(i);
  });

  if (log) console.log(path, "intermediate results:", columns, nodePlacement);
  return { columns, nodePlacement };
}

export function assignTreeColumns(
  graph: TreeNodeCollection,
  type: TreeType
): { columns: TreeColumn[]; nodePlacement: Map<string, TreeColumn> } {
  if (type == TreeType.BY_STEP) {
    let paths = listPaths(graph);

    // Simply create a column for each item in the path. Each path is assumed
    // to start at base
    let maxLength = paths.reduce(
      (prev, curr) => Math.max(prev, curr.length),
      0
    );
    let columns = new Array(maxLength).fill(0).map((_, i) => new StepColumn(i));

    let nodePlacement: NodePlacement<StepColumn> = new Map();
    paths.forEach((path) => {
      path.forEach((node, i) => nodePlacement.set(node, columns[i]));
    });

    return { columns, nodePlacement };
  } else if (type == TreeType.BY_EDGE_TYPE) {
    let paths = listPaths(graph);

    // Iteratively add columns and place nodes, one path at a time
    let columns = new Array<EdgeTypeColumn>();
    let nodePlacement: NodePlacement<EdgeTypeColumn> = new Map();
    paths.forEach((path) => {
      let result = addEdgeDataAlignedPath(graph, path, columns, nodePlacement);
      columns = result.columns;
      nodePlacement = result.nodePlacement;
    });

    return { columns, nodePlacement };
  }
  console.error(`Unrecognized tree type ${type}`);
  return { columns: [], nodePlacement: new Map() };
}

// Positioning individual nodes - after columns have been assigned

function _layoutPath<TreeColumnType extends TreeColumn>(
  graph: TreeNodeCollection,
  path: string[],
  columns: TreeColumnType[],
  nodePlacement: NodePlacement<TreeColumnType>,
  sortFn: (a: string[], b: string[]) => number,
  positions: Map<string, Position>, // mutates
  spacing: number | Spacing,
  startX: number = 0,
  startY: number = 0
): number {
  let spacingX =
    typeof spacing === "number"
      ? (spacing as number)
      : (spacing as Spacing).horizontal;
  let spacingY =
    typeof spacing === "number"
      ? (spacing as number)
      : (spacing as Spacing).vertical;

  let laidOutOneElement = false;
  for (let i = 0; i < path.length; i++) {
    let child = path[i];
    if (positions.has(child)) continue;
    if (!graph.get(child)!.visible) break;
    if (!nodePlacement.has(child))
      console.error(`Child node '${child}' not found in planned map layout`);
    let xCoord = startX + columns.indexOf(nodePlacement.get(child)!) * spacingX;
    positions.set(child, { x: xCoord, y: startY });
    laidOutOneElement = true;
  }

  if (!laidOutOneElement) return startY - spacingY; // nothing was laid out so remove the added spacing

  // Now go backwards from the end of the path and
  // at each level, sort paths starting at the current
  // node and lay them out
  let currentY = startY;
  path
    .slice()
    .reverse()
    .forEach((node) => {
      let component = _findConnectedComponent(graph, node);
      let componentPaths = listPaths(
        new Map(component.map((c) => [c, graph.get(c)!]))
      );
      componentPaths.sort(sortFn);

      componentPaths.forEach((childPath) => {
        if (childPath.length == 1 || path.includes(childPath[1])) return;
        currentY = _layoutPath(
          graph,
          childPath,
          columns,
          nodePlacement,
          sortFn,
          positions,
          spacing,
          startX,
          currentY + spacingY
        );
      });
    });

  return currentY;
}

function makeSortByColumnIndex<T extends TreeColumn>(
  columns: T[],
  nodePlacement: NodePlacement<T>
): (a: string[], b: string[]) => number {
  return (a: string[], b: string[]): number => {
    let columnIndicesA = a.map((id) => columns.indexOf(nodePlacement.get(id)!));
    let columnIndicesB = b.map((id) => columns.indexOf(nodePlacement.get(id)!));
    for (let i of Array(
      Math.min(columnIndicesA.length, columnIndicesB.length)
    ).keys()) {
      if (columnIndicesA[i] != columnIndicesB[i])
        return columnIndicesB[i] - columnIndicesA[i];
    }
    return columnIndicesB.length - columnIndicesA.length;
  };
}

export function layoutModels<TreeColumnType extends TreeColumn>(
  graph: TreeNodeCollection,
  columns: TreeColumnType[],
  nodePlacement: NodePlacement<TreeColumnType>,
  spacing: number | Spacing = 100,
  pathSortFunction: ((a: string[], b: string[]) => number) | null = null, // default is sort by column index
  startX: number = 0,
  startY: number = 0
): Map<string, Position> {
  // Assign each edge data (operation) to an x coordinate
  // let components = findConnectedComponents(graph);

  let spacingY =
    typeof spacing === "number"
      ? (spacing as number)
      : (spacing as Spacing).vertical;

  let positions = new Map<string, Position>();
  let currentY = startY;

  let allPaths = listPaths(graph);
  let baseSortFn = makeSortByColumnIndex(columns, nodePlacement);
  let sortFn: (a: string[], b: string[]) => number =
    pathSortFunction != null
      ? (a, b) => {
          let sortVal = pathSortFunction(a, b);
          if (sortVal == 0) return baseSortFn(a, b);
          return sortVal;
        }
      : baseSortFn;
  allPaths.sort(sortFn);

  let visitedRootNodes = new Set<string>();

  allPaths.forEach((path) => {
    if (visitedRootNodes.has(path[0])) return;
    visitedRootNodes.add(path[0]);

    // Layout this path first
    currentY =
      _layoutPath(
        graph,
        path,
        columns,
        nodePlacement,
        sortFn,
        positions,
        spacing,
        startX,
        currentY
      ) + spacingY;
  });

  /*// Now place each model on a grid so that models that have the same
  // delta of operations are placed on the same x coordinate
  let positions = new Map<string, Position>();
  let currentY = startY;
  components.forEach((component) => {
    let result = _layoutComponent(
      component,
      columns,
      nodePlacement,
      spacing,
      startX,
      currentY
    );
    result.positions.forEach((v, k) => positions.set(k, v));
    currentY = result.maxY + spacingY;
  });*/

  console.log("resulting positions:", positions);
  return positions;
}
