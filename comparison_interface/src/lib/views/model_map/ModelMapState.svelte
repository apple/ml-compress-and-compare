<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<svelte:options accessors />

<!-- Template-less component that keeps track of the state of the nodes and edges in the graph -->
<script lang="ts">
  import { areSetsEqual, type Position, type Rectangle } from "$lib/utils";
  import { getContext, onDestroy, onMount } from "svelte";
  import {
    LineDecoration,
    Mark,
    MarkSet,
    interpolateTo,
  } from "$lib/chart_components/animated_marks";
  import type { TreeNodeCollection } from "$lib/algorithms/map_layout";
  import * as d3 from "d3";
  import type {
    CompressionOperation,
    CompressionOperationSpec,
    Model,
    OperationParameterSpec,
  } from "$lib/data_structures/model_map";
  import type { Readable } from "svelte/store";

  const { width, height } = getContext("LayerCake") as {
    width: Readable<number>;
    height: Readable<number>;
  };

  export let operations: CompressionOperationSpec[] = [];
  export let models: Model[] = [];
  export let graph: TreeNodeCollection | null;
  export let positions: Map<string, Position> = new Map();
  export let zoomTransform: d3.ZoomTransform = d3.zoomIdentity;

  export let padding: {
    left?: number;
    right?: number;
    top?: number;
    bottom?: number;
  } = {};
  export let horizontalSpacing: number = 48;
  export let verticalSpacing: number = 24;
  export let nodeWidth: number = 96;
  export let nodeHeight: number = 96;
  export let minNodeScale: number = 0.0;
  export let maxNodeScale: number = 1000.0;

  export let hoveredIDs: string[] = [];
  export let selectedIDs: string[] = [];
  export let filteredIDs: string[] = [];

  export let colorFunction: ((id: string) => number) | null = null;
  export let colorScale = d3.scaleSequential(d3.interpolateBlues);

  export let radiusFunction: ((id: string) => number) | null = null;

  export let nodeMarks: MarkSet;

  export let animationDuration = 500;
  export let hoverAnimationDuration = 200;
  export let animationCurve = d3.easeCubicInOut;

  $: if (!!graph && positions.size > 0 && !nodeMarks) initializeMarks();

  function initializeMarks() {
    let nodes = Array.from(positions.keys()).map((id) => {
      let model = models.find((m) => m.id == id);
      if (!model) {
        console.error(`Can't find model with id ${id}`);
        return;
      }
      return new Mark(id, {
        x: {
          valueFn: () => positions.get(id)!.x,
          transform: (v: number) => gridToCanvasX(v),
          lazy: true,
        },
        y: {
          valueFn: () => positions.get(id)!.y,
          transform: (v: number) => gridToCanvasY(v),
          lazy: true,
        },
        fillStyle: {
          valueFn: () => (colorFunction != null ? colorFunction(id) : 1),
          transform: transformColor,
          cached: true,
        },
        alpha: {
          valueFn: () =>
            filteredIDs.length > 0 ? (filteredIDs.includes(id) ? 1 : 0.1) : 1,
        },
        width: {
          value: nodeWidth,
          transform: transformSize,
        },
        height: {
          value: nodeHeight,
          transform: transformSize,
        },
        radius: {
          valueFn: () =>
            Math.min(nodeWidth, nodeHeight) *
            0.5 *
            (radiusFunction != null
              ? radiusFunction(id)
              : hoveredIDs.includes(id)
              ? 1.1
              : 1),
          transform: transformSize,
        },
        strokeStyle: {
          value: "#60a5fa",
          cached: true,
        },
        lineWidth: {
          valueFn: () =>
            (selectedIDs.includes(id) ? 3 : 0) +
            (hoveredIDs.includes(id) ? 2 : 0),
          lazy: true,
        },
        parameterText: {
          value: getParameterText(model),
          cached: true,
        },
        labelAlpha: {
          value: 0.0,
        },
      });
    });

    let nodeMapping = new Map(nodes.map((mark) => [mark.id, mark]));
    let edges = Array.from(graph!.entries())
      .filter(([id, node]) => node.hasParent() && graph?.has(node.parent!))
      .map(
        ([id, node]) =>
          new LineDecoration(
            nodeMapping.get(node.parent!)!,
            nodeMapping.get(id)!,
            () => 4,
            () => 1
          )
      );
    nodeMarks = new MarkSet(nodes, edges as any);
  }

  // Grid coordinates = integer x, y pairs corresponding to cells in a grid
  // Canvas coordinates = spacing and sizes applied and multiplied by zoom transform

  export function gridToCanvasX(x: number): number {
    return zoomTransform.applyX(
      Math.round(
        (padding.left || 0) +
          x * (nodeWidth + horizontalSpacing) +
          nodeWidth * 0.5
      )
    );
  }
  export function gridToCanvasY(x: number): number {
    return zoomTransform.applyY(
      Math.round(
        (padding.top || 0) +
          x * (nodeHeight + verticalSpacing) +
          nodeHeight * 0.5
      )
    );
  }

  function transformColor(c: number): string | null {
    let color = d3.color(colorScale(c));
    if (color == null) return null;

    return color.formatHex();
  }

  function transformSize(baseSize: number): number {
    let glyphScale = Math.min(
      maxNodeScale,
      Math.max(minNodeScale, zoomTransform.k)
    );
    return baseSize * glyphScale;
  }

  export function getBoundingBoxForID(id: string): Rectangle {
    let mark = nodeMarks.getMarkByID(id);
    if (!mark)
      console.error(`can't get bounding box for node ${id} that doesn't exist`);

    let x = mark.attr("x");
    let y = mark.attr("y");
    let width = mark.attr("width");
    let height = mark.attr("height");
    return { x: x - width * 0.5, y: y - height * 0.5, width, height };
  }

  // Returns null if the given position is outside a cell (e.g. in the gap)
  export function canvasToGrid(position: {
    x: number;
    y: number;
  }): { x: number; y: number } | null {
    let canvasCoord = zoomTransform.invert([position.x, position.y]);
    let nearestX =
      (canvasCoord[0] - (padding.left || 0)) / (nodeWidth + horizontalSpacing);
    let nearestY =
      (canvasCoord[1] - (padding.top || 0)) / (nodeHeight + verticalSpacing);

    if (
      nearestX < 0 ||
      nearestY < 0 ||
      nearestX * (nodeWidth + horizontalSpacing) -
        Math.floor(nearestX) * (nodeWidth + horizontalSpacing) >=
        nodeWidth ||
      nearestY * (nodeHeight + verticalSpacing) -
        Math.floor(nearestY) * (nodeHeight + verticalSpacing) >=
        nodeHeight
    )
      return null;
    return { x: Math.floor(nearestX), y: Math.floor(nearestY) };
  }

  function getParameterText(model: Model): { [key: string]: string } | null {
    let operation = model!.operation;
    if (!operation) return null;
    let operationSpec:
      | { parameters: { [key: string]: OperationParameterSpec } }
      | undefined = !!operation
      ? operations.find((op) => op.name == operation!.name)
      : { parameters: {} };
    let paramStrings = Object.fromEntries(
      Object.entries(operation.parameters).map((param) => {
        let paramSpec = operationSpec?.parameters[param[0]];
        if (!paramSpec || !paramSpec.format) return [param[0], param[1]];
        let formatter = d3.format(paramSpec.format || ".2g");
        return [param[0], formatter(param[1])];
      })
    );
    return paramStrings;
  }

  let timer: d3.Timer | null;
  let currentTime: number = 0;
  export let needsDraw = true;

  let zoomChanged = false;

  onMount(() => {
    timer = d3.timer((elapsed) => {
      let dt = elapsed - currentTime;
      currentTime = elapsed;
      needsDraw = (!!nodeMarks && nodeMarks.advance(dt)) || zoomChanged;
      zoomChanged = false;
    });
  });

  onDestroy(() => {
    if (!!timer) timer.stop();
    timer = null;
  });

  $: zoomTransform, $width, $height, (zoomChanged = true);

  let showLabels = false;
  $: if (!!nodeMarks && nodeWidth * zoomTransform.k >= 80 != showLabels) {
    showLabels = !showLabels;
    nodeMarks.animateAll(
      "labelAlpha",
      () => interpolateTo(showLabels ? 1.0 : 0.0),
      hoverAnimationDuration,
      animationCurve
    );
  }

  let oldPositions: Map<string, Position> = new Map();
  $: if (oldPositions !== positions && !!nodeMarks) {
    console.log("updating");
    if (oldPositions.size > 0) {
      nodeMarks.animateComputed(
        "x",
        interpolateTo,
        animationDuration,
        animationCurve
      );
      nodeMarks.animateComputed(
        "y",
        interpolateTo,
        animationDuration,
        animationCurve
      );
    }
    oldPositions = positions;
  }

  $: if (!!nodeMarks) {
    colorFunction, colorScale;
    nodeMarks.animateComputed(
      "fillStyle",
      interpolateTo,
      animationDuration,
      animationCurve
    );
  }

  $: if (!!nodeMarks) {
    console.log("updating size");
    radiusFunction;
    nodeMarks.animateComputed(
      "radius",
      interpolateTo,
      animationDuration,
      animationCurve
    );
  }

  let oldHoveredIDs: Set<string> = new Set();

  $: if (!!nodeMarks && !areSetsEqual(oldHoveredIDs, new Set(hoveredIDs))) {
    nodeMarks.animateComputed(
      ["width", "height", "lineWidth"],
      interpolateTo,
      hoverAnimationDuration,
      animationCurve
    );

    oldHoveredIDs = new Set(hoveredIDs);
  }

  let oldSelectedIDs: Set<string> = new Set();
  $: if (!!nodeMarks && !areSetsEqual(oldSelectedIDs, new Set(selectedIDs))) {
    nodeMarks.animateComputed(
      "lineWidth",
      interpolateTo,
      hoverAnimationDuration,
      animationCurve
    );
    oldSelectedIDs = new Set(selectedIDs);
  }

  let oldFilteredIDs: Set<string> = new Set();
  $: if (!!nodeMarks && !areSetsEqual(oldFilteredIDs, new Set(filteredIDs))) {
    nodeMarks.animateComputed(
      "alpha",
      interpolateTo,
      hoverAnimationDuration,
      animationCurve
    );
    oldFilteredIDs = new Set(filteredIDs);
  }

  export function animateZoom() {
    nodeMarks.updateTransform("x");
    nodeMarks.updateTransform("y");
  }
</script>
