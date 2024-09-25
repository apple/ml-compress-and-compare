<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import type {
    CompressionOperationSpec,
    Metric,
    Model,
  } from "$lib/data_structures/model_map";
  import type {
    TreeColumn,
    TreeNode,
    TreeNodeCollection,
  } from "$lib/algorithms/map_layout";
  import {
    TreeType,
    assignTreeColumns,
    layoutModels,
    modelListToTrees,
    pathFromRoot,
  } from "$lib/algorithms/map_layout";
  import * as d3 from "d3";
  import { LayerCake, Canvas, Html, Svg } from "layercake";
  import ModelMapCanvas from "$lib/views/model_map/ModelMapCanvas.svelte";
  import ModelTooltip from "./ModelTooltip.svelte";
  import {
    hashPosition,
    positionsEqual,
    type Position,
    type Rectangle,
  } from "$lib/utils";
  import ModelMapState from "$lib/views/model_map/ModelMapState.svelte";
  import type { MarkSet } from "$lib/chart_components/animated_marks";
  import Multiselect from "$lib/chart_components/Multiselect.canvas.svelte";
  import { legend } from "$lib/chart_components/colorbar_legend";

  export let operations: CompressionOperationSpec[];
  export let metrics: Metric[];
  export let models: Model[];

  let positions = new Map<string, { x: number; y: number }>();
  let columns = new Array<TreeColumn>();
  let nodePlacement = new Map<string, TreeColumn>();

  let graph: TreeNodeCollection | null;
  let positionIDMap: Map<number, string> = new Map();

  let collapsedIDs = new Array<string>();

  export let columnByStep = false;
  export let colorMetric: string | null = null;
  export let hoveredIDs: string[] = [];
  export let selectedIDs: string[] = [];
  export let filteredIDs: string[] = [];
  export let sizeMetric: string | null = null;
  export let colorScale = d3.scaleSequential(d3.interpolateBlues);
  let isMultiselecting = false;
  let multiselectRegion: Rectangle | null = null;

  let mapState: ModelMapState;
  let marks: MarkSet;
  let needsDraw = false;
  let initialLayout = false;

  let hoverInfo: { xPosition: number; yPosition: number; model: Model } | null =
    null;

  $: if (!!models && models.length > 0) {
    graph = modelListToTrees(models);
    initialLayout = true;
  }

  $: if (!!graph) {
    ({ columns, nodePlacement } = assignTreeColumns(
      graph,
      columnByStep ? TreeType.BY_STEP : TreeType.BY_EDGE_TYPE
    ));
  }

  function nodeExpanded(node: TreeNode) {
    return node.parent == null || !collapsedIDs.includes(node.parent);
  }

  // Sort function to sort nodes by operation parameters. Returns zero
  // when paths are not comparable, to fall back on default sorting behavior
  function sortByOperation(path1: string[], path2: string[]): number {
    if (path1[0] != path2[0]) return 0;
    for (let index = 1; index < Math.min(path1.length, path2.length); index++) {
      if (path1[index] == path2[index]) continue;
      let model1 = models.find((m) => m.id == path1[index]);
      let model2 = models.find((m) => m.id == path2[index]);
      if (!model1 || !model2) {
        console.error(
          `Can't find models being sorted: ${path1[index]}, ${path2[index]}`
        );
        return 0;
      }
      if (model1.operation == null && model2.operation == null) return 0;
      if (
        !!model1.operation &&
        !!model2.operation &&
        model1.operation?.name == model2.operation?.name
      ) {
        // Sort by parameter values
        for (let paramName of Object.keys(model1.operation.parameters).sort()) {
          let val1 = model1!.operation!.parameters[paramName];
          let val2 = model2!.operation!.parameters[paramName];
          if (val1 == val2) continue;
          return val1 < val2 ? -1 : 1;
        }
      }
    }
    return 0;
  }

  $: if (!!graph && collapsedIDs) {
    Array.from(graph.values()).forEach((node) => {
      node.visible = nodeExpanded(node);
    });
    positions = layoutModels(
      graph,
      columns,
      nodePlacement,
      {
        horizontal: 1,
        vertical: 1,
      },
      sortByOperation
    );
    positionIDMap = new Map(
      Array.from(positions.entries()).map((entry) => [
        hashPosition(entry[1]),
        entry[0],
      ])
    );
  }

  $: if (!!marks && !!zoomer && initialLayout) {
    fitGraphOnScreen();
    initialLayout = false;
  }

  let flowchartContainer: HTMLElement;
  let flowchart: ModelMapCanvas;
  let zoomer;
  let changingZoom = false;
  let zoomTransform = d3.zoomIdentity;

  function pan(e: WheelEvent) {
    let coord = {
      x: e.clientX - (e.target as HTMLElement).getBoundingClientRect().left,
      y: e.clientY - (e.target as HTMLElement).getBoundingClientRect().top,
    };
    hoverInfo = getHoverInfo(coord);
    zoomer.translateBy(
      d3.select(flowchartContainer),
      -e.deltaX / zoomTransform.k,
      -e.deltaY / zoomTransform.k
    );
  }

  function getHoverInfo(coord: { x: number; y: number }) {
    let gridPos = mapState.canvasToGrid(coord);
    let id = getNodeAtPoint(gridPos);

    if (id) {
      // mouse on model
      hoverInfo = {
        model: models.find((m) => m.id == id),
        yPosition: mapState.gridToCanvasY(gridPos.y),
        xPosition: mapState.gridToCanvasX(gridPos.x),
      };
    } else {
      // mouse on canvas
      hoverInfo = null;
    }
    return hoverInfo;
  }

  $: if (!!flowchartContainer) {
    zoomer = d3
      .zoom()
      .scaleExtent([0.1, 3.0])
      .filter(
        (e) => (e.type !== "wheel" || e.ctrlKey) && e.type !== "mousedown"
      )
      .on("start", (e) => {
        changingZoom = true;
        mouseDown = true;
      })
      .on("end", (e) => {
        changingZoom = false;
        mouseDown = false;
      })
      .on("zoom", (e) => {
        zoomTransform = e.transform;
      });
    d3.select(flowchartContainer).call(zoomer);
  }

  const columnWidth = 72;
  const rowHeight = 72;
  const horizontalSpacing = 120;
  const verticalSpacing = 36;

  function getNodeAtPoint(point: Position | null): string | null {
    if (point != null) {
      if (positionIDMap.has(hashPosition(point))) {
        let id = positionIDMap.get(hashPosition(point))!;
        if (positionsEqual(positions.get(id)!, point)) {
          return id;
        }
      }
    }
    return null;
  }

  function handleNodeClick(e: PointerEvent) {
    if (isMultiselecting && !!multiselectRegion) {
      if (e.shiftKey)
        selectedIDs = Array.from(
          new Set([...selectedIDs, ...nodesInRegion(multiselectRegion)])
        );
      else selectedIDs = nodesInRegion(multiselectRegion);
      hoveredIDs = [];
      multiselectRegion = null;
      isMultiselecting = false;
      originalMousePosition = null;
      return;
    }

    let coord = {
      x: e.clientX - (e.target as HTMLElement).getBoundingClientRect().left,
      y: e.clientY - (e.target as HTMLElement).getBoundingClientRect().top,
    };
    let gridPos = mapState.canvasToGrid(coord);
    let nodeID = getNodeAtPoint(gridPos);
    if (!nodeID || (filteredIDs.length > 0 && !filteredIDs.includes(nodeID)))
      selectedIDs = [];
    else if (e.shiftKey) {
      if (selectedIDs.includes(nodeID)) {
        let idx = selectedIDs.indexOf(nodeID);
        selectedIDs = [
          ...selectedIDs.slice(0, idx),
          ...selectedIDs.slice(idx + 1),
        ];
      } else {
        selectedIDs = [...selectedIDs, nodeID];
      }
    } else selectedIDs = [nodeID];
    originalMousePosition = null;
  }

  let mouseDown = false;
  let originalMousePosition: { x: number; y: number } | null = null;

  function onMousedown(e: PointerEvent | MouseEvent) {
    mouseDown = true;
    originalMousePosition = {
      x: e.clientX - (e.target as HTMLElement).getBoundingClientRect().left,
      y: e.clientY - (e.target as HTMLElement).getBoundingClientRect().top,
    };
    e.stopPropagation();
  }

  function onMousemove(e: PointerEvent) {
    let coord = {
      x: e.clientX - (e.target as HTMLElement).getBoundingClientRect().left,
      y: e.clientY - (e.target as HTMLElement).getBoundingClientRect().top,
    };
    if (mouseDown && !!originalMousePosition) {
      if (
        Math.max(
          Math.abs(coord.x - originalMousePosition!.x),
          Math.abs(coord.y - originalMousePosition!.y)
        ) >= 1
      ) {
        // multiselect
        isMultiselecting = true;
        if (multiselectRegion == null)
          multiselectRegion = { x: coord.x, y: coord.y, width: 0, height: 0 };
        else
          multiselectRegion = Object.assign(
            {},
            Object.assign(multiselectRegion, {
              width: coord.x - multiselectRegion.x,
              height: coord.y - multiselectRegion.y,
            })
          );

        hoveredIDs = nodesInRegion(multiselectRegion);
        flowchartContainer.setPointerCapture(e.pointerId);
        e.stopPropagation();
      }
    } else {
      // hover
      let gridPos = mapState.canvasToGrid(coord);
      let id = getNodeAtPoint(gridPos);
      if (!id && hoveredIDs.length > 0) hoveredIDs = [];
      else if (!!id && !hoveredIDs.includes(id)) hoveredIDs = [id];

      // invoke tooltip on model hover
      hoverInfo = getHoverInfo(coord);
    }
  }

  function onMouseup(e: PointerEvent) {
    mouseDown = false;
    originalMousePosition = null;
    let appendSelection = e.shiftKey;
    setTimeout(() => {
      if (isMultiselecting && !!multiselectRegion) {
        if (appendSelection)
          selectedIDs = Array.from(
            new Set([...selectedIDs, ...nodesInRegion(multiselectRegion)])
          );
        else selectedIDs = nodesInRegion(multiselectRegion);
        hoveredIDs = [];
        multiselectRegion = null;
        isMultiselecting = false;
      }
    }, 100);
  }

  function toggleExpanded(id: string) {
    if (collapsedIDs.includes(id)) {
      let idx = collapsedIDs.indexOf(id);
      collapsedIDs = [
        ...collapsedIDs.slice(0, idx),
        ...collapsedIDs.slice(idx + 1),
      ];
    } else {
      collapsedIDs = [...collapsedIDs, id];
    }
  }

  // region is expressed in screen coordinates
  function nodesInRegion(region: Rectangle): string[] {
    if (region.width < 0)
      region = Object.assign(Object.assign({}, region), {
        x: region.x + region.width,
        width: -region.width,
      });
    if (region.height < 0)
      region = Object.assign(Object.assign({}, region), {
        y: region.y + region.height,
        height: -region.height,
      });
    return Array.from(graph!.keys()).filter((id) => {
      if (filteredIDs.length > 0 && !filteredIDs.includes(id)) return false;
      let bbox = mapState.getBoundingBoxForID(id);
      return (
        ((bbox.x >= region.x && bbox.x <= region.x + region.width) ||
          (bbox.x + bbox.width >= region.x &&
            bbox.x + bbox.width <= region.x + region.width)) &&
        ((bbox.y >= region.y && bbox.y <= region.y + region.height) ||
          (bbox.y + bbox.height >= region.y &&
            bbox.y + bbox.height <= region.y + region.height))
      );
    });
  }

  const fitGraphPadding = horizontalSpacing * 2;
  const topPadding = 72; // for navigation bar

  function fitGraphOnScreen(animated = false) {
    // Calculate bounding box and zoom to that
    let minX = 1e9;
    let maxX = -1e9;
    let minY = 0;
    let maxY = -1e9;

    marks.forEach((mark) => {
      let x = mark.attr("x");
      let y = mark.attr("y");
      minX = Math.min(x, minX);
      maxX = Math.max(x, maxX);
      minY = Math.min(y, minY);
      maxY = Math.max(y, maxY);
    });

    minX -= fitGraphPadding;
    maxX += fitGraphPadding;
    minY -= fitGraphPadding;
    maxY += fitGraphPadding;

    let width = flowchartContainer.clientWidth;
    let height = flowchartContainer.clientHeight;
    zoomer.transform(
      d3.select(flowchartContainer),
      d3.zoomIdentity
        .translate(width / 2, (height - topPadding) / 2 + topPadding)
        .scale(
          Math.min(
            2,
            1 / Math.max((maxX - minX) / width, (maxY - minY) / height)
          )
        )
        .translate(-(minX + maxX) / 2, -(minY + maxY) / 2 - topPadding)
    );
  }

  let legendContainer: SVGElement;
  $: if (!!legendContainer) {
    legendContainer.replaceChildren();
    legend(colorScale, legendContainer, {
      title: colorMetric || undefined,
      width: legendContainer.clientWidth,
      height: legendContainer.clientHeight,
    });
  }

  let sizeScale: any;
  $: if (!!metrics && !!sizeMetric) {
    let extent = d3.extent(
      models.map((m) => m.metrics[sizeMetric!]).filter((v) => v !== undefined)
    ) as [number, number];
    sizeScale = d3.scaleSqrt(extent, [0.3, 1]);
  } else sizeScale = null;

  function selectColumn(e: MouseEvent, column: TreeColumn) {
    let newSelection = Array.from(nodePlacement.keys()).filter(
      (m) =>
        nodePlacement.get(m) === column &&
        (filteredIDs.length == 0 || filteredIDs.includes(m))
    );
    if (e.shiftKey) selectedIDs = [...selectedIDs, ...newSelection];
    else selectedIDs = newSelection;
  }

  function selectParents() {
    if (!graph) return;
    let newSelection = new Set<string>(selectedIDs);
    selectedIDs.forEach((id) => {
      if (!graph?.has(id)) return;
      let node = graph!.get(id)!;
      if (!node.parent) return;
      newSelection.add(node.parent!);
    });
    selectedIDs = Array.from(newSelection);
  }

  function selectChildren() {
    if (!graph) return;
    let newSelection = new Set<string>(selectedIDs);
    selectedIDs.forEach((id) => {
      if (!graph?.has(id)) return;
      let node = graph!.get(id)!;
      node.children.forEach((child) => newSelection.add(child));
    });
    selectedIDs = Array.from(newSelection);
  }

  function selectAncestors() {
    if (!graph) return;
    let newSelection = new Set<string>();
    selectedIDs.forEach((id) => {
      pathFromRoot(graph!, id).forEach((newID) => newSelection.add(newID));
    });
    selectedIDs = Array.from(newSelection);
  }

  function selectDescendants() {
    if (!graph) return;
    let newSelection = new Set<string>();
    let stack: TreeNode[] = [];
    selectedIDs.forEach((id) => {
      let node = graph?.get(id);
      if (!node) return;
      stack.push(node);
      while (stack.length > 0) {
        node = stack.pop()!;
        newSelection.add(node.id);
        node.children.forEach((child) => {
          if (graph!.has(child)) stack.push(graph!.get(child)!);
        });
      }
    });
    selectedIDs = Array.from(newSelection);
  }
</script>

<div id="map-view" class="w-full h-full relative overflow-hidden">
  <div
    class="bg-white/50 absolute top-0 left-0 right-0 w-full flex items-center z-10 backdrop-blur-md"
    style="min-height: {topPadding}px; gap: {horizontalSpacing *
      zoomTransform.k}px; "
  >
    <div class="shrink-0 grow-0" style="margin-left: {zoomTransform.x}px;" />
    <!-- start offset for pan/zoom -->
    {#each columns as column, i (i)}
      <div
        class="{zoomTransform.k >= 0.6
          ? 'py-4 px-2'
          : ''} text-center text-gray-700 text-sm bg-transparent shrink-0 grow-0 flex justify-center items-center"
        style="width: {columnWidth * zoomTransform.k}px; line-height: 1em;"
      >
        <button
          class="origin-center hover:opacity-50"
          class:font-bold={hoveredIDs.some((id) => positions.get(id).x == i)}
          style="width: {Math.max(72, columnWidth * zoomTransform.k)}px;"
          class:-rotate-45={zoomTransform.k < 0.5 && zoomTransform.k >= 0.25}
          class:-rotate-90={zoomTransform.k < 0.25}
          class:text-xs={zoomTransform.k < 0.5}
          on:click={(e) => selectColumn(e, column)}
        >
          {column.displayName()}
        </button>
      </div>
    {/each}
  </div>
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <div
    unselectable="on"
    class="absolute w-full h-full top-0 left-0 overflow-hidden select-none {hoveredIDs.length ==
      1 &&
    (filteredIDs.length == 0 || filteredIDs.includes(hoveredIDs[0])) &&
    !isMultiselecting
      ? 'cursor-pointer'
      : 'cursor-crosshair'}"
    bind:this={flowchartContainer}
    on:wheel|stopPropagation={pan}
    on:click={(e) => handleNodeClick(e)}
    on:pointerdown={onMousedown}
    on:mousedown={onMousedown}
    on:pointermove={onMousemove}
    on:pointerup={onMouseup}
    role="button"
    tabindex="0"
  >
    <div class="absolute left-0 top-0 w-full h-full pointer-events-none">
      <LayerCake>
        <Canvas>
          <ModelMapState
            {models}
            {operations}
            bind:this={mapState}
            bind:nodeMarks={marks}
            bind:needsDraw
            {graph}
            {positions}
            {zoomTransform}
            {hoveredIDs}
            {selectedIDs}
            {filteredIDs}
            padding={{
              left: horizontalSpacing,
              top: topPadding + verticalSpacing,
            }}
            {horizontalSpacing}
            {verticalSpacing}
            {colorScale}
            nodeWidth={columnWidth}
            nodeHeight={rowHeight}
            minNodeScale={0.1}
            maxNodeScale={1}
            colorFunction={colorMetric != null
              ? (id) => models.find((m) => m.id == id).metrics[colorMetric]
              : null}
            radiusFunction={sizeMetric != null && !!sizeScale
              ? (id) => {
                  let sizeMetricValue = models.find((m) => m.id == id).metrics[
                    sizeMetric
                  ];
                  return sizeScale(sizeMetricValue);
                }
              : null}
          />
          <ModelMapCanvas
            {marks}
            {zoomTransform}
            state={mapState}
            bind:this={flowchart}
            {needsDraw}
            {hoveredIDs}
            showCrosshair={!isMultiselecting}
          />
        </Canvas>
        <Canvas>
          <Multiselect {multiselectRegion} />
        </Canvas>
        <Html pointerEvents={false}>
          <ModelTooltip
            {models}
            {metrics}
            {operations}
            {hoverInfo}
            offset={-rowHeight / 2}
          />
        </Html>
      </LayerCake>
    </div>
    {#if selectedIDs.length > 0}
      <div
        class="absolute bottom-4 right-4 flex z-10 gap-2 pointer-events-all"
        on:pointerdown={(e) => {
          // Make sure pointer events on these buttons don't propagate
          e.target.setPointerCapture(e.pointerId);
          e.stopImmediatePropagation();
        }}
      >
        <button class="tertiary-btn" on:click|stopPropagation={selectParents}
          >+ Parents</button
        >
        <button class="tertiary-btn" on:click|stopPropagation={selectChildren}
          >+ Children</button
        >
        <button class="tertiary-btn" on:click|stopPropagation={selectAncestors}
          >+ Ancestors</button
        >
        <button
          class="tertiary-btn"
          on:click|stopPropagation={selectDescendants}>+ Descendants</button
        >
      </div>
    {/if}
    <div
      class="absolute left-0 bottom-0 ml-4 mb-4 px-3 py-1 w-56 h-16 rounded bg-white/90"
    >
      <svg width="100%" height="100%" bind:this={legendContainer} />
    </div>
  </div>
</div>
