<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import { LayerCake, Svg, WebGL, Html, Canvas } from "layercake";
  import { zoomIdentity, zoom } from "d3-zoom";
  import { select } from "d3-selection";
  import * as d3 from "d3";

  import AxisX from "$lib/chart_components/AxisX.svelte";
  import AxisY from "$lib/chart_components/AxisY.svelte";
  import {
    calculateDomain,
    type Metric,
    type Model,
  } from "$lib/data_structures/model_map";
  import Scatterplot from "$lib/chart_components/Scatterplot.svg.svelte";
  import type { Rectangle } from "$lib/utils";
  import Multiselect from "$lib/chart_components/Multiselect.canvas.svelte";

  export let metrics: Metric[] = [];
  export let models: Model[] = [];

  export let xEncoding: string | null;
  export let yEncoding: string | null;
  export let colorEncoding: string | null;
  export let sizeEncoding: string | null;

  export let colorScale = d3.scaleSequential(d3.interpolateBlues);

  export let hoveredIDs: string[] = [];
  export let selectedIDs: string[] = [];
  export let filteredIDs: string[] = [];

  let zoomer: d3.ZoomBehavior<Element, unknown>;
  export let zoomTransform = zoomIdentity;
  let zooming = false;
  let panning = false;

  const chartPadding = { top: 16, left: 36, bottom: 36, right: 12 };

  $: zoomer = zoom()
    .filter((e) => (e.type !== "wheel" || e.ctrlKey) && e.type !== "mousedown")
    .scaleExtent([0.5, 32])
    .on("zoom", function (e) {
      zoomTransform = e.transform;
    })
    .on("start", (e) => {
      if (!!e.sourceEvent) {
        zooming = true;
      }
      mouseDown = true;
    })
    .on("end", (e) => {
      zooming = false;
      mouseDown = false;
    });

  $: {
    xEncoding, yEncoding;
    select(chartContainer)
      .transition()
      .duration(200)
      .call(zoomer.transform, d3.zoomIdentity);
  }

  let chartContainer: HTMLElement;
  $: if (!!chartContainer && !!zoomer) {
    select(chartContainer).call(zoomer);
  }

  let wheelEndTimeout: NodeJS.Timeout | null;
  function pan(e: WheelEvent) {
    panning = true;
    zoomer.translateBy(
      d3.select(chartContainer),
      -e.deltaX / zoomTransform.k,
      -e.deltaY / zoomTransform.k
    );
    if (!!wheelEndTimeout) clearTimeout(wheelEndTimeout);
    wheelEndTimeout = setTimeout(() => (panning = false), 100);
  }

  let xScale: d3.ScaleLinear<number, number>;
  let yScale: d3.ScaleLinear<number, number>;
  let zDomain: [number, number];
  let rDomain: [number, number];

  let xDomain: [number, number] | null;
  $: if (!!xEncoding && models.length > 0)
    xDomain = calculateDomain(metrics, models, xEncoding);
  else xDomain = null;
  let yDomain: [number, number] | null;
  // flip y domain to put 0 at the bottom
  $: if (!!yEncoding && models.length > 0)
    yDomain = calculateDomain(metrics, models, yEncoding).slice().reverse() as [
      number,
      number
    ];
  else yDomain = null;

  $: if (!!chartContainer && models.length > 0 && !!xDomain) {
    let xRange = [
      0,
      chartContainer.clientWidth - chartPadding.left - chartPadding.right,
    ];
    xScale = d3.scaleLinear().domain(xDomain).nice().range(xRange);
    xScale = xScale.domain([
      xScale.invert(zoomTransform.invertX(xRange[0])),
      xScale.invert(zoomTransform.invertX(xRange[1])),
    ]);
  }
  $: if (!!chartContainer && models.length > 0 && !!yDomain) {
    let yRange = [
      0,
      chartContainer.clientHeight - chartPadding.top - chartPadding.bottom,
    ];
    yScale = d3.scaleLinear().domain(yDomain).nice().range(yRange);
    yScale = yScale.domain([
      yScale.invert(zoomTransform.invertY(yRange[0])),
      yScale.invert(zoomTransform.invertY(yRange[1])),
    ]);
  }

  $: if (!!chartContainer && models.length > 0 && !!colorEncoding) {
    zDomain = calculateDomain(metrics, models, colorEncoding);
  } else {
    zDomain = [0, 1];
  }

  $: if (!!chartContainer && models.length > 0 && !!sizeEncoding) {
    rDomain = calculateDomain(metrics, models, sizeEncoding);
  } else {
    rDomain = [0, 1];
  }

  let isMultiselecting = false;
  let multiselectRegion: Rectangle | null = null;

  function handleNodeClick(datum: Model | null, e: PointerEvent) {
    if (isMultiselecting && !!multiselectRegion) {
      if (e.shiftKey)
        selectedIDs = Array.from(
          new Set([...selectedIDs, ...nodesInRegion(multiselectRegion)])
        );
      else selectedIDs = nodesInRegion(multiselectRegion);
      hoveredIDs = [];
      multiselectRegion = null;
      isMultiselecting = false;
      return;
    }

    let nodeID = datum != null ? datum.id : null;
    console.log("clicked", datum);
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
  }

  let mouseDown = false;

  function onMousemove(e: PointerEvent) {
    let coord = {
      x:
        e.clientX -
        chartContainer.getBoundingClientRect().left -
        (chartPadding.left || 0),
      y:
        e.clientY -
        chartContainer.getBoundingClientRect().top -
        (chartPadding.top || 0),
    };
    if (mouseDown) {
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
      e.stopPropagation();
      chartContainer.setPointerCapture(e.pointerId);
    }
  }

  function onMouseup(e: PointerEvent) {
    mouseDown = false;
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

    return models
      .filter((model) => {
        if (filteredIDs.length > 0 && !filteredIDs.includes(model.id))
          return false;
        let x = xScale(model.metrics[xEncoding]);
        let y = yScale(model.metrics[yEncoding]);
        return (
          x >= region.x &&
          x <= region.x + region.width &&
          y >= region.y &&
          y <= region.y + region.height
        );
      })
      .map((m) => m.id);
  }
</script>

<div class="w-full h-full px-2 py-3">
  <div
    unselectable="on"
    class="chart-container h-full w-full select-none {hoveredIDs.length == 1 &&
    (filteredIDs.length == 0 || filteredIDs.includes(hoveredIDs[0])) &&
    !isMultiselecting
      ? 'cursor-pointer'
      : 'cursor-crosshair'}"
    bind:this={chartContainer}
    on:wheel|stopPropagation|preventDefault={pan}
    on:click={(e) => handleNodeClick(null, e)}
    on:keydown={() => {}}
    on:pointerdown={(e) => {
      mouseDown = true;
      if (e.shiftKey) e.stopPropagation();
    }}
    on:mousedown={(e) => {
      mouseDown = true;
      if (e.shiftKey) e.stopPropagation();
    }}
    on:pointermove={onMousemove}
    on:pointerup={onMouseup}
    role="button"
    tabindex="0"
  >
    {#if models.length > 0 && xScale && yScale}
      <LayerCake
        padding={chartPadding}
        data={models}
        x={(m) => m.metrics[xEncoding]}
        y={(m) => m.metrics[yEncoding]}
        z={colorEncoding != null ? (m) => m.metrics[colorEncoding] : (d) => 1}
        r={sizeEncoding != null ? (m) => m.metrics[sizeEncoding] : (d) => 0.2}
        xDomain={xScale.domain()}
        yDomain={yScale.domain()}
        xRange={xScale.range()}
        yRange={yScale.range()}
        {zDomain}
        zScale={colorEncoding != null
          ? colorScale
          : d3.scaleSequential(d3.interpolateBlues)}
        {rDomain}
        rRange={[0.75, 2]}
      >
        <Canvas>
          <Multiselect {multiselectRegion} />
        </Canvas>

        <Svg>
          <AxisX
            gridlines
            baseline
            tickMarks
            ticks={5}
            formatTick={Math.abs(
              Math.max(xScale.domain()[0], xScale.domain()[1])
            ) >= 1000
              ? d3.format("~s")
              : d3.format(".4")}
            axisLabel={xEncoding || ""}
          />
          <AxisY
            gridlines
            baseline
            tickMarks
            dxTick={0}
            dyTick={4}
            ticks={5}
            textAnchor="end"
            formatTick={Math.abs(
              Math.max(yScale.domain()[0], yScale.domain()[1])
            ) >= 1000
              ? d3.format("~s")
              : d3.format(".4")}
            axisLabel={yEncoding || ""}
          />
          <Scatterplot
            {hoveredIDs}
            {selectedIDs}
            {filteredIDs}
            animate={!zooming && !panning}
            on:hover={(e) => {
              console.log("hover");
              hoveredIDs = e.detail ? [e.detail.id] : [];
            }}
            on:click={(e) => handleNodeClick(e.detail.datum, e.detail.event)}
          />
        </Svg>
      </LayerCake>
    {/if}
  </div>
</div>
