<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->
  
<script lang="ts">
  import { LayerCake, Svg } from "layercake";
  import * as d3 from "d3";
  import AxisX from "$lib/chart_components/AxisX.svelte";
  import AxisY from "$lib/chart_components/AxisY.svelte";
  import Column from "$lib/chart_components/Column.svg.svelte";
  import IntervalSelect from "$lib/chart_components/IntervalSelect.svg.svelte";
  import { createEventDispatcher } from "svelte";
  import { SystemBlue } from "$lib/utils";

  const dispatch = createEventDispatcher();

  export let xLabel = "";
  export let yLabel = "";
  export let dataForScales: number[] | null = null;
  export let allData: number[] = [];
  export let filteredData: number[] | null = null;

  export let filterBounds: [number, number] | null = null;

  export let xExtent: [number, number] | null = null;
  export let xFormat: ((x: number) => string) | null = null;
  export let colorScale: ((v: number) => string) | undefined = undefined;

  let chartContainer: HTMLElement;

  const chartPadding = { top: 24, left: 0, bottom: 36, right: 0 };

  const numberOfBins = 20;

  let xScale: d3.ScaleLinear<number, number>;
  let bandScale: d3.ScaleBand<string> | null = null; // used if the scale is discrete

  let histogram: d3.HistogramGeneratorNumber<number, number>;
  let fullXRange: [number, number] = [0, 1];
  let displayExtent: [number, number] = [0, 1];

  $: if (!!chartContainer && (dataForScales || allData).length > 0) {
    // Set the number of bins
    let dataToScale = dataForScales || allData;
    let uniqueValues = Array.from(new Set(dataToScale)).sort();

    displayExtent =
      xExtent != null ? xExtent : (d3.extent(dataToScale) as [number, number]);
    xScale = d3
      .scaleLinear()
      .domain(displayExtent)
      .range([
        0,
        chartContainer.clientWidth - chartPadding.left - chartPadding.right,
      ]);
    fullXRange = [
      dataToScale.reduce((a, b) => Math.min(a, b), 1e9),
      dataToScale.reduce((a, b) => Math.max(a, b), -1e9),
    ];

    // Generate the final bins array
    let ticks: number[];

    if (
      uniqueValues.length > 0 &&
      uniqueValues.indexOf(0) > 0 &&
      uniqueValues.length < numberOfBins &&
      uniqueValues.every((v) => Math.round(v) == v)
    ) {
      // integers!
      ticks = d3.range(
        uniqueValues[0],
        Math.max(uniqueValues[uniqueValues.length - 1], uniqueValues[0]) + 1
      );
      fullXRange = [ticks[0], ticks[ticks.length - 1]];
      // only use the band scale if zero is not at the beginning, as we'll want to place
      // zero at the origin otherwise
      bandScale = d3
        .scaleBand()
        .domain(ticks.map((t) => t.toString()))
        .range(xScale.range());
    } else {
      ticks = xScale
        .nice()
        .ticks(Math.max(2, Math.min(numberOfBins, uniqueValues.length)));
      fullXRange = [ticks[0], ticks[ticks.length - 1]];
      ticks = ticks.slice(0, ticks.length - 1);
      bandScale = null;
    }
    histogram = d3
      .bin()
      .domain(fullXRange)
      .value((d) => d)
      .thresholds(ticks);

    // Save the bins to a constant
    let bins = histogram(allData);
    allChartData = bins.map((bin) => ({
      x: bin.x0 || 0,
      count: bin.length,
    }));
    console.log(ticks, allChartData);
  }

  $: if (filteredData != null && !!histogram) {
    let bins = histogram(filteredData);
    filteredChartData = bins.map((bin) => ({
      x: bin.x0 || 0,
      count: bin.length,
    }));
  } else filteredChartData = null;

  let allChartData: { x: number; count: number }[] = [];
  let filteredChartData: { x: number; count: number }[] | null = null;

  let mouseDown = false;
  let initialSelectionPoint: number | null = null;

  function onMousedown(e: PointerEvent | MouseEvent) {
    mouseDown = true;
    if (e instanceof PointerEvent)
      chartContainer.setPointerCapture(e.pointerId);
  }

  function onMousemove(e: PointerEvent) {
    if (!mouseDown) return;
    if (!initialSelectionPoint) dispatch("brushstart");
    let coordX = xScale.invert(
      Math.min(
        chartContainer.clientWidth -
          (chartPadding.left || 0) -
          (chartPadding.right || 0),
        Math.max(
          0,
          e.clientX -
            chartContainer.getBoundingClientRect().left -
            (chartPadding.left || 0)
        )
      )
    );

    if (!initialSelectionPoint) {
      initialSelectionPoint = coordX;
      filterBounds = [coordX, coordX];
    }
    if (coordX > initialSelectionPoint)
      filterBounds = [filterBounds![0], coordX];
    else filterBounds = [coordX, filterBounds![1]];
  }

  function onMouseup(e: PointerEvent) {
    mouseDown = false;
    if (
      !initialSelectionPoint ||
      (filterBounds != null &&
        Math.abs(xScale(filterBounds[1]) - xScale(filterBounds[0])) <= 2)
    ) {
      // Clear selection
      filterBounds = null;
    }
    initialSelectionPoint = null;
    dispatch("brushend");
  }
</script>

<div
  unselectable="on"
  class="chart-container h-full w-full select-none cursor-crosshair"
  bind:this={chartContainer}
  on:mousedown|preventDefault={onMousedown}
  on:pointerdown|preventDefault={onMousedown}
  on:pointermove|preventDefault={onMousemove}
  on:pointerup|preventDefault={onMouseup}
>
  {#if allChartData.length > 0}
    <LayerCake
      padding={chartPadding}
      data={allChartData}
      x="x"
      y="count"
      xScale={bandScale || xScale}
      yDomain={[0, null]}
      xDomain={bandScale ? bandScale.domain() : xScale.domain()}
    >
      <Svg>
        <AxisY
          gridlines
          onRight
          baseline
          tickMarks
          dxTick={2}
          dyTick={4}
          ticks={3}
          formatTick={d3.format(".6~")}
          textAnchor="start"
          axisLabel={yLabel}
        />
        {#if filteredChartData != null}
          <Column
            data={allChartData}
            fill={"#e5e7eb"}
            useBandScale={!!bandScale}
          />
        {/if}
        <Column
          data={filteredChartData != null ? filteredChartData : allChartData}
          fill={SystemBlue}
          colorFn={colorScale ? (d) => colorScale(d.x) : null}
          stroke="#ddd"
          strokeWidth={1}
          useBandScale={!!bandScale}
        />
        {#if filterBounds != null && Math.abs(filterBounds[1] - filterBounds[0]) >= 0.01}
          <IntervalSelect
            scale={xScale}
            min={filterBounds[0]}
            max={filterBounds[1]}
            fill="#93c5fd44"
            stroke="#93c5fd"
            strokeWidth={2}
          />
        {/if}
        <AxisX
          gridlines={false}
          baseline
          tickMarks
          ticks={Math.min(5, allChartData.length)}
          formatTick={xFormat != null ? xFormat : undefined}
          axisLabel={xLabel}
        />
      </Svg>
    </LayerCake>
  {/if}
</div>
