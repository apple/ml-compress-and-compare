<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import { LayerCake, Svg, WebGL, Html, Canvas } from "layercake";
  import * as d3 from "d3";

  import AxisX from "$lib/chart_components/AxisX.svelte";
  import AxisY from "$lib/chart_components/AxisY.svelte";
  import type { Metric, Model } from "$lib/data_structures/model_map";
  import GroupedColumn from "$lib/chart_components/GroupedColumn.svg.svelte";
  import type { ModelTable } from "$lib/data_structures/model_table";
  import { schemeApple8, wrapTextByCharacterCount } from "$lib/utils";
  import CategoricalLegend from "$lib/chart_components/CategoricalLegend.svelte";
  import { createEventDispatcher } from "svelte";

  const dispatch = createEventDispatcher();

  export let metrics: Metric[] = [];
  export let models: Model[] = [];
  export let modelTable: ModelTable | null = null;

  export let xVariable: number | null;
  export let yEncoding: string | null;
  export let colorVariable: number | null = null;

  export let colorScheme = d3.schemeTableau10;

  export let showLegend = true;
  export let showXLabel = true;

  export let hoveredIDs: string[] = [];

  let colorDomain: string[] | null;

  const chartPadding = { top: 12, left: 48, bottom: 36, right: 12 };
  const maxLineLength = 15;

  let chartContainer: HTMLElement;

  let visualizationData: { x: any; y: any; color: any }[] = [];
  $: if (!!modelTable && modelTable.size > 0) {
    visualizationData = modelTable.models.map((modelInfo) => {
      let x = modelInfo.values[xVariable!];
      let color = colorVariable != null ? modelInfo.values[colorVariable!] : 1;
      let y =
        yEncoding != null
          ? models.find((m) => m.id == modelInfo.id)!.metrics[yEncoding]
          : 0.0;
      return {
        id: modelInfo.id,
        x:
          typeof x === "string"
            ? wrapTextByCharacterCount(x, maxLineLength)
            : x,
        color,
        y,
      };
    });
    colorDomain =
      colorVariable != null && modelTable.orderings.has(colorVariable)
        ? (modelTable.orderings.get(colorVariable) as string[])
        : Array.from(new Set(visualizationData.map((d) => d.color))).sort(
            sortChartValuesFn
          );
  } else {
    visualizationData = [];
    colorDomain = null;
  }

  function sortChartValuesFn(a, b) {
    let aString = `${a}`;
    let bString = `${b}`;
    if (
      aString.toLocaleLowerCase() == "none" ||
      aString.toLocaleLowerCase() == "null"
    )
      return -1;
    if (
      bString.toLocaleLowerCase() == "none" ||
      bString.toLocaleLowerCase() == "null"
    )
      return 1;
    return aString.localeCompare(bString);
  }

  let yTickFormat: any;
  $: yTickFormat =
    Math.abs(visualizationData.reduce((a, b) => Math.max(a, b.y), 0)) >= 1000
      ? d3.format(".3~s")
      : d3.format(".3~");
</script>

{#if showLegend && colorVariable != null && !!modelTable}
  <div class="mb-6 p-2 pb-1 rounded bg-gray-100">
    {#if modelTable.variables[colorVariable]}
      <div class="pb-2 font-bold" style="font-size: 0.8em;">
        {modelTable.variables[colorVariable] || ""}
      </div>
    {/if}
    <CategoricalLegend
      colorScale={d3.scaleOrdinal(colorScheme).domain(colorDomain)}
    />
  </div>
{/if}
{#if visualizationData.length > 0 && !!modelTable && xVariable != null}
  <div class="w-full h-36 max-h-full">
    <div
      unselectable="on"
      class="chart-container h-full max-w-full select-none {hoveredIDs.length >
      0
        ? 'cursor-pointer'
        : ''}"
      style="width: {modelTable.models.length * 200}px;"
      bind:this={chartContainer}
    >
      <LayerCake
        padding={chartPadding}
        data={visualizationData}
        x="x"
        y="y"
        z="color"
        xDomain={Array.from(new Set(visualizationData.map((d) => d.x))).sort(
          sortChartValuesFn
        )}
        yDomain={[0, null]}
        zDomain={colorDomain}
        xScale={d3.scaleBand().paddingInner(0.1).paddingOuter(0.05)}
        yScale={d3.scaleLinear()}
        zScale={d3.scaleOrdinal()}
        zRange={colorVariable != null ? colorScheme : schemeApple8}
      >
        <Svg>
          <AxisX
            gridlines
            baseline
            tickMarks
            ticks={5}
            axisLabel={showXLabel ? modelTable.variables[xVariable] || "" : ""}
          />
          <AxisY
            gridlines
            baseline
            tickMarks
            dxTick={0}
            dyTick={4}
            ticks={5}
            formatTick={yTickFormat}
            textAnchor="end"
            axisLabel={yEncoding || ""}
          />
          <GroupedColumn
            {hoveredIDs}
            labelFormat={yTickFormat}
            on:hover={(e) => {
              if (!!e.detail) hoveredIDs = [e.detail.id];
              else hoveredIDs = [];
            }}
            on:click={(e) => {
              dispatch("select", e.detail.id);
            }}
          />
        </Svg>
      </LayerCake>
    </div>
  </div>
{/if}
