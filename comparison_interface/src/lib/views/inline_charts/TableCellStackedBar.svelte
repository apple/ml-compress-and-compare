<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import { format } from "d3-format";
  import { LayerCake, Svg, Html } from "layercake";

  import HoverableColumn from "./HoverableColumn.svelte";
  import { scaleBand } from "d3-scale";
  import * as d3 from "d3";
  import StackedColumn from "./StackedColumn.svelte";

  export let width = 100;
  export let height = 22;

  export let histValues: { bins: [number[], number[]]; values: number[][] };
  export let mean: number | null = null;
  export let yMax: number | null = null;
  export let zMax: number | null = null;
  export let colorMap = d3.interpolateYlGnBu;

  let data: Array<{ x: number; top: number; bottom: number; color: number }> =
    [];
  let xBins: number[] = [];
  let yBins: number[] = [];

  $: if (!!histValues) {
    data = histValues.values
      .map((row, i) => {
        let cumulativeSum = 0;
        return row.map((v, j) => {
          let bottom = cumulativeSum;
          cumulativeSum += v;
          return {
            x: histValues.bins[0][i],
            color: histValues.bins[1][j],
            top: bottom + v,
            bottom,
            count: v,
          };
        });
      })
      .flat();
    xBins = histValues.bins[0];
    yBins = histValues.bins[1];
  } else {
    data = [];
    xBins = [];
    yBins = [];
  }

  let hoveredItem: any | null = null;

  let binFormat = format(".3~g");
  let countFormat = format(".2%");

  function makeTooltipText(d) {
    return `Base ${binFormat(d.x)}, change ${binFormat(d.color)}: ${countFormat(
      d.count
    )}`;
  }
</script>

{#if !!data && xBins.length > 0 && yBins.length > 0}
  <div style="width: {width}px; height: {height}px;">
    <LayerCake
      padding={{ top: 0, right: 0, bottom: 0, left: 0 }}
      x="x"
      y="top"
      z="color"
      xScale={scaleBand().round(true).align(0)}
      xDomain={xBins}
      yDomain={[0, yMax]}
      zDomain={[Math.min(...yBins), zMax]}
      zRange={[0, 1]}
      {data}
      custom={{
        hoveredGet: (d) =>
          !!hoveredItem && d.x == hoveredItem.x && d.color == hoveredItem.color,
      }}
    >
      <Svg>
        <HoverableColumn
          draw={false}
          on:hover={(e) =>
            (hoveredItem =
              e.detail != null
                ? Object.assign({ wholeColumn: true }, e.detail)
                : null)}
        />
        <StackedColumn
          baseline
          colorScale={colorMap}
          y2Get={(d) => d.bottom}
          on:hover={(e) => (hoveredItem = e.detail != null ? e.detail : null)}
        />
      </Svg>
    </LayerCake>
  </div>
  <div class="mt-1 text-xs text-slate-800 truncate">
    {#if !$$slots.caption}
      {#if hoveredItem != null}
        {makeTooltipText(
          data.find(
            (d) =>
              !!hoveredItem &&
              d.x == hoveredItem.x &&
              d.color == hoveredItem.color
          )
        )}
      {:else}
        &nbsp;{/if}
    {:else}
      <slot
        name="caption"
        hoveringItem={hoveredItem != null
          ? hoveredItem.wholeColumn
            ? { x: hoveredItem.x }
            : {
                x: hoveredItem.x,
                y: hoveredItem.color,
                value: hoveredItem.top - hoveredItem.bottom,
                proportion:
                  (hoveredItem.top - hoveredItem.bottom) /
                  data
                    .filter((d) => d.x == hoveredItem.x)
                    .reduce((a, b) => a + b.top - b.bottom, 0),
              }
          : null}
      />
    {/if}
  </div>
{/if}
