<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import { format } from "d3-format";
  import { schemeTableau10 } from "d3-scale-chromatic";
  import { LayerCake, Svg, Html } from "layercake";
  import { scaleOrdinal, scaleLinear } from "d3-scale";
  import { range } from "d3-array";

  import BarTooltip from "./BarTooltip.svelte";
  import BarSegment from "./BarSegment.svelte";

  export let width = 100;

  export let counts: { [key: string]: number } | null = null;

  export let order: Array<string> = [];

  interface Datum {
    name: string;
    start: number;
    end: number;
    index: number;
    count: number;
  }
  let data: Array<Datum> = [];

  $: if (!!counts && order.length > 0) {
    let totalCount = Object.values(counts).reduce((curr, val) => curr + val, 0);
    let runningCount = 0;
    data = order.map((d, i) => {
      let curr = runningCount;
      runningCount += counts![d] || 0;
      return {
        start: curr / totalCount,
        end: runningCount / totalCount,
        index: i,
        name: d,
        count: counts![d] || 0,
      };
    });
    console.log(data);
  } else {
    data = [];
  }

  let hoveredIndex: number;

  let countFormat = format(",");

  function makeTooltipText(d: Datum) {
    return `${d.index}: ${countFormat(d.count)} instances`;
  }
</script>

<div
  style="width: {width}px; height: 6px;"
  class="inline-block rounded overflow-hidden"
>
  <LayerCake
    padding={{ top: 0, right: 0, bottom: 0, left: 0 }}
    x="start"
    y="index"
    z="end"
    xScale={scaleLinear()}
    xDomain={[0, 1]}
    xRange={[0, width]}
    yScale={scaleOrdinal()}
    yDomain={range(counts.length)}
    yRange={schemeTableau10}
    {data}
    custom={{
      hoveredGet: (d) => d.index == hoveredIndex,
    }}
  >
    <Html>
      <BarSegment
        on:hover={(e) => (hoveredIndex = e.detail ? e.detail.index : null)}
      />
    </Html>
  </LayerCake>
</div>
<div class="text-xs text-slate-800">
  {#if $$slots.caption}
    <slot name="caption" />
  {:else if hoveredIndex != null}
    {makeTooltipText(data[hoveredIndex])}
  {:else}
    &nbsp;
  {/if}
</div>
