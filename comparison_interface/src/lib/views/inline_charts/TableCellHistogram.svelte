<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->
  
<script lang="ts">
  import { format, precisionFixed } from "d3-format";
  import { LayerCake, Svg, Html } from "layercake";

  import HoverableColumn from "./HoverableColumn.svelte";
  import { scaleBand, scaleLinear } from "d3-scale";
  import type { Histogram } from "$lib/utils";

  export let width = 100;
  export let height = 22;

  export let histValues: Histogram;
  export let mean: number | null = null;
  export let yMax: number | null = null;

  let data: Array<{ bin: number; count: number }> = [];
  let histBins: Array<number> = [];

  $: if (!!histValues) {
    data = Object.entries(histValues).map((v) => ({
      bin: parseFloat(v[0]),
      count: <number>v[1],
    }));
    data.sort((a, b) => a.bin - b.bin);
    histBins = data.map((v) => v.bin);
  } else {
    data = [];
    histBins = [];
  }

  let hoveredBin: number;

  let binFormat = format(".3g");
  let countFormat = format(",");
  $: if (data.length > 0) {
    let precision = data.reduce(
      (curr, val, i) =>
        i > 0 ? Math.min(curr, Math.abs(val.bin - data[i - 1].bin)) : curr,
      1e9
    );
    binFormat = format(`.${precisionFixed(precision)}f`);
  }

  function makeTooltipText(d) {
    return `${binFormat(d.bin)}: ${countFormat(d.count)} instances`;
  }
</script>

<div style="width: {width}px; height: {height}px;">
  <LayerCake
    padding={{ top: 0, right: 0, bottom: 0, left: 0 }}
    x="bin"
    y="count"
    xScale={scaleBand().round(true).align(0)}
    xDomain={histBins}
    yScale={scaleLinear()}
    yDomain={[0, yMax]}
    {data}
    custom={{
      hoveredGet: (d) => d.bin == hoveredBin,
    }}
  >
    <Svg>
      <HoverableColumn
        baseline
        fill="#d1d5db"
        on:hover={(e) => (hoveredBin = e.detail != null ? e.detail.bin : null)}
      />
    </Svg>
  </LayerCake>
</div>
<div class="mt-1 text-xs text-slate-800 truncate">
  {#if !$$slots.caption}
    {#if hoveredBin != null}
      {makeTooltipText(data.find((d) => d.bin == hoveredBin))}
    {:else if mean != null}
      M = {format(".3")(mean)}
    {:else}
      &nbsp;{/if}
  {:else}
    <slot
      name="caption"
      hoveringIndex={hoveredBin != null
        ? data.findIndex((d) => d.bin == hoveredBin)
        : null}
    />
  {/if}
</div>
