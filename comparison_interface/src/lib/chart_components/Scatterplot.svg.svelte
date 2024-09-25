<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->
  
<!--
  @component
  Generates an SVG scatter plot. This component can also work if the x- or y-scale is ordinal, i.e. it has a `.bandwidth` method. See the [timeplot chart](https://layercake.graphics/example/Timeplot) for an example.
 -->
<script lang="ts">
  import { createEventDispatcher, getContext } from "svelte";
  import type { Readable } from "svelte/store";
  import * as d3 from "d3";

  const dispatch = createEventDispatcher();

  const {
    data,
    xGet,
    yGet,
    zGet,
    rGet,
    xScale,
    yScale,
    zScale,
    width,
    height,
  } = getContext("LayerCake") as {
    data: Readable<any[]>;
    xGet: Readable<(d: any) => number>;
    yGet: Readable<(d: any) => number>;
    zGet: Readable<(d: any) => any>;
    rGet: Readable<(d: any) => number>;
    xScale: Readable<d3.ScaleSequentialBase<number>>;
    yScale: Readable<d3.ScaleSequentialBase<number>>;
    zScale: Readable<d3.ScaleSequentialBase<number>>;
    width: Readable<number>;
    height: Readable<number>;
  };

  /** @type {Number} [r=5] – The circle's radius. */
  export let r = 4;

  /** @type {String} [stroke='#000'] – The circle's stroke color. */
  export let stroke = "#000";

  /** @type {Number} [strokeWidth=0] – The circle's stroke width. */
  export let strokeWidth = 3;

  export let hoveredIDs: string[] = [];
  export let selectedIDs: string[] = [];
  export let filteredIDs: string[] = [];

  export let animate = false;
</script>

<defs>
  <clipPath id="inside-scatterplot">
    <rect x="0" y="0" width={$width} height={$height} />
  </clipPath>
</defs>
<g class="scatter-group" clip-path="url(#inside-scatterplot)">
  {#each $data as d (d.id)}
    {#if !selectedIDs.includes(d.id)}
      <circle
        class:animated-scatter-point={animate}
        cx={$xGet(d)}
        cy={$yGet(d)}
        r={hoveredIDs.includes(d.id) ? r * $rGet(d) * 1.5 : r * $rGet(d)}
        fill={d3.color($zGet(d))?.formatHex() +
          (filteredIDs.includes(d.id) || filteredIDs.length == 0 ? "bb" : "22")}
        stroke={"#d1d5db"}
        stroke-width={1}
        on:mouseenter={() => dispatch("hover", d)}
        on:mouseleave={() => dispatch("hover", null)}
        on:click|stopPropagation={(e) =>
          dispatch("click", { datum: d, event: e })}
      />
    {/if}
  {/each}
  {#each $data as d (d.id)}
    {#if selectedIDs.includes(d.id)}
      <circle
        class:animated-scatter-point={animate}
        cx={$xGet(d)}
        cy={$yGet(d)}
        r={hoveredIDs.includes(d.id) ? r * $rGet(d) * 1.5 : r * $rGet(d)}
        fill={d3.color($zGet(d))?.formatHex() + "bb"}
        stroke={"#60a5fa"}
        stroke-width={strokeWidth * 1.5}
        on:mouseenter={() => dispatch("hover", d)}
        on:mouseleave={() => dispatch("hover", null)}
        on:click|stopPropagation={(e) =>
          dispatch("click", { datum: d, event: e })}
      />
    {/if}
  {/each}
</g>

<style>
  .animated-scatter-point {
    transition: all 200ms ease-in-out;
  }
</style>
