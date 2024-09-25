<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->
  
<!--
  @component
  Generates an SVG column chart.
 -->
<script>
  import { createEventDispatcher, getContext } from "svelte";
  import { onMount } from "svelte";
  import * as d3 from "d3";

  const dispatch = createEventDispatcher();

  const {
    data,
    xGet,
    yGet,
    x,
    xRange,
    yRange,
    xScale,
    y,
    yScale,
    height,
    zGet,
    zScale,
    z,
    custom,
  } = getContext("LayerCake");

  export let baseline = false;

  /** @type {String} [fill='#00e047'] - The shape's fill color. */
  export let fill = "#00e047";

  export let colorScale = null;

  /** @type {Boolean} [false] - Show the numbers for each column */
  export let showLabels = false;

  /** @type {(datum: any) => number} */
  export let y2Get;

  $: columnWidth = (d) => {
    const vals = $xGet(d);
    return Math.abs(vals[1] - vals[0]);
  };

  $: columnHeight = (d) => {
    return $yScale(y2Get(d)) - $yGet(d);
  };

  export let stroke = "#ddd";
  export let strokeWidth = 1;

  const hoverStroke = "#333";
  const hoverStrokeWidth = 2;
  const selectStrokeWidth = 3;

  /** @type {number | null} */
  let hoveredIndex = null;

  // Disable transition until after loaded
  onMount(() => {
    setTimeout(() => (loaded = true), 100);
  });

  let loaded = false;
</script>

{#if baseline}
  <line
    x1={$xRange[0]}
    x2={$xRange[1]}
    y1={Math.max($yRange[0], $yRange[1])}
    y2={Math.max($yRange[0], $yRange[1])}
    {stroke}
    stroke-width={1}
  />
{/if}
<g class="column-group">
  {#each $data as d, i}
    {@const colHeight = columnHeight(d)}
    {@const xGot = $xGet(d)}
    {@const xPos = Array.isArray(xGot) ? xGot[0] : xGot}
    {@const colWidth = $xScale.bandwidth ? $xScale.bandwidth() : columnWidth(d)}
    {@const yValue = $y(d)}
    {#if colHeight > 0.0}
      <rect
        class="group-rect hover-zone"
        class:animated={loaded}
        data-id={i}
        data-range={$x(d)}
        data-count={yValue}
        x={xPos}
        y={$yGet(d)}
        width={colWidth}
        height={colHeight}
        fill={colorScale != null ? colorScale($zGet(d)) : fill}
        stroke={hoveredIndex == i ? hoverStroke : stroke}
        stroke-width={hoveredIndex == i ? hoverStrokeWidth : strokeWidth}
        on:mouseenter={() => {
          hoveredIndex = i;
          dispatch("hover", d);
        }}
        on:mouseleave={() => {
          hoveredIndex = null;
          dispatch("hover", null);
        }}
      />
      {#if showLabels && yValue}
        <text
          x={xPos + colWidth / 2}
          y={$height - colHeight - 5}
          text-anchor="middle">{yValue}</text
        >
      {/if}
    {/if}
  {/each}
</g>

<style>
  text {
    font-size: 12px;
  }
  .hover-zone {
    pointer-events: all;
  }

  .animated {
    @apply transition-all duration-300 ease-in-out;
  }
</style>
