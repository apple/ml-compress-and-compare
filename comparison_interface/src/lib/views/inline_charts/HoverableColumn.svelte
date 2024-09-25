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
    height,
    zGet,
    zScale,
    z,
    custom,
  } = getContext("LayerCake");

  export let draw = true; // set to false to only set hover zones

  /** @type {String} [fill='#00e047'] - The shape's fill color. */
  export let fill = "#00e047";

  /** @type {Boolean} [false] - Show the numbers for each column */
  export let showLabels = false;

  export let baseline = false;

  export let stroke = "#ddd";
  export let strokeWidth = 1;

  $: columnWidth = (d) => {
    const vals = $xGet(d);
    return Math.abs(vals[1] - vals[0]);
  };

  $: columnHeight = (d) => {
    return $yRange[0] - $yGet(d);
  };

  const hoverStroke = "#333";
  const hoverStrokeWidth = 1;
  const selectStrokeWidth = 3;

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
    {#if draw}
      <rect
        class="group-rect"
        class:animated={loaded}
        data-id={i}
        data-range={$x(d)}
        data-count={yValue}
        x={xPos}
        y={$yGet(d)}
        width={colWidth}
        height={colHeight}
        {fill}
        stroke={hoveredIndex == i ? hoverStroke : stroke}
        stroke-width={hoveredIndex == i ? hoverStrokeWidth : strokeWidth}
      />
    {/if}
    <rect
      class="hover-zone"
      x={xPos}
      y={0}
      width={colWidth}
      height={$height}
      fill="none"
      stroke="none"
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
