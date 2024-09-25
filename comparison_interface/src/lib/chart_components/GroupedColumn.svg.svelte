<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->
  
<!--
  @component
  Generates an SVG column chart.
 -->
<script>
  import * as d3 from "d3";
  import { createEventDispatcher, getContext } from "svelte";

  const {
    data,
    xGet,
    yGet,
    x,
    yRange,
    xScale,
    y,
    height,
    zGet,
    zScale,
    zRange,
    z,
    zDomain,
  } = getContext("LayerCake");

  const dispatch = createEventDispatcher();

  /** @type {String} [fill='#00e047'] - The shape's fill color. */
  export let fill = "#00e047";

  /** @type {String} [stroke='#000'] - The shape's stroke color. */
  export let stroke = "#000";

  /** @type {Number} [strokeWidth=0] - The shape's stroke width. */
  export let strokeWidth = 0;

  /** @type {Boolean} [false] - Show the numbers for each column */
  export let showLabels = false;

  export let labelFormat = d3.format(".3~g");

  let groupScale = d3.scaleBand();

  $: columnWidth = (d) => {
    const vals = $xGet(d);
    return Math.abs(vals[1] - vals[0]);
  };

  $: columnHeight = (d) => {
    return $yRange[0] - $yGet(d);
  };

  $: {
    groupScale = d3
      .scaleBand()
      .domain($zDomain)
      .range([0, $xScale.bandwidth()]);
  }

  /** @type {String[]} - the IDs to fade out for hover */
  export let hoveredIDs = [];
</script>

<g class="column-group">
  {#each $data as d, i}
    {@const colHeight = columnHeight(d)}
    {@const xGot = $xGet(d)}
    {@const xPos = Array.isArray(xGot) ? xGot[0] : xGot}
    {@const colWidth = groupScale.bandwidth()}
    {@const yValue = $y(d)}
    {@const fillValue = $zGet(d)}
    <rect
      class="group-rect"
      data-id={i}
      data-range={$x(d)}
      data-count={yValue}
      x={xPos + groupScale($z(d)) + 1}
      y={$yGet(d)}
      width={colWidth - 2}
      height={colHeight}
      fill={fillValue}
      opacity={hoveredIDs.includes(d.id) ? 0.7 : 1}
      {stroke}
      stroke-width={strokeWidth}
      on:mouseenter={(e) => dispatch("hover", d)}
      on:mouseleave={(e) => dispatch("hover", null)}
      on:click={(e) => dispatch("click", d)}
      on:keypress={(e) => {
        if (e.key === "Enter") dispatch("click", d);
      }}
      role="figure"
    />
    {#if (showLabels || hoveredIDs.includes(d.id)) && yValue != null}
      <text
        x={xPos + groupScale($z(d)) + colWidth * 0.5}
        y={$height - colHeight - 5}
        fill="#555"
        text-anchor="middle">{labelFormat(yValue)}</text
      >
    {/if}
  {/each}
</g>

<style>
  text {
    font-size: 12px;
  }

  rect {
    transition: opacity 200ms ease-in-out;
  }
</style>
