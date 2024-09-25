<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<!--
  @component
  Generates an SVG column chart.
 -->
<script>
  import { getContext } from "svelte";
  import * as d3 from "d3";

  const { xGet, yGet, x, xScale, yRange, xRange, y, height, zGet, zScale, z } =
    getContext("LayerCake");

  /** @type {any[]} */
  export let data = [];
  /** @type {String} [fill='#00e047'] - The shape's fill color. */
  export let fill = "#4e79a7";

  /** @type {Function | null} */
  export let colorFn = null;

  /** @type {String} [stroke='#000'] - The shape's stroke color. */
  export let stroke = "#000";

  /** @type {Number} [strokeWidth=0] - The shape's stroke width. */
  export let strokeWidth = 0;

  /** @type {Boolean} [false] - Show the numbers for each column */
  export let showLabels = false;

  $: columnWidth = (d) => {
    const vals = $xGet(d);
    if (vals.length > 0) return Math.abs(vals[1] - vals[0]);
    return 1;
  };

  $: columnHeight = (d) => {
    return $yRange[0] - $yGet(d);
  };

  export let useBandScale = false;

  let innerScale = $xScale;
  let binWidth = 1;
  $: if (useBandScale && !!data && data.length > 1) {
    binWidth = $x(data[1]) - $x(data[0]);
    innerScale = d3
      .scaleBand()
      .padding(0.05)
      .domain(data.map((d) => $x(d) + binWidth * 0.5))
      .range($xRange);
  } else {
    binWidth = $x(data[1]) - $x(data[0]);
    innerScale = $xScale;
  }
</script>

<g class="column-group">
  {#each data as d, i}
    {@const colHeight = columnHeight(d)}
    {@const xGot = innerScale($x(d) + (useBandScale ? binWidth * 0.5 : 0))}
    {@const xPos = Array.isArray(xGot) ? xGot[0] : xGot}
    {@const colWidth = innerScale.bandwidth
      ? innerScale.bandwidth()
      : $xScale(binWidth) - $xScale(0)}
    {@const yValue = $y(d)}
    <rect
      class="group-rect"
      data-id={i}
      data-range={$x(d)}
      data-count={yValue}
      x={xPos}
      y={$yGet(d)}
      width={colWidth}
      height={colHeight}
      fill={colorFn ? colorFn(d) : fill}
      {stroke}
      stroke-width={strokeWidth}
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

  .group-rect {
    transition: height 100ms ease-in-out, y 100ms ease-in-out;
  }
</style>
