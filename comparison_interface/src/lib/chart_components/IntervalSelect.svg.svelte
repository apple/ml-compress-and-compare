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

  const { xScale, yRange } = getContext("LayerCake");

  /** @type {d3.ScaleLinear<Number, Number> | null} */
  export let scale = null;

  /** @type {any[]} */
  export let data = [];
  /** @type {String} [fill='#00e047'] - The shape's fill color. */
  export let fill = "#4e79a7";

  /** @type {String} [stroke='#000'] - The shape's stroke color. */
  export let stroke = "#000";

  /** @type {Number} [strokeWidth=0] - The shape's stroke width. */
  export let strokeWidth = 0;

  /** @type {Number | Null} */
  export let min = null;
  /** @type {Number | Null} */
  export let max = null;

  /** @type {d3.ScaleLinear<Number, Number>} */
  let scaleToUse;
  $: scaleToUse = scale || $xScale;
</script>

{#if min != null && max != null}
  <g class="interval-select">
    <rect
      x={scaleToUse(min)}
      y={Math.min($yRange[0], $yRange[1])}
      width={scaleToUse(max) - scaleToUse(min)}
      height={Math.abs($yRange[1] - $yRange[0])}
      {fill}
    />
    {#if strokeWidth > 0}
      <line
        x1={scaleToUse(min)}
        y1={$yRange[0]}
        x2={scaleToUse(min)}
        y2={$yRange[1]}
        {stroke}
        stroke-width={strokeWidth}
      />
      <line
        x1={scaleToUse(max)}
        y1={$yRange[0]}
        x2={scaleToUse(max)}
        y2={$yRange[1]}
        {stroke}
        stroke-width={strokeWidth}
      />
    {/if}
  </g>
{/if}
