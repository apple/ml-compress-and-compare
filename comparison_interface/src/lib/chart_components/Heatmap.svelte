<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->
  
<!--
  @component
  Generates SVG rectangles for each data point.
 -->
<script>
  import { createEventDispatcher, getContext } from "svelte";
  import { interpolateRdBu } from "d3-scale-chromatic";

  const dispatch = createEventDispatcher();

  const { data, xGet, yGet, zGet, xDomain, yDomain, xScale, yScale, custom } =
    getContext("LayerCake");

  export let colorMap = interpolateRdBu;
  export let nullColor = "black";

  /** @type {String} [stroke='#ab00d6'] - The shape's fill color. This is technically optional because it comes with a default value but you'll likely want to replace it with your own color. */
  export let stroke = "#ab00d6";

  export let padding = 0;

  let width = 0;
  let height = 0;

  $: width = Math.abs($xScale($xDomain[1]) - $xScale($xDomain[0]));
  $: height = Math.abs($yScale($yDomain[1]) - $yScale($yDomain[0]));

  function getHovered(customVals, d) {
    let fn = (customVals || {}).hoveredGet || (() => false);
    return fn(d);
  }

  function getSelected(customVals, d) {
    let fn = (customVals || {}).selectedGet || (() => false);
    return fn(d);
  }
</script>

{#each $data as d}
  <rect
    x={$xGet(d) + padding / 2}
    y={$yGet(d) + padding / 2}
    width={width - padding}
    height={height - padding}
    style="fill:{$zGet(d) == null ? nullColor : colorMap($zGet(d))}"
    class:hovered-rect={getHovered($custom, d)}
    on:mouseenter={() => dispatch("hover", d)}
    on:mouseleave={() => dispatch("hover", null)}
    on:click={() => dispatch("click", d)}
  />
  {#if getSelected($custom, d)}
    <rect
      x={$xGet(d) + padding / 2}
      y={$yGet(d) + padding / 2}
      width={width - padding}
      height={height - padding}
      class="selected-rect"
    />
  {/if}
{/each}

<style>
  rect {
    z-index: 0;
  }
  .hovered-rect {
    stroke: gray;
    stroke-width: 2;
  }

  .selected-rect {
    fill: none;
    stroke: black;
    stroke-width: 3;
    stroke-linejoin: round;
    stroke-linecap: round;
    z-index: 10;
  }
</style>
