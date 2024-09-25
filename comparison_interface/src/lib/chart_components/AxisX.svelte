<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<!--
  @component
  Generates an SVG x-axis. This component is also configured to detect if your x-scale is an ordinal scale. If so, it will place the markers in the middle of the bandwidth.
 -->
<script>
  import { getContext } from "svelte";
  const { width, height, xScale, xRange, yRange, padding } =
    getContext("LayerCake");

  export let axisLabel = "";
  /** @type {Boolean} [gridlines=true] - Extend lines from the ticks into the chart space */
  export let gridlines = true;

  /** @type {Boolean} [tickMarks=false] - Show a vertical mark for each tick. */
  export let tickMarks = false;

  /** @type {Boolean} [baseline=false] – Show a solid line at the bottom. */
  export let baseline = false;

  /** @type {Boolean} [snapTicks=false] - Instead of centering the text on the first and the last items, align them to the edges of the chart. */
  export let snapTicks = false;

  /** @type {((v: number) => string) | undefined} [formatTick=d => d] - A function that passes the current tick value and expects a nicely formatted value in return. */
  export let formatTick = undefined;

  /** @type {Number|Number[]|Function|undefined} [ticks] - If this is a number, it passes that along to the [d3Scale.ticks](https://github.com/d3/d3-scale) function. If this is an array, hardcodes the ticks to those values. If it's a function, passes along the default tick values and expects an array of tick values in return. If nothing, it uses the default ticks supplied by the D3 function. */
  export let ticks = undefined;

  /** @type {Number} [xTick=0] - How far over to position the text marker. */
  export let xTick = 0;

  /** @type {Number} [yTick=16] - The distance from the baseline to place each tick value. */
  export let yTick = 16;

  $: isBandwidth = typeof $xScale.bandwidth === "function";

  $: tickVals = Array.isArray(ticks)
    ? ticks
    : isBandwidth
    ? $xScale.domain()
    : typeof ticks === "function"
    ? ticks($xScale.ticks())
    : $xScale.ticks(ticks);

  const labelLineHeight = 11;

  /**
   * @param {number} i
   * @returns {string} text alignment for the given tick number
   */
  function textAnchor(i) {
    if (snapTicks === true) {
      if (i === 0) {
        return "start";
      }
      if (i === tickVals.length - 1) {
        return "end";
      }
    }
    return "middle";
  }

  $: if (!formatTick) formatTick = (d) => `${d}`;
</script>

<g class="axis x-axis" class:snapTicks>
  {#each tickVals as tick, i (tick)}
    <g
      class="tick tick-{i}"
      transform="translate({$xScale(tick)},{Math.max(...$yRange)})"
    >
      {#if gridlines !== false}
        <line
          class="gridline"
          y1={$height * -1}
          y2="0"
          x1={isBandwidth ? $xScale.bandwidth() / 2 : 0}
          x2={isBandwidth ? $xScale.bandwidth() / 2 : 0}
        />
      {/if}
      {#if tickMarks === true}
        <line
          class="tick-mark"
          y1={0}
          y2={6}
          x1={isBandwidth ? $xScale.bandwidth() / 2 : 0}
          x2={isBandwidth ? $xScale.bandwidth() / 2 : 0}
        />
      {/if}
      <text dx="" dy="" text-anchor={textAnchor(i)}>
        {#each `${formatTick(tick)}`.split("\n") as line, i}
          <tspan
            x={isBandwidth ? $xScale.bandwidth() / 2 + xTick : xTick}
            y={yTick + i * labelLineHeight}>{line}</tspan
          >
        {/each}
      </text>
    </g>
  {/each}
  {#if baseline === true}
    {@const baselineY = Math.max($yRange[0], $yRange[1])}
    <line class="baseline" y1={baselineY} y2={baselineY} x1="0" x2={$width} />
  {/if}
  {#if axisLabel}
    <text
      class="axis-label"
      x={$width * 0.5}
      y={Math.max($yRange[0], $yRange[1]) + 36}
      text-anchor="middle">{axisLabel}</text
    >
  {/if}
</g>

<style>
  .tick .gridline {
    stroke: #e5e7eb;
    stroke-dasharray: 2;
  }

  line {
    stroke: #1f2937;
  }

  .baseline {
    stroke-dasharray: 0;
  }

  .axis-label {
    fill: #1f2937;
    font-size: 0.8em;
    font-weight: 500;
  }
  .tick {
    font-size: 0.725em;
  }

  /* This looks slightly better */
  .axis.snapTicks .tick:last-child text {
    transform: translateX(3px);
  }
  .axis.snapTicks .tick.tick-0 text {
    transform: translateX(-3px);
  }
</style>
