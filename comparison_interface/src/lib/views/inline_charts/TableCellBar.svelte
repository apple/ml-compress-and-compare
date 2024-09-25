<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import { format } from "d3-format";
  import * as d3 from "d3";
  import { interpolateViridis, schemeCategory10 } from "d3-scale-chromatic";
  import InlineBar from "./InlineBar.svelte";
  import { cumulativeSum } from "$lib/utils";

  export let width: number = 100;
  export let height: number = 6;
  export let scale: ((x: number) => number) | null = null;

  export let originCenter: boolean = false;

  export let value = 0.0;
  export let values: number[] | null = null;
  export let showFullBar = false;

  export let colors = schemeCategory10;
  export let hoverable = false;
  export let visible = true;

  export let labelOnBar = false; // only for single value bars
  export let labelFormat: (x: number) => string = d3.format(".3~");
  export let labelHidden = false;

  let hoveringIndex: number | null = null;

  let offsets: number[] = [];
  $: if (values != null) {
    offsets = cumulativeSum(values);
  } else offsets = [];

  function calculateBarSegment(
    value: number,
    index: number,
    center: boolean = false
  ): [number, number] {
    if (center) {
      let left = index > 0 ? (scale || ((x) => x))(offsets[index - 1]) : 0;
      left = left * 0.5 + 0.5;
      let right = left + (scale || ((x) => x))(value) * 0.5;
      return [
        Math.min(left, right),
        Math.max(left, right) - Math.min(left, right),
      ];
    }
    return [
      index > 0 ? (scale || ((x) => x))(offsets[index - 1]) : 0,
      (scale || ((x) => x))(value),
    ];
  }
</script>

<div
  class="parent-bar relative mb-1 rounded-full overflow-hidden"
  style="width: {width}px; height: {height}px;"
>
  {#if visible}
    {#if showFullBar}
      <InlineBar
        absolutePosition
        maxWidth={width}
        fraction={1.0}
        color="#e5e7eb"
        {hoverable}
        {height}
        on:mouseenter={(e) => (hoveringIndex = -1)}
        on:mouseleave={(e) => (hoveringIndex = null)}
      />
    {/if}
    {#if values != null}
      {#each values as v, i}
        {@const dimensions = calculateBarSegment(v, i, originCenter)}
        <InlineBar
          absolutePosition
          maxWidth={width}
          leftFraction={dimensions[0]}
          fraction={dimensions[1]}
          color={colors[i]}
          rounded={false}
          {hoverable}
          {height}
          on:mouseenter={(e) => (hoveringIndex = i)}
          on:mouseleave={(e) => (hoveringIndex = null)}
        />
      {/each}
    {:else}
      {@const dimensions = calculateBarSegment(value, 0, originCenter)}
      <InlineBar
        absolutePosition
        maxWidth={width}
        leftFraction={dimensions[0]}
        fraction={dimensions[1]}
        rounded={originCenter ? { left: value < 0, right: value > 0 } : true}
        {value}
        colorScale={colors}
        {hoverable}
        showCenter={originCenter}
        {height}
        on:mouseenter={(e) => (hoveringIndex = 0)}
        on:mouseleave={(e) => (hoveringIndex = null)}
      />
    {/if}
  {/if}
</div>
{#if labelOnBar && values == null}
  {@const dimensions = calculateBarSegment(value, 0, originCenter)}
  <div class="relative overflow-visible h-4" style="width: {width}px;">
    {#if !labelHidden}
      <span
        class="block absolute text-xs -translate-x-1/2 bottom-0 text-gray-500"
        style="left: {Math.max(
          12,
          Math.min(
            width - 12,
            (originCenter && value < 0
              ? dimensions[0]
              : dimensions[0] + dimensions[1]) * width
          )
        )}px;"
      >
        {labelFormat(value)}
      </span>
    {/if}
  </div>
{/if}

<div class="text-xs text-slate-800">
  {#if !$$slots.caption}
    {format(".3")(value)}
  {:else}
    <slot name="caption" {hoveringIndex} />
  {/if}
</div>
