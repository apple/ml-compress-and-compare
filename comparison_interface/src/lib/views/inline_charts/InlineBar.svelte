<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script>
  import { onMount } from "svelte";

  export let absolutePosition = false;
  export let fraction = 1.0;
  export let leftFraction = 0.0;
  /** @type {Number | Null} */
  export let value = null;
  export let maxWidth = 60;
  export let height = 6;

  export let colorScale = null;
  export let color = "lightgray";

  export let rounded = true; // or { left: boolean, right: boolean }
  export let hoverable = false;

  export let showCenter = false;

  // Disable transition until after loaded
  onMount(() => {
    setTimeout(() => (loaded = true), 100);
  });

  let loaded = false;
</script>

<span
  class="bar {absolutePosition ? 'absolute top-0' : ''} {hoverable
    ? 'hover:opacity-50'
    : ''}"
  class:rounded-l-full={rounded == true || rounded.left}
  class:rounded-r-full={rounded == true || rounded.right}
  style="height: {height}px; width: {Math.round(
    maxWidth * fraction
  )}px; {colorScale != null
    ? 'background-color: ' + colorScale(value != null ? value : fraction) + '; '
    : `background-color: ${color};`} {absolutePosition
    ? `left: ${
        leftFraction > 0 ? Math.round(maxWidth * leftFraction) + 1 : 0
      }px;`
    : ''}"
  on:mouseenter
  on:mouseleave
/>
{#if showCenter}
  <span
    class="bar rounded-full bg-gray-500 border border-white {absolutePosition
      ? 'absolute top-0'
      : ''}"
    style="height: {height}px; width: {height}px; {absolutePosition
      ? `left: ${maxWidth * 0.5 - height * 0.5}px;`
      : ''}"
  />
{/if}

<style>
  .bar {
    display: inline-block;
  }
</style>
