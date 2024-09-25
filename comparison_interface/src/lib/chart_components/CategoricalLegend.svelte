<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  export let colorScale: any = null;
</script>

<div class="flex gap-3 text-sm">
  {#if colorScale != null}
    {@const domain = colorScale.domain()}
    {#each domain as category}
      <div class="flex items-center gap-1">
        <div
          style:background={colorScale(category)}
          class="w-3 h-3 rounded-full"
        />
        <div>{category}</div>
      </div>
    {/each}
  {/if}
</div>
