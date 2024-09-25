<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->
  
<script lang="ts">
  import Icon from "$lib/icons/Icon.svelte";
  import BrushableHistogram from "$lib/chart_components/BrushableHistogram.svelte";
  import {
    calculateDomain,
    type Metric,
    type Model,
  } from "$lib/data_structures/model_map";
  import * as d3 from "d3";
  import iconTrash from "$lib/icons/icon-trash.svg";
  import { onMount } from "svelte";

  export let models: Model[] = [];
  export let filteredModelIDs: string[] = [];
  export let metrics: Metric[] = [];
  export let filtersApplied: number = 0;
  export let brushingHistogram: string | null = null;

  export let filterMetrics: string[] = [];
  export let filters: ([number, number] | null)[] = [];

  let filteredModels: Model[] = [];

  // Makes sure svelte animates the same chart for each metric
  let metricElementKeys: number[] = [];

  onMount(() => {
    if (metrics.length > 0 && filterMetrics.length == 0) {
      initializeMetrics();
    }
  });

  function initializeMetrics() {
    addMetric(metrics[0].name);
  }

  function modelsMatchingFilters(
    allMetrics: string[],
    allFilters: ([number, number] | null)[]
  ): Model[] {
    return models.filter((m) =>
      allMetrics.every((metricName, i) => {
        let filter = allFilters[i];
        if (!filter) return true;
        return (
          m.metrics[metricName] >= filter[0] &&
          m.metrics[metricName] <= filter[1]
        );
      })
    );
  }

  $: if (metrics.length > 0 && models.length > 0) {
    filteredModels = modelsMatchingFilters(filterMetrics, filters);
    filtersApplied = filters.reduce(
      (count, f) => count + (f != null ? 1 : 0),
      0
    );
    if (filtersApplied > 0) filteredModelIDs = filteredModels.map((m) => m.id);
    else filteredModelIDs = [];
  }

  function changedMetric(i: number, newName: string) {
    filterMetrics = [
      ...filterMetrics.slice(0, i),
      newName,
      ...filterMetrics.slice(i + 1),
    ];
    filters = [...filters.slice(0, i), null, ...filters.slice(i + 1)];
  }

  function addMetric(name: string) {
    metricElementKeys = [...metricElementKeys, Math.random()];
    filterMetrics = [...filterMetrics, name];
    filters = [...filters, null];
  }

  function removeMetric(i: number) {
    filterMetrics = [
      ...filterMetrics.slice(0, i),
      ...filterMetrics.slice(i + 1),
    ];
    filters = [...filters.slice(0, i), ...filters.slice(i + 1)];
    metricElementKeys = [
      ...metricElementKeys.slice(0, i),
      ...metricElementKeys.slice(i + 1),
    ];
  }

  $: if (filterMetrics.length != metricElementKeys.length) {
    while (metricElementKeys.length < filterMetrics.length)
      metricElementKeys = [...metricElementKeys, Math.random()];
  }
</script>

{#each filterMetrics as metric, i (metricElementKeys[i])}
  {@const metricSpec = metrics.find((m) => m.name == metric)}
  {@const metricFormat =
    metricSpec != null && !!metricSpec.format
      ? d3.format(metricSpec.format)
      : null}
  <div class="w-full flex items-center px-2">
    <div class="mr-2">Filter by:</div>
    <select
      class="flex-auto"
      value={filterMetrics[i]}
      on:change={(e) => changedMetric(i, e.target.value)}
    >
      {#each metrics as metricSpec}
        <option
          value={metricSpec.name}
          disabled={filterMetrics.includes(metricSpec.name)}
          >{metricSpec.name}</option
        >
      {/each}
    </select>
    <button
      class="ml-4 hover:opacity-50 text-sm animate-all"
      on:click={() => removeMetric(i)}
      ><Icon src={iconTrash} alt="Delete filter" /></button
    >
  </div>
  <div class="w-full h-32 pl-2 pr-6">
    <BrushableHistogram
      xExtent={calculateDomain(metrics, models, metric)}
      xFormat={metricFormat}
      allData={models.map((m) => m.metrics[metric])}
      filteredData={filtersApplied
        ? filteredModels.map((m) => m.metrics[metric])
        : undefined}
      bind:filterBounds={filters[i]}
      on:brushstart={() => (brushingHistogram = metric)}
      on:brushend={() => (brushingHistogram = null)}
    />
  </div>
  <div class="mb-2 text-gray-500 text-sm px-2">
    {#if brushingHistogram == metric && !!filters[i]}
      Filtering for {metric}
      {(metricFormat || d3.format(".3~"))(filters[i][0])} &mdash; {(
        metricFormat || d3.format(".3~")
      )(filters[i][1])}
    {/if}&nbsp;
  </div>
{/each}
<div class="flex w-full justify-between px-2">
  <button
    class="secondary-btn mr-4"
    on:click={() => {
      filters = filterMetrics.map((_) => null);
    }}
    disabled={filterMetrics.length == 0}>Clear All</button
  >
  <button
    class="primary-btn"
    on:click={() => {
      let nextMetric = metrics.find((m) => !filterMetrics.includes(m.name));
      if (nextMetric == null) return;
      addMetric(nextMetric.name);
    }}
    disabled={filterMetrics.length == metrics.length}>+ Add Filter</button
  >
</div>
