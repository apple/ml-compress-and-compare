<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import {
    SocketDataSource,
    getModelServerURL,
    hasModelServerURL,
  } from "$lib/datasource";
  import { traitlet } from "$lib/traitlets";
  import { onMount } from "svelte";
  import ModelMap from "$lib/views/model_map/ModelMap.svelte";
  import {
    type Model,
    type CompressionOperationSpec,
    type Metric,
    calculateDomain,
  } from "$lib/data_structures/model_map";
  import ModelScatterplot from "$lib/views/model_scatterplot/ModelScatterplot.svelte";
  import ComparisonBarChart from "$lib/views/comparison_bar_chart/ComparisonBarChart.svelte";
  import {
    computeModelExperimentTable,
    flattenExperimentTable,
    simplifyConstantValuePositions,
    simplifySmallDomainVariables,
  } from "$lib/algorithms/experiment_comparability";
  import type { ModelTable } from "$lib/data_structures/model_table";
  import * as d3 from "d3";
  import { page } from "$app/stores";
  import ComparisonRefinementMenu from "$lib/views/comparison_bar_chart/ComparisonRefinementMenu.svelte";
  import { schemeApple8 } from "$lib/utils";
  import { browser } from "$app/environment";
  import { base as baseURL } from "$app/paths";
  import MetricFilterPane from "$lib/views/metric_filter/MetricFilterPane.svelte";
  import Icon from "$lib/icons/Icon.svelte";
  import iconCompare from "$lib/icons/icon-compare.svg";
  import ModelInfoTable from "$lib/views/model_map/ModelInfoTable.svelte";

  const url = browser && $page.url;

  let dataSource = new SocketDataSource(getModelServerURL, "/model_map");

  let connected = false;
  let errorMessage = "";

  let shouldNotRedirect = false;

  onMount(async () => {
    dataSource.connect();
    dataSource.onAttach(() => {
      connected = true;
      shouldNotRedirect = true;
      errorMessage = "";
    });
    dataSource.onDetach((_: any, error: any) => {
      if (!shouldNotRedirect && !hasModelServerURL()) {
        // the server was never connected
        window.location.href = `${baseURL}/settings`;
      }
      connected = false;
      if (!!error && error.message.length > 0)
        errorMessage = `The connection to the server has been lost: ${error.message}`;
      else
        errorMessage =
          "The connection to the server has been lost. Please reload the page to reconnect.";
    });
  });

  // we can do this with traitlets
  let operations = traitlet(
    dataSource,
    "operations",
    new Array<CompressionOperationSpec>()
  );
  let metrics = traitlet(dataSource, "metrics", new Array<Metric>());
  let models = traitlet(dataSource, "models", new Array<Model>());

  let columnByStep = false;
  let xMetric: string | null = null;
  let yMetric: string | null = null;
  let colorMetric: string | null = null;
  let hoveredIDs: string[] = [];
  let selectedIDs: string[] = [];
  let filteredIDs: string[] = [];
  let sizeMetric: string | null = null;
  let colorScale = d3.scaleSequential(d3.interpolateLab("#efefef", "#232323")); // "#cedefd", "#3378F6", "#3176f6"
  $: if (colorMetric != null && colorMetric.length > 0) {
    colorScale = colorScale.domain(
      calculateDomain($metrics, $models, colorMetric)
    );
  } else {
    colorScale = colorScale.domain([0, 1]);
  }

  const MaxModelTableSize = 16;

  enum DetailPane {
    comparison = 1,
    filters = 2,
  }
  let visibleDetailPane = DetailPane.comparison;

  let filterMetrics: string[] = [];
  let filters: ([number, number] | null)[] = [];
  let filtersApplied: number = 0;
  let brushingFilterHistogram: string | null = null;

  let loadedURLParams = false;
  $: if (
    connected &&
    $models.length > 0 &&
    !loadedURLParams &&
    !!url &&
    url.searchParams.has("selection")
  ) {
    loadURLParameters();
  }

  function loadURLParameters() {
    if (loadedURLParams) return;

    let queryIDs = url.searchParams.get("selection")!.split(",");
    selectedIDs = queryIDs.filter(
      (modelID) => $models.find((m) => m.id == modelID) != null
    );
    // redirect the URL if the selected IDs aren't present
    if (selectedIDs.length != queryIDs.length) {
      let base = window.location.href.split("?")[0];
      if (selectedIDs.length == 0) window.location.href = base;
      else window.location.href = `${base}?selection=${selectedIDs.join(",")}`;
    }
    loadedURLParams = true;
  }

  let fullModelTable: ModelTable | null = null;
  let refinementAllowedValues: Map<number, any[]> = new Map();
  let refinementFilteredModels: string[] | null = null;
  let tooManyVariables = false;
  let showRefinementView = false;
  let refinementViewExpanded = true;

  let modelTable: ModelTable | null = null;
  let modelTableX: number | null = null;
  let modelTableColor: number | null = null;
  let modelTableColorScheme = schemeApple8; // d3.schemeTableau10;
  $: if (
    $models.length > 0 &&
    selectedIDs.length > 1 &&
    selectedIDs.length <= MaxModelTableSize
  ) {
    fullModelTable = computeModelExperimentTable($models, selectedIDs);
    showRefinementView = fullModelTable.numVariables > 2;
  } else {
    showRefinementView = false;
    fullModelTable = null;
  }

  let oldSelectedIDs: string[] = [];
  $: if (oldSelectedIDs !== selectedIDs) {
    refinementAllowedValues = new Map();
    oldSelectedIDs = selectedIDs;
  }

  $: if (!!fullModelTable) {
    if (selectedIDs.length > MaxModelTableSize) {
      modelTable = null;
      modelTableX = null;
      modelTableColor = null;
    } else {
      let table =
        !!refinementFilteredModels &&
        refinementFilteredModels.every((id) => selectedIDs.includes(id))
          ? computeModelExperimentTable(
              $models,
              selectedIDs.filter((m) => refinementFilteredModels!.includes(m))
            )
          : fullModelTable;
      tooManyVariables = false;

      if (table.size > 1) {
        attemptTableSimplify(table);
      } else {
        modelTable = null;
        modelTableX = null;
        modelTableColor = null;
      }
    }
  } else {
    tooManyVariables = false;
    modelTable = null;
    modelTableX = null;
    modelTableColor = null;
  }

  function attemptTableSimplify(table: ModelTable) {
    let simplerTable = table;
    if (simplerTable.numVariables > 2)
      simplerTable = simplifyConstantValuePositions(simplerTable, 1, 1e9, {
        requireCumulative: true,
      });
    else simplerTable = simplifyConstantValuePositions(simplerTable, 1, 3);
    if (simplerTable.numVariables > 2)
      simplerTable = simplifyConstantValuePositions(simplerTable, 1, 3);
    if (simplerTable.numVariables > 2)
      simplerTable = simplifySmallDomainVariables(simplerTable);

    if (
      table.numVariables > 2 &&
      table.size < 6 &&
      table.size <
        simplerTable.variables
          .map((_, i) => simplerTable.distinctValues(i).length)
          .reduce((prev, curr) => prev * curr, 1) -
          1
    ) {
      console.log("flatten", table.variables);
      simplerTable = flattenExperimentTable(table);
    }
    modelTable = simplerTable;
    if (modelTable.numVariables > 2) {
      console.log("too many encodings", modelTable);
      tooManyVariables = true;
      modelTable = null;
      modelTableX = null;
      modelTableColor = null;
    } else if (modelTable.numVariables == 1) {
      modelTableX = 0;
      modelTableColor = null;
    } else if (modelTable.numVariables == 2) {
      modelTableX = 0;
      modelTableColor = 1;
    } else {
      modelTableX = null;
      modelTableColor = null;
    }
    refinementViewExpanded = simplerTable.numVariables > 2;
  }

  let primaryMetrics: Metric[] = [];
  $: {
    let hasPrimary = $metrics.find((m) => m.primary) != null;
    primaryMetrics = $metrics.filter((m) => !hasPrimary || m.primary);
  }

  let oldPrimaryMetrics = new Array<Metric>();
  $: if (primaryMetrics !== oldPrimaryMetrics) {
    assignPreliminaryMetrics();
    oldPrimaryMetrics = primaryMetrics;
  }

  function assignPreliminaryMetrics() {
    let prelimAssignment = [
      primaryMetrics.find(
        (m) => m.name.search(/\blatency\b|\bsparsity\b/i) >= 0
      )?.name,
      primaryMetrics.find((m) => m.name.search(/\baccuracy\b|\bf1/i) >= 0)
        ?.name,
      primaryMetrics.find((m) => m.name.search(/\baccuracy\b|\bf1/i) >= 0)
        ?.name,
      primaryMetrics.find((m) => m.name.search(/\bsize|\bspars/i) >= 0)?.name,
    ]; // x, y, color, size
    prelimAssignment.forEach((_, i) => {
      if (!prelimAssignment[i])
        prelimAssignment[i] = primaryMetrics.find(
          (m) => !prelimAssignment.includes(m.name)
        )?.name;
    });
    xMetric = prelimAssignment[0] || null;
    yMetric = prelimAssignment[1] || null;
    colorMetric = prelimAssignment[2] || null;
    sizeMetric = prelimAssignment[3] || null;
  }

  function pathFromRoot(id: string): string[] {
    let node = $models.find((m) => m.id == id);
    let path: string[] = [];
    while (!!node) {
      path = [node.id, ...path];
      let parent = node.base;
      if (!!parent) {
        node = $models.find((m) => m.id == parent);
      } else break;
    }
    return path;
  }

  function openComparisonView(pane: string) {
    // For now, choose the node with the shortest path to root
    let baseModel = selectedIDs.reduce(
      (prev: string | null, curr: string) =>
        prev != null && pathFromRoot(prev).length < pathFromRoot(curr).length
          ? prev
          : curr,
      null
    );
    window.location.href = `${baseURL}/detail/${pane}?base=${baseModel}&models=${selectedIDs.join(
      ","
    )}`;
  }

  $: if (
    filteredIDs.length > 0 &&
    selectedIDs.length > 0 &&
    brushingFilterHistogram == null
  ) {
    selectedIDs = selectedIDs.filter((id) => filteredIDs.includes(id));
  }
</script>

<div class="pt-12 w-full h-full bg-gray-200">
  {#if errorMessage.length > 0}
    <div class="w-full h-full flex flex-col items-center justify-center">
      <p class="w-1/2 text-center">Model server could not be reached.</p>
      <p class="w-1/2 text-center text-gray-500 text-sm">{errorMessage}</p>
      <button class="primary-btn mt-4" on:click={() => window.location.reload()}
        >Retry</button
      >
    </div>
  {:else}
    <div class="flex flex-col w-full h-full">
      <div class="h-12 shrink-0 py-1 px-3 flex items-center">
        <label for="color-metric" class="text-gray-600 text-sm font-bold mr-2"
          >COLOR
        </label>
        <select
          class="text-sm mr-4 w-32"
          id="color-metric"
          bind:value={colorMetric}
        >
          <option value={null}>None</option>
          {#each $metrics as metric}
            <option value={metric.name}>{metric.name}</option>
          {/each}
        </select><label
          for="size-metric"
          class="text-gray-600 text-sm font-bold mr-2"
          >SIZE
        </label>
        <select
          class="text-sm mr-4 w-32"
          id="size-metric"
          bind:value={sizeMetric}
        >
          <option value={null}>None</option>
          {#each $metrics as metric}
            <option value={metric.name}>{metric.name}</option>
          {/each}
        </select>
        <button
          class="secondary-btn"
          on:click={(e) => (columnByStep = !columnByStep)}
          >{columnByStep ? "Arrange by Operation" : "Arrange by Step"}</button
        >
        <div class="flex-auto" />
        <button
          class="primary-btn mr-2"
          disabled={selectedIDs.length < 2 || selectedIDs.length > 5}
          on:click={() => openComparisonView("instances")}
          ><Icon
            src={iconCompare}
            alt="Compare"
            class="mr-2"
          />Behaviors</button
        >
        <button
          class="primary-btn"
          disabled={selectedIDs.length < 2 || selectedIDs.length > 5}
          on:click={() => openComparisonView("tensors")}
          ><Icon src={iconCompare} alt="Compare" class="mr-2" />Layers</button
        >
      </div>
      <div class="flex-auto w-full flex">
        <div class="flex-auto h-full rounded-tr-lg bg-white overflow-hidden">
          <ModelMap
            operations={$operations}
            metrics={$metrics}
            models={$models}
            {colorScale}
            bind:columnByStep
            bind:colorMetric
            bind:sizeMetric
            bind:hoveredIDs
            bind:selectedIDs
            bind:filteredIDs
          />
        </div>
        <div
          id="selection-detail-view"
          class="w-1/3 shrink-0 grow-0 h-full flex flex-col bg-gray-200 px-3 pb-3"
        >
          <div
            class="rounded-lg mb-3 bg-white shrink-0 w-full h-80 p-2 flex flex-col"
            style="max-height: 50%;"
          >
            <div class="flex-auto w-full">
              <ModelScatterplot
                models={$models}
                metrics={$metrics}
                xEncoding={xMetric}
                yEncoding={yMetric}
                colorEncoding={colorMetric}
                sizeEncoding={sizeMetric}
                {colorScale}
                bind:hoveredIDs
                bind:selectedIDs
                bind:filteredIDs
              />
            </div>
            <div class="flex items-center w-full shrink-0 px-4 mb-2">
              <label for="x-metric" class="text-gray-600 text-sm font-bold mr-2"
                >X
              </label>
              <select
                class="text-sm mr-4 flex-auto"
                id="x-metric"
                bind:value={xMetric}
              >
                <option value={null}>None</option>
                {#each $metrics as metric}
                  <option value={metric.name}>{metric.name}</option>
                {/each}
              </select>
              <label for="y-metric" class="text-gray-600 text-sm font-bold mr-2"
                >Y
              </label>
              <select
                class="text-sm flex-auto"
                id="y-metric"
                bind:value={yMetric}
              >
                <option value={null}>None</option>
                {#each $metrics as metric}
                  <option value={metric.name}>{metric.name}</option>
                {/each}
              </select>
            </div>
          </div>
          <div class="rounded-lg bg-white flex-auto basis-0">
            <div class="w-full h-full flex flex-col">
              <div class="px-4 my-4">
                <button
                  class="mr-2 {visibleDetailPane == DetailPane.comparison
                    ? 'tertiary-btn'
                    : 'transparent-btn'}"
                  on:click={() => (visibleDetailPane = DetailPane.comparison)}
                  >Compare</button
                >
                <button
                  class={visibleDetailPane == DetailPane.filters
                    ? "tertiary-btn"
                    : "transparent-btn"}
                  on:click={() => (visibleDetailPane = DetailPane.filters)}
                  >Filter{#if filtersApplied > 0}<span
                      class="ml-2 bg-blue-500 text-white rounded-full py-0.5 px-1.5 text-xs"
                      >{filteredIDs.length}</span
                    >{/if}</button
                >
              </div>
              <div class="w-full pb-4 flex-auto overflow-scroll basis-0">
                {#if visibleDetailPane == DetailPane.filters}
                  <div class="w-full px-4">
                    <MetricFilterPane
                      models={$models}
                      metrics={$metrics}
                      bind:filterMetrics
                      bind:filters
                      bind:filteredModelIDs={filteredIDs}
                      bind:filtersApplied
                      bind:brushingHistogram={brushingFilterHistogram}
                    />
                  </div>
                {:else}
                  <div class="px-4 text-normal font-bold mb-2">
                    {#if selectedIDs.length == 0}
                      Selection Details
                    {:else if selectedIDs.length == 1}
                      {@const model = $models.find(
                        (m) => m.id == selectedIDs[0]
                      )}
                      1 Model Selected:&nbsp;
                      <span class="font-mono">{model.id}</span>
                    {:else}{selectedIDs.length} Models Selected
                    {/if}
                  </div>
                  {#if selectedIDs.length < 2 || selectedIDs.length > MaxModelTableSize}
                    <p class="text-gray-600 px-4 text-sm">
                      Select between 2 and {MaxModelTableSize} models to generate
                      comparison bar charts.
                    </p>
                  {:else if showRefinementView && fullModelTable != null}
                    {#if tooManyVariables}
                      <p class="text-gray-600 px-4 mb-2 text-sm">
                        These models cannot be directly compared. Try refining
                        the comparison to filter models and reduce the number of
                        variables.
                      </p>
                    {/if}
                    <ComparisonRefinementMenu
                      modelTable={fullModelTable}
                      bind:allowedValues={refinementAllowedValues}
                      bind:filteredModelIDs={refinementFilteredModels}
                      bind:expanded={refinementViewExpanded}
                      on:hover={(e) => {
                        if (e.detail != null) hoveredIDs = e.detail;
                        else hoveredIDs = [];
                      }}
                      on:select={() =>
                        (selectedIDs = refinementFilteredModels || [])}
                    />
                  {/if}
                  {#if selectedIDs.length == 1}
                    <div class="px-4 mt-4">
                      <ModelInfoTable
                        models={$models}
                        metrics={$metrics}
                        operations={$operations}
                        model={$models.find((m) => m.id == selectedIDs[0])}
                      />
                    </div>
                  {/if}
                  {#if modelTable != null}
                    {#each primaryMetrics as metric, i}
                      <div class="mt-4 px-4">
                        <ComparisonBarChart
                          models={$models}
                          metrics={$metrics}
                          {modelTable}
                          showLegend={i == 0}
                          showXLabel
                          xVariable={modelTableX}
                          yEncoding={metric.name}
                          colorVariable={modelTableColor}
                          colorScheme={modelTableColorScheme}
                          bind:hoveredIDs
                          on:select={(e) => (selectedIDs = [e.detail])}
                        />
                      </div>
                    {/each}
                  {:else if !!refinementFilteredModels && refinementFilteredModels.length < 2}
                    <p class="text-gray-600 px-4 mb-2 text-sm">
                      Need at least two models to compare.
                    </p>
                  {/if}
                {/if}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
</style>
