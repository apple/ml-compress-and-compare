<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import type { DataSource } from "$lib/datasource";
  import * as d3 from "d3";
  import {
    countPredictions,
    type ClassPrediction,
    type InstanceData,
    type InstancePrediction,
    type PredictionDataset,
  } from "$lib/data_structures/instance_predictions";
  import TextInstanceView from "$lib/views/instance_views/TextInstanceView.svelte";
  import {
    SortDirection,
    areSetsEqual,
    interpolateSystemBlue,
    interpolateSystemRedBlue,
  } from "$lib/utils";
  import BrushableHistogram from "$lib/chart_components/BrushableHistogram.svelte";
  import Icon from "$lib/icons/Icon.svelte";
  import sortNone from "$lib/icons/sort-none.svg";
  import sortAsc from "$lib/icons/sort-asc.svg";
  import sortDesc from "$lib/icons/sort-desc.svg";
  import TableCellBar from "$lib/views/inline_charts/TableCellBar.svelte";
  import { createEventDispatcher } from "svelte";
  import ImageInstanceView from "$lib/views/instance_views/ImageInstanceView.svelte";

  const dispatch = createEventDispatcher();

  export let dataSource: DataSource | null = null;

  export let comparisonModels: string[] = [];
  export let baseModel: string = "";
  export let comparisonOptions: {
    [key: string]: { relative?: boolean; format?: string };
  } = {};

  export let predictions: PredictionDataset | null = null;

  export let comparisonName = "";

  export let hasRawValues = false;
  export let showRawValues = false;

  export let classLevel = false;
  export let selectableRows = false;
  export let filterClass: string | null = null;

  export let hoveredModelID: string | null = null;

  let instanceType: string = "text";
  let requestedInstanceIDs: any[];
  let instances: { [key: string]: InstanceData } | null = null;

  let colorScale = interpolateSystemBlue;

  const BarWidth = 200;

  function getInstanceDetails(ids: any[]) {
    if (!dataSource) return;
    requestedInstanceIDs = ids.map((id) => `${id}`);
    dataSource.request("get_instances", { ids }).then((response: any) => {
      console.log("got instances");
      if (response.error) {
        console.error("Error getting instance data:", response.error);
        return;
      }
      let newIDs = new Set(Object.keys(response.instances));
      if (!areSetsEqual(newIDs, new Set(requestedInstanceIDs))) {
        console.log("requested instance ids are stale, ignoring result");
        return;
      }
      instanceType = response.type;
      instances = response.instances;
    });
  }

  let classFilteredPredictions: PredictionDataset | null = null;
  let filteredPredictions: PredictionDataset | null = null;
  let predictionsToDisplay: PredictionDataset | null = null;
  let overallPredictionCounts: {
    [key: string]: { pred: any; base?: any; rate: number }[];
  } = {};
  let predictionPage: PredictionDataset | null = null;
  let sortModel: string | null = null;
  let sortDirection: SortDirection = SortDirection.none;
  let pageNumber = 0; // paginate results when sorted
  const numInstancesPerPage = 50;

  let brushingHistograms = false;

  // filters from each of the chart components
  let filters: ([number, number] | null)[] = [];
  let filtersApplied = false;

  let minScore = 0;
  let maxScore = 1;

  $: if (comparisonName.length > 0) {
    if (!!predictions && filterClass != null) {
      classFilteredPredictions = predictions.filterPredictions((p) =>
        predictions!.isClassLevel
          ? (p as ClassPrediction).class == filterClass!
          : (p as InstancePrediction).classes != undefined &&
            (p as InstancePrediction).classes!.includes(filterClass!)
      );
    } else classFilteredPredictions = predictions;
  } else classFilteredPredictions = null;

  $: if (!!classFilteredPredictions) {
    filteredPredictions = classFilteredPredictions.filterByComparisonMetric(
      comparisonName,
      showRawValues,
      (p, modelID, i) =>
        filters[i] == null || (p >= filters[i]![0] && p <= filters[i]![1])
    );
    filtersApplied =
      filteredPredictions.length != classFilteredPredictions.length;
  } else {
    filteredPredictions = null;
  }

  function changeSort(model: string | null) {
    if (sortModel != model) sortDirection = 0;
    sortModel = model;
    sortDirection = (sortDirection + 1) % 3;
    if (sortDirection == SortDirection.none) sortModel = null;
    console.log(sortModel, sortDirection);
  }

  let oldComparisonName = "";
  let oldRawValues = false;
  $: if (oldComparisonName != comparisonName || oldRawValues != showRawValues) {
    filters = comparisonModels.map((c) => null);
    if (!showRawValues) filters = filters.slice(1);
    oldComparisonName = comparisonName;
    oldRawValues = showRawValues;
  }

  $: if (!!predictionsToDisplay) {
    predictionPage = predictionsToDisplay.getPredictionSlice(
      pageNumber * numInstancesPerPage,
      (pageNumber + 1) * numInstancesPerPage
    );
    if (!classLevel) {
      console.log("getting instances");
      getInstanceDetails(predictionPage.getIDs());
    }
  } else {
    predictionPage = null;
  }

  $: if (!!filteredPredictions) {
    if (!brushingHistograms) {
      overallPredictionCounts = Object.fromEntries(
        comparisonModels.map((modelID) => [
          modelID,
          countInstancePredictions(
            filteredPredictions!,
            modelID,
            !showRawValues ? baseModel : null,
            numPredictionCounts
          ),
        ])
      );
      predictionsToDisplay = sortPredictions(
        classLevel
          ? filteredPredictions.aggregateClassLabels(comparisonOptions)
          : filteredPredictions,
        comparisonName,
        showRawValues,
        sortModel,
        sortDirection
      );
      console.log("predictions to display:", predictionsToDisplay);
    }
  } else {
    instances = null;
    predictionsToDisplay = null;
    overallPredictionCounts = {};
  }

  $: if (!!predictions && comparisonName.length > 0) {
    let valuesForScales = predictions.modelNames.map((modelID) =>
      predictions!.getComparisonMetrics(comparisonName, modelID, showRawValues)
    );
    maxScore = valuesForScales.reduce(
      (prev, curr) => Math.max(prev, ...curr),
      -1e9
    );
    minScore = valuesForScales.reduce(
      (prev, curr) => Math.min(prev, ...curr),
      1e9
    );
    // Expand scale if values are discrete
    let uniqueValues = Array.from(new Set(valuesForScales.flat())).sort();
    let discreteBinning = false;
    if (
      uniqueValues.length > 0 &&
      uniqueValues.length < 20 &&
      uniqueValues.indexOf(0) > 0 &&
      uniqueValues.every((v) => Math.round(v) == v)
    ) {
      // integers! where zero isn't the first value
      minScore = uniqueValues[0];
      maxScore =
        Math.max(uniqueValues[uniqueValues.length - 1], uniqueValues[0]) +
        (uniqueValues[1] - uniqueValues[0]) * 0.5;
      discreteBinning = true;
    }
    if (minScore < 0) {
      minScore = -Math.max(Math.abs(minScore), Math.abs(maxScore));
      maxScore = Math.max(Math.abs(minScore), Math.abs(maxScore));
    }
    colorScale = d3
      .scaleSequential(
        minScore < 0 ? interpolateSystemRedBlue : interpolateSystemBlue
      )
      .domain([Math.min(0, minScore), maxScore]);
    if (!discreteBinning)
      [minScore, maxScore] = d3
        .scaleLinear()
        .domain([minScore, maxScore])
        .nice()
        .domain();
  }

  function sortPredictions(
    preds: PredictionDataset,
    compName: string,
    rawValues: boolean,
    byModel: string | null,
    direction: SortDirection
  ) {
    if (byModel == null) return preds.sortByID();
    else {
      // sort by differences
      return preds.sortByComparisonMetric(
        compName,
        byModel!,
        rawValues,
        (a, b) =>
          (direction == SortDirection.ascending ? 1 : -1) *
          (rawValues ? a - b : Math.abs(a) - Math.abs(b))
      );
    }
  }

  function countInstancePredictions(
    preds: PredictionDataset,
    modelID: string,
    base: string | null,
    numToReturn: number = 5
  ): { pred: any; base?: any; rate: number }[] {
    let modelPreds = preds.predictions.map(
      (pred) => (pred as InstancePrediction).predictions[modelID]
    );
    let basePreds = !!base
      ? preds.predictions.map(
          (pred) => (pred as InstancePrediction).predictions[base]
        )
      : null;
    let totalCount = modelPreds.length;
    let counts = countPredictions(modelPreds, basePreds)
      .filter((pred) => pred.base == null || pred.pred != pred.base)
      .slice(0, numToReturn);
    return counts.map((item) => ({
      pred: item.pred,
      base: item.base,
      rate: item.count / totalCount,
    }));
  }

  function countClassPredictions(
    predictionSet: ClassPrediction,
    modelID: string,
    base: string | null,
    numToReturn: number = 5
  ): { pred: any; base?: any; rate: number }[] {
    let modelPreds = predictionSet.predictions[modelID];
    let basePreds = !!base ? predictionSet.predictions[base] : null;
    let totalCount = modelPreds.length;
    let counts = countPredictions(modelPreds, basePreds).slice(0, numToReturn);
    return counts.map((item) => ({
      pred: item.pred,
      base: item.base,
      rate: item.count / totalCount,
    }));
  }

  let scoreFormat = d3.format(".3~g");
  let scoreDifferenceFormat: (x: number) => string = d3.format("+.3~g");
  $: if (!hasRawValues) scoreDifferenceFormat = scoreFormat;
  else scoreDifferenceFormat = d3.format("+.3~g");

  const predictionRateFormat = d3.format(".1~%");

  let minRowWidth = 0;
  let instanceColumnWidth = 600;
  $: if (comparisonModels.length > 0) {
    instanceColumnWidth = instanceType == "text" && !classLevel ? 600 : 320;
    minRowWidth = instanceColumnWidth + 260 * comparisonModels.length;
  }

  let numPredictionCounts = 5;
  function resizeContainer() {
    if (window.innerHeight < 800) numPredictionCounts = 1;
    else numPredictionCounts = 5;
  }

  function getScoreDifferenceFormat(compName: string): (v: number) => string {
    if (!!comparisonOptions[compName] && comparisonOptions[compName].relative) {
      return d3.format("+.2~%");
    }
    return scoreDifferenceFormat;
  }
</script>

<svelte:window on:resize={resizeContainer} />

{#if !!baseModel && comparisonModels.length > 0 && !!classFilteredPredictions}
  <div class="min-w-full min-h-full flex justify-center">
    <div
      style="min-width: {classLevel || instanceType == 'image'
        ? '30%'
        : '60%'};"
    >
      <div
        class="header-row bg-gray-100 text-left flex items-stretch px-1 w-full header-sticky"
        style="min-width: {minRowWidth}px;"
      >
        <div
          class="rounded-tl-lg text-sm tensor-name instance-column flex-auto shrink-0"
          style="width: {instanceColumnWidth}px;"
        >
          <div class="p-2 pt-2.5 w-full h-full font-bold">Model</div>
        </div>
        {#each comparisonModels as modelID, i}
          <div
            class="text-sm model-column {i == comparisonModels.length - 1
              ? 'rounded-tr-lg'
              : ''} {modelID == baseModel
              ? 'border-r border-gray-500 border-dashed'
              : ''}"
          >
            <div class="p-2 h-full">
              <div
                class="text-sm truncate"
                class:font-bold={hoveredModelID == modelID}
                on:mouseenter={() => (hoveredModelID = modelID)}
                on:mouseleave={() => (hoveredModelID = null)}
              >
                <span class="font-mono" title={modelID}>{modelID}</span
                >{#if modelID == baseModel}&nbsp;(Base){/if}
              </div>
            </div>
          </div>
        {/each}
      </div>
      {#if !!predictionPage && !!predictionsToDisplay}
        {@const { n: totalN, comparisons: overallComparison } =
          predictionsToDisplay.overallAverageComparisonMetrics()}
        <div
          class="header-row bg-gray-100 px-1 flex items-stretch w-full"
          style="min-width: {minRowWidth}px;"
        >
          <div
            class="instance-column p-2 flex-auto shrink-0"
            style="width: {instanceColumnWidth}px;"
          >
            {predictionsToDisplay.length < predictions.length && !classLevel
              ? "Filtered"
              : ""}
            Dataset
            <span class="ml-2 text-gray-400">{totalN} instances</span>
          </div>
          {#each comparisonModels as modelID}
            {@const scoreValue = (overallComparison[comparisonName] || {})[
              modelID
            ]}

            <div
              class="model-column text-sm p-2 {modelID == baseModel
                ? 'border-r border-gray-500 border-dashed'
                : ''}"
            >
              {#if !scoreValue}
                --
              {:else}
                <div class="text-sm mb-1">
                  {#if hasRawValues && scoreValue.value != undefined}
                    {scoreFormat(scoreValue.value)}
                    <span class="text-gray-500 text-xs"
                      >&nbsp;{comparisonName}</span
                    >
                  {/if}
                </div>
                <TableCellBar
                  visible={modelID != baseModel || showRawValues}
                  value={showRawValues
                    ? scoreValue.value
                    : scoreValue.difference}
                  scale={(v) =>
                    v / Math.max(Math.abs(maxScore), Math.abs(minScore))}
                  originCenter={minScore < 0}
                  width={BarWidth}
                  height={8}
                  showFullBar
                  colors={colorScale}
                  labelOnBar={!showRawValues}
                  labelHidden={modelID == baseModel && !showRawValues}
                  labelFormat={getScoreDifferenceFormat(comparisonName)}
                >
                  <div
                    class="text-xs text-gray-500"
                    slot="caption"
                    let:hoveringIndex
                  >
                    {#if hoveringIndex != null}
                      Overall {#if showRawValues}{comparisonName} is {scoreFormat(
                          scoreValue.value
                        )} for model
                      {:else}{comparisonName}
                        {#if scoreValue.difference == 0}did not change{:else}{hasRawValues
                            ? "changed by"
                            : "is"}
                          {getScoreDifferenceFormat(comparisonName)(
                            scoreValue.difference
                          )}{/if} compared to base
                      {/if}
                    {/if}
                  </div>
                </TableCellBar>

                {#if (modelID != baseModel || showRawValues) && overallPredictionCounts.hasOwnProperty(modelID)}
                  <div class="mt-3 text-xs text-gray-500">
                    {@html showRawValues
                      ? "Model Prediction"
                      : "Base &#x25B8; Model Prediction Differences"}
                  </div>
                  {#each overallPredictionCounts[modelID] as predCount}
                    <div class="text-sm">
                      {@html !showRawValues
                        ? predCount.base + " &#x25B8; " + predCount.pred
                        : predCount.pred}
                      <span class="text-gray-500 text-xs"
                        >{predictionRateFormat(predCount.rate)}</span
                      >
                    </div>
                  {/each}
                {/if}
              {/if}
            </div>
          {/each}
        </div>
      {/if}
      <div
        class="header-row bg-gray-100 text-left flex items-end px-1 w-full header-sticky border-b pane-border"
        style="top: 36px; min-width: {minRowWidth}px;"
      >
        <div
          class="rounded-tl-lg text-sm instance-column flex-auto shrink-0"
          style="width: {instanceColumnWidth}px;"
        >
          <div class="p-2 pt-2.5 w-full h-full font-bold">
            {classLevel ? "Class" : "Instance"}
            {#if sortModel == null}
              <span class="ml-1 text-xs text-gray-500 font-normal">
                Sort
                {#if classLevel}alphabetically{:else}by instance ID{/if}
              </span>
            {/if}
          </div>
        </div>
        {#each comparisonModels as modelID, i (modelID)}
          <div
            class="text-sm model-column self-stretch {i ==
            comparisonModels.length - 1
              ? 'rounded-tr-lg'
              : ''} {modelID == baseModel
              ? 'border-r border-gray-500 border-dashed'
              : ''}"
          >
            <div class="px-2 pb-2 h-full">
              {#if modelID != baseModel || showRawValues}
                <div style="width: {BarWidth}px;" class="h-32">
                  <BrushableHistogram
                    xExtent={[minScore, maxScore]}
                    dataForScales={predictions.modelNames
                      .map((modelID) =>
                        predictions == null
                          ? []
                          : predictions.getComparisonMetrics(
                              comparisonName,
                              modelID,
                              showRawValues
                            )
                      )
                      .flat()}
                    allData={classFilteredPredictions.getComparisonMetrics(
                      comparisonName,
                      modelID,
                      showRawValues
                    )}
                    filteredData={filtersApplied
                      ? filteredPredictions?.getComparisonMetrics(
                          comparisonName,
                          modelID,
                          showRawValues
                        )
                      : null}
                    xLabel={hasRawValues && !showRawValues
                      ? "Change in " + comparisonName
                      : comparisonName}
                    {colorScale}
                    bind:filterBounds={filters[i]}
                    on:brushstart={() => (brushingHistograms = true)}
                    on:brushend={() => (brushingHistograms = false)}
                  />
                </div>
              {/if}

              <div class="mt-2 flex items-start">
                <div class="text-sm flex-auto">
                  {#if sortModel == modelID}
                    <div class="text-xs text-gray-500 font-normal">
                      Sort by
                      {#if !showRawValues && hasRawValues}change in{/if}
                      {#if sortDirection == SortDirection.ascending}{comparisonName},
                        least to greatest
                      {:else if sortDirection == SortDirection.descending}{comparisonName},
                        greatest to least{/if}
                    </div>
                  {/if}
                </div>
                {#if modelID != baseModel || showRawValues}
                  <button
                    class="hover:opacity-70 mx-1 shrink-0"
                    on:click|stopPropagation={() => changeSort(modelID)}
                    ><Icon
                      src={sortModel == modelID
                        ? sortDirection == SortDirection.ascending
                          ? sortAsc
                          : sortDesc
                        : sortNone}
                      alt="Change sort direction"
                    /></button
                  >
                {/if}
              </div>
            </div>
          </div>
        {/each}
      </div>
      {#if !!predictionPage && !!predictionsToDisplay}
        {#each predictionPage.predictions as prediction}
          <div
            class="px-1 hover:bg-gray-100 flex items-stretch bg-white"
            style="min-width: {minRowWidth}px;"
            class:cursor-pointer={selectableRows}
            on:click={() => {
              if (selectableRows) dispatch("select", prediction);
            }}
            on:keydown={(e) => {
              if (e.key === "Enter" && selectableRows)
                dispatch("select", prediction);
            }}
            role="menuitem"
            tabindex="0"
          >
            <div
              class="instance-column flex-auto shrink-0"
              style="width: {instanceColumnWidth}px;"
            >
              {#if predictionPage.isClassLevel}
                <div class="p-2 pt-4">
                  <span class="font-bold">{prediction.class}</span>
                  <span class="ml-2 text-gray-400"
                    >{prediction.predictions[baseModel].length} instances</span
                  >
                </div>
              {:else if instanceType == "text"}
                <TextInstanceView {prediction} {instances} />
              {:else if instanceType == "image"}
                <ImageInstanceView {prediction} {instances} />
              {/if}
            </div>
            {#each comparisonModels as modelID}
              {@const scoreValue =
                prediction.comparisons[comparisonName][modelID]}
              <div
                class="text-sm p-2 model-column {modelID == baseModel
                  ? 'border-r border-gray-500 border-dashed'
                  : ''}"
              >
                {#if !predictionPage.isClassLevel}
                  <div class="mb-2">
                    {prediction.predictions[modelID].pred}
                  </div>
                {/if}
                <div class="text-sm mb-1">
                  {#if hasRawValues && scoreValue.value != undefined}
                    {scoreFormat(scoreValue.value)}
                    <span class="text-gray-500 text-xs"
                      >&nbsp;{comparisonName}</span
                    >
                  {/if}
                </div>
                <TableCellBar
                  visible={modelID != baseModel || showRawValues}
                  value={showRawValues
                    ? scoreValue.value
                    : scoreValue.difference}
                  scale={(v) =>
                    v / Math.max(Math.abs(maxScore), Math.abs(minScore))}
                  originCenter={minScore < 0}
                  width={BarWidth}
                  height={8}
                  showFullBar
                  colors={colorScale}
                  labelOnBar={!showRawValues}
                  labelHidden={modelID == baseModel && !showRawValues}
                  labelFormat={getScoreDifferenceFormat(comparisonName)}
                >
                  <div
                    class="text-xs text-gray-500"
                    slot="caption"
                    let:hoveringIndex
                  >
                    {#if hoveringIndex != null}
                      {#if showRawValues}{comparisonName} is {scoreFormat(
                          scoreValue.value
                        )} for model
                      {:else}{comparisonName}
                        {#if scoreValue.difference == 0}did not change{:else}{hasRawValues
                            ? "changed by"
                            : "is"}
                          {getScoreDifferenceFormat(comparisonName)(
                            scoreValue.difference
                          )}{/if} compared to base
                      {/if}
                    {/if}
                  </div>
                </TableCellBar>
              </div>
            {/each}
          </div>
        {/each}
      {/if}
      {#if predictionsToDisplay != null}
        <div
          class="w-full flex justify-center items-center pt-2 pb-4 bg-white rounded-b-lg mb-2"
          style="min-width: {minRowWidth}px;"
        >
          <button
            class="mx-3 secondary-btn"
            disabled={pageNumber == 0}
            on:click={(e) => (pageNumber -= 1)}>Previous</button
          >
          <div class="text-sm">
            {classLevel ? "Classes" : "Instances"}
            {pageNumber * numInstancesPerPage + 1} - {Math.min(
              predictionsToDisplay.length,
              (pageNumber + 1) * numInstancesPerPage + 1
            )} of {predictionsToDisplay.length}
          </div>
          <button
            class="mx-3 secondary-btn"
            disabled={(pageNumber + 1) * numInstancesPerPage >=
              predictionsToDisplay.length}
            on:click={(e) => (pageNumber += 1)}>Next</button
          >
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  tr {
    height: 1px;
  }

  .header-row {
    z-index: 1;
    vertical-align: top;
  }

  .header-sticky {
    z-index: 10;
    position: sticky;
    position: -webkit-sticky;
    top: 0;
  }

  .model-column {
    width: 260px;
    max-width: 400px;
    flex-shrink: 0;
  }

  .instance-column {
    max-width: 60vw;
  }
</style>
