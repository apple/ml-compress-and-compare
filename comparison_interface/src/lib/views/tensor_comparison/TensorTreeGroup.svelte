<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->
  
<!-- Renders one row of the table plus potential additional child groups when expanded -->
<svelte:options accessors />

<script lang="ts">
  import TableCellBar from "$lib/views/inline_charts/TableCellBar.svelte";
  import * as d3 from "d3";

  import { ColumnWidths } from "$lib/views/column_widths";
  import type {
    TensorTree,
    TensorTreeNode,
  } from "$lib/data_structures/tensor_tree";
  import Icon from "$lib/icons/Icon.svelte";
  import ExpandArrow from "$lib/icons/icon-arrow.svg";
  import { createEventDispatcher } from "svelte";
  import TableCellHistogram from "$lib/views/inline_charts/TableCellHistogram.svelte";
  import TableCellStackedBar from "$lib/views/inline_charts/TableCellStackedBar.svelte";
  import { interpolateSystemBlue } from "$lib/utils";

  const dispatch = createEventDispatcher();

  export let baseModel: string;
  export let modelList: string[] = [];
  export let tensorTree: TensorTree;
  export let colorScale = d3.schemeBlues[2];
  export let continuousColorScale = interpolateSystemBlue; // d3.interpolateBlues;

  export let showFullName = false;

  export let moduleID: string;

  export let showChildren = true;
  export let expanded = false;
  export let selectedID: string | null = null;

  export let indentLevel = 0;

  export let lowestVisibleID: string | null = null; // used to select when navigating using arrows

  let hasChildren = false;

  let moduleInfo: TensorTreeNode | null = null;
  $: if (
    !!moduleID &&
    !!tensorTree &&
    !!tensorTree.nodes &&
    tensorTree.nodes.hasOwnProperty(moduleID)
  ) {
    moduleInfo = tensorTree.nodes[moduleID];
    hasChildren = moduleInfo.children.length > 0;
  } else {
    moduleInfo = null;
    hasChildren = false;
  }

  let moduleDisplayName = "";
  $: if (!!moduleID && !!moduleInfo) {
    let parent = moduleInfo!.parent;
    if (parent && !showFullName)
      moduleDisplayName = moduleID.replace(parent + ".", "");
    else moduleDisplayName = moduleID;
  } else moduleDisplayName = "";

  let sparsityFormat = d3.format(".1~%");
  let sparsityChangeFormat = d3.format("+.1~%");
  let weightPercentFormat = d3.format(".3~%");
  let weightChangeFormat = d3.format(".0~%");
  let weightFormat = d3.format(".3~g");
  let numParametersFormat = (v: number) =>
    d3.format(".3~s")(v).replace("G", "B").replace("T", "T");

  function select() {
    if (selectedID == moduleID) selectedID = null;
    else selectedID = moduleID;
    expanded = showChildren && hasChildren && !expanded;
  }

  function handleKey(e: KeyboardEvent) {
    if (e.key === "Enter") select();
    else if (e.key === "ArrowLeft") {
      expanded = false;
      e.preventDefault();
    } else if (e.key === "ArrowRight") {
      expanded = hasChildren;
      e.preventDefault();
    } else if (e.key === "ArrowDown") {
      if (expanded) selectedID = moduleInfo!.children[0];
      else dispatch("arrowdown");
      e.preventDefault();
    } else if (e.key === "ArrowUp") {
      dispatch("arrowup");
      e.preventDefault();
    }
  }

  let row: HTMLElement;
  $: if (!!row && selectedID == moduleID) row.focus();

  let childRows: any[] = [];
  $: if (childRows.length > 0 && !!childRows[childRows.length - 1])
    lowestVisibleID = childRows[childRows.length - 1].lowestVisibleID;
  else lowestVisibleID = moduleID;
</script>

<div class="tensor-row px-2 bg-white">
  <div
    class="rounded flex items-stretch cursor-pointer {selectedID == moduleID
      ? 'bg-blue-100'
      : 'hover:bg-gray-100'}"
    on:click|stopPropagation={select}
    on:keydown|stopPropagation={handleKey}
    role="cell"
    tabindex="0"
    bind:this={row}
  >
    <div
      class="flex items-center shrink-0 flex-1 tensor-name"
      style="min-width: {ColumnWidths.TensorName +
        ColumnWidths.DisclosureArrow}px;"
    >
      <div
        class="indent-space shrink-0"
        style="width: {indentLevel * ColumnWidths.Indent}px;"
      />
      <div class="shrink-0" style="width: {ColumnWidths.DisclosureArrow}px;">
        {#if hasChildren && showChildren}
          <div
            class="disclosure-arrow p-1.5 text-sm cursor-pointer shrink-0"
            on:click|stopPropagation={() =>
              (expanded = hasChildren && !expanded)}
            on:keydown={handleKey}
            role="cell"
            tabindex="0"
          >
            <div
              class:rotate-90={expanded}
              class="transition-all duration-200 hover:opacity-50 w-full h-full"
            >
              <Icon
                src={ExpandArrow}
                class="w-full h-full"
                alt={expanded ? "Hide children" : "Show children"}
              />
            </div>
          </div>
        {/if}
      </div>
      <div
        class="py-2 pr-2 pl-1 font-mono text-sm hover:bg-opacity-50 flex-auto"
      >
        {moduleDisplayName}
        {#if !!moduleInfo && !!baseModel && moduleInfo.data[baseModel] && !!moduleInfo.data[baseModel].num_parameters}
          <span class="ml-2 text-gray-400">
            {numParametersFormat(moduleInfo.data[baseModel].num_parameters)} parameters
          </span>
        {:else if !!moduleInfo && !!baseModel && tensorTree.type == "activations"}
          <span class="ml-2 text-gray-400">
            {#if !!moduleInfo.data[baseModel] && !!moduleInfo.data[baseModel].num_activations}
              {numParametersFormat(moduleInfo.data[baseModel].num_activations)} activations
            {:else if moduleInfo.children.length > 0}
              No activations, {moduleInfo.children.length} child{moduleInfo
                .children.length != 1
                ? "ren"
                : ""}
            {/if}
          </span>
        {/if}
      </div>
    </div>
    {#if !!moduleInfo}
      {#if !!baseModel}
        {@const modelData = (moduleInfo.data || {})[baseModel]}
        <div
          class="py-2 px-2 tensor-metric shrink-0 border-r border-gray-600 border-dashed"
          style="width: {ColumnWidths.TensorMetric}px;"
        >
          {#if tensorTree.type == "sparsity_mask"}
            <TableCellBar
              values={[modelData.num_zeros / modelData.num_parameters]}
              width={ColumnWidths.TensorMetric - 24}
              height={8}
              hoverable
              showFullBar
              colors={colorScale}
            >
              <div
                class="text-xs text-gray-500"
                slot="caption"
                let:hoveringIndex
              >
                {#if hoveringIndex == 0}
                  {sparsityFormat(
                    modelData.num_zeros / modelData.num_parameters
                  )}
                  zero in <span class="font-mono">{baseModel}</span>
                {:else if hoveringIndex == -1}
                  {sparsityFormat(
                    (modelData.num_parameters - modelData.num_zeros) /
                      modelData.num_parameters
                  )} nonzero in <span class="font-mono">{baseModel}</span>
                {:else}
                  {sparsityFormat(
                    modelData.num_zeros / modelData.num_parameters
                  )}
                  zeros
                {/if}
              </div>
            </TableCellBar>
          {:else if tensorTree.type == "weight_changes"}
            {@const hist = modelData.weight_histogram}
            <TableCellHistogram
              width={ColumnWidths.TensorMetric - 24}
              height={selectedID == moduleID ? 64 : 22}
              histValues={Object.fromEntries(
                hist.values.map((v, i) => [
                  hist.bins[i].toString(),
                  v / modelData.num_parameters,
                ])
              )}
            >
              <div
                class="text-xs text-gray-500"
                slot="caption"
                let:hoveringIndex
              >
                {#if hoveringIndex != null}
                  {weightPercentFormat(
                    hist.values[hoveringIndex] / modelData.num_parameters
                  )}
                  between {weightFormat(hist.bins[hoveringIndex])} and {weightFormat(
                    hist.bins[hoveringIndex + 1]
                  )}
                {:else}
                  L1: {weightFormat(modelData.sum / modelData.num_parameters)}
                {/if}
              </div>
            </TableCellHistogram>
          {:else if tensorTree.type == "activations"}
            {#if moduleInfo.has_activations}
              {@const hist = modelData.histogram}
              <TableCellHistogram
                width={ColumnWidths.TensorMetric - 24}
                height={selectedID == moduleID ? 64 : 22}
                histValues={Object.fromEntries(
                  hist.values.map((v, i) => [
                    hist.bins[i].toString(),
                    v / modelData.num_activations,
                  ])
                )}
              >
                <div
                  class="text-xs text-gray-500"
                  slot="caption"
                  let:hoveringIndex
                >
                  {#if hoveringIndex != null}
                    {weightPercentFormat(
                      hist.values[hoveringIndex] / modelData.num_activations
                    )}
                    between {weightFormat(hist.bins[hoveringIndex])} and {weightFormat(
                      hist.bins[hoveringIndex + 1]
                    )}
                  {:else}
                    L1: {weightFormat(
                      modelData.sum / modelData.num_activations
                    )}
                  {/if}
                </div>
              </TableCellHistogram>
            {/if}
          {/if}
        </div>
      {/if}
      {#each modelList as modelID}
        {#if modelID != baseModel}
          {@const modelData = (moduleInfo.data || {})[modelID]}
          <div
            class="py-2 px-2 tensor-metric shrink-0"
            style="width: {ColumnWidths.TensorMetric}px;"
          >
            {#if tensorTree.type == "sparsity_mask"}
              <TableCellBar
                values={[
                  modelData.zero_both / modelData.num_parameters,
                  modelData.zero_model_only / modelData.num_parameters,
                ]}
                width={ColumnWidths.TensorMetric - 24}
                height={8}
                hoverable
                showFullBar
                colors={colorScale}
              >
                <div
                  class="text-xs text-gray-500"
                  slot="caption"
                  let:hoveringIndex
                >
                  {#if hoveringIndex == 0}
                    {sparsityFormat(
                      modelData.zero_both / modelData.num_parameters
                    )}
                    zero in both <span class="font-mono">{modelID}</span> and
                    <span class="font-mono">{baseModel}</span>
                  {:else if hoveringIndex == 1}
                    {sparsityFormat(
                      modelData.zero_model_only / modelData.num_parameters
                    )} zero in <span class="font-mono">{modelID}</span> but not
                    <span class="font-mono">{baseModel}</span>
                  {:else if hoveringIndex == -1}
                    {sparsityFormat(
                      (modelData.num_parameters -
                        modelData.zero_both -
                        modelData.zero_model_only) /
                        modelData.num_parameters
                    )} nonzero in <span class="font-mono">{modelID}</span>
                  {:else}
                    {sparsityChangeFormat(
                      modelData.num_zeros / modelData.num_parameters
                    )} zeros
                  {/if}
                </div>
              </TableCellBar>
            {:else if tensorTree.type == "weight_changes"}
              {@const hist = modelData.weight_difference_histogram}
              <TableCellStackedBar
                histValues={{
                  bins: [hist.bins[0], hist.bins[1].slice(0, -1).reverse()],
                  values: hist.values.map((row) => {
                    return row
                      .map((v) => v / modelData.num_parameters)
                      .slice()
                      .reverse();
                  }),
                }}
                width={ColumnWidths.TensorMetric - 24}
                height={selectedID == moduleID ? 64 : 22}
                colorMap={continuousColorScale}
                zMax={1}
              >
                <div
                  class="text-xs text-gray-500"
                  slot="caption"
                  let:hoveringItem
                >
                  {#if !!hoveringItem}
                    {#if hoveringItem.y != undefined}
                      Value {weightFormat(hoveringItem.x)}, change by {weightChangeFormat(
                        hoveringItem.y
                      )}: {weightPercentFormat(hoveringItem.proportion)}
                    {:else}
                      {@const binIndex = hist.bins[0].indexOf(hoveringItem.x)}
                      {weightPercentFormat(
                        hist.values[binIndex].reduce((a, b) => a + b, 0) /
                          modelData.num_parameters
                      )}
                      between {weightFormat(hist.bins[0][binIndex])} and {weightFormat(
                        hist.bins[0][binIndex + 1]
                      )}
                    {/if}
                  {:else}
                    &Delta;: {weightFormat(
                      modelData.difference_sum / modelData.num_parameters
                    )}
                  {/if}
                </div>
              </TableCellStackedBar>
            {:else if tensorTree.type == "activations"}
              {#if moduleInfo.has_activations}
                {@const hist = modelData.difference_histogram}
                <TableCellStackedBar
                  histValues={{
                    bins: [hist.bins[0], hist.bins[1].slice(0, -1).reverse()],
                    values: hist.values.map((row) => {
                      return row
                        .map((v) => v / modelData.num_activations)
                        .slice()
                        .reverse();
                    }),
                  }}
                  width={ColumnWidths.TensorMetric - 24}
                  height={selectedID == moduleID ? 64 : 22}
                  colorMap={continuousColorScale}
                  zMax={1}
                >
                  <div
                    class="text-xs text-gray-500"
                    slot="caption"
                    let:hoveringItem
                  >
                    {#if !!hoveringItem}
                      {#if hoveringItem.y != undefined}
                        Value {weightFormat(hoveringItem.x)}, change by {weightChangeFormat(
                          hoveringItem.y
                        )}: {weightPercentFormat(hoveringItem.proportion)}
                      {:else}
                        {@const binIndex = hist.bins[0].indexOf(hoveringItem.x)}
                        {weightPercentFormat(
                          hist.values[binIndex].reduce((a, b) => a + b, 0) /
                            modelData.num_activations
                        )}
                        between {weightFormat(hist.bins[0][binIndex])} and {weightFormat(
                          hist.bins[0][binIndex + 1]
                        )}
                      {/if}
                    {:else}
                      &Delta;: {weightFormat(
                        modelData.difference_sum / modelData.num_activations
                      )}
                    {/if}
                  </div>
                </TableCellStackedBar>
              {/if}
            {/if}
          </div>
        {/if}
      {/each}
    {/if}
  </div>
</div>
{#if expanded && !!moduleInfo && showChildren}
  {#each moduleInfo.children as child, i (child)}
    <svelte:self
      {baseModel}
      {modelList}
      {tensorTree}
      moduleID={child}
      {colorScale}
      bind:selectedID
      indentLevel={indentLevel + 1}
      on:arrowdown={() => {
        if (!!moduleInfo && i == moduleInfo.children.length - 1)
          dispatch("arrowdown");
        else if (!!moduleInfo) selectedID = moduleInfo.children[i + 1];
      }}
      on:arrowup={() => {
        if (!!moduleInfo && i == 0) selectedID = moduleID;
        else if (!!moduleInfo) selectedID = childRows[i - 1].lowestVisibleID;
      }}
      bind:this={childRows[i]}
    />
  {/each}
{/if}
