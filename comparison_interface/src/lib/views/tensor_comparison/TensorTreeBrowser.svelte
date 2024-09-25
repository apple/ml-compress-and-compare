<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import Icon from "$lib/icons/Icon.svelte";
  import sortNone from "$lib/icons/sort-none.svg";
  import sortAsc from "$lib/icons/sort-asc.svg";
  import sortDesc from "$lib/icons/sort-desc.svg";
  import TensorTreeGroup from "./TensorTreeGroup.svelte";
  import { ColumnWidths } from "$lib/views/column_widths";
  import type { TensorTree } from "$lib/data_structures/tensor_tree";
  import * as d3 from "d3";
  import { LightBlue, SortDirection, SystemBlue } from "$lib/utils";

  export let baseModel: string;
  export let modelList: string[] = [];
  export let tensorTree: TensorTree;

  export let colorScale = [LightBlue, SystemBlue]; // d3.schemeBlues[3].slice(1); // ["#93c5fd", "#f472b6"]; // d3.schemeSet2;
  export let selectedID: string | null = null;

  export let nodeFilter: string = "";

  export let hoveredModelID: string | null = null;

  let sortModel: string | null = null;
  let sortDirection: SortDirection = SortDirection.none;

  let pageNumber = 0; // paginate results when sorted
  const numNodesPerPage = 20;

  let barScale: (x: number) => number = (x) => x;

  $: if (!!tensorTree && tensorTree.type == "sparsity_mask") {
    barScale = (x) => x;
  }

  function sortNodes(
    ids: string[],
    sortModel: string,
    sortDirection: SortDirection
  ): string[] {
    let idsWithValues = ids.map((id): [string, number] => {
      let modelData = tensorTree.nodes[id].data[sortModel];
      if (tensorTree.type == "sparsity_mask") {
        return [id, modelData.num_zeros / modelData.num_parameters];
      } else if (tensorTree.type == "weight_changes") {
        if (sortModel == baseModel)
          return [id, modelData.sum / modelData.num_parameters];
        else return [id, modelData.difference_sum / modelData.num_parameters];
      } else if (tensorTree.type == "activations") {
        if (!modelData) return [id, NaN];
        else if (sortModel == baseModel)
          return [id, modelData.sum / modelData.num_activations];
        else return [id, modelData.difference_sum / modelData.num_activations];
      }
      return [id, 0];
    });
    return idsWithValues
      .sort((a, b) =>
        sortDirection == SortDirection.ascending ? a[1] - b[1] : b[1] - a[1]
      )
      .map((v) => v[0]);
  }

  function changeSort(model: string) {
    if (sortModel != model) sortDirection = 0;
    sortModel = model;
    sortDirection = (sortDirection + 1) % 3;
    if (sortDirection == SortDirection.none) sortModel = null;
    console.log(sortModel, sortDirection);
  }

  function findRoots(
    tree: TensorTree,
    filter: string,
    sortModel: string | null,
    sortDirection: SortDirection
  ): string[] {
    if (filter.length > 0) {
      let pattern = new RegExp(filter, "i");
      let allMatchingNodes = new Set(
        Object.keys(tree.nodes).filter((id) => id.search(pattern) >= 0)
      );
      if (sortModel != null && sortDirection != SortDirection.none) {
        return sortNodes(
          Array.from(allMatchingNodes),
          sortModel,
          sortDirection
        );
      }
      // Remove nodes whose ancestors are also in the results
      return Array.from(allMatchingNodes).filter((id) => {
        let curr = tree.nodes[id].parent;
        while (!!curr && !!tree.nodes[curr]) {
          if (allMatchingNodes.has(curr)) return false;
          curr = tree.nodes[curr].parent;
        }
        return true;
      });
    }
    if (sortModel != null && sortDirection != SortDirection.none) {
      return sortNodes(Object.keys(tree.nodes), sortModel, sortDirection);
    }
    return Object.entries(tree.nodes)
      .filter(([id, n]) => !n.parent)
      .map(([id, n]) => id);
  }

  let rootNodes: string[];
  let rootElements: any[] = [];

  $: if (!!tensorTree && !!tensorTree.nodes)
    rootNodes = findRoots(tensorTree, nodeFilter, sortModel, sortDirection);
</script>

<div
  class="min-w-full min-h-full flex justify-center"
  on:click={(e) => (selectedID = null)}
  on:keypress={(e) => {
    if (e.key === "Enter") selectedID = null;
  }}
  role="button"
  tabindex="0"
>
  <div style="min-width: 60%;">
    <div
      class="text-left flex items-stretch header-row px-2 border-b pane-border bg-gray-100 text-sm"
    >
      <div
        class="button-menu rounded-tl-lg tensor-name h-full shrink-0 flex-auto"
        style="min-width: {ColumnWidths.TensorName +
          ColumnWidths.DisclosureArrow}px;"
      >
        <div class="p-2 w-full h-full font-bold">Module</div>
      </div>
      {#each modelList as modelID, i}
        <div
          class="p-2 bg-gray-100 h-full shrink-0 text-sm {modelID == baseModel
            ? 'border-r border-gray-600 border-dashed'
            : ''}"
          class:rounded-tr-lg={i == modelList.length - 1}
          style="width: {ColumnWidths.TensorMetric}px;"
        >
          <div class="flex items-center w-full">
            <div
              class="font-mono flex-auto truncate"
              class:font-bold={hoveredModelID == modelID}
              on:mouseenter={() => (hoveredModelID = modelID)}
              on:mouseleave={() => (hoveredModelID = null)}
              title={modelID}
            >
              {modelID}
            </div>
            <button
              class="hover:opacity-70 mr-1 shrink-0"
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
          </div>
          <div class="text-sm text-gray-600">
            {#if tensorTree.type == "sparsity_mask"}
              Sparsity
            {:else if tensorTree.type == "weight_changes"}
              Weight distribution
            {:else if tensorTree.type == "activations"}
              Activations
            {/if}
            {#if modelID != baseModel}
              vs. base
            {/if}
          </div>
        </div>
      {/each}
    </div>
    <div class="pb-1 bg-white" />
    {#if !!rootNodes}
      {#each !!sortModel ? rootNodes.slice(pageNumber * numNodesPerPage, (pageNumber + 1) * numNodesPerPage) : rootNodes as id, i (id)}
        <TensorTreeGroup
          {baseModel}
          {modelList}
          {tensorTree}
          {colorScale}
          showFullName
          showChildren={sortModel == null}
          moduleID={id}
          bind:selectedID
          bind:this={rootElements[i]}
          on:arrowdown={() =>
            (selectedID = rootNodes[(i + 1) % rootNodes.length])}
          on:arrowup={() =>
            (selectedID =
              rootElements[(i + rootNodes.length - 1) % rootNodes.length]
                .lowestVisibleID)}
        />
      {/each}
      {#if !!sortModel}
        <div class="w-full flex justify-center items-center pt-2 pb-2 bg-white">
          <button
            class="mx-3 secondary-btn"
            disabled={pageNumber == 0}
            on:click={(e) => (pageNumber -= 1)}>Previous</button
          >
          <div class="text-sm">
            Modules {pageNumber * numNodesPerPage + 1} - {(pageNumber + 1) *
              numNodesPerPage +
              1} of {rootNodes.length}
          </div>
          <button
            class="mx-3 secondary-btn"
            disabled={(pageNumber + 1) * numNodesPerPage >= rootNodes.length}
            on:click={(e) => (pageNumber += 1)}>Next</button
          >
        </div>
      {/if}
    {/if}
    <div class="pb-1 bg-white rounded-b-lg" />
  </div>
</div>

<style>
  .header-row {
    height: 60px;
    white-space: nowrap;

    position: sticky;
    position: -webkit-sticky;
    top: 0;
    z-index: 1;
  }

  .tensor-name {
    width: 400px;
    max-width: 60vw;
  }
</style>
