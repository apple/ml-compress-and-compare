<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import Icon from "$lib/icons/Icon.svelte";
  import Chevron from "$lib/icons/icon-chevron.svg";
  import { createEventDispatcher } from "svelte";
  import type { ModelTable } from "$lib/data_structures/model_table";

  const dispatch = createEventDispatcher();

  export let modelTable: ModelTable | null = null;

  export let filteredModelIDs: string[] | null = null;
  export let allowedValues: Map<number, any[]> = new Map();

  export let expanded = false;

  $: if (!!modelTable) {
    let allowedModels = modelTable.modelIDs().filter((id) => {
      let values = modelTable!.getModelValues(id);
      return Array.from(allowedValues.entries()).every(
        ([variableIndex, allowedList]) =>
          allowedList.length == 0 || allowedList.includes(values[variableIndex])
      );
    });
    if (allowedModels.length == modelTable.size) filteredModelIDs = null;
    else filteredModelIDs = allowedModels;
    console.log("allowed models:", filteredModelIDs);
  } else {
    filteredModelIDs = null;
  }

  function modelsFilteredByVariable(
    variableIndex: number,
    allowedList: any[]
  ): string[] {
    return modelTable.modelIDs().filter((id) => {
      let values = modelTable!.getModelValues(id);
      return (
        allowedList.length == 0 || allowedList.includes(values[variableIndex])
      );
    });
  }

  function toggleVariableValue(variableIndex: number, value: any) {
    let newValues = new Map(allowedValues.entries());
    if (!newValues.has(variableIndex)) newValues.set(variableIndex, []);
    let varValues = newValues.get(variableIndex)!;
    let index = varValues.indexOf(value);
    if (index >= 0)
      newValues.set(variableIndex, [
        ...varValues.slice(0, index),
        ...varValues.slice(index + 1),
      ]);
    else newValues.set(variableIndex, [...varValues, value]);
    allowedValues = newValues;
  }

  let filtersApplied = false;
  $: if (!!modelTable) {
    filtersApplied = Array.from(allowedValues.entries()).some(
      ([varIndex, values]) => {
        let allValues = modelTable!.distinctValues(varIndex);
        return values.length > 0 && values.length < allValues.length;
      }
    );
  } else filtersApplied = false;
</script>

{#if modelTable != null}
  <div class="rounded bg-gray-100 py-2 px-3 mx-4 mb-2">
    <div class="flex">
      <button
        class="text-normal hover:opacity-50 transition-all duration-200 flex-auto text-left"
        on:click={() => (expanded = !expanded)}
      >
        <Icon
          src={Chevron}
          width="0.75rem"
          height="0.75rem"
          class="w-6 h-6 mr-1  {expanded ? 'rotate-180' : ''}"
          alt={expanded ? "Hide refinement view" : "Show refinement view"}
        />
        Refine Comparison</button
      >
      {#if filtersApplied}
        <button
          class="text-gray-600 hover:opacity-50 transition-all duration-200 text-sm mr-2"
          on:click={(e) => (allowedValues = new Map())}>Reset</button
        >
        <button class="secondary-btn" on:click={(e) => dispatch("select")}
          >Update Selection</button
        >
      {/if}
    </div>
    {#if expanded}
      {#each modelTable.variables as variable, i (i)}
        {@const allValues = modelTable.distinctValues(i)}
        <div class="my-2 w-full">
          <p class="font-bold text-sm mb-1">
            {variable.toString().length > 0 ? variable.toString() : "Operation"}
          </p>
          <div class="flex items-center w-full overflow-scroll">
            {#each allValues as val}
              <button
                class="{allowedValues.has(i) &&
                allowedValues.get(i).includes(val)
                  ? 'bg-blue-500 text-white'
                  : 'bg-blue-100 hover:bg-blue-200'} transition-all duration-200 rounded-full mr-2 px-3 py-1 shrink-0"
                on:click={(e) => toggleVariableValue(i, val)}
                on:mouseenter={() =>
                  dispatch("hover", modelsFilteredByVariable(i, [val]))}
                on:mouseleave={() => dispatch("hover", null)}
              >
                {val}
              </button>
            {/each}
          </div>
        </div>
      {/each}
    {/if}
  </div>
{/if}
