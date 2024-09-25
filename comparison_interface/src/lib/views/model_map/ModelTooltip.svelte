<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->
  
<!--
  @component
  Generates a hover tooltip. It creates a slot with an exposed variable via `let:detail` that contains information about the event. Use the slot to populate the body of the tooltip using the exposed variable `detail`.
 -->
<script lang="ts">
  import ModelInfoTable from "./ModelInfoTable.svelte";
  import type {
    Model,
    CompressionOperation,
    CompressionOperationSpec,
    OperationParameterSpec,
    Metric,
  } from "$lib/data_structures/model_map";
  import * as d3 from "d3";

  export let operations: CompressionOperationSpec[] = [];
  export let metrics: Metric[] = [];
  export let models: Model[] = [];

  export let hoverInfo: {
    xPosition: number;
    yPosition: number;
    model: Model;
  } | null = null;
  export let offset: number = -20;
</script>

{#if hoverInfo != null}
  <div
    class="tooltip absolute text-sm bg-white/90 border border-gray-300 rounded-lg p-2 z-10"
    style="
        top:{hoverInfo.yPosition + offset}px;
        left:{hoverInfo.xPosition}px;
        "
  >
    <div class="pb-2 font-mono font-bold">
      {!!hoverInfo.model.tag ? hoverInfo.model.tag : hoverInfo.model.id}
    </div>
    <ModelInfoTable model={hoverInfo.model} {operations} {metrics} {models} />
  </div>
{/if}

<style>
  .tooltip {
    width: 350px;
    transform: translate(-50%, -100%);
  }
</style>
