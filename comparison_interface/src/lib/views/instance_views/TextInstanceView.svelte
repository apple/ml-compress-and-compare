<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import type {
    InstanceData,
    InstancePrediction,
  } from "$lib/data_structures/instance_predictions";

  export let prediction: InstancePrediction | null = null;
  export let instances: { [key: string]: InstanceData } | null = null;
</script>

{#if !!prediction}
  <div class="text-sm p-2">
    <p class="mb-2">
      <span class="font-mono text-gray-500 mr-2">{prediction.id}</span>
      <span class="text-gray-800"
        >{!!instances && !!instances[prediction.id]
          ? instances[prediction.id].text
          : "Loading..."}</span
      >
    </p>
    <p><span class="font-bold">Label:</span> {prediction.label}</p>
  </div>
{/if}

<style>
</style>
