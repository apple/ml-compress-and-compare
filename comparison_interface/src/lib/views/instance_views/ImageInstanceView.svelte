<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import { onDestroy } from "svelte";
  import type {
    InstanceData,
    InstancePrediction,
  } from "$lib/data_structures/instance_predictions";
  import { b64toBlob } from "$lib/utils";

  export let prediction: InstancePrediction | null = null;
  export let instances: { [key: string]: InstanceData } | null = null;

  let expanded = false;

  let imageBlob: Blob | null = null;
  let instanceURL: string | null = null;
  $: if (
    !!prediction &&
    !!instances &&
    !!instances[prediction.id] &&
    instances[prediction.id].image != undefined
  ) {
    imageBlob = b64toBlob(
      instances[prediction.id].image!,
      instances[prediction.id].imageFormat || "image/png"
    );
    instanceURL = URL.createObjectURL(imageBlob);
  } else {
    instanceURL = null;
  }
</script>

{#if !!prediction}
  <div class="text-sm p-2 flex">
    {#if !!instanceURL}
      <button
        class="mr-2 hover:opacity-70 transition-all duration-200"
        class:expanded
        on:click={(e) => (expanded = !expanded)}
      >
        <img
          width="100%"
          height="100%"
          src={instanceURL}
          alt="Image for instance {prediction.id}"
        />
      </button>
    {/if}
    <div class="flex-auto">
      <p class="mb-2">
        <span class="font-mono text-gray-500 mr-2">{prediction.id}</span>
      </p>
      <p><span class="font-bold">Label:</span> {prediction.label}</p>
    </div>
  </div>
{/if}

<style>
  button {
    height: 64px;
    flex-basis: 64px;
  }

  .expanded {
    height: 240px;
    flex-basis: 240px;
  }
</style>
