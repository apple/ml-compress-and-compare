<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import { SocketDataSource, getModelServerURL } from "$lib/datasource";
  import { traitlet } from "$lib/traitlets";
  import { onMount } from "svelte";
  import type { TensorTree } from "$lib/data_structures/tensor_tree";
  import TensorTreeBrowser from "$lib/views/tensor_comparison/TensorTreeBrowser.svelte";
  import { page } from "$app/stores";
  import { browser } from "$app/environment";
  import Icon from "$lib/icons/Icon.svelte";
  import iconTrash from "$lib/icons/icon-trash.svg";

  const url = browser && $page.url;

  let dataSource = new SocketDataSource(getModelServerURL, "/model_detail");

  let connected = false;

  let hoveredModelID: string | null = null;

  onMount(async () => {
    dataSource.connect();
    dataSource.onAttach(() => {
      connected = true;
      $errorMessage = "";
    });
    dataSource.onDetach((_: any, error: any) => {
      connected = false;
      if (!!error && error.message.length > 0)
        $errorMessage = `The connection to the server has been lost: ${error.message}`;
      else
        $errorMessage =
          "The connection to the server has been lost. Please reload the page to reconnect.";
    });
  });

  let modelList = traitlet<string[]>(dataSource, "model_list", []);
  let comparisonModels = traitlet<string[]>(
    dataSource,
    "comparison_models",
    [],
    { fetchInitial: false }
  );
  let baseModel = traitlet<string>(dataSource, "base_model", "", {
    fetchInitial: false,
  });
  let errorMessage = traitlet<string>(dataSource, "error_message", "");
  let loadingMessage = traitlet<string>(dataSource, "loading_message", "");
  let loadingProgress = traitlet<number>(dataSource, "loading_progress", 1);
  let tensorType = traitlet<string>(
    dataSource,
    "tensor_type",
    "sparsity_mask",
    { fetchInitial: false }
  );
  let tensorTree = traitlet<TensorTree>(dataSource, "tensor_tree", {
    nodes: {},
    type: "sparsity_mask",
  });

  $: if ($comparisonModels.length == 0 && $baseModel.length == 0 && !!url) {
    let base = url.searchParams.get("base");
    let modelsToCompare = url.searchParams.get("models");
    if (!!modelsToCompare && !!base) {
      $comparisonModels = modelsToCompare.split(",");
      $baseModel = base;
    }
  }

  function requestTensors() {
    dataSource.request("update_tensors", {
      comparison_models: $comparisonModels,
      base_model: $baseModel,
      tensor_type: $tensorType,
    });
  }

  $: if (!!url) {
    let type = url.searchParams.get("tensorType");
    if (!!type) $tensorType = type;
    else writeTensorTypeToURL();
  }

  $: if (!!url && url.searchParams.get("tensorType") != $tensorType)
    writeTensorTypeToURL($tensorType);

  function writeTensorTypeToURL(type: string | null = null) {
    type = type || $tensorType;
    const url = new URL(window.location.toString());
    url.searchParams.set("tensorType", type);
    window.history.pushState(null, "", url.toString());
  }

  let oldBaseModel = "";
  $: if (oldBaseModel != $baseModel) {
    if (oldBaseModel.length > 0) {
      const url = new URL(window.location.toString());
      url.searchParams.set("base", $baseModel);
      window.history.pushState(null, "", url.toString());
    }
    oldBaseModel = $baseModel;
  }

  $: if (
    connected &&
    $comparisonModels.length > 0 &&
    $comparisonModels.includes($baseModel)
  ) {
    // move base model to the beginning
    let index = $comparisonModels.indexOf($baseModel);
    $comparisonModels = [
      $baseModel,
      ...$comparisonModels.slice(0, index),
      ...$comparisonModels.slice(index + 1),
    ];
  }

  let nodeFilterText: string = "";
  let nodeFilterError: string = "";
  let nodeFilter: string = "";

  $: if (nodeFilterText.length > 0) {
    nodeFilterError = "";
    try {
      new RegExp(nodeFilterText);
      nodeFilter = nodeFilterText;
    } catch (e) {
      nodeFilterError = "Invalid regular expression";
    }
  } else {
    nodeFilter = nodeFilterText;
    nodeFilterError = "";
  }

  function removeComparisonModel(index: number) {
    $comparisonModels = [
      ...$comparisonModels.slice(0, index),
      ...$comparisonModels.slice(index + 1),
    ];
    if (!$comparisonModels.includes($baseModel)) {
      $baseModel = $comparisonModels[0];
    }
    $tensorTree = { nodes: {}, type: $tensorType };
    const url = new URL(window.location.toString());
    url.searchParams.set("models", $comparisonModels.join(","));
    url.searchParams.set("base", $baseModel);
    window.history.pushState(null, "", url.toString());
  }
</script>

<div class="pt-12 flex w-full h-full bg-gray-100 items-stretch">
  <div class="w-1/5 p-4 rounded-lg bg-white m-2" style="min-width: 240px;">
    <div class="text-normal font-bold mb-2">
      Comparing {$comparisonModels.length} Models
    </div>
    {#each $comparisonModels as modelID}
      <div
        class="rounded-lg bg-gray-100 mb-2 p-2 font-mono text-sm text-gray-700 break-all flex items-center"
        class:bg-gray-200={hoveredModelID == modelID}
        on:mouseenter={() => (hoveredModelID = modelID)}
        on:mouseleave={() => (hoveredModelID = null)}
      >
        <div class="flex-auto">
          {modelID}
        </div>
        <button
          class="ml-2 hover:opacity-50 w-8 h-8 shrink-0 grow-0"
          on:click={() => removeComparisonModel(i)}
          ><Icon src={iconTrash} alt="Remove model from comparison" /></button
        >
      </div>
    {/each}
    <div class="text-xs text-gray-500">
      To modify the selection, return to the Compression Overview.
    </div>
    <div class="font-bold text-sm mt-4 mb-2">Base model:</div>
    <select
      class="w-full mb-4"
      bind:value={$baseModel}
      disabled={$loadingMessage.length > 0}
    >
      {#each $comparisonModels as modelID}
        <option value={modelID}>{modelID}</option>
      {/each}
    </select>
    <div class="font-bold text-sm mb-2">Visualize:</div>
    <select
      class="w-full mb-4"
      bind:value={$tensorType}
      disabled={$loadingMessage.length > 0}
    >
      <option value="sparsity_mask">Sparsity</option>
      <option value="weight_changes">Weight Changes</option>
      <option value="activations">Activations</option>
    </select>
  </div>
  {#if $errorMessage.length > 0}
    <div class="flex-auto flex flex-col items-center justify-center">
      <p class="w-1/2 text-center text-lg">
        {#if !connected}Model server could not be reached{:else}Error retrieving
          layer information{/if}
      </p>
      <p class="w-1/2 text-center text-gray-500 text-sm">{$errorMessage}</p>
      <button class="primary-btn mt-6" on:click={() => window.location.reload()}
        >Retry</button
      >
    </div>
  {:else if (!!$tensorTree && !!$tensorTree.nodes && Object.keys($tensorTree.nodes).length > 0) || $loadingMessage.length > 0}
    <div class="flex-auto flex flex-col">
      <div class="h-12 px-3 py-1 flex items-center">
        <div class="font-bold shrink-0">Filter modules:</div>
        <input
          type="text"
          class="w-32 ml-2 mr-4 font-mono rounded p-1 border {nodeFilterError
            ? 'border-red-500'
            : 'border-gray-300'}"
          bind:value={nodeFilterText}
          placeholder="RegEx"
        />
        {#if nodeFilterError}
          <div class="text-red-500 text-sm">{nodeFilterError}</div>
        {/if}
        <div class="flex-auto" />
        {#if $loadingMessage.length > 0 && $errorMessage.length == 0}
          <div class="mr-2 text-sm text-gray-600 truncate">
            {$loadingMessage}
          </div>
          <div role="status" class="flex items-center h-full">
            {#if $loadingProgress > 0.0 && $loadingProgress < 1.0}
              <div class="w-32 bg-gray-400 rounded-full h-1.5 mr-2">
                <div
                  class="bg-blue-600 h-1.5 rounded-full transition-all duration-200"
                  style="width: {$loadingProgress * 100}%;"
                />
              </div>
            {:else}
              <svg
                aria-hidden="true"
                class="inline w-4 h-4 mr-2 text-gray-400 animate-spin fill-blue-600"
                viewBox="0 0 100 101"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
                  fill="currentColor"
                />
                <path
                  d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
                  fill="currentFill"
                />
              </svg>
            {/if}
          </div>
          <button
            class="secondary-btn"
            on:click={(e) => {
              dataSource.request("stop_updating");
            }}>Cancel</button
          >
        {/if}
      </div>
      {#if !!$tensorTree && !!$tensorTree.nodes && Object.keys($tensorTree.nodes).length > 0}
        <div class="relative flex-auto w-full bg-gray-100">
          <div class="absolute top-2 left-2 right-2 bottom-0 overflow-scroll">
            <TensorTreeBrowser
              bind:hoveredModelID
              baseModel={$baseModel}
              tensorTree={$tensorTree}
              modelList={$comparisonModels}
              {nodeFilter}
            />
          </div>
        </div>
      {/if}
    </div>
  {:else if $loadingMessage.length == 0}
    <div
      class="flex-auto h-full flex flex-col items-center justify-center bg-gray-100"
    >
      <p class="w-1/2 text-center text-lg">No data yet</p>
      <p
        class="w-1/2 text-center text-gray-500 text-sm"
        style="max-width: 480px;"
      >
        Layer information for the selected comparison has not been loaded. Click
        below to fetch it from the model server.
      </p>
      <button class="primary-btn mt-6" on:click={requestTensors}
        >Load Layers</button
      >
    </div>
  {/if}
</div>
